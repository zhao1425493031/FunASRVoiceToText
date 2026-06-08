"""WebSocket end -> summary_progress -> meeting_summary flow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import load_config
from voicetotext.server.meeting_ws_protocol import MeetingWSSession

from tests.conftest import LLM_TEST_CONFIG, ROOT, mock_summary_response


@pytest.fixture
def ws_session(tmp_path):
    raw = yaml.safe_load(LLM_TEST_CONFIG.read_text(encoding="utf-8"))
    raw["llm"]["meeting_output_dir"] = str(tmp_path / "meetings")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(raw), encoding="utf-8")
    cfg = load_config(cfg_path)
    ws = MagicMock()
    ws.send_text = AsyncMock()
    engine = create_asr_backend(cfg)
    session = MeetingWSSession(cfg, ws, engine)
    return session, ws, cfg, tmp_path


@pytest.mark.asyncio
async def test_run_meeting_summary_sends_progress_and_ok(ws_session) -> None:
    session, ws, cfg, tmp_path = ws_session
    session._session_id = "ws-sum-1"
    session._transcript_buffer = session._transcript_buffer or __import__(
        "voicetotext.llm.meeting_transcript_buffer", fromlist=["MeetingTranscriptBuffer"]
    ).MeetingTranscriptBuffer("ws-sum-1", "zh")
    session._transcript_buffer.append_final(
        {
            "type": "final",
            "text": "会议内容",
            "seg_id": "f1",
            "t_start_ms": 0,
            "t_end_ms": 1000,
        }
    )

    mock_provider = MagicMock()
    mock_provider.summarize.return_value = mock_summary_response("ws-sum-1")
    session._summary_provider = mock_provider

    ok = await session.run_meeting_summary(skip_summary=False)
    assert ok is True
    assert mock_provider.summarize.called

    sent = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
    types = [m["type"] for m in sent]
    assert "summary_progress" in types
    assert "meeting_summary" in types
    summary_msg = next(m for m in sent if m["type"] == "meeting_summary")
    assert summary_msg["status"] == "ok"
    assert summary_msg["summary"]["title"] == "测试会议"

    out_json = Path(cfg.llm_meeting_output_path) / "ws-sum-1.json"
    assert out_json.is_file()


@pytest.mark.asyncio
async def test_run_meeting_summary_warn_on_upstream_error(ws_session) -> None:
    session, ws, _cfg, _tmp = ws_session
    session._session_id = "ws-sum-err"
    from voicetotext.llm.meeting_transcript_buffer import MeetingTranscriptBuffer

    session._transcript_buffer = MeetingTranscriptBuffer("ws-sum-err", "zh")
    session._transcript_buffer.append_final(
        {"type": "final", "text": "内容", "seg_id": "e1", "t_start_ms": 0, "t_end_ms": 500}
    )
    mock_provider = MagicMock()
    mock_provider.summarize.side_effect = RuntimeError("upstream 502")
    session._summary_provider = mock_provider

    ok = await session.run_meeting_summary(skip_summary=False)
    assert ok is True
    sent = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
    summary_msg = next(m for m in sent if m["type"] == "meeting_summary")
    assert summary_msg["status"] == "error"
    assert summary_msg["summary"] is None


@pytest.mark.asyncio
async def test_run_meeting_summary_fail_mode_raises(ws_session) -> None:
    session, ws, cfg, _tmp = ws_session
    from dataclasses import replace

    from voicetotext.llm.client import LLMServerError
    from voicetotext.llm.meeting_transcript_buffer import MeetingTranscriptBuffer

    session._session_config = replace(cfg, llm_on_failure="fail")
    session._session_id = "ws-fail-1"
    session._transcript_buffer = MeetingTranscriptBuffer("ws-fail-1", "zh")
    session._transcript_buffer.append_final(
        {"type": "final", "text": "内容", "seg_id": "f1", "t_start_ms": 0, "t_end_ms": 500}
    )
    mock_provider = MagicMock()
    mock_provider.summarize.side_effect = LLMServerError("upstream 502")
    session._summary_provider = mock_provider

    with pytest.raises(LLMServerError):
        await session.run_meeting_summary(skip_summary=False)
