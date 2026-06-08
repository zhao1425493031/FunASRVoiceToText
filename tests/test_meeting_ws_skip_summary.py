"""skip_summary=true should not invoke LLM."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import load_config
from voicetotext.server.meeting_ws_protocol import MeetingWSProtocolHandler

from tests.conftest import LLM_TEST_CONFIG


@pytest.mark.asyncio
async def test_end_skip_summary_skips_llm(tmp_path) -> None:
    cfg = load_config(LLM_TEST_CONFIG)
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.headers = {}
    engine = create_asr_backend(cfg)
    handler = MeetingWSProtocolHandler(ws, engine, cfg)

    mock_session = MagicMock()
    mock_session.end = AsyncMock(return_value=True)
    mock_session.run_meeting_summary = AsyncMock(return_value=True)
    mock_session.close = AsyncMock()
    handler.asr_session = mock_session
    handler.started = True

    await handler._handle_end({"type": "end", "skip_summary": True})
    mock_session.end.assert_awaited_once()
    mock_session.run_meeting_summary.assert_awaited_once_with(skip_summary=True)
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_meeting_summary_skip_does_not_call_provider() -> None:
    from voicetotext.llm.meeting_transcript_buffer import MeetingTranscriptBuffer
    from voicetotext.server.meeting_ws_protocol import MeetingWSSession

    cfg = load_config(LLM_TEST_CONFIG)
    ws = MagicMock()
    ws.send_text = AsyncMock()
    engine = create_asr_backend(cfg)
    session = MeetingWSSession(cfg, ws, engine)
    session._transcript_buffer = MeetingTranscriptBuffer("skip-1", "zh")
    session._transcript_buffer.append_final(
        {"type": "final", "text": "x", "seg_id": "1", "t_start_ms": 0, "t_end_ms": 100}
    )
    mock_provider = MagicMock()
    session._summary_provider = mock_provider

    ok = await session.run_meeting_summary(skip_summary=True)
    assert ok is True
    mock_provider.summarize.assert_not_called()
    ws.send_text.assert_not_called()
