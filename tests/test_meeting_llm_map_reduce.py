"""Map-reduce summary for long transcripts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import yaml

from voicetotext.config import load_config
from voicetotext.llm.factory import create_summary_provider
from voicetotext.llm.schemas import SummaryRequest, SummarySegment

from tests.conftest import LLM_TEST_CONFIG, ROOT, mock_summary_dict


def test_map_reduce_triggers_multiple_http_calls() -> None:
    raw = yaml.safe_load(LLM_TEST_CONFIG.read_text(encoding="utf-8"))
    raw["llm"]["max_input_chars"] = 80
    raw["llm"]["chunk_chars"] = 40
    tmp = ROOT / "tests" / "_tmp_map_reduce.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        cfg = load_config(tmp)
        provider = create_summary_provider(cfg)
        segments = [
            SummarySegment(
                start_ms=i * 1000,
                end_ms=(i + 1) * 1000,
                speaker_id=f"SPEAKER_{i % 2:02d}",
                text=f"这是第{i}段较长的会议发言内容用于触发分块。",
            )
            for i in range(6)
        ]
        req = SummaryRequest(job_id="long-session", language="zh", segments=segments)

        map_partial = json.dumps({"partial": "chunk summary"}, ensure_ascii=False)
        reduce_body = {
            "choices": [
                {"message": {"content": json.dumps(mock_summary_dict(), ensure_ascii=False)}}
            ]
        }

        def _side_effect(*_args, **_kwargs):
            resp = MagicMock()
            resp.status_code = 200
            nonlocal call_count
            if call_count < 2:
                resp.json.return_value = {
                    "choices": [{"message": {"content": map_partial}}]
                }
            else:
                resp.json.return_value = reduce_body
            call_count += 1
            return resp

        call_count = 0
        with patch("httpx.Client") as client_cls:
            client_cls.return_value.__enter__.return_value.post.side_effect = _side_effect
            resp = provider.summarize(req)

        assert resp.status == "ok"
        assert resp.summary is not None
        assert call_count >= 3
        assert resp.meta.get("chunks", 1) >= 2
    finally:
        if tmp.is_file():
            tmp.unlink()
