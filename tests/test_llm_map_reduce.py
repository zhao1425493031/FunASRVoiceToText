"""Map-reduce summary tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.conftest_llm import mock_chat_response, sample_summary_json
from voicetotext.config import load_config
from voicetotext.llm.factory import create_summary_provider
from voicetotext.llm.schemas import SummaryRequest, SummarySegment

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_map_reduce_triggers_multiple_calls() -> None:
    cfg = load_config(FIXTURES / "config_llm_fail.yaml")
    provider = create_summary_provider(cfg)
    long_text = "あ" * 40
    req = SummaryRequest(
        job_id="j1",
        language="ja",
        segments=[SummarySegment(0, 1000, "SPEAKER_00", long_text)],
    )
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = mock_chat_response(sample_summary_json())

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        resp = provider.summarize(req)
        assert resp.status == "ok"
        assert mock_client.post.call_count >= 2
        assert resp.meta.get("chunks", 1) >= 1
