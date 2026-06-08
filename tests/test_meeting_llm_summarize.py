"""Summary provider end-to-end with mocked HTTP."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from voicetotext.llm.factory import create_summary_provider
from voicetotext.llm.schemas import SummaryRequest, SummarySegment

from tests.conftest import LLM_TEST_CONFIG, mock_llm_http_body, mock_summary_dict


def test_summarize_provider_mock_http() -> None:
    from voicetotext.config import load_config

    cfg = load_config(LLM_TEST_CONFIG)
    provider = create_summary_provider(cfg)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_llm_http_body()

    req = SummaryRequest(
        job_id="job-1",
        language="zh",
        segments=[
            SummarySegment(
                start_ms=0,
                end_ms=1000,
                speaker_id="SPEAKER_00",
                text="今天讨论项目进度",
            )
        ],
    )
    with patch("httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        resp = provider.summarize(req)

    assert resp.status == "ok"
    assert resp.summary is not None
    assert resp.summary.title == mock_summary_dict()["title"]
    assert resp.summary.markdown
    assert resp.meta.get("provider") == "openai_compatible"
