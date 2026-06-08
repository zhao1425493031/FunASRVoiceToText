"""Summary provider factory and provider tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest_llm import mock_chat_response, sample_summary_json
from voicetotext.config import load_config
from voicetotext.llm.factory import create_summary_provider
from voicetotext.llm.schemas import SummaryRequest, SummarySegment
from voicetotext.llm.summary import StubSummaryProvider, SummaryNotImplementedError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_factory_stub_when_disabled() -> None:
    cfg = load_config(FIXTURES / "config_llm_disabled.yaml")
    provider = create_summary_provider(cfg)
    assert isinstance(provider, StubSummaryProvider)


def test_factory_openai_compatible() -> None:
    cfg = load_config(FIXTURES / "config_llm_test.yaml")
    provider = create_summary_provider(cfg)
    assert provider.__class__.__name__ == "OpenAICompatibleSummaryProvider"


def test_factory_azure() -> None:
    import yaml

    raw = yaml.safe_load((FIXTURES / "config_llm_test.yaml").read_text(encoding="utf-8"))
    raw["llm"]["deployment"] = "gpt-4o"
    raw["llm"]["api_version"] = "2024-02-01"
    tmp = ROOT / "tests" / "_tmp_llm_azure_provider.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        cfg = load_config(tmp)
        provider = create_summary_provider(cfg)
        assert provider.__class__.__name__ == "AzureOpenAISummaryProvider"
    finally:
        tmp.unlink(missing_ok=True)


def test_openai_compatible_summarize_mock() -> None:
    cfg = load_config(FIXTURES / "config_llm_test.yaml")
    provider = create_summary_provider(cfg)
    req = SummaryRequest(
        job_id="j1",
        language="ja",
        segments=[SummarySegment(0, 1000, "SPEAKER_00", "お疲れ様です")],
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
        assert resp.summary is not None
        assert resp.summary.title


def test_stub_raises() -> None:
    cfg = load_config(FIXTURES / "config_llm_disabled.yaml")
    provider = create_summary_provider(cfg)
    req = SummaryRequest(job_id="j1", language="ja", segments=[])
    with pytest.raises(SummaryNotImplementedError):
        provider.summarize(req)
