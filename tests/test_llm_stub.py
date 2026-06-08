"""LLM API and stub tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest_llm import mock_chat_response, sample_summary_json
from voicetotext.llm.schemas import SummaryRequest, SummarySegment
from voicetotext.llm.summary import StubSummaryProvider, SummaryNotImplementedError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_stub_raises() -> None:
    stub = StubSummaryProvider()
    req = SummaryRequest(job_id="j1", language="ja", segments=[])
    with pytest.raises(SummaryNotImplementedError):
        stub.summarize(req)


def test_summary_schema_serializable() -> None:
    seg = SummarySegment(0, 1000, "SPEAKER_00", "hello")
    req = SummaryRequest(job_id="j1", language="ja", segments=[seg])
    d = req.to_dict()
    assert d["job_id"] == "j1"
    assert d["segments"][0]["text"] == "hello"


def test_summarize_api_disabled_501() -> None:
    os.environ["VOICETOTEXT_CONFIG"] = str(FIXTURES / "config_llm_disabled.yaml")
    from voicetotext.server import batch_app as mod

    mod.init_app(FIXTURES / "config_llm_disabled.yaml")
    client = TestClient(mod.app)
    r = client.post(
        "/api/v1/summarize",
        json={"job_id": "x", "segments": [{"start_ms": 0, "end_ms": 1, "speaker_id": "S", "text": "a"}]},
        headers={"X-API-Key": "batch-dev-8n3k1q5y"},
    )
    assert r.status_code == 501
    assert r.json()["error"] == "llm_disabled"


def test_summarize_api_enabled_mock_200() -> None:
    os.environ["VOICETOTEXT_CONFIG"] = str(FIXTURES / "config_llm_test.yaml")
    from voicetotext.server import batch_app as mod

    mod.init_app(FIXTURES / "config_llm_test.yaml")
    client = TestClient(mod.app)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = mock_chat_response(sample_summary_json())

    with patch("httpx.Client") as mock_client_cls:
        mock_http = MagicMock()
        mock_http.__enter__.return_value = mock_http
        mock_http.post.return_value = mock_resp
        mock_client_cls.return_value = mock_http

        r = client.post(
            "/api/v1/summarize",
            json={
                "job_id": "x",
                "segments": [
                    {"start_ms": 0, "end_ms": 1000, "speaker_id": "SPEAKER_00", "text": "hello"}
                ],
            },
            headers={"X-API-Key": "batch-dev-8n3k1q5y"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
