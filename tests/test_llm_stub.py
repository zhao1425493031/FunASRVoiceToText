"""LLM placeholder tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voicetotext.llm.schemas import SummaryRequest, SummarySegment
from voicetotext.llm.summary import StubSummaryProvider, SummaryNotImplementedError

ROOT = Path(__file__).resolve().parents[1]


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


def test_summarize_api_501() -> None:
    os.environ["VOICETOTEXT_CONFIG"] = str(ROOT / "config.yaml")
    from voicetotext.server import batch_app as mod

    mod.init_app(ROOT / "config.yaml")
    client = TestClient(mod.app)
    r = client.post(
        "/api/v1/summarize",
        json={"job_id": "x"},
        headers={"X-API-Key": "batch-dev-8n3k1q5y"},
    )
    assert r.status_code == 501
    assert r.json()["error"] == "llm_not_implemented"
