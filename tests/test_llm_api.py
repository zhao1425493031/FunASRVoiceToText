"""LLM HTTP API tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest_llm import mock_chat_response, sample_summary_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def llm_client():
    os.environ["VOICETOTEXT_CONFIG"] = str(FIXTURES / "config_llm_test.yaml")
    from voicetotext.server import batch_app as mod

    mod.init_app(FIXTURES / "config_llm_test.yaml")
    with patch.object(mod.pipeline, "load"), patch.object(
        mod.pipeline, "is_ready", return_value=True
    ), patch.object(mod.pipeline, "readiness_detail", return_value={"llm": {"ready": True}}):
        yield TestClient(mod.app), mod


def test_summarize_with_segments_200(llm_client) -> None:
    client, mod = llm_client
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
                "job_id": "j1",
                "language": "ja",
                "segments": [
                    {
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker_id": "SPEAKER_00",
                        "text": "お疲れ様です",
                    }
                ],
            },
            headers={"X-API-Key": "batch-dev-8n3k1q5y"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["summary"]["title"]


def test_summarize_empty_segments_400(llm_client) -> None:
    client, _ = llm_client
    r = client.post(
        "/api/v1/summarize",
        json={"job_id": "j1", "segments": []},
        headers={"X-API-Key": "batch-dev-8n3k1q5y"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "empty_segments"


def test_summarize_job_id_from_disk(llm_client, tmp_path) -> None:
    client, mod = llm_client
    job_id = "disk-job-1"
    transcript = {
        "job_id": job_id,
        "language": "ja",
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 1000,
                "speaker_id": "SPEAKER_00",
                "text": "会議開始",
            }
        ],
    }
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / f"{job_id}.json").write_text(
        json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
    )

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = mock_chat_response(sample_summary_json())

    with patch.object(mod, "PROJECT_ROOT", tmp_path), patch("httpx.Client") as mock_client_cls:
        mock_http = MagicMock()
        mock_http.__enter__.return_value = mock_http
        mock_http.post.return_value = mock_resp
        mock_client_cls.return_value = mock_http

        r = client.post(
            "/api/v1/summarize",
            json={"job_id": job_id},
            headers={"X-API-Key": "batch-dev-8n3k1q5y"},
        )
    assert r.status_code == 200


def test_summarize_upstream_502(llm_client) -> None:
    client, _ = llm_client
    mock_resp = MagicMock(status_code=500, text="boom")
    with patch("httpx.Client") as mock_client_cls:
        mock_http = MagicMock()
        mock_http.__enter__.return_value = mock_http
        mock_http.post.return_value = mock_resp
        mock_client_cls.return_value = mock_http

        r = client.post(
            "/api/v1/summarize",
            json={
                "segments": [
                    {
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker_id": "SPEAKER_00",
                        "text": "test",
                    }
                ]
            },
            headers={"X-API-Key": "batch-dev-8n3k1q5y"},
        )
    assert r.status_code == 502


def test_ready_includes_llm(llm_client) -> None:
    client, mod = llm_client
    with patch.object(
        mod.pipeline,
        "readiness_detail",
        return_value={"llm": {"enabled": True, "ready": True}},
    ):
        r = client.get("/ready")
    assert r.status_code == 200
    assert "llm" in r.json()["detail"]
