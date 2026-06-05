"""Batch HTTP API tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch_client():
    os.environ["VOICETOTEXT_CONFIG"] = str(ROOT / "config.yaml")
    os.environ["VOICETOTEXT_SKIP_HF_CHECK"] = "1"
    from voicetotext.server import batch_app as mod

    mod.init_app(ROOT / "config.yaml")
    with patch.object(mod.pipeline, "load"), patch.object(
        mod.pipeline, "is_ready", return_value=True
    ), patch.object(mod.pipeline, "readiness_detail", return_value={}):
        yield TestClient(mod.app)


def test_batch_ready(batch_client: TestClient) -> None:
    r = batch_client.get("/ready")
    assert r.status_code == 200


def test_batch_transcribe_requires_auth(batch_client: TestClient) -> None:
    r = batch_client.post("/api/v1/transcribe", files={"file": ("a.wav", b"xx")})
    assert r.status_code == 401


def test_batch_transcribe_mock(batch_client: TestClient) -> None:
    from voicetotext.asr.batch_pipeline import BatchTranscript

    fake = BatchTranscript(
        job_id="job-1",
        language="ja",
        duration_ms=1000,
        segments=[{"start_ms": 0, "end_ms": 1000, "speaker_id": "SPEAKER_00", "text": "a"}],
        meta={},
    )
    with patch.object(
        batch_client.app.state if hasattr(batch_client.app, "state") else None,
        "pipeline",
        create=True,
    ):
        from voicetotext.server import batch_app as mod

        with patch.object(mod.pipeline, "process_file", return_value=fake), patch.object(
            mod.pipeline, "export", return_value={"json": Path("out/job-1.json")}
        ):
            r = batch_client.post(
                "/api/v1/transcribe",
                files={"file": ("a.wav", b"xx")},
                headers={"X-API-Key": "batch-dev-8n3k1q5y"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == "job-1"
    assert "text" in body


def test_batch_page(batch_client: TestClient) -> None:
    r = batch_client.get("/batch")
    assert r.status_code == 200
    assert "录音文件转写" in r.text
    assert "batch.js" in r.text
