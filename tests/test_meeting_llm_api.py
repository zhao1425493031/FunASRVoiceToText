"""HTTP meeting summarize and session GET endpoints."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from voicetotext.server import meeting_app

from tests.conftest import LLM_TEST_CONFIG, mock_llm_http_body, mock_summary_response


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _use_llm_test_config(tmp_path, monkeypatch):
    from voicetotext.config import load_config

    cfg = load_config(LLM_TEST_CONFIG)
    monkeypatch.setattr(meeting_app, "config", cfg)
    mock_provider = MagicMock()
    mock_provider.summarize.return_value = mock_summary_response("api-job")
    monkeypatch.setattr(meeting_app, "_summary_provider", mock_provider)
    yield cfg, mock_provider


@pytest.mark.asyncio
async def test_summarize_with_segments_returns_200(_use_llm_test_config) -> None:
    cfg, _provider = _use_llm_test_config
    transport = ASGITransport(app=meeting_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/meeting/summarize",
            headers={"X-API-Key": cfg.api_key},
            json={
                "language": "zh",
                "segments": [
                    {
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker_id": "SPEAKER_00",
                        "text": "讨论预算",
                    }
                ],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["summary"]["title"] == "测试会议"


@pytest.mark.asyncio
async def test_summarize_with_session_id_reads_disk(_use_llm_test_config, tmp_path) -> None:
    cfg, _provider = _use_llm_test_config
    out_dir = tmp_path / "meetings"
    out_dir.mkdir(parents=True)
    session_id = "stored-session-1"
    payload = {
        "session_id": session_id,
        "language": "zh",
        "segments": [
            {"start_ms": 0, "end_ms": 500, "speaker_id": "SPEAKER_00", "text": "已存字幕"}
        ],
    }
    (out_dir / f"{session_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    cfg = replace(cfg, llm_meeting_output_dir=str(out_dir))
    meeting_app.config = cfg

    transport = ASGITransport(app=meeting_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/meeting/summarize",
            headers={"X-API-Key": cfg.api_key},
            json={"session_id": session_id},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_summarize_empty_segments_400(_use_llm_test_config) -> None:
    cfg, _ = _use_llm_test_config
    transport = ASGITransport(app=meeting_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/meeting/summarize",
            headers={"X-API-Key": cfg.api_key},
            json={"segments": []},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_session_returns_transcript(_use_llm_test_config, tmp_path) -> None:
    cfg, _ = _use_llm_test_config
    out_dir = tmp_path / "meetings"
    out_dir.mkdir(parents=True)
    session_id = "get-session-1"
    payload = {"session_id": session_id, "language": "zh", "segments": [], "summary": None}
    (out_dir / f"{session_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    cfg = replace(cfg, llm_meeting_output_dir=str(out_dir))
    meeting_app.config = cfg

    transport = ASGITransport(app=meeting_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            f"/api/v1/meeting/sessions/{session_id}",
            headers={"X-API-Key": cfg.api_key},
        )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_summarize_llm_disabled_501(monkeypatch) -> None:
    from voicetotext.config import load_config
    import yaml

    from tests.conftest import ROOT

    raw = yaml.safe_load(LLM_TEST_CONFIG.read_text(encoding="utf-8"))
    raw["llm"]["enabled"] = False
    tmp = ROOT / "tests" / "_tmp_api_disabled.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        cfg = load_config(tmp)
        monkeypatch.setattr(meeting_app, "config", cfg)
        transport = ASGITransport(app=meeting_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/meeting/summarize",
                headers={"X-API-Key": cfg.api_key},
                json={
                    "segments": [
                        {"start_ms": 0, "end_ms": 1, "speaker_id": "SPEAKER_00", "text": "x"}
                    ]
                },
            )
        assert resp.status_code == 501
    finally:
        if tmp.is_file():
            tmp.unlink()
