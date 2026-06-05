"""HTTP /health and /ready endpoint tests."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from voicetotext.server import meeting_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_health_returns_200() -> None:
    transport = ASGITransport(app=meeting_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["asr_version"] == "v3"
    assert data["backend"] == "meeting_sensevoice"


@pytest.mark.asyncio
async def test_ready_503_without_hf_token() -> None:
    os.environ.pop("HF_TOKEN", None)
    meeting_app.engine._ready = True
    meeting_app.engine._vad._ready = True
    meeting_app.engine._vad._model = object()
    meeting_app.engine._asr._ready = True
    meeting_app.engine._asr._model = object()
    meeting_app.engine._pyannote._ready = True
    meeting_app.engine._pyannote._pipeline = object()

    transport = ASGITransport(app=meeting_app.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert "HF_TOKEN" in data.get("reason", "")


@pytest.mark.asyncio
async def test_ready_200_when_engine_ready() -> None:
    os.environ["HF_TOKEN"] = "hf_test_token_ok_for_ready"
    with patch.object(
        meeting_app.engine, "check_ready", new=AsyncMock(return_value=True)
    ):
        transport = ASGITransport(app=meeting_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    del os.environ["HF_TOKEN"]
