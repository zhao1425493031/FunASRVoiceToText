"""MeetingWSProtocolHandler unit tests (no live Runtime)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import load_config
from voicetotext.server.meeting_ws_protocol import MeetingWSProtocolHandler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def meeting_config():
    return load_config(ROOT / "config.meeting.yaml")


@pytest.fixture
def handler(meeting_config):
    ws = MagicMock()
    ws.headers = {}
    engine = create_asr_backend(meeting_config)
    return MeetingWSProtocolHandler(ws, engine, meeting_config)


@pytest.mark.asyncio
async def test_start_protocol_mismatch(handler, meeting_config) -> None:
    handler.websocket.send_text = AsyncMock()
    ok = await handler.handle_text(
        json.dumps(
            {
                "type": "start",
                "protocol_version": 1,
                "mode": "meeting",
                "language": "ja",
                "api_key": meeting_config.api_key,
            }
        )
    )
    assert ok is False
    sent = handler.websocket.send_text.call_args[0][0]
    assert "protocol_mismatch" in sent


@pytest.mark.asyncio
async def test_start_unauthorized(handler, meeting_config) -> None:
    handler.websocket.send_text = AsyncMock()
    ok = await handler.handle_text(
        json.dumps(
            {
                "type": "start",
                "protocol_version": 2,
                "mode": "meeting",
                "language": "ja",
            }
        )
    )
    assert ok is False
    sent = handler.websocket.send_text.call_args[0][0]
    assert "unauthorized" in sent


@pytest.mark.asyncio
async def test_ping_pong(handler, meeting_config) -> None:
    handler.websocket.send_text = AsyncMock()
    ok = await handler.handle_text(json.dumps({"type": "ping"}))
    assert ok is True
    sent = json.loads(handler.websocket.send_text.call_args[0][0])
    assert sent["type"] == "pong"


@pytest.mark.asyncio
async def test_bytes_not_started(handler) -> None:
    handler.websocket.send_text = AsyncMock()
    await handler.handle_bytes(b"\x00" * 19200)
    sent = handler.websocket.send_text.call_args[0][0]
    assert "not_started" in sent


@pytest.mark.asyncio
async def test_invalid_chunk_size(handler, meeting_config) -> None:
    handler.websocket.send_text = AsyncMock()
    handler.started = True
    handler.asr_session = MagicMock()
    handler.asr_session.send_pcm = AsyncMock()
    await handler.handle_bytes(b"\x00" * 100)
    sent = handler.websocket.send_text.call_args[0][0]
    assert "invalid_chunk" in sent


@pytest.mark.asyncio
async def test_unsupported_language(handler, meeting_config) -> None:
    handler.websocket.send_text = AsyncMock()
    ok = await handler.handle_text(
        json.dumps(
            {
                "type": "start",
                "protocol_version": 2,
                "mode": "meeting",
                "language": "fr",
                "api_key": meeting_config.api_key,
            }
        )
    )
    assert ok is False
    assert "unsupported_language" in handler.websocket.send_text.call_args[0][0]


@pytest.mark.asyncio
async def test_start_success_mock_session(handler, meeting_config) -> None:
    handler.websocket.send_text = AsyncMock()
    with patch(
        "voicetotext.server.meeting_ws_protocol.MeetingWSSession.start",
        new_callable=AsyncMock,
    ):
        ok = await handler.handle_text(
            json.dumps(
                {
                    "type": "start",
                    "protocol_version": 2,
                    "mode": "meeting",
                    "language": "ja",
                    "api_key": meeting_config.api_key,
                }
            )
        )
    assert ok is True
    assert handler.started is True
