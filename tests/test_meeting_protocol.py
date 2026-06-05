"""Meeting protocol helper tests."""

from __future__ import annotations

import json

import pytest

from voicetotext.server.meeting_ws_protocol import SUPPORTED_LANGUAGES
from voicetotext.server.protocol_common import encode_message, parse_client_message


def test_parse_meeting_start() -> None:
    msg = parse_client_message(
        json.dumps(
            {
                "type": "start",
                "protocol_version": 2,
                "mode": "meeting",
                "language": "ja",
            }
        )
    )
    assert msg["protocol_version"] == 2
    assert msg["mode"] == "meeting"


def test_supported_languages() -> None:
    assert "ja" in SUPPORTED_LANGUAGES
    assert "fr" not in SUPPORTED_LANGUAGES


def test_ping_message_type() -> None:
    msg = parse_client_message(json.dumps({"type": "ping"}))
    assert msg["type"] == "ping"


def test_encode_message() -> None:
    raw = encode_message({"type": "pong", "protocol_version": 2})
    data = json.loads(raw)
    assert data["type"] == "pong"


def test_parse_session_ready() -> None:
    msg = parse_client_message(
        json.dumps({"type": "session_ready", "protocol_version": 2})
    )
    assert msg["type"] == "session_ready"
