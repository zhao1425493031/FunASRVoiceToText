"""Tests for WebSocket protocol helpers."""

from __future__ import annotations

import json

import pytest

from voicetotext.server.ws_protocol import encode_message, parse_client_message


def test_encode_message() -> None:
    raw = encode_message({"type": "partial", "text": "你好"})
    data = json.loads(raw)
    assert data["type"] == "partial"
    assert data["text"] == "你好"


def test_parse_start_message() -> None:
    msg = parse_client_message(
        json.dumps(
            {
                "type": "start",
                "wav_name": "web_mic",
                "audio_fs": 16000,
                "wav_format": "pcm",
                "chunk_size": [0, 10, 5],
            }
        )
    )
    assert msg["type"] == "start"
    assert msg["chunk_size"] == [0, 10, 5]


def test_parse_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_client_message("not json")


def test_parse_non_object() -> None:
    with pytest.raises(ValueError):
        parse_client_message('"string"')
