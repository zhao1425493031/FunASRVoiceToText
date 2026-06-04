"""Unit tests for FunASR Runtime WebSocket client mapping."""

from __future__ import annotations

import pytest

from voicetotext.asr.runtime_client import RuntimeWSClient
from voicetotext.config import load_config


@pytest.fixture
def client():
    return RuntimeWSClient(load_config())


def test_parse_chunk_size_default(client: RuntimeWSClient) -> None:
    assert client._parse_chunk_size() == [5, 10, 5]


def test_parse_chunk_size_custom() -> None:
    cfg = load_config()
    from dataclasses import replace

    custom = replace(cfg, runtime_chunk_size="8,16,8")
    client = RuntimeWSClient(custom)
    assert client._parse_chunk_size() == [8, 16, 8]


def test_map_2pass_online_to_partial() -> None:
    mapped = RuntimeWSClient.map_runtime_to_client(
        {"mode": "2pass-online", "text": "こんにちは"}
    )
    assert mapped is not None
    assert mapped["type"] == "partial"
    assert mapped["text"] == "こんにちは"
    assert mapped["is_final"] is False


def test_map_2pass_offline_to_final() -> None:
    mapped = RuntimeWSClient.map_runtime_to_client(
        {"mode": "2pass-offline", "text": "こんにちは世界"}
    )
    assert mapped is not None
    assert mapped["type"] == "final"
    assert mapped["is_final"] is True


def test_map_empty_text_returns_none() -> None:
    assert RuntimeWSClient.map_runtime_to_client({"mode": "2pass-online", "text": ""}) is None


def test_uri_ws(client: RuntimeWSClient) -> None:
    assert client.uri.startswith("ws://")
