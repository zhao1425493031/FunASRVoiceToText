"""Tests for API key authentication."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi import HTTPException

from voicetotext.config import load_config
from voicetotext.server.auth import (
    extract_ws_api_key,
    require_http_api_key,
    validate_api_key,
)


def test_validate_no_key_config() -> None:
    cfg = load_config()
    assert validate_api_key(cfg, None) is True
    assert validate_api_key(cfg, "anything") is True


def test_validate_with_key_config() -> None:
    cfg = replace(load_config(), api_key="secret")
    assert validate_api_key(cfg, "secret") is True
    assert validate_api_key(cfg, "wrong") is False
    assert validate_api_key(cfg, None) is False


def test_extract_ws_api_key_from_message() -> None:
    msg = {"api_key": "from-msg"}
    assert extract_ws_api_key(msg, {}) == "from-msg"


def test_extract_ws_api_key_from_header() -> None:
    assert extract_ws_api_key({}, {"x-api-key": "hdr"}) == "hdr"


def test_require_http_api_key_raises() -> None:
    cfg = replace(load_config(), api_key="secret")

    class FakeRequest:
        headers = {}

    with pytest.raises(HTTPException) as exc:
        require_http_api_key(FakeRequest(), cfg)
    assert exc.value.status_code == 401
