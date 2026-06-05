"""Auth tests for batch API."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from voicetotext.config import load_config
from voicetotext.server.auth import verify_api_key

ROOT = Path(__file__).resolve().parents[1]


class FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def test_verify_api_key_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["VOICETOTEXT_CONFIG"] = str(ROOT / "config.yaml")
    cfg = load_config()
    req = FakeRequest({"X-API-Key": cfg.api_key or ""})
    verify_api_key(req)  # no raise


def test_verify_api_key_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["VOICETOTEXT_CONFIG"] = str(ROOT / "config.yaml")
    req = FakeRequest({"X-API-Key": "wrong"})
    with pytest.raises(HTTPException) as exc:
        verify_api_key(req)
    assert exc.value.status_code == 401
