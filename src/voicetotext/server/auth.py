"""API Key authentication for HTTP and WebSocket."""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import HTTPException, Request, status

from voicetotext.config import AppConfig


def is_auth_required(config: AppConfig) -> bool:
    return config.auth_enabled


def validate_api_key(config: AppConfig, provided: str | None) -> bool:
    if not config.auth_enabled:
        return True
    if not provided:
        return False
    return provided == config.api_key


def validate_meeting_api_key(config: AppConfig, provided: str | None) -> bool:
    if not validate_api_key(config, provided):
        return False
    if not config.is_meeting_service:
        return True
    scopes = config.api_key_scopes
    if not scopes:
        return True
    return "meeting" in scopes


def extract_header_api_key(headers: Mapping[str, str]) -> str | None:
    for name in ("x-api-key", "X-API-Key", "X-Api-Key"):
        value = headers.get(name)
        if value:
            return value
    return None


def extract_ws_api_key(msg: dict[str, Any], headers: Mapping[str, str]) -> str | None:
    if msg.get("api_key"):
        return str(msg["api_key"])
    return extract_header_api_key(headers)


def require_http_api_key(request: Request, config: AppConfig) -> None:
    if not config.auth_enabled:
        return
    key = extract_header_api_key(request.headers)
    if not validate_api_key(config, key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
