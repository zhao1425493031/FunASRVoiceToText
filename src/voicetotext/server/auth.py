"""API Key authentication for batch HTTP routes."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from voicetotext.config import load_config, resolve_config_path


def extract_header_api_key(headers) -> str | None:
    for name in ("x-api-key", "X-API-Key", "X-Api-Key"):
        value = headers.get(name)
        if value:
            return value
    return None


def verify_api_key(request: Request) -> None:
    config = load_config(resolve_config_path())
    if not config.auth_enabled:
        return
    key = extract_header_api_key(request.headers)
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    if not key or key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
