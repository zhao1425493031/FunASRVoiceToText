"""Shared WebSocket JSON helpers for meeting protocol."""

from __future__ import annotations

import json
import uuid
from typing import Any


def new_seg_id() -> str:
    return uuid.uuid4().hex[:12]

from fastapi import WebSocket


def encode_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def parse_client_message(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Message must be a JSON object")
    return data


def is_websocket_disconnected(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("WebSocketDisconnect", "ClientDisconnected"):
        return True
    if isinstance(exc, RuntimeError) and "close message has been sent" in str(exc):
        return True
    return False


async def send_json_safe(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """Send JSON text; return False if client already disconnected."""
    from starlette.websockets import WebSocketDisconnect

    try:
        await websocket.send_text(encode_message(payload))
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        if is_websocket_disconnected(exc):
            return False
        raise


async def send_error(websocket: WebSocket, code: str, message: str) -> None:
    await send_json_safe(
        websocket, {"type": "error", "code": code, "message": message}
    )
