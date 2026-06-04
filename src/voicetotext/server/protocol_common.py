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


async def send_error(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.send_text(
        encode_message({"type": "error", "code": code, "message": message})
    )
