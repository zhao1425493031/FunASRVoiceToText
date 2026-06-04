"""FunASR Runtime WebSocket 2pass client."""

from __future__ import annotations

import json
import ssl
from typing import Any
import websockets
from websockets.asyncio.client import connect

from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class RuntimeWSClient:
    """One session with FunASR runtime websocket server."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._ws: Any = None
        self._recv_task = None

    @property
    def uri(self) -> str:
        host = self.config.runtime_host
        port = self.config.runtime_port
        use_ssl = self.config.runtime_ssl
        scheme = "wss" if use_ssl else "ws"
        return f"{scheme}://{host}:{port}"

    async def connect(self) -> None:
        ssl_context = None
        if self.config.runtime_ssl:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        self._ws = await connect(self.uri, ssl=ssl_context)
        handshake = {
            "mode": self.config.runtime_mode,
            "wav_name": "gateway_mic",
            "is_speaking": True,
            "wav_format": "pcm",
            "audio_fs": self.config.sample_rate,
            "chunk_size": self._parse_chunk_size(),
            "itn": True,
        }
        await self._ws.send(json.dumps(handshake, ensure_ascii=False))
        logger.info("Runtime WS connected to %s", self.uri)

    def _parse_chunk_size(self) -> list[int]:
        parts = [int(x.strip()) for x in self.config.runtime_chunk_size.split(",")]
        if len(parts) == 3:
            return parts
        return [5, 10, 5]

    async def send_pcm(self, pcm_bytes: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("Runtime client not connected")
        await self._ws.send(pcm_bytes)

    async def end_speaking(self) -> None:
        if self._ws is None:
            return
        await self._ws.send(
            json.dumps({"is_speaking": False}, ensure_ascii=False)
        )

    async def recv_message(self) -> dict[str, Any] | None:
        if self._ws is None:
            return None
        try:
            raw = await self._ws.recv()
        except websockets.exceptions.ConnectionClosed:
            return None
        if isinstance(raw, bytes):
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    @staticmethod
    def map_runtime_to_client(msg: dict[str, Any]) -> dict[str, Any] | None:
        """Map FunASR 2pass message to gateway partial/final JSON."""
        mode = str(msg.get("mode", ""))
        text = str(msg.get("text", "") or "").strip()
        if not text:
            return None
        if "online" in mode or mode == "2pass-online":
            return {
                "type": "partial",
                "mode": "online",
                "text": text,
                "is_final": False,
            }
        if "offline" in mode or mode == "2pass-offline":
            return {
                "type": "final",
                "mode": "offline_punc",
                "text": text,
                "is_final": True,
            }
        if msg.get("is_final"):
            return {
                "type": "final",
                "mode": "offline_punc",
                "text": text,
                "is_final": True,
            }
        return {
            "type": "partial",
            "mode": "online",
            "text": text,
            "is_final": False,
        }

    async def ping_ready(self) -> bool:
        try:
            async with connect(
                self.uri,
                open_timeout=3,
                close_timeout=1,
            ) as ws:
                await ws.close()
            return True
        except Exception as exc:
            logger.warning("Runtime not reachable at %s: %s", self.uri, exc)
            return False
