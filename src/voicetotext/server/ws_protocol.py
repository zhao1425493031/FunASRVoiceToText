"""WebSocket ASR protocol handler (SenseVoice / Paraformer / Runtime gateway)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.runtime_client import RuntimeWSClient
from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger
from voicetotext.server.auth import extract_ws_api_key, validate_api_key
from voicetotext.stream_session import StreamSession

logger = get_logger(__name__)


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


class RuntimeWSSession:
    """Bridge one client WebSocket to FunASR Runtime 2pass service."""

    def __init__(self, config: AppConfig, websocket: WebSocket) -> None:
        self.config = config
        self.websocket = websocket
        self._client = RuntimeWSClient(config)
        self._recv_task: asyncio.Task | None = None

    async def start(self, msg: dict[str, Any]) -> None:
        await self._client.connect()
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("Runtime session started wav_name=%s", msg.get("wav_name", "web_mic"))

    async def _recv_loop(self) -> None:
        try:
            while True:
                raw_msg = await self._client.recv_message()
                if raw_msg is None:
                    break
                mapped = RuntimeWSClient.map_runtime_to_client(raw_msg)
                if mapped:
                    await self.websocket.send_text(encode_message(mapped))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Runtime recv loop error: %s", exc)

    async def send_pcm(self, data: bytes) -> None:
        await self._client.send_pcm(data)

    async def end(self) -> None:
        await self._client.end_speaking()

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        await self._client.close()


class WSProtocolHandler:
    def __init__(
        self,
        websocket: WebSocket,
        engine: ASRBackend,
        config: AppConfig,
        inference_lock: asyncio.Lock,
    ) -> None:
        self.websocket = websocket
        self.engine = engine
        self.config = config
        self.inference_lock = inference_lock
        self.session: StreamSession | None = None
        self.runtime_session: RuntimeWSSession | None = None
        self.started = False
        self._use_runtime = config.asr_backend.lower() == "runtime"

    async def handle_text(self, raw: str) -> bool:
        """Handle JSON control message. Returns False to close connection."""
        try:
            msg = parse_client_message(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            await send_error(self.websocket, "invalid_json", str(exc))
            return False

        msg_type = msg.get("type")
        if msg_type == "start":
            return await self._handle_start(msg)
        if msg_type == "end":
            await self._handle_end()
            return False
        await send_error(self.websocket, "unknown_type", f"Unknown type: {msg_type}")
        return True

    async def _handle_start(self, msg: dict[str, Any]) -> bool:
        api_key = extract_ws_api_key(msg, self.websocket.headers)
        if not validate_api_key(self.config, api_key):
            await send_error(self.websocket, "unauthorized", "Invalid or missing API key")
            return False

        if self._use_runtime:
            self.runtime_session = RuntimeWSSession(self.config, self.websocket)
            try:
                await self.runtime_session.start(msg)
            except Exception as exc:
                logger.exception("Runtime connect failed: %s", exc)
                await send_error(self.websocket, "runtime_unavailable", str(exc))
                return False
        else:
            chunk_size = msg.get("chunk_size", self.config.chunk_size)
            if chunk_size != self.config.chunk_size:
                logger.warning(
                    "Client chunk_size %s differs from server %s",
                    chunk_size,
                    self.config.chunk_size,
                )
            self.session = StreamSession(engine=self.engine, config=self.config)

        self.started = True
        logger.info("Session started backend=%s", self.config.asr_backend)
        return True

    async def handle_bytes(self, data: bytes) -> None:
        if not self.started:
            await send_error(self.websocket, "not_started", "Send start message before audio")
            return

        expected = self.config.chunk_stride_bytes
        if len(data) != expected:
            await send_error(
                self.websocket,
                "invalid_chunk",
                f"Expected {expected} bytes, got {len(data)}",
            )
            return

        if self._use_runtime:
            if self.runtime_session is None:
                return
            try:
                await self.runtime_session.send_pcm(data)
            except Exception as exc:
                logger.exception("Runtime send failed: %s", exc)
                await send_error(self.websocket, "runtime_error", str(exc))
            return

        if self.session is None:
            return

        async with self.inference_lock:
            result = await asyncio.to_thread(self.session.feed_pcm, data)

        if result.session_too_long:
            await send_error(
                self.websocket,
                "session_too_long",
                f"Recording exceeds {self.config.session_pcm_max_seconds}s limit",
            )
            return

        if result.partial:
            await self.websocket.send_text(
                encode_message(
                    {
                        "type": "partial",
                        "mode": "online",
                        "text": result.partial,
                        "is_final": False,
                    }
                )
            )
        if result.final:
            await self._send_final(result.final)

    async def _handle_end(self) -> None:
        if self._use_runtime:
            if self.runtime_session is not None:
                await self.runtime_session.end()
                await asyncio.sleep(0.5)
                await self.runtime_session.close()
                self.runtime_session = None
        elif self.session is not None:
            async with self.inference_lock:
                result = await asyncio.to_thread(self.session.finalize)
            if result.final:
                try:
                    await self._send_final(result.final)
                except Exception as exc:
                    logger.warning("Could not send final (client may have disconnected): %s", exc)
            else:
                logger.warning("Finalize produced no final text for client")
            self.session.reset()

        self.started = False
        logger.info("Session ended by client")

    async def _send_final(self, text: str) -> None:
        await self.websocket.send_text(
            encode_message(
                {
                    "type": "final",
                    "mode": "offline_punc",
                    "text": text,
                    "is_final": True,
                }
            )
        )

    async def cleanup(self) -> None:
        if self.runtime_session is not None:
            await self.runtime_session.close()
            self.runtime_session = None
        if self.session is not None:
            self.session.reset()
            self.session = None
