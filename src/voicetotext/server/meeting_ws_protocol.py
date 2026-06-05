"""WebSocket protocol v2 for meeting (multi-speaker) ASR."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.meeting_stream_session import MeetingStreamSession
from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger
from voicetotext.server.auth import extract_ws_api_key, validate_meeting_api_key
from voicetotext.server.protocol_common import (
    encode_message,
    is_websocket_disconnected,
    parse_client_message,
    send_error,
    send_json_safe,
)

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = frozenset({"ja", "zh"})


class MeetingWSSession:
    """In-process meeting_qwen streaming session."""

    def __init__(
        self,
        config: AppConfig,
        websocket: WebSocket,
        engine: ASRBackend,
    ) -> None:
        self.config = config
        self.websocket = websocket
        self.engine = engine
        self._stream: MeetingStreamSession | None = None
        self._session_start = time.time()
        self._session_config = config
        self._pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self, msg: dict[str, Any]) -> None:
        if not self.engine.is_loaded:
            raise RuntimeError("ASR model not loaded; wait for server startup to finish")
        self._session_start = time.time()
        lang = str(msg.get("language", self.config.language)).lower()
        if lang in ("ja", "zh"):
            self._session_config = replace(self.config, language=lang)
        else:
            self._session_config = self.config
        set_lang = getattr(self.engine, "set_session_language", None)
        if callable(set_lang):
            set_lang(self._session_config.language)
        self._stream = MeetingStreamSession(
            self.engine,
            self._session_config,
            session_start=self._session_start,
        )
        self._closed = False
        self._pcm_queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._pcm_worker())
        logger.info("Meeting session started (v2 lang=%s)", self._session_config.language)

    async def _send_messages(self, messages: list[dict[str, Any]]) -> bool:
        for mapped in messages:
            if not await send_json_safe(self.websocket, mapped):
                return False
        return True

    async def _pcm_worker(self) -> None:
        """Process PCM off the WebSocket receive loop (Qwen CPU infer can take 10s+)."""
        while not self._closed:
            data = await self._pcm_queue.get()
            try:
                if data is None:
                    break
                if self._stream is None:
                    continue
                try:
                    messages = await asyncio.to_thread(self._stream.feed_pcm, data)
                except Exception as exc:
                    logger.warning("Meeting feed_pcm failed: %s", exc)
                    continue
                if messages and not self._closed:
                    if not await self._send_messages(messages):
                        logger.warning(
                            "Meeting subtitle send failed (client disconnected)"
                        )
                        break
            finally:
                self._pcm_queue.task_done()

    async def send_pcm(self, data: bytes) -> bool:
        if self._stream is None or self._closed:
            return True
        await self._pcm_queue.put(data)
        return True

    async def end(self) -> bool:
        if self._stream is None:
            return True
        await self._pcm_queue.join()
        await self._pcm_queue.put(None)
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None
        messages = await asyncio.to_thread(self._stream.finalize_all)
        return await self._send_messages(messages)

    async def close(self) -> None:
        self._closed = True
        clear_lang = getattr(self.engine, "set_session_language", None)
        if callable(clear_lang):
            clear_lang(None)
        if self._worker_task is not None:
            await self._pcm_queue.put(None)
            try:
                await asyncio.wait_for(self._worker_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
            self._worker_task = None
        self._stream = None


class MeetingWSProtocolHandler:
    """Meeting v2 handler (meeting_qwen in-process ASR)."""

    def __init__(
        self,
        websocket: WebSocket,
        engine: ASRBackend,
        config: AppConfig,
    ) -> None:
        self.websocket = websocket
        self.engine = engine
        self.config = config
        self.asr_session: MeetingWSSession | None = None
        self.started = False
        self._session_start = time.time()

    def _session_duration_s(self) -> float:
        return time.time() - self._session_start

    async def handle_text(self, raw: str) -> bool:
        try:
            msg = parse_client_message(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            await send_error(self.websocket, "invalid_json", str(exc))
            return False

        msg_type = msg.get("type")
        if msg_type == "ping":
            await self.websocket.send_text(
                encode_message({"type": "pong", "protocol_version": 2})
            )
            return True
        if msg_type == "start":
            return await self._handle_start(msg)
        if msg_type == "end":
            await self._handle_end()
            return False
        await send_error(self.websocket, "unknown_type", f"Unknown type: {msg_type}")
        return True

    async def _handle_start(self, msg: dict[str, Any]) -> bool:
        api_key = extract_ws_api_key(msg, self.websocket.headers)
        if not validate_meeting_api_key(self.config, api_key):
            await send_error(self.websocket, "unauthorized", "Invalid or missing API key")
            return False

        if msg.get("protocol_version") != 2:
            await send_error(
                self.websocket,
                "protocol_mismatch",
                "protocol_version must be 2",
            )
            return False
        if msg.get("mode") != "meeting":
            await send_error(self.websocket, "protocol_mismatch", "mode must be meeting")
            return False

        lang = str(msg.get("language", self.config.language)).lower()
        if lang not in SUPPORTED_LANGUAGES:
            await send_error(
                self.websocket,
                "unsupported_language",
                f"Unsupported language: {lang}",
            )
            return False

        max_spk = int(msg.get("max_speakers", self.config.meeting_max_speakers))
        if max_spk < 1 or max_spk > self.config.meeting_max_speakers:
            await send_error(
                self.websocket,
                "protocol_mismatch",
                f"max_speakers must be 1..{self.config.meeting_max_speakers}",
            )
            return False

        self.asr_session = MeetingWSSession(self.config, self.websocket, self.engine)
        try:
            await self.asr_session.start(msg)
        except Exception as exc:
            logger.exception("Meeting ASR start failed: %s", exc)
            await send_error(self.websocket, "backend_unavailable", str(exc))
            return False

        self.started = True
        self._session_start = time.time()
        logger.info("Meeting session started session_id=%s", msg.get("session_id"))
        await self.websocket.send_text(
            encode_message(
                {
                    "type": "session_ready",
                    "protocol_version": 2,
                    "message": "ok",
                }
            )
        )
        return True

    async def handle_bytes(self, data: bytes) -> None:
        if not self.started:
            await send_error(self.websocket, "not_started", "Send start message before audio")
            return

        max_s = self.config.meeting_session_max_seconds
        if max_s > 0 and self._session_duration_s() > max_s:
            await send_error(
                self.websocket,
                "session_too_long",
                f"Meeting session exceeds {max_s}s limit",
            )
            return

        expected = self.config.chunk_stride_bytes
        if len(data) != expected:
            await send_error(
                self.websocket,
                "invalid_chunk",
                f"Expected {expected} bytes, got {len(data)}",
            )
            return

        if self.asr_session is None:
            return
        try:
            if not await self.asr_session.send_pcm(data):
                self.started = False
        except WebSocketDisconnect:
            self.started = False
        except Exception as exc:
            if is_websocket_disconnected(exc):
                self.started = False
                return
            logger.warning("Meeting ASR send failed: %s", exc)
            try:
                await send_error(self.websocket, "asr_error", str(exc))
            except Exception:
                self.started = False

    async def _handle_end(self) -> None:
        if self.asr_session is not None:
            await self.asr_session.end()
            await self.asr_session.close()
            self.asr_session = None
        self.started = False
        logger.info("Meeting session ended by client")

    async def cleanup(self) -> None:
        if self.asr_session is not None:
            await self.asr_session.close()
            self.asr_session = None
