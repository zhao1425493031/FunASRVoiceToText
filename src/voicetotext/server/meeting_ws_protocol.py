"""WebSocket protocol v2 for meeting (multi-speaker) ASR."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import replace
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.meeting_stream_session import MeetingStreamSession
from voicetotext.config import AppConfig
from voicetotext.llm.factory import create_summary_provider
from voicetotext.llm.meeting_export import write_meeting_transcript
from voicetotext.llm.meeting_transcript_buffer import MeetingTranscriptBuffer
from voicetotext.llm.schemas import MeetingTranscript, SummaryRequest, SummarySegment
from voicetotext.llm.summary import SummaryNotImplementedError, SummaryProvider
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
    """In-process meeting_sensevoice streaming session."""

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
        self._participant_id: str | None = None
        self._speaker_ctx: Any = None
        self._session_id = str(uuid.uuid4())
        self._transcript_buffer: MeetingTranscriptBuffer | None = None
        self._summary_provider: SummaryProvider = create_summary_provider(config)
        self._summary_completed = False

    async def start(self, msg: dict[str, Any]) -> None:
        if not self.engine.is_loaded:
            raise RuntimeError("ASR model not loaded; wait for server startup to finish")
        self._session_start = time.time()
        lang = str(msg.get("language", self.config.language)).lower()
        if lang in ("ja", "zh"):
            self._session_config = replace(self.config, language=lang)
        else:
            self._session_config = self.config
        session_id = str(msg.get("session_id", "")).strip() or str(uuid.uuid4())
        self._session_id = session_id
        self._transcript_buffer = MeetingTranscriptBuffer(
            session_id,
            self._session_config.language,
            single_speaker_mode=self._session_config.meeting_spk_mode == "single",
        )
        open_ctx = getattr(self.engine, "open_speaker_context", None)
        if callable(open_ctx):
            self._speaker_ctx = open_ctx(session_id)
        else:
            begin_spk = getattr(self.engine, "begin_speaker_session", None)
            if callable(begin_spk):
                self._speaker_ctx = begin_spk()

        participant_id = str(msg.get("participant_id", "")).strip() or None
        self._participant_id = participant_id
        client_speaker_id: int | None = None
        register = getattr(self.engine, "register_participant", None)
        if callable(register) and participant_id:
            client_speaker_id = register(participant_id)

        self._stream = MeetingStreamSession(
            self.engine,
            self._session_config,
            session_start=self._session_start,
            client_speaker_id=client_speaker_id,
            speaker_ctx=self._speaker_ctx,
        )
        self._closed = False
        self._pcm_queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._pcm_worker())
        logger.info(
            "Meeting session started (v2 lang=%s spk_source=%s participant=%s speaker_id=%s session_id=%s)",
            self._session_config.language,
            self._session_config.meeting_spk_source,
            participant_id or "-",
            client_speaker_id if client_speaker_id is not None else "-",
            self._session_id,
        )

    async def _send_messages(self, messages: list[dict[str, Any]]) -> bool:
        for mapped in messages:
            if mapped.get("type") == "final" and self._transcript_buffer is not None:
                self._transcript_buffer.append_final(mapped)
            if not await send_json_safe(self.websocket, mapped):
                return False
        return True

    async def _pcm_worker(self) -> None:
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

    async def run_meeting_summary(self, *, skip_summary: bool = False) -> bool:
        if self._summary_completed:
            return True
        self._summary_completed = True

        cfg = self._session_config
        buffer = self._transcript_buffer
        if not cfg.llm_enabled or skip_summary or buffer is None or not buffer:
            return True

        await send_json_safe(
            self.websocket,
            {
                "type": "summary_progress",
                "protocol_version": 2,
                "stage": "generating",
            },
        )

        llm_meta: dict[str, Any] = {
            "enabled": True,
            "provider": cfg.llm_provider,
            "model": cfg.llm_model,
        }
        summary_dict: dict[str, Any] | None = None
        error_msg: str | None = None
        resp_meta: dict[str, Any] = {}

        try:
            req = SummaryRequest(
                job_id=self._session_id,
                language=buffer.language,
                segments=[
                    SummarySegment(
                        start_ms=int(s["start_ms"]),
                        end_ms=int(s["end_ms"]),
                        speaker_id=str(s["speaker_id"]),
                        text=str(s["text"]),
                    )
                    for s in buffer.segments
                ],
            )
            resp = await asyncio.wait_for(
                asyncio.to_thread(self._summary_provider.summarize, req),
                timeout=cfg.llm_meeting_ws_wait_sec,
            )
            if resp.status == "ok" and resp.summary is not None:
                summary_dict = resp.summary.to_dict()
                llm_meta.update({"status": "ok", **resp.meta})
                resp_meta = resp.meta
            else:
                error_msg = resp.error or "summary generation failed"
                llm_meta.update({"status": "error", "error": error_msg})
        except SummaryNotImplementedError:
            return True
        except asyncio.TimeoutError:
            error_msg = "llm_timeout"
            llm_meta.update({"status": "error", "error": error_msg})
            if cfg.llm_on_failure == "fail":
                await send_error(self.websocket, "server_error", error_msg)
                raise RuntimeError(error_msg) from None
        except Exception as exc:
            logger.exception("Meeting LLM summary failed session=%s", self._session_id)
            error_msg = str(exc)
            llm_meta.update({"status": "error", "error": error_msg})
            if cfg.llm_on_failure == "fail":
                await send_error(self.websocket, "server_error", error_msg)
                raise

        transcript = MeetingTranscript(
            session_id=self._session_id,
            language=buffer.language,
            duration_ms=buffer.duration_ms(),
            segments=buffer.segments,
            meta={
                "service": "meeting",
                "asr_model": cfg.asr_model,
                "llm": llm_meta,
            },
            summary=summary_dict,
        )
        paths = write_meeting_transcript(
            cfg.llm_meeting_output_path,
            transcript,
            write_summary_md=cfg.llm_output_summary_md,
        )
        files = {k: str(v) for k, v in paths.items()}

        if summary_dict is not None:
            await send_json_safe(
                self.websocket,
                {
                    "type": "meeting_summary",
                    "protocol_version": 2,
                    "status": "ok",
                    "session_id": self._session_id,
                    "summary": summary_dict,
                    "meta": resp_meta,
                    "files": files,
                },
            )
        else:
            await send_json_safe(
                self.websocket,
                {
                    "type": "meeting_summary",
                    "protocol_version": 2,
                    "status": "error",
                    "session_id": self._session_id,
                    "error": error_msg or "summary_failed",
                    "summary": None,
                    "files": files,
                },
            )
        return True

    async def close(self) -> None:
        self._closed = True
        release = getattr(self.engine, "release_participant", None)
        if callable(release) and self._participant_id:
            release(self._participant_id)
        self._participant_id = None
        close_ctx = getattr(self.engine, "close_speaker_context", None)
        if callable(close_ctx) and self._speaker_ctx is not None:
            close_ctx(self._speaker_ctx)
        else:
            end_spk = getattr(self.engine, "end_speaker_session", None)
            if callable(end_spk):
                end_spk(self._speaker_ctx)
        self._speaker_ctx = None
        if self._worker_task is not None:
            await self._pcm_queue.put(None)
            try:
                await asyncio.wait_for(self._worker_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
            self._worker_task = None
        self._stream = None


class MeetingWSProtocolHandler:
    """Meeting v3 handler (meeting_sensevoice in-process ASR)."""

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
        self._ended_normally = False

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
            await self._handle_end(msg)
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
        self._ended_normally = False
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

    async def _handle_end(self, msg: dict[str, Any]) -> None:
        skip_summary = bool(msg.get("skip_summary", False))
        if self.asr_session is not None:
            try:
                await self.asr_session.end()
                await self.asr_session.run_meeting_summary(skip_summary=skip_summary)
            except Exception as exc:
                logger.exception("Meeting end/summary failed: %s", exc)
                if self.config.llm_on_failure == "fail":
                    try:
                        await send_error(self.websocket, "server_error", str(exc))
                    except Exception:
                        pass
            await self.asr_session.close()
            self.asr_session = None
        self._ended_normally = True
        self.started = False
        logger.info("Meeting session ended by client")

    async def cleanup(self) -> None:
        if self.asr_session is not None:
            if not self._ended_normally:
                try:
                    await self.asr_session.end()
                    await self.asr_session.run_meeting_summary(skip_summary=False)
                except Exception as exc:
                    logger.warning("Meeting cleanup summary failed: %s", exc)
            await self.asr_session.close()
            self.asr_session = None
