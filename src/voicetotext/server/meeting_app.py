"""FastAPI application for meeting-only service (v3: meeting_sensevoice)."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import AppConfig, load_config
from voicetotext.llm.client import (
    LLMAuthError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from voicetotext.llm.factory import create_summary_provider
from voicetotext.llm.schemas import SummaryRequest, SummarySegment
from voicetotext.llm.summary import SummaryNotImplementedError, SummaryProvider
from voicetotext.logging_setup import get_logger, setup_logging
from voicetotext.server.auth import (
    extract_header_api_key,
    require_http_api_key,
    validate_meeting_api_key,
)
from voicetotext.server.meeting_ws_protocol import MeetingWSProtocolHandler
from voicetotext.server.protocol_common import send_error

_summary_provider: SummaryProvider | None = None

_config_path: Path | None = None
config: AppConfig = load_config()
if not config.is_meeting_service:
    raise RuntimeError("meeting_app requires service_mode=meeting")
logger = setup_logging(config)
engine: ASRBackend = create_asr_backend(config)
_summary_provider = create_summary_provider(config)
_active_meeting_ws_connections = 0
_ws_connection_lock = asyncio.Lock()


def init_app(config_path: Path | None = None) -> None:
    global _config_path, config, logger, engine, _summary_provider
    _config_path = config_path
    config = load_config(config_path)
    if not config.is_meeting_service:
        raise ValueError("meeting_app requires config with service_mode: meeting")
    logger = setup_logging(config)
    engine = create_asr_backend(config)
    _summary_provider = create_summary_provider(config)


def _cfg() -> AppConfig:
    return config


def _summary() -> SummaryProvider:
    if _summary_provider is None:
        init_app(_config_path)
    assert _summary_provider is not None
    return _summary_provider


def _verify_meeting_http(request: Request) -> None:
    require_http_api_key(request, config)
    key = extract_header_api_key(request.headers)
    if not validate_meeting_api_key(config, key):
        raise HTTPException(status_code=401, detail="Invalid or missing meeting API key")


def _load_meeting_session_json(session_id: str) -> dict[str, Any] | None:
    path = config.llm_meeting_output_path / f"{session_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _segments_from_body(body: dict[str, Any]) -> list[SummarySegment]:
    segments: list[SummarySegment] = []
    for item in body.get("segments") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            SummarySegment(
                start_ms=int(item.get("start_ms", 0)),
                end_ms=int(item.get("end_ms", 0)),
                speaker_id=str(item.get("speaker_id", "")),
                text=text,
            )
        )
    return segments


def _llm_detail() -> dict[str, Any]:
    return {
        "enabled": config.llm_enabled,
        "ready": config.llm_ready,
        "provider": config.llm_provider,
        "model": config.llm_model,
    }


def _meeting_url_with_key(request: Request | None = None) -> str:
    key = config.api_key or ""
    if request is not None:
        base = str(request.base_url).rstrip("/")
    else:
        base = f"http://{config.host}:{config.port}"
    if key:
        return f"{base}/meeting?key={key}"
    return f"{base}/meeting"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    resolved = getattr(engine, "device", config.device)
    logger.info(
        "Meeting v3 ASR model=%s device=%s (resolved=%s)",
        config.asr_model,
        config.device,
        resolved,
    )
    engine.load()
    ready = await engine.check_ready()
    if not ready:
        logger.warning("Meeting ASR not fully ready after load(); check HF_TOKEN for Pyannote")
    logger.info("Meeting server ready http://%s:%s", config.host, config.port)
    yield
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    logger.info("Meeting server shutdown")


app = FastAPI(title="VoiceToText-Meeting-v3", lifespan=lifespan)
web_root = config.web_root


@app.get("/")
async def root_redirect(request: Request) -> RedirectResponse:
    return RedirectResponse(url=_meeting_url_with_key(request), status_code=302)


@app.get("/meeting", response_model=None)
async def meeting_page(request: Request) -> HTMLResponse | RedirectResponse:
    if config.auth_enabled and config.api_key and not request.query_params.get("key"):
        return RedirectResponse(url=_meeting_url_with_key(request), status_code=302)
    html_path = web_root / "meeting.html"
    html = html_path.read_text(encoding="utf-8")
    key_js = (config.api_key or "").replace("\\", "\\\\").replace('"', '\\"')
    spk_mode = config.meeting_spk_mode.replace("\\", "\\\\").replace('"', '\\"')
    spk_source = config.meeting_spk_source.replace("\\", "\\\\").replace('"', '\\"')
    lang = config.language.replace("\\", "\\\\").replace('"', '\\"')
    inject = (
        f'<script>window.__MEETING_API_KEY__="{key_js}";'
        f'window.__MEETING_SPK_MODE__="{spk_mode}";'
        f'window.__MEETING_SPK_SOURCE__="{spk_source}";'
        f'window.__MEETING_LANGUAGE__="{lang}";</script>'
    )
    if "</head>" in html:
        html = html.replace("</head>", f"{inject}\n</head>", 1)
    else:
        html = inject + html
    return HTMLResponse(html)


if web_root.is_dir():
    app.mount("/static", StaticFiles(directory=web_root), name="static")


@app.get("/health")
async def health() -> dict[str, str | int]:
    resolved = getattr(engine, "device", config.device)
    active = getattr(engine, "active_session_count", 0)
    return {
        "status": "ok",
        "device": resolved,
        "device_config": config.device,
        "backend": config.asr_backend,
        "language": config.language,
        "service_mode": config.service_mode,
        "asr_version": "v3",
        "active_speaker_sessions": active,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    ok = await engine.check_ready()
    detail = getattr(engine, "readiness_detail", lambda: {})()
    if not ok:
        reason = "ASR stack not ready"
        if (
            config.meeting_spk_source in ("pyannote", "hybrid")
            and config.meeting_use_diarization
            and config.meeting_spk_mode == "multi"
        ):
            token_env = config.pyannote_hf_token_env
            if not os.environ.get(token_env):
                reason = f"Missing {token_env} for Pyannote Community-1"
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "backend": config.asr_backend,
                "service_mode": "meeting",
                "reason": reason,
                "detail": detail,
            },
        )
    if isinstance(detail, dict):
        detail = {**detail, "llm": _llm_detail()}
    else:
        detail = {"llm": _llm_detail()}
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "backend": config.asr_backend,
            "service_mode": "meeting",
            "detail": detail,
        },
    )


@app.post("/api/v1/meeting/summarize")
async def meeting_summarize(
    body: dict[str, Any],
    _: None = Depends(_verify_meeting_http),
) -> JSONResponse:
    if not config.llm_enabled:
        return JSONResponse(
            status_code=501,
            content={"error": "llm_disabled", "message": "LLM summary is disabled"},
        )

    session_id = str(body.get("session_id", "")).strip()
    language = str(body.get("language", config.language))

    if "segments" in body:
        segments = _segments_from_body(body)
        if not segments:
            return JSONResponse(
                status_code=400,
                content={"error": "empty_segments", "message": "segments required"},
            )
    elif session_id:
        stored = _load_meeting_session_json(session_id)
        if stored is None:
            return JSONResponse(
                status_code=400,
                content={"error": "session_not_found", "message": f"No session {session_id}"},
            )
        segments = _segments_from_body(stored)
        language = str(stored.get("language", language))
        if not segments:
            return JSONResponse(
                status_code=400,
                content={"error": "empty_segments", "message": "segments required"},
            )
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "empty_segments", "message": "segments required"},
        )

    req = SummaryRequest(
        job_id=session_id or "meeting-summary",
        language=language,
        segments=segments,
    )
    try:
        resp = _summary().summarize(req)
    except SummaryNotImplementedError:
        return JSONResponse(
            status_code=501,
            content={"error": "llm_disabled", "message": "LLM summary is disabled"},
        )
    except LLMTimeoutError as exc:
        return JSONResponse(
            status_code=504,
            content={"error": "llm_timeout", "message": str(exc)},
        )
    except (LLMAuthError, LLMRateLimitError, LLMServerError, RuntimeError) as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "llm_upstream_error", "message": str(exc)},
        )

    if resp.status != "ok" or resp.summary is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": "summary_failed",
                "message": resp.error or "summary generation failed",
            },
        )
    return JSONResponse(status_code=200, content=resp.to_dict())


@app.get("/api/v1/meeting/sessions/{session_id}")
async def get_meeting_session(
    session_id: str,
    _: None = Depends(_verify_meeting_http),
) -> JSONResponse:
    data = _load_meeting_session_json(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(status_code=200, content=data)


@app.websocket("/ws/meeting/asr")
async def websocket_meeting_asr(websocket: WebSocket) -> None:
    global _active_meeting_ws_connections

    async with _ws_connection_lock:
        if _active_meeting_ws_connections >= config.max_ws_connections:
            await websocket.accept()
            await send_error(
                websocket,
                "too_many_connections",
                f"Max meeting WebSocket connections ({config.max_ws_connections}) reached",
            )
            await websocket.close()
            return
        _active_meeting_ws_connections += 1

    await websocket.accept()
    client = websocket.client
    logger.info("Meeting WebSocket connected from %s", client)
    handler = MeetingWSProtocolHandler(websocket, engine, config)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"] is not None:
                should_continue = await handler.handle_text(message["text"])
                if not should_continue:
                    break
            elif "bytes" in message and message["bytes"] is not None:
                await handler.handle_bytes(message["bytes"])
    except WebSocketDisconnect:
        logger.info("Meeting WebSocket disconnected from %s", client)
    except Exception as exc:
        from voicetotext.server.protocol_common import is_websocket_disconnected

        if is_websocket_disconnected(exc):
            logger.info("Meeting WebSocket closed during send from %s", client)
        else:
            logger.exception("Meeting WebSocket error: %s", exc)
            try:
                await send_error(websocket, "server_error", str(exc))
            except Exception:
                pass
    finally:
        await handler.cleanup()
        async with _ws_connection_lock:
            _active_meeting_ws_connections = max(0, _active_meeting_ws_connections - 1)
        logger.info("Meeting WebSocket session closed for %s", client)
