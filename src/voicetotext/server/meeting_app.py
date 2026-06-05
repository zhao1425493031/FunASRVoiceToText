"""FastAPI application for meeting-only service (v3: meeting_sensevoice)."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import AppConfig, load_config
from voicetotext.logging_setup import get_logger, setup_logging
from voicetotext.server.meeting_ws_protocol import MeetingWSProtocolHandler
from voicetotext.server.protocol_common import send_error

_config_path: Path | None = None
config: AppConfig = load_config()
if not config.is_meeting_service:
    raise RuntimeError("meeting_app requires service_mode=meeting")
logger = setup_logging(config)
engine: ASRBackend = create_asr_backend(config)
_active_meeting_ws_connections = 0
_ws_connection_lock = asyncio.Lock()


def init_app(config_path: Path | None = None) -> None:
    global _config_path, config, logger, engine
    _config_path = config_path
    config = load_config(config_path)
    if not config.is_meeting_service:
        raise ValueError("meeting_app requires config with service_mode: meeting")
    logger = setup_logging(config)
    engine = create_asr_backend(config)


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
    logger.info(
        "Meeting v3 ASR model=%s device=%s",
        config.asr_model,
        config.device,
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
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "device": engine.device,
        "backend": config.asr_backend,
        "language": config.language,
        "service_mode": config.service_mode,
        "asr_version": "v3",
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
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "backend": config.asr_backend,
            "service_mode": "meeting",
            "detail": detail,
        },
    )


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
