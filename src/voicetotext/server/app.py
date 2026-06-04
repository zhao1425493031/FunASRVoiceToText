"""FastAPI application: static web UI, WebSocket ASR, batch transcribe."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import AppConfig, load_config, resolve_config_path
from voicetotext.logging_setup import get_logger, setup_logging
from voicetotext.server.routes_transcribe import create_transcribe_router
from voicetotext.server.ws_protocol import WSProtocolHandler, send_error

_config_path: Path | None = None
config: AppConfig = load_config()
logger = setup_logging(config)
engine: ASRBackend = create_asr_backend(config)
inference_lock = asyncio.Lock()
_active_ws_connections = 0
_ws_connection_lock = asyncio.Lock()


def init_app(config_path: Path | None = None) -> None:
    """Reload module-level config and engine (used by run_server --config)."""
    global _config_path, config, logger, engine
    _config_path = config_path
    config = load_config(config_path)
    logger = setup_logging(config)
    engine = create_asr_backend(config)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    backend = config.asr_backend.lower()
    if backend == "runtime":
        logger.info("Runtime gateway mode — skipping in-process model load")
        engine.load()
        ready = await engine.check_ready()
        if not ready:
            logger.warning(
                "FunASR Runtime not reachable at %s:%s",
                config.runtime_host,
                config.runtime_port,
            )
    else:
        logger.info("Preloading ASR models (%s)...", backend)
        await asyncio.to_thread(engine.load)
    logger.info(
        "Server ready http://%s:%s (LAN: use host machine IP)",
        config.host,
        config.port,
    )
    yield
    logger.info("Server shutdown")


app = FastAPI(title="VoiceToText", lifespan=lifespan)
web_root = config.web_root


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(web_root / "index.html")


if web_root.is_dir():
    app.mount("/static", StaticFiles(directory=web_root), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "device": engine.device,
        "backend": config.asr_backend,
        "language": config.language,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    if config.asr_backend.lower() == "runtime":
        ok = await engine.check_ready()
        if not ok:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "backend": "runtime",
                    "runtime": f"{config.runtime_host}:{config.runtime_port}",
                    "reason": "FunASR Runtime unreachable",
                },
            )
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "backend": "runtime"},
        )

    if not engine.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "backend": config.asr_backend},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "backend": config.asr_backend},
    )


app.include_router(create_transcribe_router(config, engine, inference_lock))


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket) -> None:
    global _active_ws_connections

    async with _ws_connection_lock:
        if _active_ws_connections >= config.max_ws_connections:
            await websocket.accept()
            await send_error(
                websocket,
                "too_many_connections",
                f"Max WebSocket connections ({config.max_ws_connections}) reached",
            )
            await websocket.close()
            return
        _active_ws_connections += 1

    await websocket.accept()
    client = websocket.client
    logger.info("WebSocket connected from %s", client)
    handler = WSProtocolHandler(websocket, engine, config, inference_lock)

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
        logger.info("WebSocket disconnected from %s", client)
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await send_error(websocket, "server_error", str(exc))
        except Exception:
            pass
    finally:
        await handler.cleanup()
        async with _ws_connection_lock:
            _active_ws_connections = max(0, _active_ws_connections - 1)
        logger.info("WebSocket session closed for %s", client)


def create_app(config_path: Path | None = None) -> FastAPI:
    if config_path is not None:
        init_app(config_path)
    return app
