"""FastAPI application for offline batch transcription service."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voicetotext.asr.batch_pipeline import BatchPipeline, transcript_to_plain
from voicetotext.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    AppConfig,
    apply_secrets,
    load_config,
)
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
from voicetotext.server.auth import verify_api_key

_config_path: Path = DEFAULT_CONFIG_PATH
config: AppConfig | None = None
logger = get_logger(__name__)
pipeline: BatchPipeline | None = None
_jobs: dict[str, dict[str, Any]] = {}
_summary_provider: SummaryProvider | None = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-job")


def init_app(config_path: Path | None = None) -> None:
    global _config_path, config, logger, pipeline, _summary_provider
    _config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_config(_config_path)
    logger = setup_logging(config)
    apply_secrets(_config_path)
    pipeline = BatchPipeline(config)
    _summary_provider = create_summary_provider(config)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if config is None:
        init_app(_config_path)
    assert config is not None and pipeline is not None
    logger.info("Batch server loading ASR=%s", config.asr_model)
    pipeline.load()
    yield
    _executor.shutdown(wait=False, cancel_futures=True)
    logger.info("Batch server shutdown")


app = FastAPI(title="VoiceToText-Batch", lifespan=lifespan)


def _cfg() -> AppConfig:
    if config is None:
        init_app(_config_path)
    assert config is not None
    return config


def _pipe() -> BatchPipeline:
    if pipeline is None:
        init_app(_config_path)
    assert pipeline is not None
    return pipeline


def _summary() -> SummaryProvider:
    if _summary_provider is None:
        init_app(_config_path)
    assert _summary_provider is not None
    return _summary_provider


def _load_transcript_json(job_id: str) -> dict[str, Any] | None:
    import json

    for base in (Path("out"), Path("out") / "jobs", PROJECT_ROOT / "out"):
        path = base / f"{job_id}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _segments_from_body(body: dict[str, Any]) -> list[SummarySegment]:
    raw_segments = body.get("segments") or []
    segments: list[SummarySegment] = []
    for item in raw_segments:
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


def _update_job(job_id: str, **fields: Any) -> None:
    entry = _jobs.setdefault(job_id, {})
    entry.update(fields)


def _run_job_sync(job_id: str, input_path: Path, out_dir: Path) -> None:
    pipe = _pipe()

    def on_progress(stage: str, pct: int) -> None:
        _update_job(job_id, status="running", stage=stage, progress=pct)

    try:
        _update_job(job_id, status="running", stage="start", progress=5)
        transcript = pipe.process_file(
            input_path,
            out_dir,
            job_id=job_id,
            on_progress=on_progress,
        )
        data = transcript.to_dict()
        _update_job(
            job_id,
            status="completed",
            stage="done",
            progress=100,
            transcript=data,
            text=transcript_to_plain(data),
            summary=transcript.summary,
        )
    except Exception as exc:
        logger.exception("Batch job %s failed: %s", job_id, exc)
        _update_job(job_id, status="failed", stage="error", progress=0, error=str(exc))
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict[str, str]:
    cfg = _cfg()
    return {
        "status": "ok",
        "service": "batch",
        "asr_model": cfg.asr_model,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    pipe = _pipe()
    detail = pipe.readiness_detail()
    if not pipe.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "batch",
                "reason": "Batch stack not ready (check HF_TOKEN and models)",
                "detail": detail,
            },
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "service": "batch", "detail": detail},
    )


@app.get("/batch", response_model=None)
async def batch_page() -> HTMLResponse:
    cfg = _cfg()
    html_path = cfg.web_root / "batch.html"
    html = html_path.read_text(encoding="utf-8")
    key_js = (cfg.api_key or "").replace("\\", "\\\\").replace('"', '\\"')
    lang = cfg.language.replace("\\", "\\\\").replace('"', '\\"')
    inject = (
        f'<script>window.__BATCH_API_KEY__="{key_js}";'
        f'window.__BATCH_LANGUAGE__="{lang}";</script>'
    )
    if "</head>" in html:
        html = html.replace("</head>", f"{inject}\n</head>", 1)
    else:
        html = inject + html
    return HTMLResponse(html)


_web_root = PROJECT_ROOT / "web"
if _web_root.is_dir():
    app.mount("/static", StaticFiles(directory=_web_root), name="static")


@app.post("/api/v1/transcribe/async")
async def transcribe_async(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
) -> dict[str, str]:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    job_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)

    out_dir = Path("out") / "jobs"
    _jobs[job_id] = {
        "status": "queued",
        "stage": "upload",
        "progress": 0,
        "filename": file.filename or "audio",
    }
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_job_sync, job_id, tmp_path, out_dir)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/v1/jobs/{job_id}/status")
async def job_status(
    job_id: str,
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job not found")
    entry = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": entry.get("status", "unknown"),
        "stage": entry.get("stage", ""),
        "progress": entry.get("progress", 0),
        "filename": entry.get("filename"),
        "error": entry.get("error"),
        "text": entry.get("text"),
        "transcript": entry.get("transcript"),
        "summary": entry.get("summary"),
    }


@app.post("/api/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    pipe = _pipe()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        out_dir = Path("out") / "jobs"
        transcript = pipe.process_file(tmp_path, out_dir)
        data = transcript.to_dict()
        _jobs[transcript.job_id] = {
            "status": "completed",
            "progress": 100,
            "transcript": data,
            "text": transcript_to_plain(data),
        }
        paths = pipe.export(transcript, out_dir)
        return {
            "job_id": transcript.job_id,
            "status": "completed",
            "output_dir": str(out_dir),
            "files": {k: str(v) for k, v in paths.items()},
            "transcript": data,
            "text": transcript_to_plain(data),
        }
    except Exception as exc:
        logger.exception("Batch transcribe failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/v1/jobs/{job_id}")
async def get_job(
    job_id: str,
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job not found")
    entry = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": entry.get("status"),
        "transcript": entry.get("transcript"),
        "text": entry.get("text"),
        "summary": entry.get("summary"),
    }


@app.get("/")
async def root_redirect(request: Request) -> HTMLResponse:
    return await batch_page()


@app.post("/api/v1/summarize")
async def summarize(
    body: dict[str, Any],
    _: None = Depends(verify_api_key),
) -> JSONResponse:
    cfg = _cfg()
    if not cfg.llm_enabled:
        return JSONResponse(
            status_code=501,
            content={"error": "llm_disabled", "message": "LLM summary is disabled"},
        )

    job_id = str(body.get("job_id", "")).strip()
    language = str(body.get("language", cfg.language))

    if "segments" in body:
        segments = _segments_from_body(body)
        if not segments:
            return JSONResponse(
                status_code=400,
                content={"error": "empty_segments", "message": "segments required"},
            )
    elif job_id:
        stored = _load_transcript_json(job_id)
        if stored is None:
            return JSONResponse(
                status_code=400,
                content={"error": "job_not_found", "message": f"No transcript for {job_id}"},
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

    req = SummaryRequest(job_id=job_id or "summary", language=language, segments=segments)
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

    data = resp.to_dict()
    return JSONResponse(status_code=200, content=data)
