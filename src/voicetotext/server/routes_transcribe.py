"""Batch transcription HTTP API."""

from __future__ import annotations

import asyncio
import io
import wave
from typing import TYPE_CHECKING

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from voicetotext.server.auth import require_http_api_key

if TYPE_CHECKING:
    from voicetotext.asr.base import ASRBackend
    from voicetotext.config import AppConfig

router = APIRouter(prefix="/api", tags=["transcribe"])


def _read_wav_pcm(data: bytes, expected_sr: int) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid WAV file: {exc}",
        ) from exc

    if sample_width != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="WAV must be 16-bit PCM",
        )
    if channels != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="WAV must be mono",
        )

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate != expected_sr:
        duration = len(audio) / sample_rate
        target_len = int(duration * expected_sr)
        x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0, duration, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sample_rate = expected_sr

    return audio, sample_rate


def create_transcribe_router(
    config: AppConfig,
    engine: ASRBackend,
    inference_lock: asyncio.Lock,
) -> APIRouter:
    @router.post("/transcribe")
    async def transcribe(
        request: Request,
        file: UploadFile = File(...),
    ) -> dict[str, str]:
        require_http_api_key(request, config)
        if config.asr_backend.lower() == "runtime":
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Batch transcribe is not available in runtime backend mode",
            )
        if not engine.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ASR model not loaded",
            )

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

        audio, sr = _read_wav_pcm(raw, config.sample_rate)

        async with inference_lock:
            text = await asyncio.to_thread(engine.transcribe_file, audio, sr)

        return {"text": text, "language": config.language, "backend": config.asr_backend}

    return router
