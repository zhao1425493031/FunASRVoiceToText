"""Transcribe diarization windows in parallel for diar-first batch pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from voicetotext.asr.sensevoice_funasr_engine import SenseVoiceFunASREngine
from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscribedWindow:
    start_ms: int
    end_ms: int
    speaker_label: str
    text: str


def _transcribe_one(
    audio: np.ndarray,
    window: DiarizationSegment,
    asr: SenseVoiceFunASREngine,
    sample_rate: int,
    language: str,
) -> TranscribedWindow | None:
    s0 = int(window.start_ms * sample_rate / 1000)
    s1 = int(window.end_ms * sample_rate / 1000)
    chunk = audio[s0:s1]
    if chunk.size == 0:
        return None
    cache: dict = {}
    text = asr.transcribe(chunk, cache, is_final=True, language=language)
    if not text.strip():
        return None
    return TranscribedWindow(
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        speaker_label=window.speaker_label,
        text=text.strip(),
    )


def transcribe_diar_windows(
    audio: np.ndarray,
    windows: list[DiarizationSegment],
    asr: SenseVoiceFunASREngine,
    sample_rate: int,
    language: str,
    workers: int,
) -> list[TranscribedWindow]:
    """Transcribe each diar window; drop empty results; preserve time order."""
    if not windows:
        return []

    max_workers = max(1, workers)
    results: list[TranscribedWindow] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _transcribe_one,
                audio,
                window,
                asr,
                sample_rate,
                language,
            )
            for window in windows
        ]
        for fut in futures:
            item = fut.result()
            if item is not None:
                results.append(item)

    results.sort(key=lambda w: w.start_ms)
    logger.info(
        "Transcribed %d/%d diar windows (workers=%d)",
        len(results),
        len(windows),
        max_workers,
    )
    return results
