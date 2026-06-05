"""FSMN-VAD via FunASR AutoModel."""

from __future__ import annotations

from typing import Any

import numpy as np

from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


def _extract_vad_segments(result: Any) -> list[tuple[int, int]]:
    """Parse FunASR VAD output to list of (start_ms, end_ms)."""
    segments: list[tuple[int, int]] = []
    if not result:
        return segments
    items = result if isinstance(result, list) else [result]
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value") or item.get("text")
        if isinstance(value, list):
            for seg in value:
                if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    segments.append((int(seg[0]), int(seg[1])))
        ts = item.get("timestamp")
        if isinstance(ts, list):
            for seg in ts:
                if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    segments.append((int(seg[0]), int(seg[1])))
    return segments


class FunASRVAD:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._model: Any = None
        self._ready = False
        self._last_has_speech = False

    @property
    def is_loaded(self) -> bool:
        return self._ready and self._model is not None

    def load(self) -> None:
        from funasr import AutoModel

        logger.info("Loading VAD model=%s device=%s", self.config.vad_model, self.device)
        self._model = AutoModel(
            model=self.config.vad_model,
            device=self.device,
            disable_update=True,
        )
        self._ready = True

    def detect_speech_frame(self, audio: np.ndarray) -> bool:
        """Frame-level speech: RMS fallback when VAD batch not run."""
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio * audio)))
        return rms >= self.config.vad_energy_threshold

    def detect_speech_in_utterance(self, audio: np.ndarray) -> bool:
        """Utterance-level: FunASR VAD if loaded, else RMS."""
        if self._model is None or audio.size == 0:
            return self.detect_speech_frame(audio)
        try:
            segs = self._vad_segments(audio)
            if segs:
                self._last_has_speech = True
                return True
            self._last_has_speech = False
            return self.detect_speech_frame(audio)
        except Exception as exc:
            logger.debug("VAD generate failed, fallback RMS: %s", exc)
            return self.detect_speech_frame(audio)

    def _vad_segments(self, audio: np.ndarray) -> list[tuple[int, int]]:
        if self._model is None or audio.size == 0:
            return []
        res = self._model.generate(input=audio, batch_size_s=300)
        return _extract_vad_segments(res)

    def segment_utterances(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Full-file VAD segmentation as (start_ms, end_ms) list."""
        return self._vad_segments(audio)

    def utterance_endpoint_reached(
        self,
        audio: np.ndarray,
        *,
        tail_margin_ms: int = 320,
    ) -> bool:
        """
        True when FSMN-VAD shows speech ended before the utterance tail.
        Industry endpoint gate: avoid cutting on brief RMS dips mid-sentence.
        """
        if audio.size == 0:
            return True
        duration_ms = int(audio.size / self.config.sample_rate * 1000)
        if duration_ms < 200:
            return False
        try:
            segs = self._vad_segments(audio)
        except Exception as exc:
            logger.debug("FSMN endpoint check failed, fallback silence timer: %s", exc)
            return True
        if not segs:
            # No speech detected — not an endpoint; wait for more audio or silence timer.
            return False
        last_end = max(end for _, end in segs)
        return last_end <= max(0, duration_ms - tail_margin_ms)
