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
            res = self._model.generate(input=audio, batch_size_s=300)
            segs = _extract_vad_segments(res)
            if segs:
                self._last_has_speech = True
                return True
            self._last_has_speech = False
            return self.detect_speech_frame(audio)
        except Exception as exc:
            logger.debug("VAD generate failed, fallback RMS: %s", exc)
            return self.detect_speech_frame(audio)
