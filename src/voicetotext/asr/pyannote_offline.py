"""Full-file Pyannote Community-1 diarization for batch pipeline."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


def _parse_pyannote_output(output: Any, window_start_ms: int) -> list[DiarizationSegment]:
    segments: list[DiarizationSegment] = []
    try:
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive is None and hasattr(output, "speaker_diarization"):
            ann = output.speaker_diarization
        else:
            ann = exclusive
        if ann is None:
            return segments
        for turn, _, speaker in ann.itertracks(yield_label=True):
            start_ms = window_start_ms + int(turn.start * 1000)
            end_ms = window_start_ms + int(turn.end * 1000)
            segments.append(
                DiarizationSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_label=str(speaker),
                )
            )
    except Exception as exc:
        logger.warning("Parse pyannote output failed: %s", exc)
    return segments


class PyannoteOfflineDiarizer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._pipeline: Any = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._pipeline is not None

    def load(self) -> None:
        env_name = self.config.pyannote_hf_token_env
        token = os.environ.get(env_name, "").strip()
        if not token:
            logger.warning("Pyannote offline: missing %s", env_name)
            return

        from pyannote.audio import Pipeline

        logger.info(
            "Loading Pyannote offline model=%s device=%s",
            self.config.pyannote_model,
            self.device,
        )
        self._pipeline = Pipeline.from_pretrained(
            self.config.pyannote_model,
            token=token,
        )
        if self.device.startswith("cuda"):
            import torch

            self._pipeline.to(torch.device(self.device))
        self._ready = True

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        if not self.is_ready or self._pipeline is None or audio.size == 0:
            return []
        import torch

        waveform = torch.from_numpy(audio).unsqueeze(0)
        kwargs: dict[str, Any] = {}
        if self.config.pyannote_min_speakers > 0:
            kwargs["min_speakers"] = self.config.pyannote_min_speakers
        if self.config.pyannote_max_speakers > 0:
            kwargs["max_speakers"] = self.config.pyannote_max_speakers

        output = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **kwargs,
        )
        return _parse_pyannote_output(output, window_start_ms=0)
