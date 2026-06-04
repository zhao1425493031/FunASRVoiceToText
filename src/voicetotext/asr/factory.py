"""Construct ASR backend (meeting: embedded Python only)."""

from __future__ import annotations

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.embedded_engine import EmbeddedSenseVoiceEngine
from voicetotext.config import AppConfig


def create_asr_backend(config: AppConfig) -> ASRBackend:
    backend = config.asr_backend.lower()
    if backend != "embedded":
        raise ValueError(
            f"Unsupported asr_backend: {config.asr_backend} (meeting requires embedded)"
        )
    return EmbeddedSenseVoiceEngine(config)
