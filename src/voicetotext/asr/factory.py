"""Construct ASR backend (meeting v2: meeting_qwen only)."""

from __future__ import annotations

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.meeting_qwen_engine import MeetingQwenEngine
from voicetotext.config import AppConfig


def create_asr_backend(config: AppConfig) -> ASRBackend:
    backend = config.asr_backend.lower()
    if backend != "meeting_qwen":
        raise ValueError(
            f"Unsupported asr_backend: {config.asr_backend} (meeting v2 requires meeting_qwen)"
        )
    return MeetingQwenEngine(config)
