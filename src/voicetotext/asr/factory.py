"""Construct ASR backend (meeting v3: meeting_sensevoice only)."""

from __future__ import annotations

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.meeting_sensevoice_engine import MeetingSenseVoiceEngine
from voicetotext.config import AppConfig


def create_asr_backend(config: AppConfig) -> ASRBackend:
    backend = config.asr_backend.lower()
    if backend != "meeting_sensevoice":
        raise ValueError(
            f"Unsupported asr_backend: {config.asr_backend} "
            "(meeting v3 requires meeting_sensevoice)"
        )
    return MeetingSenseVoiceEngine(config)
