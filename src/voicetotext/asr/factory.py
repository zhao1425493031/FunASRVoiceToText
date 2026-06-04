"""Construct ASR backend from configuration."""

from __future__ import annotations

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.paraformer_engine import ParaformerEngine
from voicetotext.asr.runtime_gateway import RuntimeGatewayEngine
from voicetotext.asr.sensevoice_engine import SenseVoiceEngine
from voicetotext.config import AppConfig


def create_asr_backend(config: AppConfig) -> ASRBackend:
    backend = config.asr_backend.lower()
    if backend == "sensevoice":
        return SenseVoiceEngine(config)
    if backend == "runtime":
        return RuntimeGatewayEngine(config)
    if backend == "paraformer":
        return ParaformerEngine(config)
    raise ValueError(f"Unknown asr_backend: {config.asr_backend}")
