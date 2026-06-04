"""Gateway engine that delegates to FunASR Runtime via WebSocket."""

from __future__ import annotations

import numpy as np

from voicetotext.asr.runtime_client import RuntimeWSClient
from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class RuntimeGatewayEngine:
    """No in-process model; forwards audio to runtime service."""

    backend_name = "runtime"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._ready = False

    @property
    def is_loaded(self) -> bool:
        return self._ready

    def load(self) -> None:
        self._ready = True
        logger.info("Runtime gateway mode: inference delegated to %s:%s", self.config.runtime_host, self.config.runtime_port)

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def detect_speech(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return False
        return float(np.sqrt(np.mean(audio * audio))) >= 0.01

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        raise NotImplementedError("Use RuntimeWSSession in ws_protocol for runtime backend")

    def finalize_text(self, text: str) -> str:
        return text.strip()

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        return self.finalize_text(draft_fallback)

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError("Batch transcribe not supported in runtime gateway mode")

    async def check_ready(self) -> bool:
        client = RuntimeWSClient(self.config)
        return await client.ping_ready()
