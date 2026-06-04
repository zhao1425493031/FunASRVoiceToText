"""ASR backend protocol for streaming and batch transcription."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASRBackend(Protocol):
    """Unified interface for SenseVoice, Paraformer, and Runtime gateway."""

    device: str

    @property
    def backend_name(self) -> str: ...

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray: ...

    def detect_speech(self, audio: np.ndarray) -> bool: ...

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        """Transcribe an audio window (float32 mono 16kHz)."""
        ...

    def finalize_text(self, text: str) -> str:
        """Apply post-processing (punctuation / ITN) for final output."""
        ...

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        """Authoritative final: full audio ITN when possible, else draft + punc fallback."""
        ...

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe full utterance (batch API / CLI file mode)."""
        ...

    async def check_ready(self) -> bool:
        """Return True if backend is ready to serve traffic."""
        ...
