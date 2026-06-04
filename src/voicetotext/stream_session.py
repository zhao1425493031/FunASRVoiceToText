"""Per-connection streaming ASR session with window accumulation and VAD."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from voicetotext.asr.base import ASRBackend
from voicetotext.config import AppConfig

# Punctuation-only fragments (noise hallucinations) are not sent to clients.
_PUNCT_ONLY = re.compile(r"^[\s。．.,、!?！？…・\-_'\"]+$")


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def is_meaningful_text(text: str, min_chars: int = 2) -> bool:
    """Return True if text has enough non-punctuation content to show users."""
    stripped = text.strip()
    if not stripped:
        return False
    if _PUNCT_ONLY.match(stripped):
        return False
    core = re.sub(r"[\s。．.,、!?！？…・\-_'\"]", "", stripped)
    return len(core) >= min_chars


@dataclass
class SessionResult:
    partial: str | None = None
    final: str | None = None


@dataclass
class StreamSession:
    engine: ASRBackend
    config: AppConfig
    cache: dict = field(default_factory=dict)
    confirmed: str = ""
    partial: str = ""
    draft: str = ""
    last_voice_ts: float = field(default_factory=time.time)
    _pending_pcm: bytearray = field(default_factory=bytearray)
    _window_pcm: bytearray = field(default_factory=bytearray)

    def reset(self) -> None:
        self.cache.clear()
        self.confirmed = ""
        self.partial = ""
        self.draft = ""
        self.last_voice_ts = time.time()
        self._pending_pcm.clear()
        self._window_pcm.clear()

    def feed_pcm(self, pcm_bytes: bytes) -> SessionResult:
        self._pending_pcm.extend(pcm_bytes)
        out = SessionResult()
        stride_bytes = self.config.chunk_stride_bytes

        while len(self._pending_pcm) >= stride_bytes:
            chunk_bytes = bytes(self._pending_pcm[:stride_bytes])
            del self._pending_pcm[:stride_bytes]
            chunk_result = self._ingest_chunk(chunk_bytes)
            if chunk_result.partial:
                out.partial = chunk_result.partial
            if chunk_result.final:
                out.final = chunk_result.final

        if self.config.auto_finalize_on_silence:
            silence_ms = (time.time() - self.last_voice_ts) * 1000
            if (
                self.draft
                and silence_ms >= self.config.vad_silence_ms
                and not self._pending_pcm
                and not self._window_pcm
            ):
                final_result = self.finalize()
                if final_result.final:
                    out.final = final_result.final
        return out

    def _ingest_chunk(self, pcm_bytes: bytes) -> SessionResult:
        audio = self.engine.pcm_bytes_to_float32(pcm_bytes)
        if self._chunk_has_speech(audio):
            self.last_voice_ts = time.time()

        if self.config.asr_backend.lower() == "paraformer":
            return self._process_paraformer_chunk(audio)

        return self._process_window_chunk(pcm_bytes)

    def _chunk_has_speech(self, audio: np.ndarray) -> bool:
        return audio_rms(audio) >= self.config.vad_energy_threshold

    def _window_has_speech(self, window_audio: np.ndarray) -> bool:
        return audio_rms(window_audio) >= self.config.vad_energy_threshold

    def _process_paraformer_chunk(self, audio: np.ndarray) -> SessionResult:
        text = self.engine.transcribe_window(audio, self.cache, is_final=False)
        out = SessionResult()
        if text:
            self.partial = text
            out.partial = text
        return out

    def _process_window_chunk(self, pcm_bytes: bytes) -> SessionResult:
        self._window_pcm.extend(pcm_bytes)
        out = SessionResult()
        if len(self._window_pcm) < self.config.stream_window_bytes:
            return out
        window_bytes = bytes(self._window_pcm)
        self._window_pcm.clear()
        window_audio = self.engine.pcm_bytes_to_float32(window_bytes)
        if not self._window_has_speech(window_audio):
            return out

        text = self.engine.transcribe_window(window_audio, self.cache, is_final=False)
        if not text or not is_meaningful_text(text, self.config.min_partial_chars):
            return out

        merged = self._merge_segment(self.draft, text)
        if merged == self.draft:
            return out

        self.draft = merged
        self.partial = self.draft
        out.partial = self.draft
        return out

    @staticmethod
    def _merge_segment(draft: str, segment: str) -> str:
        """Append or extend session draft for independent window transcripts."""
        segment = segment.strip()
        if not segment:
            return draft
        if not draft:
            return segment
        if segment == draft:
            return draft
        if draft in segment and len(segment) > len(draft):
            return segment
        if segment in draft:
            return draft
        return draft + segment

    def finalize(self) -> SessionResult:
        out = SessionResult()

        if self._pending_pcm:
            remainder = bytes(self._pending_pcm)
            self._pending_pcm.clear()
            self._window_pcm.extend(remainder)

        if self._window_pcm:
            window_audio = self.engine.pcm_bytes_to_float32(bytes(self._window_pcm))
            self._window_pcm.clear()
            if self._window_has_speech(window_audio):
                text = self.engine.transcribe_window(
                    window_audio, self.cache, is_final=True
                )
                if text and is_meaningful_text(text, self.config.min_partial_chars):
                    self.draft = self._merge_segment(self.draft, text)

        if self.draft:
            final_text = self.engine.finalize_text(self.draft)
            if final_text and is_meaningful_text(final_text, self.config.min_partial_chars):
                self.confirmed = (
                    f"{self.confirmed} {final_text}".strip()
                    if self.confirmed
                    else final_text
                )
                out.final = final_text

        self.partial = ""
        self.draft = ""
        self.cache.clear()
        self.last_voice_ts = time.time()
        return out

    def stream_file_chunks(
        self,
        pcm_chunks: list[bytes],
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
    ) -> str:
        for chunk in pcm_chunks:
            result = self.feed_pcm(chunk)
            if result.partial and on_partial:
                on_partial(result.partial)
            if result.final and on_final:
                on_final(result.final)
        result = self.finalize()
        if result.final and on_final:
            on_final(result.final)
        return self.confirmed
