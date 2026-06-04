"""In-process meeting stream: PCM chunks -> protocol v2 partial/final."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from voicetotext.asr.base import ASRBackend
from voicetotext.asr.meeting_speaker import MeetingSpeakerAssigner
from voicetotext.server.protocol_common import new_seg_id
from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


def _synthetic_runtime_msg(
    text: str,
    *,
    t_start_ms: int,
    t_end_ms: int | None,
    is_final: bool,
) -> dict[str, Any]:
    end = t_end_ms if t_end_ms is not None else t_start_ms
    return {
        "text": text,
        "mode": "2pass-offline" if is_final else "2pass-online",
        "is_final": is_final,
        "timestamp": f"[[{t_start_ms},{end}]]",
        "stamp_sents": [{"start": t_start_ms, "end": end}],
    }


class MeetingStreamSession:
    """
    Buffers one utterance (speech until silence), runs ASR once at end.

    SenseVoice is not true streaming: partial per chunk caused duplicate/wrong lines.
    Default: only emit ``final`` after ``vad_silence_ms`` silence.
    """

    def __init__(
        self,
        engine: ASRBackend,
        config: AppConfig,
        *,
        session_start: float | None = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self._assigner = MeetingSpeakerAssigner(config)
        self._session_start = session_start or time.time()
        self._utterance_chunks: list[np.ndarray] = []
        self._pcm_ring = bytearray()
        self._ring_max_bytes = config.sample_rate * 2 * 30
        self._last_voice_ts = time.time()
        self._utterance_seg_id = new_seg_id()
        self._utterance_start_ms = 0
        self._last_partial_at = 0.0

    def _elapsed_ms(self) -> int:
        return int((time.time() - self._session_start) * 1000)

    def _append_ring(self, pcm_bytes: bytes) -> None:
        self._pcm_ring.extend(pcm_bytes)
        if len(self._pcm_ring) > self._ring_max_bytes:
            del self._pcm_ring[: len(self._pcm_ring) - self._ring_max_bytes]

    def _concat_utterance(self) -> np.ndarray:
        if not self._utterance_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._utterance_chunks)

    def _silence_elapsed_ms(self) -> float:
        return (time.time() - self._last_voice_ts) * 1000.0

    def _map_text(
        self,
        text: str,
        *,
        is_final: bool,
        t_start_ms: int,
        t_end_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        text = text.strip()
        if not text or len(text) < self.config.min_partial_chars and not is_final:
            return []

        runtime_msg = _synthetic_runtime_msg(
            text,
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms or self._elapsed_ms(),
            is_final=is_final,
        )
        pcm_window = bytes(self._pcm_ring) if is_final else None
        speaker_id, t_start, t_end, speaker_changed = self._assigner.assign(
            runtime_msg, pcm_window=pcm_window
        )
        if t_start is None:
            t_start = t_start_ms
        if is_final and t_end is None:
            t_end = t_end_ms or self._elapsed_ms()

        out: list[dict[str, Any]] = []
        if speaker_changed and is_final:
            out.append(
                {
                    "type": "speaker_change",
                    "protocol_version": 2,
                    "speaker_id": speaker_id,
                    "t_ms": t_end or t_start or self._elapsed_ms(),
                }
            )

        msg_type = "final" if is_final else "partial"
        out.append(
            {
                "type": msg_type,
                "protocol_version": 2,
                "seg_id": self._utterance_seg_id,
                "speaker_id": speaker_id,
                "text": text,
                "t_start_ms": t_start,
                "t_end_ms": t_end if is_final else None,
                "mode": "embedded",
                "is_final": is_final,
            }
        )
        return out

    def _maybe_emit_partial(self, utterance: np.ndarray) -> list[dict[str, Any]]:
        if not self.config.meeting_emit_partial:
            return []
        now = time.time()
        if now - self._last_partial_at < 1.5:
            return []
        self._last_partial_at = now
        cache: dict = {}
        partial = self.engine.transcribe_window(utterance, cache, is_final=False)
        if not partial:
            return []
        return self._map_text(
            partial,
            is_final=False,
            t_start_ms=self._utterance_start_ms,
        )

    def feed_pcm(self, pcm_bytes: bytes) -> list[dict[str, Any]]:
        self._append_ring(pcm_bytes)
        audio = self.engine.pcm_bytes_to_float32(pcm_bytes)
        out: list[dict[str, Any]] = []

        if self.engine.detect_speech(audio):
            if not self._utterance_chunks:
                self._utterance_start_ms = self._elapsed_ms()
                self._utterance_seg_id = new_seg_id()
            self._last_voice_ts = time.time()
            self._utterance_chunks.append(audio)

        utterance = self._concat_utterance()
        if utterance.size == 0:
            return out

        out.extend(self._maybe_emit_partial(utterance))

        if self._silence_elapsed_ms() >= self.config.vad_silence_ms:
            out.extend(self._finalize_utterance())

        return out

    def _finalize_utterance(self) -> list[dict[str, Any]]:
        utterance = self._concat_utterance()
        if utterance.size == 0:
            return []

        text = self.engine.finalize_utterance(utterance, "")
        if not text:
            self._reset_utterance()
            return []

        out = self._map_text(
            text,
            is_final=True,
            t_start_ms=self._utterance_start_ms,
            t_end_ms=self._elapsed_ms(),
        )
        self._reset_utterance()
        return out

    def _reset_utterance(self) -> None:
        self._utterance_chunks.clear()
        self._utterance_seg_id = new_seg_id()
        self._last_voice_ts = time.time()
        self._last_partial_at = 0.0

    def finalize_all(self) -> list[dict[str, Any]]:
        return self._finalize_utterance()
