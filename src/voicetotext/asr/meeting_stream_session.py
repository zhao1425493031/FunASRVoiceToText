"""In-process meeting stream v2: PCM -> partial/final + Pyannote speaker_id."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from voicetotext.asr.base import ASRBackend
from voicetotext.server.protocol_common import new_seg_id
from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class MeetingStreamSession:
    """Utterance buffer + throttled partial + final on silence (FSMN frame VAD)."""

    def __init__(
        self,
        engine: ASRBackend,
        config: AppConfig,
        *,
        session_start: float | None = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self._session_start = session_start or time.time()
        self._utterance_chunks: list[np.ndarray] = []
        self._last_voice_ts = time.time()
        self._utterance_seg_id = new_seg_id()
        self._utterance_start_ms = 0
        self._last_partial_at = 0.0
        self._last_partial_text = ""
        self._last_speaker_id = 0

    def _elapsed_ms(self) -> int:
        return int((time.time() - self._session_start) * 1000)

    def _utterance_duration_ms(self) -> float:
        samples = sum(c.size for c in self._utterance_chunks)
        return samples / self.config.sample_rate * 1000.0

    def _append_diarization_pcm(self, pcm_bytes: bytes) -> None:
        fn = getattr(self.engine, "append_pcm_for_diarization", None)
        if callable(fn):
            fn(pcm_bytes)

    def _concat_utterance(self) -> np.ndarray:
        if not self._utterance_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._utterance_chunks)

    def _silence_elapsed_ms(self) -> float:
        return (time.time() - self._last_voice_ts) * 1000.0

    def _should_defer_finalize(self, text: str) -> bool:
        spoken_ms = self._utterance_duration_ms()
        min_chars = self.config.meeting_min_finalize_chars
        min_ms = self.config.meeting_min_utterance_ms
        if len(text.strip()) < min_chars and spoken_ms < min_ms:
            return True
        return False

    def _can_finalize(self) -> bool:
        silence = self._silence_elapsed_ms()
        spoken_ms = self._utterance_duration_ms()
        if spoken_ms < self.config.meeting_min_utterance_ms:
            return silence >= self.config.vad_silence_long_ms
        return silence >= self.config.vad_silence_ms

    def _should_emit_partial_text(self, new_text: str) -> bool:
        old = self._last_partial_text
        if not old:
            return True
        if new_text == old:
            return False
        if new_text.startswith(old):
            return True
        return len(new_text) >= len(old)

    def _resolve_speaker(
        self,
        t_start_ms: int,
        t_end_ms: int,
        *,
        is_final: bool,
    ) -> tuple[int, bool]:
        if is_final:
            fn = getattr(self.engine, "resolve_speaker", None)
            if callable(fn):
                speaker_id, changed = fn(t_start_ms, t_end_ms)
                self._last_speaker_id = speaker_id
                return speaker_id, changed
        fn_last = getattr(self.engine, "last_speaker_id", None)
        if callable(fn_last):
            self._last_speaker_id = int(fn_last())
        if self.config.meeting_spk_mode != "multi" or not self.config.meeting_use_diarization:
            return 0, False
        return self._last_speaker_id, False

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

        end_ms = t_end_ms or self._elapsed_ms()
        speaker_id, speaker_changed = self._resolve_speaker(
            t_start_ms, end_ms, is_final=is_final
        )

        out: list[dict[str, Any]] = []
        if (
            speaker_changed
            and is_final
            and self.config.meeting_spk_mode == "multi"
            and self.config.meeting_use_diarization
        ):
            out.append(
                {
                    "type": "speaker_change",
                    "protocol_version": 2,
                    "speaker_id": speaker_id,
                    "t_ms": end_ms,
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
                "t_start_ms": t_start_ms,
                "t_end_ms": end_ms if is_final else None,
                "mode": getattr(self.engine, "backend_name", "meeting_qwen"),
                "is_final": is_final,
            }
        )
        logger.info(
            "Meeting subtitle %s seg=%s chars=%d",
            msg_type,
            self._utterance_seg_id,
            len(text),
        )
        return out

    def _maybe_emit_partial(self, utterance: np.ndarray) -> list[dict[str, Any]]:
        if not self.config.meeting_emit_partial:
            return []
        if self._utterance_duration_ms() < self.config.meeting_partial_min_ms:
            return []
        now = time.time()
        interval_s = self.config.meeting_partial_interval_ms / 1000.0
        if now - self._last_partial_at < interval_s:
            return []
        self._last_partial_at = now
        cache: dict = {}
        partial = self.engine.transcribe_window(utterance, cache, is_final=False)
        if not partial or not self._should_emit_partial_text(partial):
            return []
        self._last_partial_text = partial
        return self._map_text(
            partial,
            is_final=False,
            t_start_ms=self._utterance_start_ms,
        )

    def feed_pcm(self, pcm_bytes: bytes) -> list[dict[str, Any]]:
        self._append_diarization_pcm(pcm_bytes)
        audio = self.engine.pcm_bytes_to_float32(pcm_bytes)
        out: list[dict[str, Any]] = []

        if self.engine.detect_speech(audio):
            if not self._utterance_chunks:
                self._utterance_start_ms = self._elapsed_ms()
                self._utterance_seg_id = new_seg_id()
                self._last_partial_text = ""
            self._last_voice_ts = time.time()
            self._utterance_chunks.append(audio)

        utterance = self._concat_utterance()
        if utterance.size == 0:
            return out

        out.extend(self._maybe_emit_partial(utterance))

        if self._can_finalize():
            out.extend(self._finalize_utterance())

        return out

    def _finalize_utterance(self) -> list[dict[str, Any]]:
        utterance = self._concat_utterance()
        if utterance.size == 0:
            return []

        text = self.engine.finalize_utterance(utterance, self._last_partial_text)
        if not text:
            self._reset_utterance()
            return []

        if self._should_defer_finalize(text):
            logger.debug("Defer finalize (short fragment): %s", text[:32])
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
        self._last_partial_text = ""

    def finalize_all(self) -> list[dict[str, Any]]:
        return self._finalize_utterance()
