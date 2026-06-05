"""In-process meeting stream: PCM -> partial/final + Pyannote speaker_id."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from typing import TYPE_CHECKING

from voicetotext.asr.base import ASRBackend
from voicetotext.server.protocol_common import new_seg_id
from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

if TYPE_CHECKING:
    from voicetotext.asr.speaker_session import SpeakerSessionContext

logger = get_logger(__name__)


class MeetingStreamSession:
    """
    Standard streaming session:
    - RMS frame VAD + hangover keeps audio through brief dips
    - FSMN-VAD confirms utterance endpoint before final
    - partial/final share one seg_id; final always runs full-utterance ASR
    - speaker_id from Pyannote timeline overlap (audio sample clock)
    """

    def __init__(
        self,
        engine: ASRBackend,
        config: AppConfig,
        *,
        session_start: float | None = None,
        client_speaker_id: int | None = None,
        speaker_ctx: SpeakerSessionContext | None = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self._speaker_ctx = speaker_ctx
        self._client_speaker_id = client_speaker_id
        self._session_start = session_start or time.time()
        self._asr_cache: dict = {}
        self._utterance_chunks: list[np.ndarray] = []
        self._session_samples = 0
        self._last_voice_ts = 0.0
        self._utterance_seg_id = new_seg_id()
        self._utterance_start_ms = 0
        self._last_partial_at = 0.0
        self._last_partial_text = ""
        self._last_speaker_id = 0

    def _elapsed_ms(self) -> int:
        return int(self._session_samples / self.config.sample_rate * 1000)

    def _utterance_duration_ms(self) -> float:
        samples = sum(c.size for c in self._utterance_chunks)
        return samples / self.config.sample_rate * 1000.0

    def _silence_ms(self) -> float:
        if self._last_voice_ts <= 0:
            return float("inf")
        return (time.time() - self._last_voice_ts) * 1000.0

    def _in_speech_hangover(self) -> bool:
        return self._silence_ms() < self.config.vad_speech_hangover_ms

    def _chunk_is_speech(self, audio: np.ndarray) -> bool:
        return self.engine.detect_speech(audio)

    def _speech_active(self, audio: np.ndarray) -> bool:
        """Frame VAD with hangover: standard open-utterance gating."""
        loud = self._chunk_is_speech(audio)
        if loud:
            self._last_voice_ts = time.time()
        if self._utterance_chunks:
            return loud or self._in_speech_hangover()
        return loud

    def _append_diarization_pcm(self, pcm_bytes: bytes) -> None:
        if self._speaker_ctx is None:
            return
        fn = getattr(self.engine, "append_pcm_for_diarization", None)
        if callable(fn):
            fn(self._speaker_ctx, pcm_bytes)

    def _concat_utterance(self) -> np.ndarray:
        if not self._utterance_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._utterance_chunks)

    def _speech_audio_for_asr(self, utterance: np.ndarray) -> np.ndarray:
        if utterance.size == 0:
            return utterance
        silence_ms = self._silence_ms()
        if not np.isfinite(silence_ms):
            return utterance
        tail_ms = max(0.0, silence_ms - self.config.vad_speech_hangover_ms)
        if tail_ms <= 0:
            return utterance
        trim_samples = int(tail_ms / 1000.0 * self.config.sample_rate)
        if trim_samples <= 0 or trim_samples >= utterance.size:
            return utterance
        return utterance[:-trim_samples]

    def _asr_audio_window(self, utterance: np.ndarray, *, is_final: bool) -> np.ndarray:
        if is_final:
            return utterance
        max_sec = self.config.meeting_partial_max_sec
        if max_sec <= 0:
            return utterance
        max_samples = int(self.config.sample_rate * max_sec)
        if utterance.size <= max_samples:
            return utterance
        return utterance[-max_samples:]

    def _can_finalize(self, utterance: np.ndarray) -> bool:
        if self._utterance_duration_ms() >= self.config.meeting_max_utterance_ms:
            return True
        if self._silence_ms() < self.config.vad_silence_ms:
            return False
        fn = getattr(self.engine, "utterance_endpoint_reached", None)
        if callable(fn) and self.config.meeting_use_fsmn_endpoint:
            return bool(fn(utterance))
        return True

    def _should_emit_partial_text(self, new_text: str) -> bool:
        old = self._last_partial_text
        if not old:
            return True
        if new_text == old:
            return False
        if new_text.startswith(old):
            return True
        return len(new_text) >= len(old)

    def _uses_pyannote_speaker(self) -> bool:
        if self.config.meeting_spk_mode != "multi":
            return False
        if not self.config.meeting_use_diarization:
            return False
        return self.config.meeting_spk_source in ("pyannote", "hybrid")

    def _uses_client_speaker(self) -> bool:
        if self.config.meeting_spk_mode != "multi":
            return False
        return self.config.meeting_spk_source in ("client", "hybrid")

    def _resolve_speaker(
        self,
        t_start_ms: int,
        t_end_ms: int | None = None,
        *,
        is_final: bool,
        audio: np.ndarray | None = None,
    ) -> tuple[int, bool]:
        if self.config.meeting_spk_mode != "multi":
            return 0, False

        if self._uses_client_speaker() and self._client_speaker_id is not None:
            changed = self._last_speaker_id != self._client_speaker_id
            self._last_speaker_id = self._client_speaker_id
            return self._client_speaker_id, changed and is_final

        if self._uses_pyannote_speaker() and self._speaker_ctx is not None:
            end_ms = t_end_ms if t_end_ms is not None else self._elapsed_ms()
            point_fn = getattr(self.engine, "speaker_at_ms", None)
            if not is_final and callable(point_fn):
                point_spk = point_fn(self._speaker_ctx, end_ms)
                if point_spk is not None:
                    self._last_speaker_id = point_spk
                    return point_spk, False
            fn = getattr(self.engine, "resolve_speaker", None)
            if callable(fn):
                kwargs: dict[str, Any] = {}
                if is_final and audio is not None and audio.size > 0:
                    kwargs["audio"] = audio
                speaker_id, changed = fn(
                    self._speaker_ctx, t_start_ms, end_ms, **kwargs
                )
                self._last_speaker_id = speaker_id
                return speaker_id, changed if is_final else False

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
        utterance_audio: np.ndarray | None = None
        if is_final:
            raw = self._concat_utterance()
            if raw.size > 0:
                utterance_audio = self._speech_audio_for_asr(raw)
        speaker_id, speaker_changed = self._resolve_speaker(
            t_start_ms,
            end_ms,
            is_final=is_final,
            audio=utterance_audio,
        )

        out: list[dict[str, Any]] = []
        if (
            speaker_changed
            and is_final
            and self.config.meeting_spk_mode == "multi"
            and self._uses_pyannote_speaker()
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
                "mode": getattr(self.engine, "backend_name", "meeting_sensevoice"),
                "is_final": is_final,
            }
        )
        logger.info(
            "Meeting subtitle %s seg=%s spk=%d chars=%d",
            msg_type,
            self._utterance_seg_id,
            speaker_id,
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
        speech_audio = self._speech_audio_for_asr(utterance)
        partial_audio = self._asr_audio_window(speech_audio, is_final=False)
        cache = (
            self._asr_cache
            if self.config.meeting_partial_max_sec <= 0
            else {}
        )
        partial = self.engine.transcribe_window(
            partial_audio,
            cache,
            is_final=False,
            language=self.config.language,
        )
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
        self._session_samples += audio.size
        out: list[dict[str, Any]] = []

        speech = self._speech_active(audio)
        open_utterance = bool(self._utterance_chunks)
        if not open_utterance and speech:
            self._utterance_start_ms = self._elapsed_ms() - int(
                audio.size / self.config.sample_rate * 1000
            )
            self._utterance_seg_id = new_seg_id()
            self._last_partial_text = ""
            self._asr_cache = {}

        if open_utterance or speech:
            self._utterance_chunks.append(audio)

        utterance = self._concat_utterance()
        if utterance.size == 0:
            return out

        out.extend(self._maybe_emit_partial(utterance))

        if self._can_finalize(utterance):
            out.extend(self._finalize_utterance())

        return out

    def _finalize_utterance(self) -> list[dict[str, Any]]:
        utterance = self._concat_utterance()
        if utterance.size == 0:
            return []

        speech_audio = self._speech_audio_for_asr(utterance)
        try:
            text = self.engine.finalize_utterance(
                speech_audio,
                self._last_partial_text,
                language=self.config.language,
            )
        except TypeError:
            text = self.engine.finalize_utterance(speech_audio, self._last_partial_text)
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
        self._last_partial_at = 0.0
        self._last_partial_text = ""
        self._asr_cache = {}

    def finalize_all(self) -> list[dict[str, Any]]:
        return self._finalize_utterance()
