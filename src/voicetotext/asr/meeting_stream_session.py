"""In-process meeting stream v2: PCM -> partial/final + Pyannote speaker_id."""

from __future__ import annotations

import re
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
    Industry-style streaming session:
    - RMS + hangover keeps audio continuous through brief dips
    - FSMN-VAD confirms utterance endpoint before final
    - partial/final share one seg_id (same subtitle row)
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
        self._utterance_chunks: list[np.ndarray] = []
        # 0 = no voice yet; avoids hangover bypassing vad_speech_onset_chunks at session start
        self._last_rms_voice_ts = 0.0
        self._utterance_seg_id = new_seg_id()
        self._utterance_start_ms = 0
        self._last_partial_at = 0.0
        self._last_partial_text = ""
        self._last_speaker_id = 0
        self._finalize_deferred = False
        self._speech_onset_streak = 0

    def _elapsed_ms(self) -> int:
        return int((time.time() - self._session_start) * 1000)

    def _utterance_duration_ms(self) -> float:
        samples = sum(c.size for c in self._utterance_chunks)
        return samples / self.config.sample_rate * 1000.0

    def _rms_silence_ms(self) -> float:
        return (time.time() - self._last_rms_voice_ts) * 1000.0

    def _in_speech_capture(self) -> bool:
        return self._rms_silence_ms() < self.config.vad_speech_hangover_ms

    def _chunk_has_voice_energy(self, audio: np.ndarray) -> bool:
        return self.engine.detect_speech(audio)

    def _update_speech_gate(self, audio: np.ndarray) -> bool:
        """
        Require consecutive loud chunks before treating input as speech (noise gate).
        Once speech is open, a single loud chunk refreshes the voice timestamp.
        """
        loud = self._chunk_has_voice_energy(audio)
        need = max(1, self.config.vad_speech_onset_chunks)
        open_utterance = bool(self._utterance_chunks)

        if loud:
            self._speech_onset_streak += 1
        else:
            self._speech_onset_streak = 0

        if open_utterance or self._in_speech_capture():
            if loud:
                self._last_rms_voice_ts = time.time()
            return loud or self._in_speech_capture()

        if self._speech_onset_streak >= need:
            self._last_rms_voice_ts = time.time()
            return True
        return False

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
        """Drop trailing silence from open utterance before ASR (buffer keeps it for FSMN)."""
        if utterance.size == 0:
            return utterance
        tail_ms = max(0.0, self._rms_silence_ms() - self.config.vad_speech_hangover_ms)
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

    def _is_discardable_fragment(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        if len(text) < self.config.meeting_min_finalize_chars:
            return True
        return bool(re.fullmatch(r"[\s。，、．,.!?！？…・]+", text))

    def _should_defer_finalize(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if len(text) >= self.config.meeting_min_finalize_chars:
            return False
        return self._rms_silence_ms() < self.config.vad_silence_long_ms

    def _finalize_silence_threshold_ms(self) -> int:
        spoken_ms = self._utterance_duration_ms()
        if spoken_ms < self.config.meeting_min_utterance_ms:
            return self.config.vad_silence_long_ms
        return self.config.vad_silence_ms

    def _can_finalize(self, utterance: np.ndarray) -> bool:
        spoken_ms = self._utterance_duration_ms()
        if spoken_ms >= self.config.meeting_max_utterance_ms:
            return True

        silence = self._rms_silence_ms()
        threshold = self._finalize_silence_threshold_ms()
        if self._finalize_deferred:
            threshold = max(threshold, self.config.vad_silence_long_ms)
        if silence < threshold:
            return False

        fn = getattr(self.engine, "utterance_endpoint_reached", None)
        if callable(fn):
            if bool(fn(utterance)):
                return True
            # Long silence with open buffer: FSMN may lag; force endpoint.
            return silence >= self.config.vad_silence_long_ms * 2
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
            if not is_final:
                current_fn = getattr(self.engine, "current_speaker_id", None)
                if callable(current_fn):
                    current_spk = current_fn(self._speaker_ctx)
                    if current_spk is not None:
                        self._last_speaker_id = current_spk
                        return current_spk, False
                point_fn = getattr(self.engine, "speaker_at_ms", None)
                if callable(point_fn):
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

    def _should_finalize_on_speaker_change(self) -> bool:
        if not self.config.meeting_spk_change_finalize:
            return False
        if not self._uses_pyannote_speaker():
            return False
        if not self._utterance_chunks:
            return False
        if self._utterance_duration_ms() < self.config.meeting_min_utterance_ms:
            return False
        if self._speaker_ctx is None:
            return False
        fn = getattr(self.engine, "speaker_at_ms", None)
        if not callable(fn):
            return False
        start_spk = fn(self._speaker_ctx, self._utterance_start_ms)
        now_spk = fn(self._speaker_ctx, self._elapsed_ms())
        if start_spk is None or now_spk is None:
            return False
        return start_spk != now_spk

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
        cache: dict = {}
        speech_audio = self._speech_audio_for_asr(utterance)
        partial_audio = self._asr_audio_window(speech_audio, is_final=False)
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
        out: list[dict[str, Any]] = []

        speech = self._update_speech_gate(audio)

        open_utterance = bool(self._utterance_chunks)
        if not open_utterance and (speech or self._in_speech_capture()):
            self._utterance_start_ms = self._elapsed_ms()
            self._utterance_seg_id = new_seg_id()
            self._last_partial_text = ""
            self._finalize_deferred = False

        # Keep buffering through trailing silence so FSMN can see utterance endpoint.
        if open_utterance or speech or self._in_speech_capture():
            self._utterance_chunks.append(audio)

        utterance = self._concat_utterance()
        if utterance.size == 0:
            return out

        if self._should_finalize_on_speaker_change():
            partial_text = self._last_partial_text.strip()
            if len(partial_text) >= self.config.meeting_min_finalize_chars:
                logger.info(
                    "Speaker change boundary at %dms (seg=%s)",
                    self._elapsed_ms(),
                    self._utterance_seg_id,
                )
                out.extend(self._finalize_utterance())
                self._utterance_start_ms = self._elapsed_ms()
                self._utterance_seg_id = new_seg_id()
                self._last_partial_text = ""
                self._finalize_deferred = False
                self._utterance_chunks = [audio]
                utterance = self._concat_utterance()

        out.extend(self._maybe_emit_partial(utterance))

        if self._can_finalize(utterance):
            out.extend(self._finalize_utterance())

        return out

    def _finalize_utterance(self) -> list[dict[str, Any]]:
        utterance = self._concat_utterance()
        if utterance.size == 0:
            return []

        was_deferred = self._finalize_deferred
        if self._finalize_deferred:
            text = self._last_partial_text.strip()
            if not text or self._should_defer_finalize(text):
                return []
        else:
            speech_audio = self._speech_audio_for_asr(utterance)
            text = self.engine.finalize_utterance(
                speech_audio,
                self._last_partial_text,
                language=self.config.language,
            )
            if not text:
                self._reset_utterance()
                return []
            if self._should_defer_finalize(text):
                self._finalize_deferred = True
                self._last_partial_text = text
                logger.debug("Defer finalize (short fragment): %s", text[:32])
                return []

        if not was_deferred and self._is_discardable_fragment(text):
            logger.debug("Discard junk fragment: %s", text[:32])
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
        self._finalize_deferred = False
        self._speech_onset_streak = 0

    def finalize_all(self) -> list[dict[str, Any]]:
        return self._finalize_utterance()
