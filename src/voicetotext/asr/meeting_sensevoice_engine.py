"""Meeting v3 ASR backend: FSMN-VAD + SenseVoice + Pyannote + utterance embedding."""

from __future__ import annotations

import os
import threading
import uuid

import numpy as np

from voicetotext.asr.funasr_vad import FunASRVAD
from voicetotext.asr.participant_registry import ParticipantSpeakerRegistry
from voicetotext.asr.pyannote_worker import PyannoteWorker
from voicetotext.asr.sensevoice_funasr_engine import SenseVoiceFunASREngine
from voicetotext.asr.speaker_session import SpeakerSessionContext
from voicetotext.asr.utterance_speaker import UtteranceSpeakerEngine
from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class MeetingSenseVoiceEngine:
    backend_name = "meeting_sensevoice"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        requested = config.device
        self.device = resolve_device(config.device)
        if requested.lower().startswith("cuda") and self.device == "cpu":
            logger.warning(
                "CUDA requested (device=%s) but unavailable; falling back to CPU",
                requested,
            )
        self._vad = FunASRVAD(config)
        self._asr = SenseVoiceFunASREngine(config)
        self._pyannote = PyannoteWorker(config, device=self.device)
        self._embedding: UtteranceSpeakerEngine | None = None
        if self._uses_utterance_embedding():
            self._embedding = UtteranceSpeakerEngine(config, self.device)
        self._participants = ParticipantSpeakerRegistry(config.meeting_max_speakers)
        self._sessions: dict[str, SpeakerSessionContext] = {}
        self._sessions_lock = threading.Lock()
        self._active_session_count = 0
        self._inference_lock = threading.Lock()
        self._ready = False

    @property
    def is_loaded(self) -> bool:
        return self._ready

    def _reuse_partial_for_final(self) -> bool:
        return self.config.meeting_reuse_partial_for_final

    def uses_pyannote(self) -> bool:
        if self.config.meeting_spk_mode != "multi" or not self.config.meeting_use_diarization:
            return False
        return self.config.meeting_spk_source in ("pyannote", "hybrid")

    def _uses_utterance_embedding(self) -> bool:
        if self.config.meeting_spk_mode != "multi":
            return False
        if not self.config.meeting_use_diarization:
            return False
        if not self.config.meeting_use_utterance_embedding:
            return False
        return self.config.meeting_spk_source in ("pyannote", "hybrid")

    def uses_client_speaker(self) -> bool:
        if self.config.meeting_spk_mode != "multi":
            return False
        return self.config.meeting_spk_source in ("client", "hybrid")

    def open_speaker_context(self, session_id: str | None = None) -> SpeakerSessionContext:
        """Create isolated speaker state for one WebSocket meeting."""
        sid = (session_id or "").strip() or str(uuid.uuid4())
        ctx = SpeakerSessionContext.create(self.config, sid)
        with self._sessions_lock:
            self._sessions[sid] = ctx
            self._active_session_count += 1
        if self.uses_pyannote():
            self._pyannote.bind_session(ctx)
        logger.info(
            "Speaker context opened session_id=%s active=%d",
            sid,
            self._active_session_count,
        )
        return ctx

    def close_speaker_context(self, ctx: SpeakerSessionContext) -> None:
        with self._sessions_lock:
            self._sessions.pop(ctx.session_id, None)
            self._active_session_count = max(0, self._active_session_count - 1)
            active = self._active_session_count
        if self.uses_pyannote():
            self._pyannote.unbind_session(ctx)
        ctx.reset()
        logger.info(
            "Speaker context closed session_id=%s active=%d",
            ctx.session_id,
            active,
        )

    def begin_speaker_session(self) -> SpeakerSessionContext:
        """Backward-compatible alias for open_speaker_context()."""
        return self.open_speaker_context()

    def end_speaker_session(self, ctx: SpeakerSessionContext | None = None) -> None:
        if ctx is not None:
            self.close_speaker_context(ctx)

    @property
    def active_session_count(self) -> int:
        with self._sessions_lock:
            return self._active_session_count

    def register_participant(self, participant_id: str | None) -> int | None:
        if not self.uses_client_speaker() or not participant_id:
            return None
        return self._participants.speaker_id_for(participant_id.strip())

    def release_participant(self, participant_id: str | None) -> None:
        if participant_id:
            self._participants.release(participant_id.strip())

    def _tune_cpu_threads(self) -> None:
        if not self.device.startswith("cpu"):
            return
        try:
            import torch

            cores = os.cpu_count() or 4
            threads = max(2, min(4, cores // 2))
            torch.set_num_threads(threads)
            logger.info("MeetingSenseVoiceEngine CPU threads=%s", threads)
        except ImportError:
            pass

    def _utterance_seconds(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return audio.size / self.config.sample_rate

    def _can_reuse_partial_draft(self, audio: np.ndarray, draft: str) -> bool:
        if not self._reuse_partial_for_final() or not draft:
            return False
        max_sec = self.config.meeting_partial_max_sec
        if max_sec <= 0:
            return True
        return self._utterance_seconds(audio) <= max_sec

    def load(self) -> None:
        self._tune_cpu_threads()
        self._vad.load()
        self._asr.load()
        if self.uses_pyannote():
            self._pyannote.load()
            self._pyannote.start()
        if self._embedding is not None:
            self._embedding.load()
        self._ready = True
        logger.info(
            "MeetingSenseVoiceEngine ready (asr=%s device=%s embedding=%s primary=%s)",
            self.config.asr_model,
            self.device,
            self._embedding is not None,
            self.config.meeting_spk_primary,
        )

    def shutdown(self) -> None:
        if self.uses_pyannote():
            self._pyannote.stop()

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def append_pcm_for_diarization(
        self,
        ctx: SpeakerSessionContext,
        pcm_bytes: bytes,
    ) -> None:
        if self.uses_pyannote():
            self._pyannote.append_pcm(ctx, pcm_bytes)

    def sync_timeline(self, ctx: SpeakerSessionContext) -> None:
        if self.uses_pyannote():
            ctx.merger.update_segments(ctx.pyannote_segments)

    def detect_speech(self, audio: np.ndarray) -> bool:
        return self._vad.detect_speech_frame(audio)

    def utterance_endpoint_reached(self, audio: np.ndarray) -> bool:
        if not self.config.meeting_use_fsmn_endpoint:
            return True
        return self._vad.utterance_endpoint_reached(audio)

    def transcribe_window(
        self,
        audio: np.ndarray,
        cache: dict,
        *,
        is_final: bool,
        language: str | None = None,
    ) -> str:
        lang = language or self.config.language
        with self._inference_lock:
            return self._asr.transcribe(audio, cache, is_final=is_final, language=lang)

    def finalize_text(self, text: str) -> str:
        return text.strip()

    def finalize_utterance(
        self,
        audio: np.ndarray,
        draft_fallback: str,
        *,
        language: str | None = None,
    ) -> str:
        draft = draft_fallback.strip()
        if self._can_reuse_partial_draft(audio, draft):
            return self.finalize_text(draft)
        cache: dict = {}
        lang = language or self.config.language
        with self._inference_lock:
            text = self._asr.transcribe(audio, cache, is_final=True, language=lang)
        if text:
            return self.finalize_text(text)
        return self.finalize_text(draft)

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.config.sample_rate:
            logger.warning("Expected sample_rate=%s", self.config.sample_rate)
        return self.finalize_utterance(audio, "")

    def _pyannote_assignment(
        self,
        ctx: SpeakerSessionContext,
        t_start_ms: int,
        t_end_ms: int,
    ) -> tuple[int, int, bool]:
        self.sync_timeline(ctx)
        merger = ctx.merger
        single = self.config.meeting_spk_mode != "multi"
        if single:
            changed = merger.last_speaker_id != 0
            merger.force_speaker_id(0)
            return 0, 0, changed

        label, overlap_ms = merger.best_overlap_label(t_start_ms, t_end_ms)
        if not label or overlap_ms <= 0:
            return merger.last_speaker_id, 0, False

        speaker_id = merger._label_to_speaker_id(label)
        changed = speaker_id != merger.last_speaker_id
        merger._last_speaker_id = speaker_id
        logger.info(
            "assign_speaker [%d,%d] label=%s id=%d overlap_ms=%d (pyannote)",
            t_start_ms,
            t_end_ms,
            label,
            speaker_id,
            overlap_ms,
        )
        return speaker_id, overlap_ms, changed

    def _embedding_assignment(
        self,
        ctx: SpeakerSessionContext,
        audio: np.ndarray | None,
    ) -> int | None:
        if self._embedding is None or audio is None or audio.size == 0:
            return None
        with self._inference_lock:
            return self._embedding.assign_from_audio(audio, ctx.registry)

    def resolve_speaker(
        self,
        ctx: SpeakerSessionContext,
        t_start_ms: int,
        t_end_ms: int,
        *,
        audio: np.ndarray | None = None,
    ) -> tuple[int, bool]:
        if not self.uses_pyannote():
            return 0, False

        merger = ctx.merger
        primary = self.config.meeting_spk_primary
        min_overlap = self.config.meeting_pyannote_min_overlap_ms
        py_id, py_overlap, py_changed = self._pyannote_assignment(ctx, t_start_ms, t_end_ms)
        py_reliable = py_overlap >= min_overlap
        emb_id = self._embedding_assignment(ctx, audio)

        if primary == "pyannote":
            if py_reliable:
                return py_id, py_changed
            if emb_id is not None:
                logger.info("Speaker pyannote-unreliable -> embedding id=%d", emb_id)
                return merger.force_speaker_id(emb_id)
            return py_id, py_changed

        if primary == "fusion":
            if emb_id is not None and py_reliable:
                if emb_id != py_id:
                    logger.info(
                        "Speaker fusion: embedding=%d pyannote=%d -> embedding",
                        emb_id,
                        py_id,
                    )
                return merger.force_speaker_id(emb_id)
            if emb_id is not None:
                return merger.force_speaker_id(emb_id)
            if py_reliable:
                return py_id, py_changed
            return merger.last_speaker_id, False

        if emb_id is not None:
            if py_reliable and emb_id != py_id:
                logger.info(
                    "Speaker embedding-primary id=%d (pyannote=%d overlap=%dms)",
                    emb_id,
                    py_id,
                    py_overlap,
                )
            return merger.force_speaker_id(emb_id)
        if py_reliable:
            logger.info("Speaker embedding-miss -> pyannote id=%d", py_id)
            return py_id, py_changed
        return merger.last_speaker_id, False

    def current_speaker_id(self, ctx: SpeakerSessionContext) -> int | None:
        if ctx.registry.speaker_count > 0:
            return ctx.registry.last_speaker_id
        if ctx.merger.last_speaker_id >= 0:
            return ctx.merger.last_speaker_id
        return None

    def speaker_at_ms(self, ctx: SpeakerSessionContext, t_ms: int) -> int | None:
        if not self.uses_pyannote():
            return None
        if self.config.meeting_spk_primary in ("embedding", "fusion"):
            current = self.current_speaker_id(ctx)
            if current is not None:
                return current
        self.sync_timeline(ctx)
        point = ctx.merger.speaker_at_ms(t_ms)
        if point is not None:
            return point
        return self.current_speaker_id(ctx)

    async def check_ready(self) -> bool:
        if not self._ready:
            return False
        if self.uses_pyannote():
            if not self._pyannote.hf_token():
                return False
            if not self._pyannote.is_loaded:
                return False
        if self._embedding is not None and not self._embedding.is_loaded:
            return False
        return self._vad.is_loaded and self._asr.is_loaded

    def readiness_detail(self) -> dict[str, str | bool | int]:
        token_ok = (
            bool(self._pyannote.hf_token()) if self.uses_pyannote() else True
        )
        return {
            "vad_loaded": self._vad.is_loaded,
            "asr_loaded": self._asr.is_loaded,
            "pyannote_loaded": self._pyannote.is_loaded if self.uses_pyannote() else None,
            "embedding_loaded": (
                self._embedding.is_loaded if self._embedding is not None else None
            ),
            "hf_token_present": token_ok,
            "spk_source": self.config.meeting_spk_source,
            "spk_primary": self.config.meeting_spk_primary,
            "device_resolved": self.device,
            "active_speaker_sessions": self.active_session_count,
            "engine_ready": self._ready,
        }
