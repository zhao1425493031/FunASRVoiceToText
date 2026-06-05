"""Meeting v2 ASR backend: FSMN-VAD + Qwen3 partial/final + Pyannote."""

from __future__ import annotations

import os
import threading

import numpy as np

from voicetotext.asr.funasr_vad import FunASRVAD
from voicetotext.asr.participant_registry import ParticipantSpeakerRegistry
from voicetotext.asr.pyannote_worker import PyannoteWorker
from voicetotext.asr.qwen_funasr_engine import QwenFunASREngine
from voicetotext.asr.speaker_timeline import SpeakerTimelineMerger
from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class MeetingQwenEngine:
    backend_name = "meeting_qwen"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._vad = FunASRVAD(config)
        self._partial = QwenFunASREngine(config, model_id=config.qwen_partial_model)
        self._final = QwenFunASREngine(config, model_id=config.qwen_final_model)
        self._pyannote = PyannoteWorker(config)
        self._merger = SpeakerTimelineMerger(max_speakers=config.meeting_max_speakers)
        self._participants = ParticipantSpeakerRegistry(config.meeting_max_speakers)
        self._session_language: str | None = None
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

    def uses_client_speaker(self) -> bool:
        if self.config.meeting_spk_mode != "multi":
            return False
        return self.config.meeting_spk_source in ("client", "hybrid")

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
            logger.info("MeetingQwenEngine CPU threads=%s", threads)
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
        self._partial.load()
        if self._reuse_partial_for_final():
            self._final = self._partial
            logger.info(
                "MeetingQwenEngine CPU-fast: reusing partial model for final (%s)",
                self.config.qwen_partial_model,
            )
        else:
            self._final.load()
        if self.uses_pyannote():
            self._pyannote.load()
            self._pyannote.start()
        self._ready = True
        final_model = (
            self.config.qwen_partial_model
            if self._reuse_partial_for_final()
            else self.config.qwen_final_model
        )
        logger.info("MeetingQwenEngine ready (partial=%s final=%s)",
                    self.config.qwen_partial_model, final_model)

    def shutdown(self) -> None:
        if self.uses_pyannote():
            self._pyannote.stop()

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def append_pcm_for_diarization(self, pcm_bytes: bytes) -> None:
        if self.uses_pyannote():
            self._pyannote.append_pcm(pcm_bytes)

    def sync_timeline(self) -> None:
        if self.uses_pyannote():
            self._merger.update_segments(self._pyannote.get_segments())

    def detect_speech(self, audio: np.ndarray) -> bool:
        return self._vad.detect_speech_frame(audio)

    def utterance_endpoint_reached(self, audio: np.ndarray) -> bool:
        if not self.config.meeting_use_fsmn_endpoint:
            return True
        return self._vad.utterance_endpoint_reached(audio)

    def set_session_language(self, language: str | None) -> None:
        self._session_language = language

    def _active_language(self) -> str:
        return self._session_language or self.config.language

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        lang = self._active_language()
        with self._inference_lock:
            if is_final:
                return self._final.transcribe(audio, cache, is_final=True, language=lang)
            return self._partial.transcribe(audio, cache, is_final=False, language=lang)

    def finalize_text(self, text: str) -> str:
        return text.strip()

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        draft = draft_fallback.strip()
        if self._can_reuse_partial_draft(audio, draft):
            return self.finalize_text(draft)
        cache: dict = {}
        lang = self._active_language()
        with self._inference_lock:
            text = self._final.transcribe(audio, cache, is_final=True, language=lang)
        if text:
            return self.finalize_text(text)
        return self.finalize_text(draft)

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.config.sample_rate:
            logger.warning("Expected sample_rate=%s", self.config.sample_rate)
        cache: dict = {}
        return self.finalize_utterance(audio, "")

    def resolve_speaker(
        self,
        t_start_ms: int,
        t_end_ms: int,
    ) -> tuple[int, bool]:
        if not self.uses_pyannote():
            return 0, False
        self.sync_timeline()
        single = self.config.meeting_spk_mode != "multi"
        return self._merger.assign_speaker(
            t_start_ms,
            t_end_ms,
            single_speaker_mode=single,
        )

    def last_speaker_id(self) -> int:
        return self._merger._last_speaker_id

    async def check_ready(self) -> bool:
        if not self._ready:
            return False
        if self.uses_pyannote():
            if not self._pyannote.hf_token():
                return False
            if not self._pyannote.is_loaded:
                return False
        return (
            self._vad.is_loaded
            and self._partial.is_loaded
            and self._final.is_loaded
        )

    def readiness_detail(self) -> dict[str, str | bool]:
        token_ok = (
            bool(self._pyannote.hf_token()) if self.uses_pyannote() else True
        )
        return {
            "vad_loaded": self._vad.is_loaded,
            "partial_loaded": self._partial.is_loaded,
            "final_loaded": self._final.is_loaded,
            "pyannote_loaded": self._pyannote.is_loaded if self.uses_pyannote() else None,
            "hf_token_present": token_ok,
            "spk_source": self.config.meeting_spk_source,
            "engine_ready": self._ready,
        }
