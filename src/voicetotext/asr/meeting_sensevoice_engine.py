"""Meeting v3 ASR backend: FSMN-VAD + SenseVoice + Pyannote + utterance embedding."""

from __future__ import annotations

import os
import threading

import numpy as np

from voicetotext.asr.funasr_vad import FunASRVAD
from voicetotext.asr.participant_registry import ParticipantSpeakerRegistry
from voicetotext.asr.pyannote_worker import PyannoteWorker
from voicetotext.asr.sensevoice_funasr_engine import SenseVoiceFunASREngine
from voicetotext.asr.speaker_timeline import SpeakerTimelineMerger
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

    def begin_speaker_session(self) -> None:
        """Reset diarization state for a new WebSocket meeting session."""
        self._merger.clear()
        if self.uses_pyannote():
            self._pyannote.reset_session()
        if self._embedding is not None:
            self._embedding.reset_session()
        logger.info("Speaker session begun (timeline + pyannote + embedding reset)")

    def end_speaker_session(self) -> None:
        if self.uses_pyannote():
            self._pyannote.reset_session()
        if self._embedding is not None:
            self._embedding.reset_session()

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
            "MeetingSenseVoiceEngine ready (asr=%s device=%s embedding=%s)",
            self.config.asr_model,
            self.device,
            self._embedding is not None,
        )

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
            return self._asr.transcribe(audio, cache, is_final=is_final, language=lang)

    def finalize_text(self, text: str) -> str:
        return text.strip()

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        draft = draft_fallback.strip()
        if self._can_reuse_partial_draft(audio, draft):
            return self.finalize_text(draft)
        cache: dict = {}
        lang = self._active_language()
        with self._inference_lock:
            text = self._asr.transcribe(audio, cache, is_final=True, language=lang)
        if text:
            return self.finalize_text(text)
        return self.finalize_text(draft)

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.config.sample_rate:
            logger.warning("Expected sample_rate=%s", self.config.sample_rate)
        return self.finalize_utterance(audio, "")

    def _should_use_embedding(
        self,
        audio: np.ndarray | None,
        pyannote_labels: int,
    ) -> bool:
        if self._embedding is None or audio is None or audio.size == 0:
            return False
        if pyannote_labels >= 2:
            return False
        return True

    def resolve_speaker(
        self,
        t_start_ms: int,
        t_end_ms: int,
        *,
        audio: np.ndarray | None = None,
    ) -> tuple[int, bool]:
        if not self.uses_pyannote():
            return 0, False
        self.sync_timeline()
        single = self.config.meeting_spk_mode != "multi"
        pyannote_labels = self._pyannote.speaker_label_count()
        speaker_id, changed = self._merger.assign_speaker(
            t_start_ms,
            t_end_ms,
            single_speaker_mode=single,
        )

        if self._should_use_embedding(audio, pyannote_labels):
            emb_id = self._embedding.assign_from_audio(audio)  # type: ignore[union-attr]
            if emb_id is not None:
                logger.info(
                    "Speaker fallback embedding id=%d (pyannote_labels=%d)",
                    emb_id,
                    pyannote_labels,
                )
                return self._merger.force_speaker_id(emb_id)

        return speaker_id, changed

    def speaker_at_ms(self, t_ms: int) -> int | None:
        if not self.uses_pyannote():
            return None
        self.sync_timeline()
        point = self._merger.speaker_at_ms(t_ms)
        if point is not None:
            return point
        if self._embedding is not None and self._embedding.registry.speaker_count > 0:
            return self._embedding.registry.last_speaker_id
        return None

    def last_speaker_id(self) -> int:
        return self._merger.last_speaker_id

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

    def readiness_detail(self) -> dict[str, str | bool]:
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
            "device_resolved": self.device,
            "engine_ready": self._ready,
        }
