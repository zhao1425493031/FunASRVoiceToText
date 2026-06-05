"""Meeting v2 ASR backend: FSMN-VAD + Qwen3 partial/final + Pyannote."""

from __future__ import annotations

import numpy as np

from voicetotext.asr.funasr_vad import FunASRVAD
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
        self._ready = False

    @property
    def is_loaded(self) -> bool:
        return self._ready

    def load(self) -> None:
        self._vad.load()
        self._partial.load()
        self._final.load()
        self._pyannote.load()
        self._pyannote.start()
        self._ready = True
        logger.info("MeetingQwenEngine ready (partial=%s final=%s)", 
                    self.config.qwen_partial_model, self.config.qwen_final_model)

    def shutdown(self) -> None:
        self._pyannote.stop()

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def append_pcm_for_diarization(self, pcm_bytes: bytes) -> None:
        if self.config.meeting_use_diarization and self.config.meeting_spk_mode == "multi":
            self._pyannote.append_pcm(pcm_bytes)

    def sync_timeline(self) -> None:
        if self.config.meeting_use_diarization and self.config.meeting_spk_mode == "multi":
            self._merger.update_segments(self._pyannote.get_segments())

    def detect_speech(self, audio: np.ndarray) -> bool:
        return self._vad.detect_speech_frame(audio)

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        if is_final:
            return self._final.transcribe(audio, cache, is_final=True)
        return self._partial.transcribe(audio, cache, is_final=False)

    def finalize_text(self, text: str) -> str:
        return text.strip()

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        cache: dict = {}
        text = self._final.transcribe(audio, cache, is_final=True)
        if text:
            return self.finalize_text(text)
        return self.finalize_text(draft_fallback)

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
        self.sync_timeline()
        single = self.config.meeting_spk_mode != "multi" or not self.config.meeting_use_diarization
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
        if self.config.meeting_use_diarization and self.config.meeting_spk_mode == "multi":
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
        token_ok = bool(self._pyannote.hf_token()) if self.config.meeting_use_diarization else True
        return {
            "vad_loaded": self._vad.is_loaded,
            "partial_loaded": self._partial.is_loaded,
            "final_loaded": self._final.is_loaded,
            "pyannote_loaded": self._pyannote.is_loaded,
            "hf_token_present": token_ok,
            "engine_ready": self._ready,
        }
