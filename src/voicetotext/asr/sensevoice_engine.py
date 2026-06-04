"""SenseVoiceSmall Japanese ASR backend with window-based pseudo-streaming."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class SenseVoiceEngine:
    backend_name = "sensevoice"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel

        logger.info(
            "Loading SenseVoice model=%s language=%s device=%s",
            self.config.asr_model,
            self.config.language,
            self.device,
        )
        t0 = time.perf_counter()
        self._model = AutoModel(
            model=self.config.asr_model,
            trust_remote_code=self.config.trust_remote_code,
            device=self.device,
            disable_update=True,
        )
        logger.info("SenseVoice loaded in %.2fs", time.perf_counter() - t0)

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def detect_speech(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return False
        return float(np.sqrt(np.mean(audio * audio))) >= 0.01

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        if audio.size == 0:
            return ""
        t0 = time.perf_counter()
        try:
            result = self.model.generate(
                input=audio,
                cache=cache,
                language=self.config.language,
                use_itn=True,
                is_final=is_final,
            )
        except TypeError:
            result = self.model.generate(
                input=audio,
                cache=cache,
                language=self.config.language,
                use_itn=True,
            )
        text = self._extract_text(result)
        text = self._postprocess(text)
        if text:
            logger.info(
                "SenseVoice window %.0fms is_final=%s text=%r",
                (time.perf_counter() - t0) * 1000,
                is_final,
                text,
            )
        return text

    def finalize_text(self, text: str) -> str:
        return self._postprocess(text.strip())

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)
        cache: dict = {}
        text = self.transcribe_window(audio, cache, is_final=True)
        return self.finalize_text(text)

    async def check_ready(self) -> bool:
        return self.is_loaded

    def _postprocess(self, text: str) -> str:
        if not text:
            return ""
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            return rich_transcription_postprocess(text)
        except Exception:
            return text

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio
        duration = len(audio) / src_sr
        target_len = int(duration * dst_sr)
        x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0, duration, num=target_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    @staticmethod
    def _extract_text(result: Any) -> str:
        if not result:
            return ""
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                return str(item.get("text", "") or "")
            return str(item)
        if isinstance(result, dict):
            return str(result.get("text", "") or "")
        return str(result)
