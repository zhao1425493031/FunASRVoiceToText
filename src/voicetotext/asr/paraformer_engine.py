"""Paraformer streaming engine (development / regression only)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class ParaformerEngine:
    backend_name = "paraformer"

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

        kwargs: dict[str, Any] = {
            "model": self.config.asr_model,
            "device": self.device,
            "disable_update": True,
        }
        if self.config.punc_model:
            kwargs["punc_model"] = self.config.punc_model
        logger.info("Loading Paraformer model=%s device=%s", self.config.asr_model, self.device)
        t0 = time.perf_counter()
        self._model = AutoModel(**kwargs)
        logger.info("Paraformer loaded in %.2fs", time.perf_counter() - t0)

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
        from voicetotext.stream_session import audio_rms

        return audio_rms(audio) >= self.config.vad_energy_threshold

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        result = self.model.generate(
            input=audio,
            cache=cache,
            is_final=is_final,
            chunk_size=self.config.chunk_size,
            encoder_chunk_look_back=self.config.encoder_chunk_look_back,
            decoder_chunk_look_back=self.config.decoder_chunk_look_back,
        )
        return self._extract_text(result)

    def finalize_text(self, text: str) -> str:
        stripped = text.strip()
        if not stripped or not self.config.punc_model:
            return stripped
        try:
            result = self.model.generate(input=stripped, task="punc")
            return self._extract_text(result) or stripped
        except Exception:
            logger.exception("Punctuation failed")
            return stripped

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)
        cache: dict = {}
        stride = self.config.chunk_stride_samples
        texts: list[str] = []
        for offset in range(0, len(audio), stride):
            segment = audio[offset : offset + stride]
            if len(segment) < stride:
                pad = np.zeros(stride - len(segment), dtype=np.float32)
                segment = np.concatenate([segment, pad])
            is_final = offset + stride >= len(audio)
            text = self.transcribe_window(segment, cache, is_final=is_final)
            if text:
                texts.append(text)
        combined = "".join(texts)
        return self.finalize_text(combined)

    async def check_ready(self) -> bool:
        return self.is_loaded

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
