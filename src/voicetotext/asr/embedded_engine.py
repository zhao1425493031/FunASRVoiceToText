"""In-process SenseVoice ASR for meeting (Windows-friendly, no Docker Runtime)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)

_SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+\|>")


def _is_sensevoice_model(model_name: str) -> bool:
    return "sensevoice" in model_name.lower()


def _clean_sensevoice_text(text: str) -> str:
    if not text:
        return ""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(text).strip()
    except Exception:
        return _SENSEVOICE_TAG_RE.sub("", text).strip()


def _extract_text(result: Any, *, sensevoice: bool = False) -> str:
    if not result:
        return ""
    if isinstance(result, str):
        raw = result.strip()
    elif isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict):
                t = item.get("text") or item.get("value")
                if t:
                    parts.append(str(t).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        raw = "".join(parts).strip()
    elif isinstance(result, dict):
        t = result.get("text") or result.get("value")
        raw = str(t).strip() if t else ""
    else:
        raw = str(result).strip()

    if sensevoice:
        return _clean_sensevoice_text(raw)
    return raw


class EmbeddedSenseVoiceEngine:
    """FunASR AutoModel (SenseVoice) in the gateway process."""

    backend_name = "embedded"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._model: Any = None
        self._ready = False
        self._sensevoice = _is_sensevoice_model(config.asr_model)

    @property
    def is_loaded(self) -> bool:
        return self._ready and self._model is not None

    def load(self) -> None:
        from funasr import AutoModel

        kwargs: dict[str, Any] = {
            "model": self.config.asr_model,
            "device": self.device,
            "disable_update": True,
            "trust_remote_code": self.config.trust_remote_code,
        }
        # SenseVoice + 内置 fsmn-vad 与 Paraformer 流式 chunk_size 不兼容
        if not self._sensevoice:
            vad = self.config.vad_model or self.config.meeting_vad_model
            if vad:
                kwargs["vad_model"] = vad
            if self.config.punc_model:
                kwargs["punc_model"] = self.config.punc_model

        logger.info(
            "Loading embedded ASR model=%s device=%s sensevoice=%s",
            self.config.asr_model,
            self.device,
            self._sensevoice,
        )
        self._model = AutoModel(**kwargs)
        self._ready = True
        logger.info("Embedded ASR model ready")

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def detect_speech(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio * audio)))
        return rms >= self.config.vad_energy_threshold

    def _generate_kwargs(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> dict[str, Any]:
        if self._sensevoice:
            lang = self.config.language.lower()
            language = "ja" if lang == "ja" else ("zh" if lang == "zh" else "auto")
            return {
                "input": audio,
                "cache": cache,
                "language": language,
                "use_itn": True,
                "batch_size_s": 300,
            }

        kw: dict[str, Any] = {
            "input": audio,
            "cache": cache,
            "is_final": is_final,
            "chunk_size": self.config.chunk_size,
            "encoder_chunk_look_back": self.config.encoder_chunk_look_back,
            "decoder_chunk_look_back": self.config.decoder_chunk_look_back,
        }
        lang = self.config.language.lower()
        if lang == "ja":
            kw["language"] = "ja"
        elif lang == "zh":
            kw["language"] = "zh"
        return kw

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        if self._model is None or audio.size == 0:
            return ""
        try:
            res = self._model.generate(**self._generate_kwargs(audio, cache, is_final=is_final))
            return _extract_text(res, sensevoice=self._sensevoice)
        except Exception as exc:
            logger.warning("Embedded transcribe failed is_final=%s: %s", is_final, exc)
            return ""

    def finalize_text(self, text: str) -> str:
        return text.strip()

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        cache: dict = {}
        text = self.transcribe_window(audio, cache, is_final=True)
        if text:
            return self.finalize_text(text)
        return self.finalize_text(draft_fallback)

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.config.sample_rate:
            logger.warning("Resampling not implemented; expected %s Hz", self.config.sample_rate)
        cache: dict = {}
        return self.transcribe_window(audio, cache, is_final=True)

    async def check_ready(self) -> bool:
        return self.is_loaded
