"""SenseVoice via FunASR AutoModel (meeting v3 ASR sub-engine)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)

_SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+\|>")


def _clean_sensevoice_text(text: str) -> str:
    if not text:
        return ""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(text).strip()
    except Exception:
        return _SENSEVOICE_TAG_RE.sub("", text).strip()


def extract_text(result: Any) -> str:
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
    return _clean_sensevoice_text(raw)


def _language_kw(language: str | None, default: str) -> str:
    lang = (language or default).lower()
    if lang == "ja":
        return "ja"
    if lang == "zh":
        return "zh"
    return "auto"


class SenseVoiceFunASREngine:
    """FunASR AutoModel wrapper for SenseVoice (single model partial + final)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_id = config.asr_model
        self.device = resolve_device(config.device)
        self._model: Any = None
        self._ready = False

    @property
    def is_loaded(self) -> bool:
        return self._ready and self._model is not None

    def load(self) -> None:
        from funasr import AutoModel

        logger.info(
            "Loading SenseVoice ASR model=%s device=%s",
            self.model_id,
            self.device,
        )
        self._model = AutoModel(
            model=self.model_id,
            device=self.device,
            disable_update=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._ready = True

    def transcribe(
        self,
        audio: np.ndarray,
        cache: dict,
        *,
        is_final: bool,
        language: str | None = None,
    ) -> str:
        if self._model is None or audio.size == 0:
            return ""
        del is_final  # SenseVoice uses window-level recognition; no Paraformer streaming.
        try:
            lang = _language_kw(language, self.config.language)
            res = self._model.generate(
                input=audio,
                cache=cache,
                language=lang,
                use_itn=True,
                batch_size_s=300,
            )
            return extract_text(res)
        except Exception as exc:
            logger.warning("SenseVoice transcribe failed: %s", exc)
            return ""
