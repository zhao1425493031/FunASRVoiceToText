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


def extract_stamp_sents(result: Any) -> list[dict[str, int | str]]:
    """Parse FunASR stamp_sents into [{start_ms, end_ms, text}, ...]."""
    items: list[Any] = []
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        items = [result]

    stamps: list[dict[str, int | str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        stamp_sents = item.get("stamp_sents")
        if not isinstance(stamp_sents, list):
            continue
        for sent in stamp_sents:
            if not isinstance(sent, dict):
                continue
            start = sent.get("start")
            end = sent.get("end")
            text = sent.get("text") or sent.get("value") or ""
            if start is None or end is None:
                continue
            cleaned = _clean_sensevoice_text(str(text))
            if not cleaned:
                continue
            stamps.append(
                {
                    "start_ms": int(start),
                    "end_ms": int(end),
                    "text": cleaned,
                }
            )
    return stamps


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
            stamp_sents = extract_stamp_sents(res)
            if stamp_sents:
                logger.debug("SenseVoice stamp_sents count=%d", len(stamp_sents))
            return extract_text(res)
        except Exception as exc:
            logger.warning("SenseVoice transcribe failed: %s", exc)
            return ""
