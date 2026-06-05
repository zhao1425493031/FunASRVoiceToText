"""Qwen3-ASR via FunASR AutoModel (partial / final models)."""

from __future__ import annotations

from typing import Any

import numpy as np

from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


def _extract_text(result: Any) -> str:
    if not result:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict):
                t = item.get("text") or item.get("value")
                if t:
                    parts.append(str(t).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "".join(parts).strip()
    if isinstance(result, dict):
        t = result.get("text") or result.get("value")
        return str(t).strip() if t else ""
    return str(result).strip()


class QwenFunASREngine:
    def __init__(self, config: AppConfig, *, model_id: str) -> None:
        self.config = config
        self.model_id = model_id
        self.device = resolve_device(config.device)
        self._model: Any = None
        self._ready = False

    @property
    def is_loaded(self) -> bool:
        return self._ready and self._model is not None

    def load(self) -> None:
        from funasr import AutoModel

        hub = self.config.funasr_hub or "hf"
        logger.info(
            "Loading Qwen ASR model=%s hub=%s device=%s",
            self.model_id,
            hub,
            self.device,
        )
        self._model = AutoModel(
            model=self.model_id,
            hub=hub,
            trust_remote_code=self.config.trust_remote_code,
            device=self.device,
            disable_update=True,
        )
        self._ready = True

    def _language_kw(self) -> dict[str, Any]:
        lang = self.config.language.lower()
        if lang == "ja":
            return {"language": "ja"}
        if lang == "zh":
            return {"language": "zh"}
        return {}

    def transcribe(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        if self._model is None or audio.size == 0:
            return ""
        try:
            kw: dict[str, Any] = {
                "input": audio,
                "cache": cache,
                "batch_size_s": 300,
            }
            kw.update(self._language_kw())
            if not is_final:
                kw["chunk_size"] = self.config.chunk_size
                kw["encoder_chunk_look_back"] = self.config.encoder_chunk_look_back
                kw["decoder_chunk_look_back"] = self.config.decoder_chunk_look_back
            else:
                kw["is_final"] = True
            res = self._model.generate(**kw)
            return _extract_text(res)
        except TypeError:
            try:
                res = self._model.generate(input=audio, batch_size_s=300)
                return _extract_text(res)
            except Exception as exc:
                logger.warning("Qwen transcribe failed: %s", exc)
                return ""
        except Exception as exc:
            logger.warning("Qwen transcribe failed is_final=%s: %s", is_final, exc)
            return ""
