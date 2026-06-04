"""Lazy-loaded ct-punc model for finalize fallback punctuation."""

from __future__ import annotations

from typing import Any

from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger
from voicetotext.text_utils import strip_model_tags

logger = get_logger(__name__)


class PuncRestorer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.config.punc_model:
            raise ValueError("punc_model is not configured")
        from funasr import AutoModel

        logger.info("Loading punctuation model=%s", self.config.punc_model)
        self._model = AutoModel(
            model=self.config.punc_model,
            device="cpu",
            disable_update=True,
        )
        logger.info("Punctuation model loaded")

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    def restore(self, text: str) -> str:
        stripped = strip_model_tags(text.strip())
        if not stripped or not self.config.punc_model:
            return stripped
        try:
            result = self.model.generate(input=stripped, task="punc")
            return _extract_text(result) or stripped
        except Exception:
            logger.exception("ct-punc restore failed")
            return stripped


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
