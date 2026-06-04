"""SenseVoiceSmall ASR backend with window-based pseudo-streaming."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from voicetotext.asr.punc_restorer import PuncRestorer
from voicetotext.config import AppConfig, resolve_device
from voicetotext.logging_setup import get_logger
from voicetotext.text_utils import (
    apply_japanese_punctuation,
    audio_rms,
    is_meaningful_text,
    normalize_trailing_punctuation,
    strip_model_tags,
)

logger = get_logger(__name__)


class SenseVoiceEngine:
    backend_name = "sensevoice"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._model: Any = None
        self._punc_restorer = PuncRestorer(config)

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
        return audio_rms(audio) >= self.config.vad_energy_threshold

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        if audio.size == 0:
            return ""
        use_itn = is_final
        t0 = time.perf_counter()
        try:
            result = self.model.generate(
                input=audio,
                cache=cache,
                language=self.config.language,
                use_itn=use_itn,
                is_final=is_final,
            )
        except TypeError:
            result = self.model.generate(
                input=audio,
                cache=cache,
                language=self.config.language,
                use_itn=use_itn,
            )
        text = self._extract_text(result)
        if is_final:
            text = self._postprocess(text)
        else:
            text = strip_model_tags(text)
        if text:
            logger.info(
                "SenseVoice window %.0fms is_final=%s use_itn=%s text=%r",
                (time.perf_counter() - t0) * 1000,
                is_final,
                use_itn,
                text,
            )
        return text

    def _generate_utterance(self, audio: np.ndarray, cache: dict) -> Any:
        base = dict(
            input=audio,
            cache=cache,
            language=self.config.language,
            use_itn=True,
            is_final=True,
        )
        optional = dict(
            batch_size_s=min(self.config.session_pcm_max_seconds, 300),
            merge_vad=True,
            merge_length_s=15,
        )
        try:
            return self.model.generate(**base, **optional)
        except TypeError:
            try:
                return self.model.generate(**base)
            except TypeError:
                base.pop("is_final", None)
                return self.model.generate(**base)

    def transcribe_utterance(self, audio: np.ndarray) -> str:
        """Full-session pass with ITN for authoritative final text."""
        if audio.size == 0:
            return ""
        t0 = time.perf_counter()
        cache: dict = {}
        result = self._generate_utterance(audio, cache)
        text = self._postprocess(self._extract_text(result))
        if text:
            logger.info(
                "SenseVoice utterance %.0fms text=%r",
                (time.perf_counter() - t0) * 1000,
                text[:80] + ("..." if len(text) > 80 else ""),
            )
        return text

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        """Prefer full audio ITN; fall back to ct-punc on streaming draft."""
        min_chars = self.config.min_partial_chars
        if audio.size > 0 and audio_rms(audio) >= self.config.vad_energy_threshold:
            text = self.transcribe_utterance(audio)
            if text and is_meaningful_text(text, min_chars):
                return text
            logger.warning("Full utterance ASR returned empty or trivial text")

        fallback = strip_model_tags(draft_fallback.strip())
        if fallback and self.config.punc_model and self.config.language.lower() in (
            "zh",
            "yue",
            "en",
        ):
            logger.warning("Applying ct-punc fallback on streaming draft len=%d", len(fallback))
            restored = self._punc_restorer.restore(fallback)
            if restored and is_meaningful_text(restored, min_chars):
                return restored

        if fallback:
            logger.warning("Finalize fallback: applying draft postprocess (lang=%s)", self.config.language)
            return self.finalize_text(fallback)
        return ""

    def finalize_text(self, text: str) -> str:
        return self._postprocess(strip_model_tags(text.strip()))

    def transcribe_file(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)
        return self.finalize_utterance(audio, "")

    async def check_ready(self) -> bool:
        return self.is_loaded

    def _postprocess(self, text: str) -> str:
        if not text:
            return ""
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            text = rich_transcription_postprocess(text)
        except Exception:
            pass
        text = normalize_trailing_punctuation(text)
        if self._should_apply_ja_punctuation():
            text = apply_japanese_punctuation(text)
        return text

    def _should_apply_ja_punctuation(self) -> bool:
        return (
            self.config.ja_apply_punctuation
            and self.config.language.lower() in ("ja", "jp", "japanese")
        )

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
