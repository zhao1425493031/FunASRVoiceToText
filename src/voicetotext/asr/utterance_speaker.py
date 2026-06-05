"""Per-utterance speaker embedding (FunASR campplus) for real-time fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def _extract_embedding(result: Any) -> np.ndarray | None:
    if not result:
        return None
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            for key in ("spk_embedding", "embedding", "embeddings"):
                val = item.get(key)
                if val is not None:
                    arr = np.asarray(val, dtype=np.float32).reshape(-1)
                    if arr.size > 0:
                        return arr
    if isinstance(result, dict):
        for key in ("spk_embedding", "embedding"):
            val = result.get(key)
            if val is not None:
                arr = np.asarray(val, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    return arr
    return None


class UtteranceSpeakerRegistry:
    """Online cosine clustering on utterance embeddings (dynamic speaker count)."""

    def __init__(self, *, max_speakers: int, threshold: float) -> None:
        self.max_speakers = max(1, max_speakers)
        self.threshold = threshold
        self._centroids: list[np.ndarray] = []
        self._last_speaker_id = 0

    def reset(self) -> None:
        self._centroids.clear()
        self._last_speaker_id = 0

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    @property
    def last_speaker_id(self) -> int:
        return self._last_speaker_id

    def assign(self, embedding: np.ndarray) -> int:
        emb = embedding.astype(np.float32).reshape(-1)
        if emb.size == 0:
            return self._last_speaker_id

        if not self._centroids:
            self._centroids.append(emb)
            self._last_speaker_id = 0
            logger.info("Utterance embedding: new speaker_id=0 (first centroid)")
            return 0

        best_idx = 0
        best_sim = -1.0
        for i, centroid in enumerate(self._centroids):
            sim = _cosine_similarity(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.threshold:
            # EMA update centroid for drift tolerance
            updated = 0.85 * self._centroids[best_idx] + 0.15 * emb
            self._centroids[best_idx] = updated
            self._last_speaker_id = best_idx
            logger.info(
                "Utterance embedding: matched speaker_id=%d sim=%.3f",
                best_idx,
                best_sim,
            )
            return best_idx

        if len(self._centroids) >= self.max_speakers:
            self._last_speaker_id = best_idx
            logger.info(
                "Utterance embedding: max speakers reached, fallback id=%d sim=%.3f",
                best_idx,
                best_sim,
            )
            return best_idx

        new_id = len(self._centroids)
        self._centroids.append(emb)
        self._last_speaker_id = new_id
        logger.info(
            "Utterance embedding: new speaker_id=%d (sim=%.3f < %.3f)",
            new_id,
            best_sim,
            self.threshold,
        )
        return new_id


class UtteranceSpeakerEngine:
    """Lazy-loaded FunASR campplus embedding extractor."""

    def __init__(self, config: AppConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.registry = UtteranceSpeakerRegistry(
            max_speakers=config.meeting_max_speakers,
            threshold=config.meeting_spk_embedding_threshold,
        )
        self._model: Any = None
        self._ready = False

    @property
    def is_loaded(self) -> bool:
        return self._ready and self._model is not None

    def load(self) -> None:
        from funasr import AutoModel

        model_id = self.config.meeting_spk_embedding_model
        logger.info("Loading utterance embedding model=%s device=%s", model_id, self.device)
        self._model = AutoModel(
            model=model_id,
            device=self.device,
            disable_update=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._ready = True

    def reset_session(self) -> None:
        self.registry.reset()

    def assign_from_audio(self, audio: np.ndarray) -> int | None:
        if self._model is None or audio.size == 0:
            return None
        min_samples = int(self.config.sample_rate * 0.3)
        if audio.size < min_samples:
            return None
        try:
            with np.errstate(all="ignore"):
                res = self._model.generate(input=audio)
            emb = _extract_embedding(res)
            if emb is None:
                return None
            return self.registry.assign(emb)
        except Exception as exc:
            logger.warning("Utterance embedding failed: %s", exc)
            return None
