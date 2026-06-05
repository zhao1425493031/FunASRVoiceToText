"""Per-utterance speaker embedding (FunASR campplus) — primary real-time diarization."""

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


def _to_numpy_vector(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    elif hasattr(val, "cpu"):
        val = val.cpu().numpy()
    while isinstance(val, (list, tuple)) and len(val) == 1:
        val = val[0]
    try:
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return arr if arr.size > 0 else None


def _extract_embedding(result: Any) -> np.ndarray | None:
    if not result:
        return None
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            for key in ("spk_embedding", "embedding", "embeddings"):
                emb = _to_numpy_vector(item.get(key))
                if emb is not None:
                    return emb
    if isinstance(result, dict):
        for key in ("spk_embedding", "embedding"):
            emb = _to_numpy_vector(result.get(key))
            if emb is not None:
                return emb
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
    """Lazy-loaded FunASR campplus embedding extractor (model shared per process)."""

    def __init__(self, config: AppConfig, device: str) -> None:
        self.config = config
        self.device = device
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

    def assign_from_audio(
        self,
        audio: np.ndarray,
        registry: UtteranceSpeakerRegistry,
    ) -> int | None:
        if self._model is None or audio.size == 0:
            return None
        min_samples = int(self.config.sample_rate * 0.3)
        if audio.size < min_samples:
            logger.debug(
                "Utterance embedding skipped: audio %.2fs < 0.3s",
                audio.size / self.config.sample_rate,
            )
            return None
        try:
            with np.errstate(all="ignore"):
                try:
                    res = self._model.generate(input=audio, embedding=True)
                except TypeError:
                    res = self._model.generate(input=audio)
            emb = _extract_embedding(res)
            if emb is None:
                logger.warning("Utterance embedding: no vector in model output")
                return None
            return registry.assign(emb)
        except Exception as exc:
            logger.warning("Utterance embedding failed: %s", exc)
            return None
