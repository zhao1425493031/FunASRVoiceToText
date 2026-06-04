"""Enterprise meeting speaker assignment (session-level embedding clustering)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)

_MODEL_LOCK = threading.Lock()
_SPEAKER_MODEL: Any = None


def _load_spk_model(config: AppConfig) -> Any:
    global _SPEAKER_MODEL
    with _MODEL_LOCK:
        if _SPEAKER_MODEL is not None:
            return _SPEAKER_MODEL
        from funasr import AutoModel

        logger.info("Loading meeting spk model %s", config.meeting_spk_model)
        _SPEAKER_MODEL = AutoModel(
            model=config.meeting_spk_model,
            device="cpu",
            disable_update=True,
        )
        return _SPEAKER_MODEL


def parse_runtime_timestamps(msg: dict[str, Any]) -> tuple[int | None, int | None]:
    stamp_sents = msg.get("stamp_sents")
    if isinstance(stamp_sents, list) and stamp_sents:
        first = stamp_sents[0]
        if isinstance(first, dict):
            start = first.get("start")
            end = first.get("end")
            if start is not None and end is not None:
                return int(start), int(end)
    ts_raw = msg.get("timestamp")
    if not ts_raw:
        return None, None
    try:
        if isinstance(ts_raw, str):
            pairs = json.loads(ts_raw)
        else:
            pairs = ts_raw
        if pairs and isinstance(pairs[0], (list, tuple)) and len(pairs[0]) >= 2:
            return int(pairs[0][0]), int(pairs[-1][1])
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.debug("Could not parse timestamp: %s", exc)
    return None, None


def extract_runtime_speaker_id(msg: dict[str, Any]) -> int | None:
    for key in ("spk", "speaker_id", "speaker", "spk_id"):
        if key in msg and msg[key] is not None:
            try:
                return int(msg[key])
            except (TypeError, ValueError):
                pass
    return None


def _tensor_to_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return None
    return arr / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _extract_spk_embedding(config: AppConfig, pcm_bytes: bytes) -> np.ndarray | None:
    min_bytes = int(config.sample_rate * (config.meeting_spk_min_audio_ms / 1000.0)) * 2
    if len(pcm_bytes) < min_bytes:
        return None
    try:
        model = _load_spk_model(config)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        res = model.generate(input=audio, batch_size_s=300)
        if not res or not isinstance(res, list):
            return None
        for item in res:
            if not isinstance(item, dict):
                continue
            for key in ("spk_embedding", "embedding", "spk_emb"):
                vec = _tensor_to_vector(item.get(key))
                if vec is not None:
                    return vec
    except Exception as exc:
        logger.warning("spk embedding extract failed: %s", exc)
    return None


@dataclass
class _SpeakerProfile:
    speaker_id: int
    centroid: np.ndarray
    hit_count: int = 0


@dataclass
class _SpeakerRegistry:
    """Session-stable speaker IDs via embedding similarity."""

    max_speakers: int
    similarity_threshold: float
    new_speaker_max_sim: float
    centroid_momentum: float = 0.75
    profiles: list[_SpeakerProfile] = field(default_factory=list)

    def _update_centroid(self, profile: _SpeakerProfile, embedding: np.ndarray) -> None:
        profile.centroid = (
            self.centroid_momentum * profile.centroid
            + (1.0 - self.centroid_momentum) * embedding
        )
        norm = float(np.linalg.norm(profile.centroid))
        if norm > 1e-8:
            profile.centroid = profile.centroid / norm
        profile.hit_count += 1

    def assign(self, embedding: np.ndarray) -> int:
        if not self.profiles:
            self.profiles.append(
                _SpeakerProfile(speaker_id=0, centroid=embedding.copy(), hit_count=1)
            )
            return 0

        best_id = 0
        best_sim = -1.0
        for profile in self.profiles:
            sim = _cosine_similarity(embedding, profile.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = profile.speaker_id

        # 同一人音色波动：相似度不够高但仍明显高于「新人」阈值 → 归到最近档案
        if best_sim >= self.similarity_threshold or best_sim >= self.new_speaker_max_sim:
            for profile in self.profiles:
                if profile.speaker_id == best_id:
                    self._update_centroid(profile, embedding)
                    break
            return best_id

        if len(self.profiles) >= self.max_speakers:
            logger.debug(
                "Speaker registry full, closest id=%s sim=%.3f",
                best_id,
                best_sim,
            )
            return best_id

        new_id = len(self.profiles)
        self.profiles.append(
            _SpeakerProfile(speaker_id=new_id, centroid=embedding.copy(), hit_count=1)
        )
        logger.info(
            "New speaker profile id=%s (sim=%.3f < %.3f)",
            new_id,
            best_sim,
            self.new_speaker_max_sim,
        )
        return new_id


class MeetingSpeakerAssigner:
    """
    meeting_spk_mode=single → 固定 [話者1]（演讲/单人测试）
    meeting_spk_mode=multi  → cam++ 声纹 + 会话聚类（多人会议）
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._last_speaker = 0
        self._max_speakers = max(1, config.meeting_max_speakers)
        self._registry: _SpeakerRegistry | None = None
        self._use_clustering = (
            config.meeting_spk_mode == "multi" and config.meeting_use_diarization
        )
        if self._use_clustering:
            self._registry = _SpeakerRegistry(
                max_speakers=self._max_speakers,
                similarity_threshold=config.meeting_spk_similarity_threshold,
                new_speaker_max_sim=config.meeting_spk_new_speaker_max_sim,
            )

    def assign(
        self,
        runtime_msg: dict[str, Any],
        *,
        pcm_utterance: bytes | None = None,
    ) -> tuple[int, int | None, int | None, bool]:
        t_start, t_end = parse_runtime_timestamps(runtime_msg)
        external = extract_runtime_speaker_id(runtime_msg)
        speaker_changed = False

        if external is not None:
            speaker_id = external % self._max_speakers
        elif not self._use_clustering:
            speaker_id = 0
        elif pcm_utterance:
            speaker_id = self._assign_from_utterance_pcm(pcm_utterance)
        else:
            speaker_id = self._last_speaker

        if speaker_id != self._last_speaker:
            speaker_changed = True
        self._last_speaker = speaker_id
        return speaker_id, t_start, t_end, speaker_changed

    def _assign_from_utterance_pcm(self, pcm_utterance: bytes) -> int:
        embedding = _extract_spk_embedding(self.config, pcm_utterance)
        if embedding is None:
            return self._last_speaker
        if self._registry is None:
            return self._last_speaker
        return self._registry.assign(embedding)
