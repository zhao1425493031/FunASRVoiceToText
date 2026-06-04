"""Speaker assignment for meeting ASR (cam++ or timestamp heuristics)."""

from __future__ import annotations

import json
import threading
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
    """Extract t_start_ms, t_end_ms from Runtime timestamp or stamp_sents."""
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
    stamp_sents = msg.get("stamp_sents")
    if isinstance(stamp_sents, list):
        for item in stamp_sents:
            if isinstance(item, dict) and "spk" in item:
                try:
                    return int(item["spk"])
                except (TypeError, ValueError):
                    pass
    return None


class MeetingSpeakerAssigner:
    """Assign speaker_id per utterance; optional cam++ embedding clustering."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._last_speaker = 0
        self._last_end_ms: int | None = None
        self._embeddings: list[np.ndarray] = []
        self._max_speakers = max(1, config.meeting_max_speakers)

    def assign(
        self,
        runtime_msg: dict[str, Any],
        pcm_window: bytes | None = None,
    ) -> tuple[int, int | None, int | None, bool]:
        """
        Returns (speaker_id, t_start_ms, t_end_ms, speaker_changed).
        """
        t_start, t_end = parse_runtime_timestamps(runtime_msg)
        spk = extract_runtime_speaker_id(runtime_msg)
        speaker_changed = False

        if spk is not None:
            speaker_id = spk % self._max_speakers
        elif (
            self.config.meeting_use_diarization
            and pcm_window
            and len(pcm_window) >= self.config.sample_rate
        ):
            speaker_id = self._assign_from_audio(pcm_window)
        else:
            speaker_id = self._heuristic_speaker(t_start, t_end)

        if speaker_id != self._last_speaker:
            speaker_changed = True
        self._last_speaker = speaker_id
        if t_end is not None:
            self._last_end_ms = t_end

        return speaker_id, t_start, t_end, speaker_changed

    def _heuristic_speaker(self, t_start: int | None, t_end: int | None) -> int:
        gap_ms = 2000
        if (
            t_start is not None
            and self._last_end_ms is not None
            and t_start - self._last_end_ms > gap_ms
        ):
            return (self._last_speaker + 1) % self._max_speakers
        return self._last_speaker

    def _assign_from_audio(self, pcm_window: bytes) -> int:
        try:
            model = _load_spk_model(self.config)
            audio = np.frombuffer(pcm_window, dtype=np.int16).astype(np.float32) / 32768.0
            res = model.generate(input=audio, batch_size_s=300)
            if res and isinstance(res, list):
                for item in res:
                    if isinstance(item, dict) and "spk" in item:
                        return int(item["spk"]) % self._max_speakers
                    sentence_info = item.get("sentence_info") if isinstance(item, dict) else None
                    if sentence_info and isinstance(sentence_info, list) and sentence_info:
                        spk = sentence_info[0].get("spk")
                        if spk is not None:
                            return int(spk) % self._max_speakers
        except Exception as exc:
            logger.warning("cam++ assign failed, using heuristic: %s", exc)
        return self._heuristic_speaker(None, None)
