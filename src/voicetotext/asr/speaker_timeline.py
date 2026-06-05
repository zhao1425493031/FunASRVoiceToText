"""Merge Pyannote exclusive diarization segments with ASR timestamps."""

from __future__ import annotations

from dataclasses import dataclass, field

from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


def parse_runtime_timestamps(msg: dict) -> tuple[int | None, int | None]:
    stamp_sents = msg.get("stamp_sents")
    if isinstance(stamp_sents, list) and stamp_sents:
        first = stamp_sents[0]
        if isinstance(first, dict):
            start = first.get("start")
            end = first.get("end")
            if start is not None and end is not None:
                return int(start), int(end)
    return None, None


@dataclass(frozen=True)
class DiarizationSegment:
    start_ms: int
    end_ms: int
    speaker_label: str


def merge_diarization_windows(
    existing: list[DiarizationSegment],
    incoming: list[DiarizationSegment],
    window_start_ms: int,
    window_end_ms: int,
    *,
    max_history_ms: int = 3_600_000,
) -> list[DiarizationSegment]:
    """
    Merge a new Pyannote sliding-window result into the session timeline.

    Segments overlapping [window_start, window_end] are replaced by ``incoming``;
    segments outside the window are retained (enterprise cumulative diarization).
    """
    if window_end_ms < window_start_ms:
        window_end_ms = window_start_ms
    kept = [
        seg
        for seg in existing
        if seg.end_ms <= window_start_ms or seg.start_ms >= window_end_ms
    ]
    merged = sorted(kept + list(incoming), key=lambda s: s.start_ms)
    if not merged or max_history_ms <= 0:
        return merged
    cutoff = merged[-1].end_ms - max_history_ms
    return [seg for seg in merged if seg.end_ms >= cutoff]


@dataclass
class SpeakerTimelineMerger:
    """Map ASR [t_start, t_end] to stable integer speaker_id."""

    max_speakers: int
    segments: list[DiarizationSegment] = field(default_factory=list)
    _label_to_id: dict[str, int] = field(default_factory=dict)
    _last_speaker_id: int = 0

    def clear(self) -> None:
        self.segments.clear()
        self._label_to_id.clear()
        self._last_speaker_id = 0

    @property
    def last_speaker_id(self) -> int:
        return self._last_speaker_id

    def force_speaker_id(self, speaker_id: int) -> tuple[int, bool]:
        """Override last speaker (e.g. utterance embedding fallback)."""
        changed = speaker_id != self._last_speaker_id
        self._last_speaker_id = speaker_id
        return speaker_id, changed

    def update_segments(self, segments: list[DiarizationSegment]) -> None:
        self.segments = sorted(segments, key=lambda s: s.start_ms)

    def merge_window(
        self,
        incoming: list[DiarizationSegment],
        window_start_ms: int,
        window_end_ms: int,
    ) -> None:
        self.segments = merge_diarization_windows(
            self.segments,
            incoming,
            window_start_ms,
            window_end_ms,
        )

    def _label_to_speaker_id(self, label: str) -> int:
        if label in self._label_to_id:
            return self._label_to_id[label]
        if len(self._label_to_id) >= self.max_speakers:
            return min(self._label_to_id.values()) if self._label_to_id else 0
        new_id = len(self._label_to_id)
        self._label_to_id[label] = new_id
        return new_id

    def _overlap_ms(self, a_start: int, a_end: int, b_start: int, b_end: int) -> int:
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        return max(0, end - start)

    def speaker_at_ms(self, t_ms: int) -> int | None:
        """Return speaker_id for the diarization segment covering t_ms."""
        for seg in self.segments:
            if seg.start_ms <= t_ms < seg.end_ms:
                return self._label_to_speaker_id(seg.speaker_label)
        return None

    def best_overlap_label(
        self,
        t_start_ms: int,
        t_end_ms: int,
    ) -> tuple[str, int]:
        """Return (best_speaker_label, overlap_ms). Label empty when no overlap."""
        if not self.segments:
            return "", 0
        best_label = ""
        best_overlap = 0
        for seg in self.segments:
            ov = self._overlap_ms(t_start_ms, t_end_ms, seg.start_ms, seg.end_ms)
            if ov > best_overlap:
                best_overlap = ov
                best_label = seg.speaker_label
        return best_label, best_overlap

    def assign_speaker(
        self,
        t_start_ms: int,
        t_end_ms: int,
        *,
        single_speaker_mode: bool = False,
    ) -> tuple[int, bool]:
        """
        Return (speaker_id, speaker_changed).
        Uses maximum temporal overlap with exclusive diarization segments.
        """
        if single_speaker_mode:
            changed = self._last_speaker_id != 0
            self._last_speaker_id = 0
            return 0, changed

        best_label, best_overlap = self.best_overlap_label(t_start_ms, t_end_ms)
        if not best_label or best_overlap <= 0:
            logger.debug(
                "assign_speaker: no overlap [%d,%d] segs=%d last=%d",
                t_start_ms,
                t_end_ms,
                len(self.segments),
                self._last_speaker_id,
            )
            return self._last_speaker_id, False

        speaker_id = self._label_to_speaker_id(best_label)
        changed = speaker_id != self._last_speaker_id
        self._last_speaker_id = speaker_id
        logger.info(
            "assign_speaker [%d,%d] label=%s id=%d overlap_ms=%d segs=%d changed=%s",
            t_start_ms,
            t_end_ms,
            best_label,
            speaker_id,
            best_overlap,
            len(self.segments),
            changed,
        )
        return speaker_id, changed
