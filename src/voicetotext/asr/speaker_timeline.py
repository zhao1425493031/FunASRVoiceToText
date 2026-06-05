"""Merge Pyannote exclusive diarization segments with ASR timestamps."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def update_segments(self, segments: list[DiarizationSegment]) -> None:
        self.segments = sorted(segments, key=lambda s: s.start_ms)

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

        if not self.segments:
            return self._last_speaker_id, False

        best_label = ""
        best_overlap = -1
        for seg in self.segments:
            ov = self._overlap_ms(t_start_ms, t_end_ms, seg.start_ms, seg.end_ms)
            if ov > best_overlap:
                best_overlap = ov
                best_label = seg.speaker_label

        if not best_label or best_overlap <= 0:
            return self._last_speaker_id, False

        speaker_id = self._label_to_speaker_id(best_label)
        changed = speaker_id != self._last_speaker_id
        self._last_speaker_id = speaker_id
        return speaker_id, changed
