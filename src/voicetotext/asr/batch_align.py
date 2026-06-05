"""Align ASR segments with offline diarization for batch export."""

from __future__ import annotations

from dataclasses import dataclass

from voicetotext.asr.speaker_timeline import DiarizationSegment, SpeakerTimelineMerger
from voicetotext.config import AppConfig


@dataclass(frozen=True)
class AsrSegment:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class AlignedSegment:
    start_ms: int
    end_ms: int
    speaker_id: str
    text: str


def _label_to_export_id(label: str, numeric_id: int) -> str:
    if label:
        return label if label.startswith("SPEAKER_") else f"SPEAKER_{numeric_id:02d}"
    return f"SPEAKER_{numeric_id:02d}"


def align_batch_segments(
    asr_segments: list[AsrSegment],
    diar_segments: list[DiarizationSegment],
    config: AppConfig,
    *,
    max_speakers: int = 16,
) -> list[AlignedSegment]:
    merger = SpeakerTimelineMerger(max_speakers=max_speakers)
    merger.update_segments(diar_segments)
    min_overlap = config.batch_align_min_overlap_ms

    aligned: list[AlignedSegment] = []
    label_map: dict[int, str] = {}

    for seg in asr_segments:
        if not seg.text.strip():
            continue
        best_label, overlap = merger.best_overlap_label(seg.start_ms, seg.end_ms)
        if overlap < min_overlap:
            speaker_num = merger.last_speaker_id
            export_label = label_map.get(speaker_num, f"SPEAKER_{speaker_num:02d}")
        else:
            speaker_num, _ = merger.assign_speaker(seg.start_ms, seg.end_ms)
            export_label = _label_to_export_id(best_label, speaker_num)
            label_map[speaker_num] = export_label

        aligned.append(
            AlignedSegment(
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                speaker_id=export_label,
                text=seg.text.strip(),
            )
        )
    return aligned
