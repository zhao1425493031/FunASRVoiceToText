"""Normalize Pyannote diarization segments for diar-first batch ASR."""

from __future__ import annotations

from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.config import AppConfig


class DiarizationEmptyError(RuntimeError):
    """Raised when Pyannote returns no speaker segments."""


def merge_adjacent_segments(
    segments: list[DiarizationSegment],
    merge_gap_ms: int,
) -> list[DiarizationSegment]:
    """Merge consecutive segments with the same speaker when gap <= merge_gap_ms."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.start_ms)
    merged: list[DiarizationSegment] = [ordered[0]]
    for seg in ordered[1:]:
        prev = merged[-1]
        gap = seg.start_ms - prev.end_ms
        if seg.speaker_label == prev.speaker_label and gap <= merge_gap_ms:
            merged[-1] = DiarizationSegment(
                start_ms=prev.start_ms,
                end_ms=max(prev.end_ms, seg.end_ms),
                speaker_label=prev.speaker_label,
            )
        else:
            merged.append(seg)
    return merged


def absorb_short_segments(
    segments: list[DiarizationSegment],
    min_ms: int,
) -> list[DiarizationSegment]:
    """Absorb segments shorter than min_ms into a neighbor (same speaker preferred)."""
    if not segments or min_ms <= 0:
        return list(segments)

    working = list(sorted(segments, key=lambda s: s.start_ms))
    changed = True
    while changed and working:
        changed = False
        if len(working) == 1:
            break
        next_working: list[DiarizationSegment] = []
        i = 0
        while i < len(working):
            seg = working[i]
            duration = seg.end_ms - seg.start_ms
            if duration >= min_ms:
                next_working.append(seg)
                i += 1
                continue

            # Short segment: merge into neighbor
            if i > 0 and working[i - 1].speaker_label == seg.speaker_label:
                prev = next_working[-1]
                next_working[-1] = DiarizationSegment(
                    start_ms=prev.start_ms,
                    end_ms=max(prev.end_ms, seg.end_ms),
                    speaker_label=prev.speaker_label,
                )
                changed = True
            elif i + 1 < len(working):
                nxt = working[i + 1]
                if seg.speaker_label == nxt.speaker_label:
                    working[i + 1] = DiarizationSegment(
                        start_ms=min(seg.start_ms, nxt.start_ms),
                        end_ms=nxt.end_ms,
                        speaker_label=nxt.speaker_label,
                    )
                else:
                    # Different speaker: extend previous segment to cover gap
                    if next_working:
                        prev = next_working[-1]
                        next_working[-1] = DiarizationSegment(
                            start_ms=prev.start_ms,
                            end_ms=max(prev.end_ms, seg.end_ms),
                            speaker_label=prev.speaker_label,
                        )
                    else:
                        next_working.append(
                            DiarizationSegment(
                                start_ms=seg.start_ms,
                                end_ms=seg.end_ms,
                                speaker_label=seg.speaker_label,
                            )
                        )
                changed = True
            elif next_working:
                prev = next_working[-1]
                next_working[-1] = DiarizationSegment(
                    start_ms=prev.start_ms,
                    end_ms=max(prev.end_ms, seg.end_ms),
                    speaker_label=prev.speaker_label,
                )
                changed = True
            else:
                next_working.append(seg)
            i += 1
        working = next_working if next_working else working

    return working


def split_long_segments(
    segments: list[DiarizationSegment],
    max_ms: int,
) -> list[DiarizationSegment]:
    """Split segments longer than max_ms; speaker label unchanged per chunk."""
    if max_ms <= 0:
        return list(segments)
    out: list[DiarizationSegment] = []
    for seg in segments:
        cursor = seg.start_ms
        while cursor < seg.end_ms:
            chunk_end = min(seg.end_ms, cursor + max_ms)
            out.append(
                DiarizationSegment(
                    start_ms=cursor,
                    end_ms=chunk_end,
                    speaker_label=seg.speaker_label,
                )
            )
            cursor = chunk_end
    return out


def normalize_diar_segments(
    segments: list[DiarizationSegment],
    config: AppConfig,
) -> list[DiarizationSegment]:
    """Sort → merge → absorb_short → split_long → sort."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.start_ms)
    merged = merge_adjacent_segments(ordered, config.batch_diar_merge_gap_ms)
    absorbed = absorb_short_segments(merged, config.batch_diar_min_segment_ms)
    split = split_long_segments(absorbed, config.batch_max_segment_ms)
    return sorted(split, key=lambda s: s.start_ms)
