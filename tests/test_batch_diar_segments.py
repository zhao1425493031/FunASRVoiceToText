"""Diarization segment normalization tests."""

from __future__ import annotations

from pathlib import Path

from voicetotext.asr.batch_diar_segments import (
    absorb_short_segments,
    drop_leading_isolated_segment,
    merge_adjacent_segments,
    normalize_diar_segments,
    reassign_misplaced_short_segments,
    split_long_segments,
)
from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_merge_adjacent_same_speaker() -> None:
    segs = [
        DiarizationSegment(0, 1000, "A"),
        DiarizationSegment(1200, 3000, "A"),
        DiarizationSegment(4000, 5000, "B"),
    ]
    merged = merge_adjacent_segments(segs, merge_gap_ms=500)
    assert len(merged) == 2
    assert merged[0].start_ms == 0
    assert merged[0].end_ms == 3000
    assert merged[0].speaker_label == "A"


def test_merge_does_not_merge_different_speaker() -> None:
    segs = [
        DiarizationSegment(0, 1000, "A"),
        DiarizationSegment(1100, 2000, "B"),
    ]
    merged = merge_adjacent_segments(segs, merge_gap_ms=500)
    assert len(merged) == 2


def test_absorb_short_into_same_speaker_neighbor() -> None:
    segs = [
        DiarizationSegment(0, 2000, "A"),
        DiarizationSegment(2000, 2150, "A"),
        DiarizationSegment(3000, 5000, "B"),
    ]
    out = absorb_short_segments(segs, min_ms=300)
    assert out[0].end_ms >= 2150


def test_split_long_segments() -> None:
    segs = [DiarizationSegment(0, 65000, "A")]
    out = split_long_segments(segs, max_ms=30000)
    assert len(out) == 3
    assert all(s.speaker_label == "A" for s in out)
    assert out[-1].end_ms == 65000


def test_reassign_misplaced_short_to_next_speaker() -> None:
    segs = [
        DiarizationSegment(70000, 76800, "SPEAKER_00"),
        DiarizationSegment(77600, 78500, "SPEAKER_00"),
        DiarizationSegment(78500, 82700, "SPEAKER_01"),
    ]
    out = reassign_misplaced_short_segments(segs, max_ms=1200)
    assert out[1].speaker_label == "SPEAKER_01"


def test_drop_leading_isolated_segment() -> None:
    segs = [
        DiarizationSegment(30, 1000, "SPEAKER_01"),
        DiarizationSegment(8147, 15235, "SPEAKER_01"),
    ]
    out = drop_leading_isolated_segment(segs)
    assert len(out) == 1
    assert out[0].start_ms == 8147


def test_normalize_diar_segments_pipeline() -> None:
    cfg = load_config(ROOT / "config.yaml")
    raw = [
        DiarizationSegment(0, 800, "SPEAKER_00"),
        DiarizationSegment(900, 35000, "SPEAKER_00"),
        DiarizationSegment(35100, 40000, "SPEAKER_01"),
    ]
    out = normalize_diar_segments(raw, cfg)
    assert out == sorted(out, key=lambda s: s.start_ms)
    assert all(s.end_ms - s.start_ms <= cfg.batch_max_segment_ms for s in out)
