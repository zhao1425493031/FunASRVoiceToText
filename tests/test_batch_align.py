"""Batch alignment tests."""

from __future__ import annotations

from voicetotext.asr.batch_align import AsrSegment, align_batch_segments
from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.config import load_config
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_align_batch_segments_overlap() -> None:
    cfg = load_config(ROOT / "config.yaml")
    diar = [
        DiarizationSegment(0, 5000, "SPEAKER_00"),
        DiarizationSegment(5000, 10000, "SPEAKER_01"),
    ]
    asr = [
        AsrSegment(100, 4000, "こんにちは"),
        AsrSegment(5100, 9000, "はい"),
    ]
    aligned = align_batch_segments(asr, diar, cfg)
    assert len(aligned) == 2
    assert aligned[0].speaker_id.startswith("SPEAKER_")
    assert aligned[0].text == "こんにちは"
    assert aligned[1].text == "はい"
