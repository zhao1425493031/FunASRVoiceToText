"""Batch alignment tests."""

from __future__ import annotations

from voicetotext.asr.batch_align import (
    AsrSegment,
    align_batch_segments,
    build_aligned_from_diar,
)
from voicetotext.asr.batch_transcribe import TranscribedWindow
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


def test_build_aligned_from_diar_direct_labels() -> None:
    windows = [
        TranscribedWindow(0, 2000, "SPEAKER_00", "こんにちは"),
        TranscribedWindow(2100, 4000, "SPEAKER_01", "はい"),
        TranscribedWindow(4100, 6000, "SPEAKER_00", "ありがとう"),
    ]
    aligned = build_aligned_from_diar(windows)
    assert len(aligned) == 3
    assert aligned[0].speaker_id == "SPEAKER_00"
    assert aligned[1].speaker_id == "SPEAKER_01"
    assert aligned[2].speaker_id == "SPEAKER_00"
    assert aligned[0].text == "こんにちは"
