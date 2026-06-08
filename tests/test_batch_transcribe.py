"""Diar window transcription tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from voicetotext.asr.batch_transcribe import transcribe_diar_windows
from voicetotext.asr.speaker_timeline import DiarizationSegment

ROOT = Path(__file__).resolve().parents[1]


def test_transcribe_diar_windows_parallel_and_filter_empty() -> None:
    audio = np.zeros(32000, dtype=np.float32)
    windows = [
        DiarizationSegment(0, 500, "SPEAKER_00"),
        DiarizationSegment(600, 1200, "SPEAKER_01"),
        DiarizationSegment(1300, 2000, "SPEAKER_00"),
    ]
    asr = MagicMock()
    asr.transcribe.side_effect = ["こんにちは", "", "さようなら"]

    out = transcribe_diar_windows(audio, windows, asr, 16000, "ja", workers=2)
    assert len(out) == 2
    assert out[0].text == "こんにちは"
    assert out[1].text == "さようなら"
    assert out[0].speaker_label == "SPEAKER_00"
    assert out[1].speaker_label == "SPEAKER_00"
    assert asr.transcribe.call_count == 3
