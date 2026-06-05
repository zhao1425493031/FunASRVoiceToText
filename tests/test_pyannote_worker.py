"""Pyannote worker ring buffer and segment parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from voicetotext.asr.pyannote_worker import AudioRingBuffer, _parse_pyannote_output
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_audio_ring_buffer_snapshot() -> None:
    ring = AudioRingBuffer(16000, max_seconds=30)
    pcm = (np.full(16000, 1000, dtype=np.int16)).tobytes()
    ring.append(pcm)
    audio, start_ms = ring.snapshot_window(1.0)
    assert audio.size == 16000
    assert start_ms >= 0


def test_parse_pyannote_output_empty() -> None:
    assert _parse_pyannote_output(None, 0) == []


def test_parse_pyannote_output_with_mock_annotation() -> None:
    turn = MagicMock()
    turn.start = 1.0
    turn.end = 2.5
    track = MagicMock()
    ann = MagicMock()
    ann.itertracks = MagicMock(return_value=[(turn, track, "SPEAKER_00")])
    output = MagicMock()
    output.exclusive_speaker_diarization = ann
    segs = _parse_pyannote_output(output, 5000)
    assert len(segs) == 1
    assert segs[0].speaker_label == "SPEAKER_00"
    assert segs[0].start_ms == 6000


def test_pyannote_hf_token_env() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    from voicetotext.asr.pyannote_worker import PyannoteWorker

    worker = PyannoteWorker(cfg)
    assert worker.hf_token() is None or isinstance(worker.hf_token(), str)
