"""Batch pipeline unit tests (mocked models)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.asr.batch_diar_segments import DiarizationEmptyError
from voicetotext.asr.batch_pipeline import BatchPipeline
from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch_config():
    return load_config(ROOT / "config.yaml")


def test_batch_config_load(batch_config) -> None:
    assert batch_config.language == "ja"
    assert "SenseVoice" in batch_config.asr_model
    assert "json" in batch_config.output_formats
    assert batch_config.pyannote_min_speakers == 2
    assert batch_config.batch_asr_parallel_workers >= 1


@patch("voicetotext.asr.batch_pipeline.load_audio_file")
@patch.object(BatchPipeline, "load")
def test_process_file_diar_first(mock_load, mock_audio, batch_config, tmp_path) -> None:
    mock_load.return_value = None
    audio = np.zeros(16000, dtype=np.float32)
    mock_audio.return_value = (audio, 1000)

    pipeline = BatchPipeline(batch_config, skip_summary=True)
    pipeline._loaded = True
    pipeline._asr = MagicMock()
    pipeline._asr.is_loaded = True
    pipeline._asr.transcribe.return_value = "テスト"
    pipeline._diarizer = MagicMock()
    pipeline._diarizer.is_ready = True
    pipeline._diarizer.diarize.return_value = [
        DiarizationSegment(0, 500, "SPEAKER_00"),
        DiarizationSegment(500, 1000, "SPEAKER_01"),
    ]

    wav = tmp_path / "t.wav"
    wav.write_bytes(b"x")
    out = tmp_path / "out"
    transcript = pipeline.process_file(wav, out)

    assert transcript.job_id
    assert transcript.summary is None
    assert transcript.meta["pipeline"] == "diar_first"
    assert len(transcript.segments) >= 1
    assert (out / f"{transcript.job_id}.json").is_file()
    pipeline._asr.transcribe.assert_called()
    assert not hasattr(pipeline, "_vad") or pipeline.__dict__.get("_vad") is None


@patch("voicetotext.asr.batch_pipeline.load_audio_file")
@patch.object(BatchPipeline, "load")
def test_process_file_empty_diar_raises(
    mock_load, mock_audio, batch_config, tmp_path
) -> None:
    mock_load.return_value = None
    audio = np.zeros(16000, dtype=np.float32)
    mock_audio.return_value = (audio, 1000)

    pipeline = BatchPipeline(batch_config, skip_summary=True)
    pipeline._loaded = True
    pipeline._asr = MagicMock()
    pipeline._diarizer = MagicMock()
    pipeline._diarizer.diarize.return_value = []

    wav = tmp_path / "t.wav"
    wav.write_bytes(b"x")

    with pytest.raises(DiarizationEmptyError):
        pipeline.process_file(wav, tmp_path / "out")
