"""Batch pipeline LLM integration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest_llm import sample_summary_json
from voicetotext.asr.batch_pipeline import BatchPipeline
from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.config import load_config
from voicetotext.llm.schemas import MeetingSummary, SummaryResponse

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _mock_summary_response() -> SummaryResponse:
    import json

    data = json.loads(sample_summary_json())
    return SummaryResponse(
        job_id="job",
        summary=MeetingSummary(**data),
        status="ok",
        meta={"latency_ms": 10, "chunks": 1},
    )


@patch("voicetotext.asr.batch_pipeline.load_audio_file")
@patch.object(BatchPipeline, "load")
def test_pipeline_generates_summary(mock_load, mock_audio, tmp_path) -> None:
    cfg = load_config(FIXTURES / "config_llm_test.yaml")
    mock_load.return_value = None
    mock_audio.return_value = (np.zeros(16000, dtype=np.float32), 1000)

    pipeline = BatchPipeline(cfg)
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
    pipeline._summary_provider = MagicMock()
    pipeline._summary_provider.summarize.return_value = _mock_summary_response()

    wav = tmp_path / "t.wav"
    wav.write_bytes(b"x")
    out = tmp_path / "out"
    transcript = pipeline.process_file(wav, out)

    assert transcript.summary is not None
    assert transcript.summary["title"]
    assert transcript.meta["llm"]["status"] == "ok"
    assert (out / f"{transcript.job_id}.summary.md").is_file()


@patch("voicetotext.asr.batch_pipeline.load_audio_file")
@patch.object(BatchPipeline, "load")
def test_pipeline_skip_summary(mock_load, mock_audio, tmp_path) -> None:
    cfg = load_config(FIXTURES / "config_llm_test.yaml")
    mock_load.return_value = None
    mock_audio.return_value = (np.zeros(16000, dtype=np.float32), 1000)

    pipeline = BatchPipeline(cfg, skip_summary=True)
    pipeline._loaded = True
    pipeline._asr = MagicMock()
    pipeline._asr.transcribe.return_value = "テスト"
    pipeline._diarizer = MagicMock()
    pipeline._diarizer.diarize.return_value = [DiarizationSegment(0, 1000, "SPEAKER_00")]
    pipeline._summary_provider = MagicMock()

    wav = tmp_path / "t.wav"
    wav.write_bytes(b"x")
    out = tmp_path / "out"
    transcript = pipeline.process_file(wav, out)

    assert transcript.summary is None
    pipeline._summary_provider.summarize.assert_not_called()


@patch("voicetotext.asr.batch_pipeline.load_audio_file")
@patch.object(BatchPipeline, "load")
def test_pipeline_on_failure_warn(mock_load, mock_audio, tmp_path) -> None:
    cfg = load_config(FIXTURES / "config_llm_test.yaml")
    mock_load.return_value = None
    mock_audio.return_value = (np.zeros(16000, dtype=np.float32), 1000)

    pipeline = BatchPipeline(cfg)
    pipeline._loaded = True
    pipeline._asr = MagicMock()
    pipeline._asr.transcribe.return_value = "テスト"
    pipeline._diarizer = MagicMock()
    pipeline._diarizer.diarize.return_value = [DiarizationSegment(0, 1000, "SPEAKER_00")]
    pipeline._summary_provider = MagicMock()
    pipeline._summary_provider.summarize.side_effect = RuntimeError("502")

    wav = tmp_path / "t.wav"
    wav.write_bytes(b"x")
    out = tmp_path / "out"
    transcript = pipeline.process_file(wav, out)

    assert transcript.summary is None
    assert transcript.meta["llm"]["status"] == "error"
    assert (out / f"{transcript.job_id}.json").is_file()


@patch("voicetotext.asr.batch_pipeline.load_audio_file")
@patch.object(BatchPipeline, "load")
def test_pipeline_on_failure_fail(mock_load, mock_audio, tmp_path) -> None:
    cfg = load_config(FIXTURES / "config_llm_fail.yaml")
    mock_load.return_value = None
    mock_audio.return_value = (np.zeros(16000, dtype=np.float32), 1000)

    pipeline = BatchPipeline(cfg)
    pipeline._loaded = True
    pipeline._asr = MagicMock()
    pipeline._asr.transcribe.return_value = "テスト"
    pipeline._diarizer = MagicMock()
    pipeline._diarizer.diarize.return_value = [DiarizationSegment(0, 1000, "SPEAKER_00")]
    pipeline._summary_provider = MagicMock()
    pipeline._summary_provider.summarize.side_effect = RuntimeError("502")

    wav = tmp_path / "t.wav"
    wav.write_bytes(b"x")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="502"):
        pipeline.process_file(wav, out)
