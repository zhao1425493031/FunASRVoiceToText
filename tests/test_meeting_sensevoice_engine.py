"""MeetingSenseVoiceEngine tests with mocked sub-engines."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from voicetotext.asr.meeting_sensevoice_engine import MeetingSenseVoiceEngine
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def meeting_config():
    return load_config(ROOT / "config.meeting.yaml")


@patch("voicetotext.asr.meeting_sensevoice_engine.UtteranceSpeakerEngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_load_starts_pyannote(
    mock_py, mock_asr, mock_vad, mock_emb, meeting_config
) -> None:
    os.environ["HF_TOKEN"] = "hf_test_token"
    mock_vad.return_value.is_loaded = True
    mock_asr.return_value.is_loaded = True
    mock_py.return_value.is_loaded = True
    mock_py.return_value.hf_token.return_value = "hf_test_token"

    engine = MeetingSenseVoiceEngine(meeting_config)
    engine.load()
    assert engine.is_loaded
    mock_py.return_value.start.assert_called_once()
    mock_emb.return_value.load.assert_not_called()
    del os.environ["HF_TOKEN"]


def test_readiness_detail_without_token(meeting_config) -> None:
    os.environ.pop("HF_TOKEN", None)
    engine = MeetingSenseVoiceEngine(meeting_config)
    detail = engine.readiness_detail()
    assert "hf_token_present" in detail
    assert "asr_loaded" in detail
    assert "active_speaker_sessions" in detail


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_resolve_speaker_single_mode(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    cfg = replace(meeting_config, meeting_spk_mode="single")
    engine = MeetingSenseVoiceEngine(cfg)
    engine._ready = True
    ctx = engine.open_speaker_context("test-single")
    spk, _ = engine.resolve_speaker(ctx, 0, 1000)
    assert spk == 0
    engine.close_speaker_context(ctx)


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_finalize_always_runs_full_asr(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    mock_asr.return_value.transcribe.return_value = "full sentence"
    audio = np.zeros(1600, dtype=np.float32)

    out = engine.finalize_utterance(audio, "partial draft")

    assert out == "full sentence"
    mock_asr.return_value.transcribe.assert_called_once()


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_finalize_falls_back_to_draft_when_asr_empty(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    mock_asr.return_value.transcribe.return_value = ""
    audio = np.zeros(1600, dtype=np.float32)

    out = engine.finalize_utterance(audio, "partial draft")

    assert out == "partial draft"
    mock_asr.return_value.transcribe.assert_called_once()


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_transcribe_window_passes_language(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    asr = mock_asr.return_value
    engine.transcribe_window(
        np.zeros(1600, dtype=np.float32), {}, is_final=False, language="zh"
    )
    asr.transcribe.assert_called_once()
    assert asr.transcribe.call_args.kwargs["language"] == "zh"


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_open_speaker_context_binds_pyannote(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    engine._ready = True
    ctx = engine.open_speaker_context("sess-a")
    mock_py.return_value.bind_session.assert_called_once_with(ctx)
    engine.close_speaker_context(ctx)
    mock_py.return_value.unbind_session.assert_called_once_with(ctx)


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_resolve_speaker_uses_pyannote_when_overlap_reliable(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    engine._ready = True
    ctx = engine.open_speaker_context("sess-py")

    from voicetotext.asr.speaker_timeline import DiarizationSegment

    ctx.pyannote_segments = [
        DiarizationSegment(start_ms=0, end_ms=2000, speaker_label="SPEAKER_01"),
    ]
    ctx.merger._label_to_id = {"SPEAKER_01": 1}
    spk, changed = engine.resolve_speaker(ctx, 100, 1500)
    assert spk == 1
    assert changed is True
    engine.close_speaker_context(ctx)


@patch("voicetotext.asr.meeting_sensevoice_engine.UtteranceSpeakerEngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_resolve_speaker_falls_back_to_embedding_when_overlap_low(
    mock_py, mock_asr, mock_vad, mock_emb, meeting_config
) -> None:
    cfg = replace(meeting_config, meeting_use_utterance_embedding=True)
    engine = MeetingSenseVoiceEngine(cfg)
    engine._ready = True
    engine._embedding = mock_emb.return_value
    mock_emb.return_value.assign_from_audio.return_value = 2
    ctx = engine.open_speaker_context("sess-emb")
    audio = np.zeros(8000, dtype=np.float32)

    def fake_overlap(_s: int, _e: int) -> tuple[str, int]:
        return "SPEAKER_00", 50

    ctx.merger.best_overlap_label = fake_overlap  # type: ignore[method-assign]
    spk, _ = engine.resolve_speaker(ctx, 0, 1000, audio=audio)
    assert spk == 2
    mock_emb.return_value.assign_from_audio.assert_called_once()
    engine.close_speaker_context(ctx)
