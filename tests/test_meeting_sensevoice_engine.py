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
    mock_emb.return_value.is_loaded = True

    engine = MeetingSenseVoiceEngine(meeting_config)
    engine.load()
    assert engine.is_loaded
    mock_py.return_value.start.assert_called_once()
    mock_emb.return_value.load.assert_called_once()
    del os.environ["HF_TOKEN"]


def test_readiness_detail_without_token(meeting_config) -> None:
    os.environ.pop("HF_TOKEN", None)
    engine = MeetingSenseVoiceEngine(meeting_config)
    detail = engine.readiness_detail()
    assert "hf_token_present" in detail
    assert "asr_loaded" in detail


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_resolve_speaker_single_mode(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    cfg = replace(meeting_config, meeting_spk_mode="single")
    engine = MeetingSenseVoiceEngine(cfg)
    engine._ready = True
    spk, changed = engine.resolve_speaker(0, 1000)
    assert spk == 0


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_finalize_reuses_partial_when_configured(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    cfg = replace(
        meeting_config,
        meeting_reuse_partial_for_final=True,
        meeting_partial_max_sec=4.0,
    )
    engine = MeetingSenseVoiceEngine(cfg)
    out = engine.finalize_utterance(
        np.zeros(1600, dtype=np.float32),
        "draft from partial",
    )
    assert out == "draft from partial"
    mock_asr.return_value.transcribe.assert_not_called()


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_finalize_reruns_asr_when_utterance_longer_than_partial_window(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    cfg = replace(
        meeting_config,
        meeting_reuse_partial_for_final=True,
        meeting_partial_max_sec=4.0,
        sample_rate=16000,
    )
    engine = MeetingSenseVoiceEngine(cfg)
    mock_asr.return_value.transcribe.return_value = "full sentence"
    audio = np.zeros(int(16000 * 8), dtype=np.float32)

    out = engine.finalize_utterance(audio, "tail only draft")

    assert out == "full sentence"
    mock_asr.return_value.transcribe.assert_called_once()


@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_set_session_language_passed_to_asr(
    mock_py, mock_asr, mock_vad, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    engine.set_session_language("zh")
    asr = mock_asr.return_value
    engine.transcribe_window(np.zeros(1600, dtype=np.float32), {}, is_final=False)
    asr.transcribe.assert_called_once()
    assert asr.transcribe.call_args.kwargs["language"] == "zh"


@patch("voicetotext.asr.meeting_sensevoice_engine.UtteranceSpeakerEngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_begin_speaker_session_resets_workers(
    mock_py, mock_asr, mock_vad, mock_emb, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    engine._ready = True
    engine.begin_speaker_session()
    mock_py.return_value.reset_session.assert_called_once()
    mock_emb.return_value.reset_session.assert_called_once()


@patch("voicetotext.asr.meeting_sensevoice_engine.UtteranceSpeakerEngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_sensevoice_engine.SenseVoiceFunASREngine")
@patch("voicetotext.asr.meeting_sensevoice_engine.PyannoteWorker")
def test_resolve_speaker_uses_embedding_when_pyannote_single_label(
    mock_py, mock_asr, mock_vad, mock_emb, meeting_config
) -> None:
    engine = MeetingSenseVoiceEngine(meeting_config)
    engine._ready = True
    mock_py.return_value.speaker_label_count.return_value = 1
    mock_py.return_value.get_segments.return_value = []
    mock_emb.return_value.assign_from_audio.return_value = 1
    audio = np.zeros(8000, dtype=np.float32)
    spk, changed = engine.resolve_speaker(0, 1000, audio=audio)
    assert spk == 1
    mock_emb.return_value.assign_from_audio.assert_called_once()
