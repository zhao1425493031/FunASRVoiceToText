"""MeetingQwenEngine tests with mocked sub-engines."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.asr.meeting_qwen_engine import MeetingQwenEngine
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def meeting_config():
    return load_config(ROOT / "config.meeting.yaml")


@patch("voicetotext.asr.meeting_qwen_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_qwen_engine.QwenFunASREngine")
@patch("voicetotext.asr.meeting_qwen_engine.PyannoteWorker")
def test_load_starts_pyannote(mock_py, mock_qwen, mock_vad, meeting_config) -> None:
    os.environ["HF_TOKEN"] = "hf_test_token"
    mock_vad.return_value.is_loaded = True
    mock_qwen.return_value.is_loaded = True
    mock_py.return_value.is_loaded = True
    mock_py.return_value.hf_token.return_value = "hf_test_token"

    engine = MeetingQwenEngine(meeting_config)
    engine.load()
    assert engine.is_loaded
    mock_py.return_value.start.assert_called_once()
    del os.environ["HF_TOKEN"]


def test_readiness_detail_without_token(meeting_config) -> None:
    os.environ.pop("HF_TOKEN", None)
    engine = MeetingQwenEngine(meeting_config)
    detail = engine.readiness_detail()
    assert "hf_token_present" in detail


@patch("voicetotext.asr.meeting_qwen_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_qwen_engine.QwenFunASREngine")
@patch("voicetotext.asr.meeting_qwen_engine.PyannoteWorker")
def test_resolve_speaker_single_mode(
    mock_py, mock_qwen, mock_vad, meeting_config
) -> None:
    from dataclasses import replace

    cfg = replace(meeting_config, meeting_spk_mode="single")
    engine = MeetingQwenEngine(cfg)
    engine._ready = True
    spk, changed = engine.resolve_speaker(0, 1000)
    assert spk == 0


@patch("voicetotext.asr.meeting_qwen_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_qwen_engine.QwenFunASREngine")
@patch("voicetotext.asr.meeting_qwen_engine.PyannoteWorker")
def test_finalize_reuses_partial_when_configured(
    mock_py, mock_qwen, mock_vad, meeting_config
) -> None:
    from dataclasses import replace

    cfg = replace(meeting_config, meeting_reuse_partial_for_final=True)
    engine = MeetingQwenEngine(cfg)
    import numpy as np

    out = engine.finalize_utterance(
        np.zeros(1600, dtype=np.float32),
        "draft from partial",
    )
    assert out == "draft from partial"
    mock_qwen.return_value.transcribe.assert_not_called()


@patch("voicetotext.asr.meeting_qwen_engine.FunASRVAD")
@patch("voicetotext.asr.meeting_qwen_engine.QwenFunASREngine")
@patch("voicetotext.asr.meeting_qwen_engine.PyannoteWorker")
def test_set_session_language_passed_to_partial(
    mock_py, mock_qwen, mock_vad, meeting_config
) -> None:
    engine = MeetingQwenEngine(meeting_config)
    engine.set_session_language("zh")
    partial = mock_qwen.return_value
    import numpy as np

    engine.transcribe_window(np.zeros(1600, dtype=np.float32), {}, is_final=False)
    partial.transcribe.assert_called_once()
    assert partial.transcribe.call_args.kwargs["language"] == "zh"
