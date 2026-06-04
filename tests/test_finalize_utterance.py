"""Tests for SenseVoice finalize_utterance and transcribe_utterance."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.asr.sensevoice_engine import SenseVoiceEngine
from voicetotext.config import load_config


@pytest.fixture
def engine():
    return SenseVoiceEngine(load_config())


def _load_mock(engine: SenseVoiceEngine, generate_side_effect) -> MagicMock:
    mock_model = MagicMock()
    mock_model.generate.side_effect = generate_side_effect
    with patch("funasr.AutoModel", return_value=mock_model):
        engine.load()
    return mock_model


def test_transcribe_utterance_uses_itn(engine: SenseVoiceEngine) -> None:
    mock_model = _load_mock(engine, [{"text": "定稿句子"}])
    audio = np.ones(1600, dtype=np.float32) * 0.5
    text = engine.transcribe_utterance(audio)
    assert text == "定稿句子"
    assert mock_model.generate.call_args.kwargs.get("use_itn") is True


def test_finalize_utterance_falls_back_to_punc(engine: SenseVoiceEngine) -> None:
    cfg = replace(load_config(), punc_model="ct-punc")
    engine = SenseVoiceEngine(cfg)
    _load_mock(engine, [{"text": ""}])

    mock_punc = MagicMock()
    mock_punc.restore.return_value = "草稿，带标点。"

    with patch.object(engine._punc_restorer, "restore", mock_punc.restore):
        audio = np.ones(1600, dtype=np.float32) * 0.5
        text = engine.finalize_utterance(audio, "草稿无标点")

    assert text == "草稿，带标点。"
    mock_punc.restore.assert_called_once_with("草稿无标点")


def test_finalize_utterance_prefers_full_audio(engine: SenseVoiceEngine) -> None:
    _load_mock(engine, [{"text": "整段识别"}])
    audio = np.ones(1600, dtype=np.float32) * 0.5
    text = engine.finalize_utterance(audio, "草稿")
    assert text == "整段识别"
