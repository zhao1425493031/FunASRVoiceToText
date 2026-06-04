"""Unit tests for SenseVoiceEngine (mocked, no model download)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.asr.sensevoice_engine import SenseVoiceEngine
from voicetotext.config import load_config


@pytest.fixture
def engine():
    return SenseVoiceEngine(load_config())


def test_sensevoice_transcribe_window_mock(engine: SenseVoiceEngine) -> None:
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "<|ja|>こんにちは"}]

    with patch("funasr.AutoModel", return_value=mock_model):
        engine.load()

    audio = np.zeros(3200, dtype=np.float32)
    cache: dict = {}
    text = engine.transcribe_window(audio, cache, is_final=False)
    mock_model.generate.assert_called()
    call_kwargs = mock_model.generate.call_args.kwargs
    assert call_kwargs.get("language") == "ja"
    assert text  # postprocess may strip tags


def test_sensevoice_finalize_text(engine: SenseVoiceEngine) -> None:
    with patch.object(engine, "_postprocess", side_effect=lambda t: t.strip()):
        assert engine.finalize_text("  test  ") == "test"


def test_sensevoice_is_loaded_before_load(engine: SenseVoiceEngine) -> None:
    assert engine.is_loaded is False
