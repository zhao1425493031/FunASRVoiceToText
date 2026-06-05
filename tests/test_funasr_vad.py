"""FunASRVAD tests with mocked AutoModel."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from voicetotext.asr.funasr_vad import FunASRVAD
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_detect_speech_frame_rms() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    vad = FunASRVAD(cfg)
    loud = np.full(1600, 0.5, dtype=np.float32)
    quiet = np.zeros(1600, dtype=np.float32)
    assert vad.detect_speech_frame(loud)
    assert not vad.detect_speech_frame(quiet)


@patch("funasr.AutoModel")
def test_vad_load(mock_auto_model) -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    mock_auto_model.return_value = MagicMock()
    vad = FunASRVAD(cfg)
    vad.load()
    assert vad.is_loaded
    mock_auto_model.assert_called_once()
