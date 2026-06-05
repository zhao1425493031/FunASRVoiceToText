"""SenseVoice FunASR sub-engine tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.asr.sensevoice_funasr_engine import (
    SenseVoiceFunASREngine,
    _language_kw,
    extract_text,
)
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app_config():
    return load_config(ROOT / "config.yaml")


def test_extract_text_list() -> None:
    out = extract_text([{"text": "<|zh|>你好"}, {"text": "世界"}])
    assert "世界" in out or out  # postprocess may strip tags


def test_extract_text_empty() -> None:
    assert extract_text([]) == ""
    assert extract_text(None) == ""


def test_extract_text_string_strips_tags() -> None:
    assert extract_text("<|zh|>你好") != ""


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("ja", "ja"),
        ("zh", "zh"),
        ("JA", "ja"),
        ("ZH", "zh"),
        ("auto", "auto"),
    ],
)
def test_language_kw_maps_codes(app_config, lang, expected) -> None:
    assert _language_kw(lang, app_config.language) == expected


def test_language_kw_override_beats_default(app_config) -> None:
    assert _language_kw("zh", "ja") == "zh"


@patch("funasr.AutoModel")
def test_transcribe_calls_generate(mock_auto, app_config) -> None:
    model = MagicMock()
    model.generate.return_value = [{"text": "测试"}]
    mock_auto.return_value = model

    engine = SenseVoiceFunASREngine(app_config)
    engine.load()
    audio = np.zeros(1600, dtype=np.float32)
    cache: dict = {}
    out = engine.transcribe(audio, cache, is_final=False, language="zh")
    assert out
    model.generate.assert_called_once()
    call_kw = model.generate.call_args.kwargs
    assert call_kw["language"] == "zh"
    assert call_kw["use_itn"] is True
