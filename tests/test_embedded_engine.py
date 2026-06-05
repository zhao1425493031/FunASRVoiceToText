"""Qwen FunASR text extraction helpers (v2; embedded SenseVoice removed from meeting path)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from voicetotext.asr.qwen_funasr_engine import QwenFunASREngine, _extract_text
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def meeting_config():
    return load_config(ROOT / "config.meeting.yaml")


def test_extract_text_list() -> None:
    out = _extract_text([{"text": "hello"}, {"text": "世界"}])
    assert out == "hello世界"


def test_extract_text_empty() -> None:
    assert _extract_text([]) == ""
    assert _extract_text(None) == ""


def test_extract_text_string() -> None:
    assert _extract_text("  trimmed  ") == "trimmed"


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("ja", "Japanese"),
        ("zh", "Chinese"),
        ("JA", "Japanese"),
        ("ZH", "Chinese"),
    ],
)
def test_language_kw_maps_to_qwen_names(meeting_config, lang, expected) -> None:
    cfg = replace(meeting_config, language=lang)
    engine = QwenFunASREngine(cfg, model_id="test-model")
    assert engine._language_kw() == {"language": expected}


def test_language_kw_override_beats_config(meeting_config) -> None:
    cfg = replace(meeting_config, language="ja")
    engine = QwenFunASREngine(cfg, model_id="test-model")
    assert engine._language_kw("zh") == {"language": "Chinese"}
