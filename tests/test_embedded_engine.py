"""Qwen FunASR text extraction helpers (v2; embedded SenseVoice removed from meeting path)."""

from __future__ import annotations

from voicetotext.asr.qwen_funasr_engine import _extract_text


def test_extract_text_list() -> None:
    out = _extract_text([{"text": "hello"}, {"text": "世界"}])
    assert out == "hello世界"


def test_extract_text_empty() -> None:
    assert _extract_text([]) == ""
    assert _extract_text(None) == ""


def test_extract_text_string() -> None:
    assert _extract_text("  trimmed  ") == "trimmed"
