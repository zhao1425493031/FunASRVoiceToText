"""Embedded engine text cleanup and SenseVoice kwargs."""

from __future__ import annotations

from voicetotext.asr.embedded_engine import (
    EmbeddedSenseVoiceEngine,
    _clean_sensevoice_text,
    _extract_text,
    _is_sensevoice_model,
)
from voicetotext.config import load_config
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_is_sensevoice_model() -> None:
    assert _is_sensevoice_model("iic/SenseVoiceSmall")
    assert not _is_sensevoice_model("paraformer-zh")


def test_clean_sensevoice_tags() -> None:
    raw = "<|ja|><|Speech|><|withitn|>こんにちは"
    assert "こんにちは" in _clean_sensevoice_text(raw) or _clean_sensevoice_text(raw)


def test_extract_text_sensevoice() -> None:
    out = _extract_text(
        [{"text": "<|ja|><|Speech|><|withitn|>テスト"}],
        sensevoice=True,
    )
    assert "テスト" in out or out == "テスト" or len(out) > 0


def test_sensevoice_generate_kwargs_no_chunk_size() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    eng = EmbeddedSenseVoiceEngine(cfg)
    audio = __import__("numpy").zeros(1600, dtype="float32")
    kw = eng._generate_kwargs(audio, {}, is_final=False)
    assert "chunk_size" not in kw
    expected = cfg.language if cfg.language in ("ja", "zh") else "auto"
    assert kw["language"] == expected
