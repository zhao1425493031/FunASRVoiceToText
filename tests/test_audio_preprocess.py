"""Audio path sanitization tests."""

from __future__ import annotations

from voicetotext.asr.audio_preprocess import sanitize_audio_path


def test_sanitize_strips_bidi_mark() -> None:
    # U+202A LEFT-TO-RIGHT EMBEDDING before path
    dirty = "\u202aC:\\Users\\yong\\test.wav"
    clean = sanitize_audio_path(dirty)
    assert "\u202a" not in str(clean)
    assert clean.name == "test.wav"


def test_sanitize_strips_quotes() -> None:
    assert sanitize_audio_path('"C:\\a\\b.wav"').name == "b.wav"
