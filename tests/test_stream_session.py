"""Tests for StreamSession (mocked ASR engine)."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from voicetotext.config import load_config
from voicetotext.stream_session import (
    StreamSession,
    is_meaningful_text,
)


class MockEngine:
    backend_name = "sensevoice"

    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def detect_speech(self, _audio: np.ndarray) -> bool:
        return True

    def transcribe_window(self, _audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        self.calls.append((is_final, len(cache)))
        cache["n"] = cache.get("n", 0) + 1
        if cache["n"] == 1:
            return "partial-text"
        return "more-text"

    def finalize_text(self, text: str) -> str:
        return text + "。"


class SilentAwareEngine(MockEngine):
    """Only transcribe when audio has non-zero energy."""

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        if float(np.max(np.abs(audio))) < 1e-6:
            return ""
        return super().transcribe_window(audio, cache, is_final=is_final)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def session(config):
    return StreamSession(engine=MockEngine(), config=config)


def test_is_meaningful_text_filters_punctuation() -> None:
    assert is_meaningful_text(".") is False
    assert is_meaningful_text("。") is False
    assert is_meaningful_text("こんにちは。") is True


def test_reset_clears_state(session: StreamSession) -> None:
    session.draft = "x"
    session.cache["k"] = 1
    session.reset()
    assert session.draft == ""
    assert session.cache == {}


def test_feed_pcm_emits_accumulated_partial(session: StreamSession) -> None:
    chunk = b"\x00\x80" * session.config.chunk_stride_samples
    per_window = (
        session.config.stream_window_bytes // session.config.chunk_stride_bytes + 1
    )
    result = None
    for _ in range(per_window * 2):
        result = session.feed_pcm(chunk)
    assert result is not None
    assert result.partial == "partial-textmore-text"


def test_silent_window_skips_transcribe(config) -> None:
    engine = SilentAwareEngine()
    session = StreamSession(engine=engine, config=config)
    silent = b"\x00\x00" * (
        session.config.stream_window_bytes // 2
    )
    chunk = silent[: session.config.chunk_stride_bytes]
    chunks_needed = (
        session.config.stream_window_bytes // session.config.chunk_stride_bytes + 1
    )
    for _ in range(chunks_needed):
        session.feed_pcm(chunk)
    assert engine.calls == []


def test_dot_segment_not_appended(config) -> None:
    class DotEngine(MockEngine):
        def transcribe_window(self, _audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
            cache["n"] = cache.get("n", 0) + 1
            return "." if cache["n"] > 1 else "hello"

    cfg = replace(config, vad_silence_ms=60_000)
    session = StreamSession(engine=DotEngine(), config=cfg)
    chunk = b"\x00\x80" * session.config.chunk_stride_samples
    per_window = (
        session.config.stream_window_bytes // session.config.chunk_stride_bytes + 1
    )
    last = None
    for _ in range(per_window * 2):
        last = session.feed_pcm(chunk)
    assert session.draft == "hello"
    # Second window returns "." which is filtered; draft must stay unchanged.
    assert last is not None and last.partial is None


def test_finalize_uses_full_draft(session: StreamSession) -> None:
    session.draft = "测试"
    result = session.finalize()
    assert result.final == "测试。"
    assert session.draft == ""


def test_no_auto_finalize_on_silence_by_default(config) -> None:
    cfg = replace(config, auto_finalize_on_silence=False, vad_silence_ms=1)
    session = StreamSession(engine=MockEngine(), config=cfg)
    session.draft = "hello"
    session.last_voice_ts = time.time() - 10
    result = session.feed_pcm(b"")
    assert result.final is None
    assert session.draft == "hello"


def test_paraformer_chunk_path() -> None:
    cfg = replace(load_config(), asr_backend="paraformer")
    session = StreamSession(engine=MockEngine(), config=cfg)
    chunk = b"\x00\x80" * cfg.chunk_stride_samples
    result = session.feed_pcm(chunk)
    assert result.partial == "partial-text"
