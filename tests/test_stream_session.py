"""Tests for StreamSession (mocked ASR engine)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from voicetotext.config import load_config
from voicetotext.stream_session import StreamSession


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
        return "partial-text" if cache["n"] == 1 else "more-text"

    def finalize_text(self, text: str) -> str:
        return text + "。"


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def session(config):
    return StreamSession(engine=MockEngine(), config=config)


def test_reset_clears_state(session: StreamSession) -> None:
    session.partial = "x"
    session.cache["k"] = 1
    session.reset()
    assert session.partial == ""
    assert session.cache == {}


def test_feed_pcm_emits_partial_after_window(session: StreamSession) -> None:
    chunk = b"\x00\x01" * session.config.chunk_stride_samples
    result = None
    chunks_needed = (
        session.config.stream_window_bytes // session.config.chunk_stride_bytes + 1
    )
    for _ in range(chunks_needed):
        result = session.feed_pcm(chunk)
    assert result is not None
    assert result.partial == "partial-text"


def test_finalize_adds_punctuation(session: StreamSession) -> None:
    session.partial = "测试"
    result = session.finalize()
    assert result.final == "测试。"
    assert session.partial == ""


def test_paraformer_chunk_path() -> None:
    cfg = replace(load_config(), asr_backend="paraformer")
    session = StreamSession(engine=MockEngine(), config=cfg)
    chunk = b"\x00\x00" * cfg.chunk_stride_samples
    result = session.feed_pcm(chunk)
    assert result.partial == "partial-text"
