"""MeetingStreamSession with mock ASR engine."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest

from dataclasses import replace

from voicetotext.asr.meeting_stream_session import MeetingStreamSession
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class MockEngine:
    device = "cpu"
    backend_name = "mock"

    def __init__(self) -> None:
        self._calls = 0
        self.config = load_config(ROOT / "config.meeting.yaml")

    @property
    def is_loaded(self) -> bool:
        return True

    def pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def detect_speech(self, audio: np.ndarray) -> bool:
        return float(np.max(np.abs(audio))) > 0.05

    def transcribe_window(self, audio: np.ndarray, cache: dict, *, is_final: bool) -> str:
        self._calls += 1
        if is_final:
            return "final sentence"
        return "partial"

    def finalize_utterance(self, audio: np.ndarray, draft_fallback: str) -> str:
        return "final sentence"

    def finalize_text(self, text: str) -> str:
        return text.strip()


@pytest.fixture
def meeting_config():
    return load_config(ROOT / "config.meeting.yaml")


def _pcm_chunk(config, *, loud: bool = True) -> bytes:
    n = config.chunk_stride_samples
    amp = 8000 if loud else 0
    return (np.full(n, amp, dtype=np.int16)).tobytes()


def test_feed_pcm_no_partial_by_default(meeting_config) -> None:
    engine = MockEngine()
    session = MeetingStreamSession(engine, meeting_config)
    msgs = session.feed_pcm(_pcm_chunk(meeting_config))
    assert not any(m["type"] == "partial" for m in msgs)


def test_feed_pcm_emits_partial_when_enabled(meeting_config) -> None:
    cfg = replace(meeting_config, meeting_emit_partial=True)
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg)
    msgs = session.feed_pcm(_pcm_chunk(cfg))
    assert any(m["type"] == "partial" for m in msgs)


def test_silence_emits_final(meeting_config) -> None:
    engine = MockEngine()
    meeting_config_vad = meeting_config
    session = MeetingStreamSession(engine, meeting_config_vad)
    session.feed_pcm(_pcm_chunk(meeting_config, loud=True))
    session._last_voice_ts = time.time() - 2.0
    msgs = session.feed_pcm(_pcm_chunk(meeting_config, loud=False))
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"] == "final sentence"
    assert "partial" not in [m["type"] for m in msgs]
