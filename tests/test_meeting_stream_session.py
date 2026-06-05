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

    def __init__(self, *, finalize_text: str = "final sentence") -> None:
        self._calls = 0
        self._finalize_calls = 0
        self._finalize_text = finalize_text
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
        self._finalize_calls += 1
        return self._finalize_text

    def finalize_text(self, text: str) -> str:
        return text.strip()

    def utterance_endpoint_reached(self, audio: np.ndarray) -> bool:
        return True


@pytest.fixture
def meeting_config():
    return load_config(ROOT / "config.meeting.yaml")


def _pcm_chunk(config, *, loud: bool = True) -> bytes:
    n = config.chunk_stride_samples
    amp = 8000 if loud else 0
    return (np.full(n, amp, dtype=np.int16)).tobytes()


def test_feed_pcm_partial_after_min_duration(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        meeting_emit_partial=True,
        meeting_partial_min_ms=100,
        meeting_partial_interval_ms=0,
    )
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg)
    msgs = session.feed_pcm(_pcm_chunk(cfg))
    partials = [m for m in msgs if m["type"] == "partial"]
    assert partials
    assert partials[0]["protocol_version"] == 2


def test_feed_pcm_no_partial_when_disabled(meeting_config) -> None:
    cfg = replace(meeting_config, meeting_emit_partial=False)
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg)
    msgs = session.feed_pcm(_pcm_chunk(cfg))
    assert not any(m["type"] == "partial" for m in msgs)


def test_silence_emits_final(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        meeting_min_finalize_chars=5,
        meeting_min_utterance_ms=500,
    )
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg)
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    session._last_rms_voice_ts = time.time() - 3.0
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=False))
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"] == "final sentence"


def test_partial_uses_tail_window_only(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        meeting_emit_partial=True,
        meeting_partial_min_ms=100,
        meeting_partial_interval_ms=0,
        meeting_partial_max_sec=2.0,
        sample_rate=16000,
    )
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg)
    chunk_samples = cfg.chunk_stride_samples
    loud = np.full(chunk_samples, 0.5, dtype=np.float32)
    session._utterance_chunks = [loud.copy() for _ in range(20)]
    session._utterance_start_ms = 0
    session._last_partial_at = 0.0

    def _capture(audio, cache, *, is_final=False):
        session._captured_partial_samples = audio.size
        return "partial tail"

    engine.transcribe_window = _capture  # type: ignore[method-assign]
    msgs = session._maybe_emit_partial(session._concat_utterance())
    assert msgs
    assert session._captured_partial_samples == int(2.0 * 16000)


def test_short_fragment_defers_until_long_silence(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        meeting_min_finalize_chars=20,
        meeting_min_utterance_ms=500,
        vad_silence_ms=1500,
        vad_silence_long_ms=3000,
        meeting_emit_partial=False,
    )
    engine = MockEngine(finalize_text="早上好")
    session = MeetingStreamSession(engine, cfg)
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    session._last_rms_voice_ts = time.time() - 2.0
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=False))
    assert not any(m["type"] == "final" for m in msgs)
    assert session._finalize_deferred

    session._last_rms_voice_ts = time.time() - 3.5
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=False))
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert engine._finalize_calls == 1


def test_client_speaker_id_on_partial_and_final(meeting_config) -> None:
    cfg = replace(meeting_config, meeting_spk_source="client", meeting_spk_mode="multi")
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg, client_speaker_id=2)
    msgs = session._map_text("你好", is_final=False, t_start_ms=0)
    assert msgs[0]["speaker_id"] == 2
    finals = session._map_text("你好世界", is_final=True, t_start_ms=0, t_end_ms=1000)
    assert finals[0]["speaker_id"] == 2


def test_open_utterance_keeps_buffer_through_trailing_silence(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        vad_speech_hangover_ms=100,
        vad_silence_ms=8000,
        meeting_emit_partial=False,
        meeting_use_fsmn_endpoint=False,
    )
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg)
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    before = len(session._utterance_chunks)
    session._last_rms_voice_ts = time.time() - 2.0
    session.feed_pcm(_pcm_chunk(cfg, loud=False))
    assert len(session._utterance_chunks) > before


def test_discardable_fragment_not_emitted(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        meeting_min_finalize_chars=6,
        meeting_emit_partial=False,
        meeting_use_fsmn_endpoint=False,
    )
    engine = MockEngine(finalize_text="。")
    session = MeetingStreamSession(engine, cfg)
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    session._last_rms_voice_ts = time.time() - 3.0
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=False))
    assert not any(m["type"] == "final" for m in msgs)


class SpeakerChangeMockEngine(MockEngine):
    def __init__(self) -> None:
        super().__init__(finalize_text="第一句")
        self._speaker_at: dict[int, int] = {}

    def speaker_at_ms(self, t_ms: int) -> int | None:
        return self._speaker_at.get(t_ms)

    def resolve_speaker(
        self, t_start_ms: int, t_end_ms: int, **kwargs: object
    ) -> tuple[int, bool]:
        start = self._speaker_at.get(t_start_ms, 0)
        end = self._speaker_at.get(t_end_ms, start)
        changed = end != getattr(self, "_last", start)
        self._last = end
        return end, changed


def test_speaker_change_triggers_early_finalize(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        meeting_spk_change_finalize=True,
        meeting_emit_partial=True,
        meeting_partial_min_ms=100,
        meeting_partial_interval_ms=0,
        meeting_min_finalize_chars=2,
        meeting_min_utterance_ms=100,
    )
    engine = SpeakerChangeMockEngine()
    session = MeetingStreamSession(engine, cfg)
    session._utterance_start_ms = 0
    engine._speaker_at[0] = 0
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    session._last_partial_text = "第一句内容"
    session._elapsed_ms = lambda: 2000  # type: ignore[method-assign]
    engine._speaker_at[2000] = 1
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=True))
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) >= 1
    assert finals[0]["text"] == "第一句"


def test_short_pause_does_not_finalize(meeting_config) -> None:
    cfg = replace(
        meeting_config,
        vad_silence_ms=1100,
        vad_silence_long_ms=2200,
        meeting_min_utterance_ms=900,
        meeting_emit_partial=False,
    )
    engine = MockEngine()
    session = MeetingStreamSession(engine, cfg)
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    session._last_rms_voice_ts = time.time() - 0.6
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=False))
    assert not any(m["type"] == "final" for m in msgs)
