"""Finalize on VAD silence (no short-fragment defer)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from voicetotext.asr.meeting_stream_session import MeetingStreamSession
from voicetotext.config import load_config
from tests.test_meeting_stream_session import MockEngine, _pcm_chunk

ROOT = Path(__file__).resolve().parents[1]


def test_short_text_finalized_on_silence() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    cfg = replace(
        cfg,
        meeting_emit_partial=False,
        vad_silence_ms=500,
        meeting_use_fsmn_endpoint=False,
    )
    engine = MockEngine()

    def short_final(_audio, _draft):
        return "关于上周。"

    engine.finalize_utterance = short_final  # type: ignore[method-assign]

    session = MeetingStreamSession(engine, cfg)
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    session._last_voice_ts = time.time() - 2.5
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=False))
    assert any(m["type"] == "final" for m in msgs)
