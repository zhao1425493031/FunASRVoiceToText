"""Defer short fragment finalization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from voicetotext.asr.meeting_stream_session import MeetingStreamSession
from voicetotext.config import load_config
from tests.test_meeting_stream_session import MockEngine, _pcm_chunk

ROOT = Path(__file__).resolve().parents[1]


def test_short_text_not_finalized_early() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    cfg = replace(
        cfg,
        meeting_min_finalize_chars=12,
        meeting_min_utterance_ms=2000,
        meeting_emit_partial=False,
        vad_silence_long_ms=3000,
    )
    engine = MockEngine()

    def short_final(_audio, _draft):
        return "关于上周。"

    engine.finalize_utterance = short_final  # type: ignore[method-assign]

    session = MeetingStreamSession(engine, cfg)
    session.feed_pcm(_pcm_chunk(cfg, loud=True))
    session._last_rms_voice_ts = __import__("time").time() - 2.5
    msgs = session.feed_pcm(_pcm_chunk(cfg, loud=False))
    assert not any(m["type"] == "final" for m in msgs)
