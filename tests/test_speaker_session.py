"""SpeakerSessionContext isolation tests."""

from __future__ import annotations

from pathlib import Path

from voicetotext.asr.speaker_session import SpeakerSessionContext
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_speaker_context_create_and_reset() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    ctx = SpeakerSessionContext.create(cfg, "test-session")
    assert ctx.session_id == "test-session"
    assert ctx.registry.speaker_count == 0
    ctx.registry.assign(__import__("numpy").random.randn(8).astype("float32"))
    assert ctx.registry.speaker_count == 1
    ctx.reset()
    assert ctx.registry.speaker_count == 0
    assert ctx.merger.last_speaker_id == 0
