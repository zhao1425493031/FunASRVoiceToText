"""Meeting /ready logic tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voicetotext.asr.meeting_sensevoice_engine import MeetingSenseVoiceEngine
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_check_ready_false_without_load() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    engine = MeetingSenseVoiceEngine(cfg)
    assert await engine.check_ready() is False


@pytest.mark.asyncio
async def test_check_ready_true_when_mocked() -> None:
    os.environ["HF_TOKEN"] = "hf_test"
    cfg = load_config(ROOT / "config.meeting.yaml")
    engine = MeetingSenseVoiceEngine(cfg)
    engine._ready = True
    engine._vad._ready = True
    engine._vad._model = object()
    engine._asr._ready = True
    engine._asr._model = object()
    engine._pyannote._ready = True
    engine._pyannote._pipeline = object()
    if engine._embedding is not None:
        engine._embedding._ready = True
        engine._embedding._model = object()
    assert await engine.check_ready() is True
    del os.environ["HF_TOKEN"]


@pytest.mark.asyncio
async def test_check_ready_false_without_hf_token_multi() -> None:
    os.environ.pop("HF_TOKEN", None)
    cfg = load_config(ROOT / "config.meeting.yaml")
    engine = MeetingSenseVoiceEngine(cfg)
    engine._ready = True
    engine._vad._ready = True
    engine._vad._model = object()
    engine._asr._ready = True
    engine._asr._model = object()
    engine._pyannote._ready = True
    engine._pyannote._pipeline = object()
    if engine._embedding is not None:
        engine._embedding._ready = True
        engine._embedding._model = object()
    assert await engine.check_ready() is False
