"""ASR factory tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from voicetotext.asr.factory import create_asr_backend
from voicetotext.asr.meeting_sensevoice_engine import MeetingSenseVoiceEngine
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_create_meeting_sensevoice_backend() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    eng = create_asr_backend(cfg)
    assert isinstance(eng, MeetingSenseVoiceEngine)
    assert eng.backend_name == "meeting_sensevoice"


def test_reject_embedded_backend() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["asr_backend"] = "embedded"
    tmp = ROOT / "tests" / "_tmp_factory.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="meeting_sensevoice"):
            create_asr_backend(load_config(tmp))
    finally:
        if tmp.is_file():
            tmp.unlink()
