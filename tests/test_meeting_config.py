"""Meeting configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from voicetotext.config import load_config, DEFAULT_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[1]


def test_load_meeting_config() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    assert cfg.service_mode == "meeting"
    assert cfg.port == 8766
    assert cfg.asr_backend == "embedded"
    assert cfg.asr_model == "iic/SenseVoiceSmall"
    assert cfg.is_meeting_service is True
    assert cfg.meeting_max_speakers == 8
    assert cfg.meeting_session_max_seconds == 7200
    assert cfg.ja_apply_punctuation is False
    assert cfg.device == "cpu"
    assert "meeting" in cfg.api_key_scopes


def test_default_config_is_meeting_yaml() -> None:
    assert DEFAULT_CONFIG_PATH.name == "config.meeting.yaml"
    cfg = load_config()
    assert cfg.is_meeting_service is True


def test_meeting_rejects_sensevoice() -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["asr_backend"] = "sensevoice"
    tmp = ROOT / "tests" / "_tmp_meeting_bad.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="embedded"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()
