"""Meeting v2 configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from voicetotext.config import load_config, DEFAULT_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[1]


def test_load_meeting_config() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    assert cfg.service_mode == "meeting"
    assert cfg.port == 8766
    assert cfg.asr_backend == "meeting_qwen"
    assert cfg.is_meeting_service is True
    assert cfg.meeting_max_speakers == 8
    assert cfg.meeting_session_max_seconds == 7200
    assert cfg.device == "cpu"
    assert cfg.language == "ja"
    assert cfg.qwen_partial_model == "Qwen/Qwen3-ASR-0.6B"
    assert cfg.qwen_final_model == "Qwen/Qwen3-ASR-1.7B"
    assert cfg.vad_model == "fsmn-vad"
    assert "meeting" in cfg.api_key_scopes


def test_default_config_is_meeting_yaml() -> None:
    assert DEFAULT_CONFIG_PATH.name == "config.meeting.yaml"
    cfg = load_config()
    assert cfg.is_meeting_service is True


def test_meeting_rejects_embedded_backend() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["asr_backend"] = "embedded"
    tmp = ROOT / "tests" / "_tmp_meeting_bad.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="meeting_qwen"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_rejects_sensevoice_model_key() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["asr_model"] = "iic/SenseVoiceSmall"
    tmp = ROOT / "tests" / "_tmp_meeting_sv.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="SenseVoice"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_rejects_campplus_config() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["meeting_spk_model"] = "iic/speech_campplus_sv_zh-cn_16k-common"
    tmp = ROOT / "tests" / "_tmp_meeting_camp.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="cam"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_language_must_be_ja_or_zh() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["language"] = "auto"
    tmp = ROOT / "tests" / "_tmp_meeting_lang.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="language"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()
