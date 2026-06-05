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
    assert cfg.language in ("ja", "zh")
    assert cfg.meeting_spk_source == "pyannote"
    assert cfg.meeting_reuse_partial_for_final is True
    assert cfg.meeting_partial_max_sec == 0.0
    assert cfg.meeting_use_fsmn_endpoint is True
    assert cfg.vad_speech_hangover_ms >= 400
    assert cfg.vad_silence_ms >= 800
    assert cfg.meeting_min_finalize_chars >= 6
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


def test_apply_meeting_secrets_from_file() -> None:
    import os

    from voicetotext.config import apply_meeting_secrets

    secrets = ROOT / "tests" / "_tmp_secrets.yaml"
    secrets.write_text('hf_token: "hf_from_file_test_token_ok"\n', encoding="utf-8")
    cfg = ROOT / "tests" / "_tmp_cfg_secrets.yaml"
    raw = __import__("yaml").safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["secrets_file"] = "tests/_tmp_secrets.yaml"
    cfg.write_text(__import__("yaml").dump(raw), encoding="utf-8")
    old = os.environ.pop("HF_TOKEN", None)
    try:
        assert apply_meeting_secrets(cfg) is True
        assert os.environ.get("HF_TOKEN") == "hf_from_file_test_token_ok"
    finally:
        if old:
            os.environ["HF_TOKEN"] = old
        else:
            os.environ.pop("HF_TOKEN", None)
        for p in (secrets, cfg):
            if p.is_file():
                p.unlink()


def test_meeting_spk_source_must_be_valid() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["meeting_spk_source"] = "invalid"
    tmp = ROOT / "tests" / "_tmp_meeting_spk.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="meeting_spk_source"):
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
