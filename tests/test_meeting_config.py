"""Meeting v3 configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from voicetotext.config import load_config, DEFAULT_CONFIG_PATH, resolve_device

ROOT = Path(__file__).resolve().parents[1]


def test_load_meeting_config() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    assert cfg.service_mode == "meeting"
    assert cfg.port == 8766
    assert cfg.asr_backend == "meeting_sensevoice"
    assert cfg.is_meeting_service is True
    assert cfg.meeting_max_speakers == 8
    assert cfg.meeting_session_max_seconds == 7200
    assert cfg.device == "auto"
    assert cfg.language in ("ja", "zh")
    assert cfg.meeting_spk_source == "pyannote"
    assert cfg.meeting_reuse_partial_for_final is True
    assert cfg.meeting_partial_max_sec == 0.0
    assert cfg.meeting_use_fsmn_endpoint is True
    assert cfg.vad_speech_hangover_ms >= 250
    assert cfg.vad_silence_ms >= 500
    assert cfg.meeting_spk_change_finalize is True
    assert cfg.pyannote_step_sec == 2.0
    assert cfg.meeting_min_finalize_chars >= 6
    assert cfg.asr_model == "iic/SenseVoiceSmall"
    assert cfg.vad_model == "fsmn-vad"
    assert "meeting" in cfg.api_key_scopes
    assert cfg.pyannote_context_sec == 60.0
    assert cfg.meeting_use_utterance_embedding is True
    assert "campplus" in cfg.meeting_spk_embedding_model
    assert cfg.meeting_spk_embedding_threshold == 0.72


def test_resolve_device_cuda_fallback_when_unavailable() -> None:
    assert resolve_device("cuda") == "cpu" or resolve_device("cuda").startswith("cuda")
    assert resolve_device("auto") in ("cpu", "cuda:0")


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
        with pytest.raises(ValueError, match="meeting_sensevoice"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_rejects_qwen_backend() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["asr_backend"] = "meeting_qwen"
    tmp = ROOT / "tests" / "_tmp_meeting_qwen.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="meeting_sensevoice"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_rejects_qwen_model_keys() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["qwen_partial_model"] = "Qwen/Qwen3-ASR-0.6B"
    tmp = ROOT / "tests" / "_tmp_meeting_qwen_key.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="qwen_partial_model"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_rejects_missing_asr_model() -> None:
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    del raw["asr_model"]
    tmp = ROOT / "tests" / "_tmp_meeting_no_asr.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="asr_model"):
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
    from voicetotext.config import apply_meeting_secrets

    secrets = ROOT / "tests" / "_tmp_secrets.yaml"
    secrets.write_text('hf_token: "hf_from_file_test_token_ok"\n', encoding="utf-8")
    cfg = ROOT / "tests" / "_tmp_cfg_secrets.yaml"
    raw = yaml.safe_load((ROOT / "config.meeting.yaml").read_text(encoding="utf-8"))
    raw["secrets_file"] = "tests/_tmp_secrets.yaml"
    cfg.write_text(yaml.dump(raw), encoding="utf-8")
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
