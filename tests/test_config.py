"""Configuration loading tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from voicetotext.config import DEFAULT_CONFIG_PATH, load_config, resolve_device

ROOT = Path(__file__).resolve().parents[1]


def test_load_config() -> None:
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.port == 8767
    assert cfg.language == "ja"
    assert "SenseVoice" in cfg.asr_model
    assert "json" in cfg.output_formats
    assert cfg.llm_enabled is False
    assert cfg.pyannote_min_speakers == 2
    assert cfg.pyannote_max_speakers == 2
    assert cfg.batch_diar_merge_gap_ms == 500
    assert cfg.batch_diar_min_segment_ms == 300
    assert cfg.batch_asr_parallel_workers == 4


def test_resolve_device() -> None:
    assert resolve_device("auto") in ("cpu", "cuda:0")


def test_default_config_path() -> None:
    assert DEFAULT_CONFIG_PATH.name == "config.yaml"


def test_rejects_sensevoice_large() -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["asr_model"] = "iic/SenseVoiceLarge"
    tmp = ROOT / "tests" / "_tmp_large.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not published"):
            load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def test_rejects_invalid_pyannote_speaker_range() -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["pyannote_min_speakers"] = 3
    raw["pyannote_max_speakers"] = 2
    tmp = ROOT / "tests" / "_tmp_speakers.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="pyannote_min_speakers"):
            load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def test_rejects_invalid_batch_workers() -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["batch_asr_parallel_workers"] = 0
    tmp = ROOT / "tests" / "_tmp_workers.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="batch_asr_parallel_workers"):
            load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)
