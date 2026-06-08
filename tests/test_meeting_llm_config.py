"""LLM configuration parsing and validation for meeting service."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from voicetotext.config import load_config

from tests.conftest import LLM_TEST_CONFIG, ROOT


def test_meeting_llm_config_loads_all_fields() -> None:
    cfg = load_config(LLM_TEST_CONFIG)
    assert cfg.llm_enabled is True
    assert cfg.llm_provider == "openai_compatible"
    assert cfg.llm_model == "qwen-plus-test"
    assert cfg.llm_api_url == "https://mock-llm.test/v1"
    assert cfg.llm_api_key == "sk-test-fake-meeting-llm-key-abcdef"
    assert cfg.llm_temperature == 0.3
    assert cfg.llm_max_tokens == 4096
    assert cfg.llm_timeout_sec == 30
    assert cfg.llm_max_input_chars == 120000
    assert cfg.llm_chunk_chars == 30000
    assert cfg.llm_retry_max == 1
    assert cfg.llm_retry_backoff_sec == 0.01
    assert cfg.llm_on_failure == "warn"
    assert cfg.llm_output_summary_md is True
    assert cfg.llm_meeting_output_dir == "tests/_tmp_meetings"
    assert cfg.llm_meeting_ws_wait_sec == 60
    assert cfg.llm_ready is True


def test_meeting_llm_enabled_without_api_key_rejected() -> None:
    raw = yaml.safe_load(LLM_TEST_CONFIG.read_text(encoding="utf-8"))
    raw["llm"]["api_key"] = ""
    tmp = ROOT / "tests" / "_tmp_llm_no_key.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="api_key"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_llm_azure_requires_deployment() -> None:
    raw = yaml.safe_load(LLM_TEST_CONFIG.read_text(encoding="utf-8"))
    raw["llm"]["deployment"] = "gpt-4o"
    raw["llm"]["api_url"] = "https://example.openai.azure.com"
    tmp = ROOT / "tests" / "_tmp_llm_azure.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="api_version"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_llm_placeholder_key_rejected() -> None:
    raw = yaml.safe_load(LLM_TEST_CONFIG.read_text(encoding="utf-8"))
    raw["llm"]["api_key"] = "sk-你的密钥"
    tmp = ROOT / "tests" / "_tmp_llm_placeholder.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="api_key"):
            load_config(tmp)
    finally:
        if tmp.is_file():
            tmp.unlink()


def test_meeting_llm_disabled_uses_stub_provider() -> None:
    raw = yaml.safe_load(LLM_TEST_CONFIG.read_text(encoding="utf-8"))
    raw["llm"]["enabled"] = False
    tmp = ROOT / "tests" / "_tmp_llm_disabled.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        cfg = load_config(tmp)
        assert cfg.llm_enabled is False
        assert cfg.llm_provider == "stub"
        assert cfg.llm_ready is False
    finally:
        if tmp.is_file():
            tmp.unlink()
