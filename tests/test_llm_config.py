"""LLM configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_load_llm_fields_from_config() -> None:
    cfg = load_config(FIXTURES / "config_llm_test.yaml")
    assert cfg.llm_enabled is True
    assert cfg.llm_provider == "openai_compatible"
    assert cfg.llm_model == "qwen-plus"
    assert cfg.llm_api_url == "https://example.test/compatible-mode/v1"
    assert cfg.llm_api_key == "sk-testkey1234567890abcdef"
    assert cfg.llm_temperature == 0.3
    assert cfg.llm_max_tokens == 4096
    assert cfg.llm_timeout_sec == 30
    assert cfg.llm_max_input_chars == 120000
    assert cfg.llm_chunk_chars == 30000
    assert cfg.llm_on_failure == "warn"
    assert cfg.llm_output_summary_md is True
    assert cfg.llm_ready is True


def test_main_config_llm_enabled() -> None:
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.llm_enabled is True
    assert cfg.llm_provider == "openai_compatible"
    assert cfg.llm_model == "qwen-plus"
    assert cfg.llm_timeout_sec == 120


def test_rejects_enabled_without_api_key() -> None:
    raw = yaml.safe_load((FIXTURES / "config_llm_test.yaml").read_text(encoding="utf-8"))
    raw["llm"]["api_key"] = ""
    tmp = ROOT / "tests" / "_tmp_llm_no_key.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="llm.api_key"):
            load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def test_rejects_azure_without_deployment() -> None:
    raw = yaml.safe_load((FIXTURES / "config_llm_test.yaml").read_text(encoding="utf-8"))
    raw["llm"]["deployment"] = "gpt-4o"
    raw["llm"]["api_version"] = ""
    tmp = ROOT / "tests" / "_tmp_llm_azure.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="api_version"):
            load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def test_rejects_placeholder_api_key() -> None:
    raw = yaml.safe_load((FIXTURES / "config_llm_test.yaml").read_text(encoding="utf-8"))
    raw["llm"]["api_key"] = "sk-xxx"
    tmp = ROOT / "tests" / "_tmp_llm_placeholder.yaml"
    tmp.write_text(yaml.dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="llm.api_key"):
            load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)
