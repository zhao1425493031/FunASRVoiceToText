"""Tests for configuration loading."""

from __future__ import annotations

from voicetotext.config import load_config, resolve_device


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert cfg.port == 8765
    assert cfg.asr_backend == "sensevoice"
    assert cfg.language in ("ja", "zh", "auto", "en", "ko", "yue")
    assert cfg.asr_model == "iic/SenseVoiceSmall"
    assert cfg.punc_model == ""
    assert cfg.ja_apply_punctuation is True
    assert cfg.session_pcm_max_seconds == 600
    assert cfg.session_pcm_max_bytes == 600 * 16000 * 2
    assert cfg.chunk_size == [0, 10, 5]
    assert cfg.chunk_stride_samples == 9600
    assert cfg.stream_window_ms == 2000
    assert cfg.stream_window_bytes == 64000
    assert cfg.ssl_certfile == "certs/cert.pem"
    assert cfg.ssl_keyfile == "certs/key.pem"
    assert cfg.max_ws_connections == 20


def test_resolve_device_auto() -> None:
    device = resolve_device("auto")
    assert device in ("cpu", "cuda:0")
