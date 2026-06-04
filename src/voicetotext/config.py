"""Load and validate application configuration from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    asr_backend: str
    asr_model: str
    vad_model: str
    punc_model: str
    language: str
    device: str
    chunk_size: list[int]
    encoder_chunk_look_back: int
    decoder_chunk_look_back: int
    vad_silence_ms: int
    stream_window_ms: int
    log_dir: str
    sample_rate: int
    ssl_certfile: str | None
    ssl_keyfile: str | None
    trust_remote_code: bool
    api_key: str | None
    max_ws_connections: int
    runtime_host: str
    runtime_port: int
    runtime_mode: str
    runtime_chunk_size: str
    runtime_ssl: bool
    model_hub: str

    @property
    def chunk_stride_samples(self) -> int:
        return self.chunk_size[1] * 960

    @property
    def chunk_stride_bytes(self) -> int:
        return self.chunk_stride_samples * 2

    @property
    def stream_window_bytes(self) -> int:
        return int(self.sample_rate * (self.stream_window_ms / 1000.0)) * 2

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def log_path(self) -> Path:
        return PROJECT_ROOT / self.log_dir

    @property
    def web_root(self) -> Path:
        return PROJECT_ROOT / "web"

    @property
    def ssl_enabled(self) -> bool:
        return bool(self.ssl_certfile and self.ssl_keyfile)

    def resolve_ssl_paths(self) -> tuple[Path, Path] | None:
        if not self.ssl_enabled:
            return None
        cert = PROJECT_ROOT / self.ssl_certfile
        key = PROJECT_ROOT / self.ssl_keyfile
        if not cert.is_file():
            raise FileNotFoundError(f"SSL cert not found: {cert}")
        if not key.is_file():
            raise FileNotFoundError(f"SSL key not found: {key}")
        return cert, key


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def _coerce_int_list(value: Any, key: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must be a list of three integers")
    return [int(v) for v in value]


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    env_path = os.environ.get("VOICETOTEXT_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None) -> AppConfig:
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    port = int(raw.get("port", 8765))
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")

    return AppConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=port,
        asr_backend=str(raw.get("asr_backend", "sensevoice")),
        asr_model=str(raw.get("asr_model", "iic/SenseVoiceSmall")),
        vad_model=str(raw.get("vad_model", "fsmn-vad")),
        punc_model=str(raw.get("punc_model", "")),
        language=str(raw.get("language", "ja")),
        device=str(raw.get("device", "auto")),
        chunk_size=_coerce_int_list(raw.get("chunk_size", [0, 10, 5]), "chunk_size"),
        encoder_chunk_look_back=int(raw.get("encoder_chunk_look_back", 4)),
        decoder_chunk_look_back=int(raw.get("decoder_chunk_look_back", 1)),
        vad_silence_ms=int(raw.get("vad_silence_ms", 800)),
        stream_window_ms=int(raw.get("stream_window_ms", 2000)),
        log_dir=str(raw.get("log_dir", "logs")),
        sample_rate=int(raw.get("sample_rate", 16000)),
        ssl_certfile=_optional_str(raw.get("ssl_certfile")),
        ssl_keyfile=_optional_str(raw.get("ssl_keyfile")),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        api_key=_optional_str(raw.get("api_key")),
        max_ws_connections=int(raw.get("max_ws_connections", 20)),
        runtime_host=str(raw.get("runtime_host", "127.0.0.1")),
        runtime_port=int(raw.get("runtime_port", 10095)),
        runtime_mode=str(raw.get("runtime_mode", "2pass")),
        runtime_chunk_size=str(raw.get("runtime_chunk_size", "5,10,5")),
        runtime_ssl=bool(raw.get("runtime_ssl", False)),
        model_hub=str(raw.get("model_hub", "ms")),
    )
