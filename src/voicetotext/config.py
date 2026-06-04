"""Load and validate application configuration from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.meeting.yaml"


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
    auto_finalize_on_silence: bool
    vad_energy_threshold: float
    min_partial_chars: int
    stream_window_ms: int
    log_dir: str
    sample_rate: int
    ssl_certfile: str | None
    ssl_keyfile: str | None
    trust_remote_code: bool
    api_key: str | None
    max_ws_connections: int
    session_pcm_max_seconds: int
    ja_apply_punctuation: bool
    service_mode: str
    api_key_scopes: tuple[str, ...]
    meeting_max_speakers: int
    meeting_session_max_seconds: int
    meeting_use_diarization: bool
    meeting_emit_partial: bool
    meeting_spk_model: str
    meeting_vad_model: str

    @property
    def is_meeting_service(self) -> bool:
        return self.service_mode.lower() == "meeting"

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
    def session_pcm_max_bytes(self) -> int:
        return self.session_pcm_max_seconds * self.sample_rate * 2

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


def _coerce_str_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return ()


def _validate_config(raw: dict[str, Any]) -> None:
    service_mode = str(raw.get("service_mode", "meeting")).lower()
    asr_backend = str(raw.get("asr_backend", "embedded")).lower()
    if service_mode != "meeting":
        raise ValueError("This project only supports service_mode=meeting")
    if asr_backend != "embedded":
        raise ValueError("Meeting service requires asr_backend=embedded")
    lang = str(raw.get("language", "ja")).lower()
    if lang not in ("ja", "zh", "auto"):
        raise ValueError(f"Unsupported language: {lang}")


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

    _validate_config(raw)

    service_mode = str(raw.get("service_mode", "meeting")).lower()
    is_meeting = True

    port = int(raw.get("port", 8765))
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")

    return AppConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=port,
        asr_backend=str(raw.get("asr_backend", "embedded")),
        asr_model=str(raw.get("asr_model", "iic/SenseVoiceSmall")),
        vad_model=str(raw.get("vad_model", "fsmn-vad")),
        punc_model=str(raw.get("punc_model", "")),
        language=str(raw.get("language", "ja")),
        device=str(raw.get("device", "cpu" if is_meeting else "auto")),
        chunk_size=_coerce_int_list(raw.get("chunk_size", [0, 10, 5]), "chunk_size"),
        encoder_chunk_look_back=int(raw.get("encoder_chunk_look_back", 4)),
        decoder_chunk_look_back=int(raw.get("decoder_chunk_look_back", 1)),
        vad_silence_ms=int(raw.get("vad_silence_ms", 800)),
        auto_finalize_on_silence=bool(raw.get("auto_finalize_on_silence", False)),
        vad_energy_threshold=float(raw.get("vad_energy_threshold", 0.02)),
        min_partial_chars=int(raw.get("min_partial_chars", 2)),
        stream_window_ms=int(raw.get("stream_window_ms", 2000)),
        log_dir=str(raw.get("log_dir", "logs")),
        sample_rate=int(raw.get("sample_rate", 16000)),
        ssl_certfile=_optional_str(raw.get("ssl_certfile")),
        ssl_keyfile=_optional_str(raw.get("ssl_keyfile")),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        api_key=_optional_str(raw.get("api_key")),
        max_ws_connections=int(raw.get("max_ws_connections", 20)),
        session_pcm_max_seconds=int(raw.get("session_pcm_max_seconds", 600)),
        ja_apply_punctuation=bool(raw.get("ja_apply_punctuation", False)),
        service_mode=service_mode,
        api_key_scopes=_coerce_str_list(raw.get("api_key_scopes")),
        meeting_max_speakers=int(raw.get("meeting_max_speakers", 8)),
        meeting_session_max_seconds=int(raw.get("meeting_session_max_seconds", 7200)),
        meeting_use_diarization=bool(raw.get("meeting_use_diarization", True)),
        meeting_emit_partial=bool(raw.get("meeting_emit_partial", False)),
        meeting_spk_model=str(
            raw.get("meeting_spk_model", "iic/speech_campplus_sv_zh-cn_16k-common")
        ),
        meeting_vad_model=str(raw.get("meeting_vad_model", "fsmn-vad")),
    )
