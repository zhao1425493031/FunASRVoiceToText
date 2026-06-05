"""Load and validate application configuration from config.meeting.yaml."""

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
    language: str
    device: str
    chunk_size: list[int]
    encoder_chunk_look_back: int
    decoder_chunk_look_back: int
    vad_silence_ms: int
    vad_energy_threshold: float
    min_partial_chars: int
    log_dir: str
    sample_rate: int
    ssl_certfile: str | None
    ssl_keyfile: str | None
    trust_remote_code: bool
    api_key: str | None
    max_ws_connections: int
    service_mode: str
    api_key_scopes: tuple[str, ...]
    meeting_max_speakers: int
    meeting_session_max_seconds: int
    meeting_use_diarization: bool
    meeting_spk_mode: str
    meeting_min_finalize_chars: int
    meeting_emit_partial: bool
    meeting_partial_interval_ms: int
    meeting_partial_min_ms: int
    meeting_min_utterance_ms: int
    vad_silence_long_ms: int
    vad_model: str
    qwen_partial_model: str
    qwen_final_model: str
    funasr_hub: str
    pyannote_model: str
    pyannote_window_sec: float
    pyannote_step_sec: float
    pyannote_hf_token_env: str

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
    if service_mode != "meeting":
        raise ValueError("This project only supports service_mode=meeting")

    asr_backend = str(raw.get("asr_backend", "meeting_qwen")).lower()
    if asr_backend != "meeting_qwen":
        raise ValueError("Meeting v2 requires asr_backend=meeting_qwen")

    for forbidden in ("sensevoice", "campplus"):
        blob = str(raw).lower()
        if forbidden in blob and any(
            k in raw for k in ("asr_model", "meeting_spk_model") if raw.get(k)
        ):
            pass
    if raw.get("asr_model") and "sensevoice" in str(raw.get("asr_model", "")).lower():
        raise ValueError("Meeting v2 does not support SenseVoice; use qwen_*_model")
    if raw.get("meeting_spk_model"):
        raise ValueError("meeting_spk_model (cam++) removed in v2; use pyannote_*")

    lang = str(raw.get("language", "ja")).lower()
    if lang not in ("ja", "zh"):
        raise ValueError(f"Unsupported language: {lang} (use ja or zh)")

    if raw.get("meeting_use_diarization", True) and raw.get("meeting_spk_mode", "multi") == "multi":
        env_name = str(raw.get("pyannote_hf_token_env", "HF_TOKEN"))
        if not os.environ.get(env_name):
            # Allow missing token at load time for tests via VOICETOTEXT_SKIP_HF_CHECK=1
            if os.environ.get("VOICETOTEXT_SKIP_HF_CHECK") != "1":
                pass  # ready endpoint will report not ready


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

    port = int(raw.get("port", 8766))
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")

    return AppConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=port,
        asr_backend=str(raw.get("asr_backend", "meeting_qwen")),
        language=str(raw.get("language", "ja")),
        device=str(raw.get("device", "cpu")),
        chunk_size=_coerce_int_list(raw.get("chunk_size", [0, 10, 5]), "chunk_size"),
        encoder_chunk_look_back=int(raw.get("encoder_chunk_look_back", 4)),
        decoder_chunk_look_back=int(raw.get("decoder_chunk_look_back", 1)),
        vad_silence_ms=int(raw.get("vad_silence_ms", 1500)),
        vad_energy_threshold=float(raw.get("vad_energy_threshold", 0.008)),
        min_partial_chars=int(raw.get("min_partial_chars", 1)),
        log_dir=str(raw.get("log_dir", "logs")),
        sample_rate=int(raw.get("sample_rate", 16000)),
        ssl_certfile=_optional_str(raw.get("ssl_certfile")),
        ssl_keyfile=_optional_str(raw.get("ssl_keyfile")),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        api_key=_optional_str(raw.get("api_key")),
        max_ws_connections=int(raw.get("max_ws_connections", 10)),
        service_mode=str(raw.get("service_mode", "meeting")),
        api_key_scopes=_coerce_str_list(raw.get("api_key_scopes")),
        meeting_max_speakers=int(raw.get("meeting_max_speakers", 8)),
        meeting_session_max_seconds=int(raw.get("meeting_session_max_seconds", 7200)),
        meeting_use_diarization=bool(raw.get("meeting_use_diarization", True)),
        meeting_spk_mode=str(raw.get("meeting_spk_mode", "multi")).lower(),
        meeting_min_finalize_chars=int(raw.get("meeting_min_finalize_chars", 15)),
        meeting_emit_partial=bool(raw.get("meeting_emit_partial", True)),
        meeting_partial_interval_ms=int(raw.get("meeting_partial_interval_ms", 2000)),
        meeting_partial_min_ms=int(raw.get("meeting_partial_min_ms", 800)),
        meeting_min_utterance_ms=int(raw.get("meeting_min_utterance_ms", 1600)),
        vad_silence_long_ms=int(raw.get("vad_silence_long_ms", 2600)),
        vad_model=str(raw.get("vad_model", "fsmn-vad")),
        qwen_partial_model=str(raw.get("qwen_partial_model", "Qwen/Qwen3-ASR-0.6B")),
        qwen_final_model=str(raw.get("qwen_final_model", "Qwen/Qwen3-ASR-1.7B")),
        funasr_hub=str(raw.get("funasr_hub", "hf")),
        pyannote_model=str(
            raw.get("pyannote_model", "pyannote/speaker-diarization-community-1")
        ),
        pyannote_window_sec=float(raw.get("pyannote_window_sec", 10)),
        pyannote_step_sec=float(raw.get("pyannote_step_sec", 5)),
        pyannote_hf_token_env=str(raw.get("pyannote_hf_token_env", "HF_TOKEN")),
    )
