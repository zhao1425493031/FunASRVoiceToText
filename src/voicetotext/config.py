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
    vad_speech_hangover_ms: int
    meeting_max_utterance_ms: int
    meeting_use_fsmn_endpoint: bool
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
    meeting_spk_source: str
    meeting_min_finalize_chars: int
    meeting_emit_partial: bool
    meeting_partial_interval_ms: int
    meeting_partial_min_ms: int
    meeting_min_utterance_ms: int
    meeting_reuse_partial_for_final: bool
    meeting_spk_change_finalize: bool
    meeting_partial_max_sec: float
    vad_silence_long_ms: int
    vad_model: str
    asr_model: str
    pyannote_model: str
    pyannote_window_sec: float
    pyannote_context_sec: float
    pyannote_step_sec: float
    pyannote_min_speakers: int
    pyannote_max_speakers: int
    pyannote_hf_token_env: str
    meeting_use_utterance_embedding: bool
    meeting_spk_embedding_model: str
    meeting_spk_embedding_threshold: float

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
    """Resolve config device with safe CUDA fallback for enterprise deployment."""
    raw = str(device).strip().lower()
    if raw == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"
    if raw.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
        return raw if ":" in raw else "cuda:0"
    return raw


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

    asr_backend = str(raw.get("asr_backend", "meeting_sensevoice")).lower()
    if asr_backend != "meeting_sensevoice":
        raise ValueError("Meeting v3 requires asr_backend=meeting_sensevoice")

    for forbidden_key in ("qwen_partial_model", "qwen_final_model", "funasr_hub"):
        if raw.get(forbidden_key):
            raise ValueError(
                f"Meeting v3 removed {forbidden_key}; use asr_model for SenseVoice"
            )

    if not raw.get("asr_model"):
        raise ValueError("Meeting v3 requires asr_model (e.g. iic/SenseVoiceSmall)")

    if raw.get("meeting_spk_model"):
        raise ValueError("meeting_spk_model (cam++) removed in v2; use pyannote_*")

    lang = str(raw.get("language", "ja")).lower()
    if lang not in ("ja", "zh"):
        raise ValueError(f"Unsupported language: {lang} (use ja or zh)")

    spk_source = str(raw.get("meeting_spk_source", "hybrid")).lower()
    if spk_source not in ("client", "pyannote", "hybrid"):
        raise ValueError("meeting_spk_source must be client, pyannote, or hybrid")

    needs_pyannote = spk_source in ("pyannote", "hybrid")
    if (
        needs_pyannote
        and raw.get("meeting_use_diarization", True)
        and raw.get("meeting_spk_mode", "multi") == "multi"
    ):
        env_name = str(raw.get("pyannote_hf_token_env", "HF_TOKEN"))
        if not os.environ.get(env_name):
            if os.environ.get("VOICETOTEXT_SKIP_HF_CHECK") != "1":
                pass


def _valid_hf_token(token: str) -> bool:
    t = str(token).strip()
    if not t.startswith("hf_") or len(t) < 20:
        return False
    placeholders = ("你的", "在这里粘贴", "paste your", "placeholder", "hf_xxxx")
    lower = t.lower()
    return not any(p in lower for p in placeholders)


def apply_meeting_secrets(config_path: Path | None = None) -> bool:
    """
    Load HF token from secrets.meeting.yaml into os.environ before server start.
    Skips if env already set. Returns True if token is available after apply.
    """
    path = resolve_config_path(config_path)
    if not path.is_file():
        return bool(_valid_hf_token(os.environ.get("HF_TOKEN", "")))

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    env_name = str(raw.get("pyannote_hf_token_env", "HF_TOKEN"))
    existing = os.environ.get(env_name, "").strip()
    if _valid_hf_token(existing):
        return True

    secrets_rel = str(raw.get("secrets_file", "secrets.meeting.yaml"))
    secrets_path = PROJECT_ROOT / secrets_rel
    if not secrets_path.is_file():
        return False

    with secrets_path.open(encoding="utf-8") as f:
        secrets = yaml.safe_load(f) or {}

    token = secrets.get("hf_token") or secrets.get("HF_TOKEN")
    if token is None or not _valid_hf_token(str(token)):
        return False

    os.environ[env_name] = str(token).strip()
    return True


def require_hf_token_for_meeting(config_path: Path | None = None) -> None:
    """Raise SystemExit with setup hints if Pyannote needs HF token but none configured."""
    path = resolve_config_path(config_path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    spk_source = str(raw.get("meeting_spk_source", "hybrid")).lower()
    if spk_source not in ("pyannote", "hybrid"):
        return
    if not raw.get("meeting_use_diarization", True):
        return
    if str(raw.get("meeting_spk_mode", "multi")).lower() != "multi":
        return

    env_name = str(raw.get("pyannote_hf_token_env", "HF_TOKEN"))
    if apply_meeting_secrets(path):
        return

    secrets_rel = str(raw.get("secrets_file", "secrets.meeting.yaml"))
    secrets_path = PROJECT_ROOT / secrets_rel
    msg = f"""
错误：缺少 HuggingFace Token（环境变量 {env_name}）

多人说话人分离需要 Pyannote，请先配置 Token（只需做一次）：

  1. 复制模板：
     copy secrets.meeting.yaml.example secrets.meeting.yaml

  2. 打开 https://huggingface.co/pyannote/speaker-diarization-community-1 点击同意许可

  3. 在 https://huggingface.co/settings/tokens 创建 Read 类型 Token

  4. 编辑 {secrets_path.name} ，将 hf_token 改为你的真实 Token（形如 hf_xxxxxxxx...）

  5. 再运行：python scripts/run_meeting.py

若暂不需要多人分离，可在 config.meeting.yaml 设置：
  meeting_spk_mode: single
"""
    raise SystemExit(msg.strip())


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
        asr_backend=str(raw.get("asr_backend", "meeting_sensevoice")),
        language=str(raw.get("language", "ja")),
        device=str(raw.get("device", "auto")),
        chunk_size=_coerce_int_list(raw.get("chunk_size", [0, 10, 5]), "chunk_size"),
        encoder_chunk_look_back=int(raw.get("encoder_chunk_look_back", 4)),
        decoder_chunk_look_back=int(raw.get("decoder_chunk_look_back", 1)),
        vad_silence_ms=int(raw.get("vad_silence_ms", 1000)),
        vad_speech_hangover_ms=int(raw.get("vad_speech_hangover_ms", 450)),
        meeting_max_utterance_ms=int(raw.get("meeting_max_utterance_ms", 60000)),
        meeting_use_fsmn_endpoint=bool(raw.get("meeting_use_fsmn_endpoint", True)),
        vad_energy_threshold=float(raw.get("vad_energy_threshold", 0.005)),
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
        meeting_spk_source=str(raw.get("meeting_spk_source", "hybrid")).lower(),
        meeting_min_finalize_chars=int(raw.get("meeting_min_finalize_chars", 2)),
        meeting_emit_partial=bool(raw.get("meeting_emit_partial", True)),
        meeting_partial_interval_ms=int(raw.get("meeting_partial_interval_ms", 1200)),
        meeting_partial_min_ms=int(raw.get("meeting_partial_min_ms", 500)),
        meeting_min_utterance_ms=int(raw.get("meeting_min_utterance_ms", 400)),
        meeting_reuse_partial_for_final=bool(
            raw.get("meeting_reuse_partial_for_final", False)
        ),
        meeting_spk_change_finalize=bool(
            raw.get("meeting_spk_change_finalize", True)
        ),
        meeting_partial_max_sec=float(raw.get("meeting_partial_max_sec", 0)),
        vad_silence_long_ms=int(raw.get("vad_silence_long_ms", 1600)),
        vad_model=str(raw.get("vad_model", "fsmn-vad")),
        asr_model=str(raw.get("asr_model", "iic/SenseVoiceSmall")),
        pyannote_model=str(
            raw.get("pyannote_model", "pyannote/speaker-diarization-community-1")
        ),
        pyannote_window_sec=float(raw.get("pyannote_window_sec", 10)),
        pyannote_context_sec=float(raw.get("pyannote_context_sec", 60)),
        pyannote_step_sec=float(raw.get("pyannote_step_sec", 4)),
        pyannote_min_speakers=int(raw.get("pyannote_min_speakers", 1)),
        pyannote_max_speakers=int(raw.get("pyannote_max_speakers", 0)),
        pyannote_hf_token_env=str(raw.get("pyannote_hf_token_env", "HF_TOKEN")),
        meeting_use_utterance_embedding=bool(
            raw.get("meeting_use_utterance_embedding", True)
        ),
        meeting_spk_embedding_model=str(
            raw.get(
                "meeting_spk_embedding_model",
                "iic/speech_campplus_sv_zh-cn_16k-common",
            )
        ),
        meeting_spk_embedding_threshold=float(
            raw.get("meeting_spk_embedding_threshold", 0.72)
        ),
    )
