"""Load and validate batch ASR configuration."""

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
    language: str
    device: str
    log_dir: str
    sample_rate: int
    ssl_certfile: str | None
    ssl_keyfile: str | None
    trust_remote_code: bool
    api_key: str | None
    vad_model: str
    vad_energy_threshold: float
    asr_model: str
    pyannote_model: str
    pyannote_min_speakers: int
    pyannote_max_speakers: int
    pyannote_hf_token_env: str
    secrets_file: str
    output_formats: tuple[str, ...]
    max_audio_duration_sec: int
    batch_align_min_overlap_ms: int
    batch_max_segment_ms: int
    batch_diar_merge_gap_ms: int
    batch_diar_min_segment_ms: int
    batch_asr_parallel_workers: int
    llm_enabled: bool
    llm_provider: str
    llm_model: str

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


def _coerce_str_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return ()


def _parse_llm(raw: dict[str, Any]) -> tuple[bool, str, str]:
    llm = raw.get("llm") or {}
    if not isinstance(llm, dict):
        llm = {}
    return (
        bool(llm.get("enabled", False)),
        str(llm.get("provider", "stub")),
        str(llm.get("model", "")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _validate_config(raw: dict[str, Any]) -> None:
    asr_model = str(raw.get("asr_model", ""))
    if not asr_model:
        raise ValueError("config requires asr_model (e.g. iic/SenseVoiceSmall)")
    norm = asr_model.lower().replace("_", "").replace("-", "")
    if "sensevoicelarge" in norm:
        raise ValueError(
            "iic/SenseVoiceLarge is not published on FunASR; use iic/SenseVoiceSmall"
        )
    lang = str(raw.get("language", "ja")).lower()
    if lang not in ("ja", "zh"):
        raise ValueError(f"Unsupported language: {lang} (use ja or zh)")

    min_spk = int(raw.get("pyannote_min_speakers", 2))
    max_spk = int(raw.get("pyannote_max_speakers", 2))
    if min_spk < 1:
        raise ValueError(f"pyannote_min_speakers must be >= 1, got {min_spk}")
    if max_spk > 0 and min_spk > max_spk:
        raise ValueError(
            f"pyannote_min_speakers ({min_spk}) must be <= "
            f"pyannote_max_speakers ({max_spk})"
        )

    merge_gap = int(raw.get("batch_diar_merge_gap_ms", 500))
    if merge_gap < 0:
        raise ValueError(f"batch_diar_merge_gap_ms must be >= 0, got {merge_gap}")

    min_seg = int(raw.get("batch_diar_min_segment_ms", 300))
    if min_seg < 100:
        raise ValueError(f"batch_diar_min_segment_ms must be >= 100, got {min_seg}")

    workers = int(raw.get("batch_asr_parallel_workers", 4))
    if workers < 1:
        raise ValueError(f"batch_asr_parallel_workers must be >= 1, got {workers}")


def _valid_hf_token(token: str) -> bool:
    t = str(token).strip()
    if not t.startswith("hf_") or len(t) < 20:
        return False
    placeholders = ("你的", "在这里粘贴", "paste your", "placeholder", "hf_xxxx")
    lower = t.lower()
    return not any(p in lower for p in placeholders)


def apply_secrets(config_path: Path | None = None) -> bool:
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


def require_hf_token(config_path: Path | None = None) -> None:
    path = resolve_config_path(config_path)
    if apply_secrets(path):
        return
    env_name = "HF_TOKEN"
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        env_name = str(raw.get("pyannote_hf_token_env", "HF_TOKEN"))
        secrets_rel = str(raw.get("secrets_file", "secrets.meeting.yaml"))
    else:
        secrets_rel = "secrets.meeting.yaml"
    raise SystemExit(
        f"缺少 HuggingFace Token。请配置 {secrets_rel} 或环境变量 {env_name}，"
        f"并访问 https://huggingface.co/pyannote/speaker-diarization-community-1 接受许可。"
    )


# 兼容旧脚本名
apply_meeting_secrets = apply_secrets
require_hf_token_for_meeting = require_hf_token


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
    llm_enabled, llm_provider, llm_model = _parse_llm(raw)
    output_formats = _coerce_str_list(raw.get("output_formats", ("json", "md", "srt")))
    if not output_formats:
        output_formats = ("json", "md", "srt")

    port = int(raw.get("port", 8767))
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")

    return AppConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=port,
        language=str(raw.get("language", "ja")),
        device=str(raw.get("device", "auto")),
        log_dir=str(raw.get("log_dir", "logs")),
        sample_rate=int(raw.get("sample_rate", 16000)),
        ssl_certfile=_optional_str(raw.get("ssl_certfile")),
        ssl_keyfile=_optional_str(raw.get("ssl_keyfile")),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        api_key=_optional_str(raw.get("api_key")),
        vad_model=str(raw.get("vad_model", "fsmn-vad")),
        vad_energy_threshold=float(raw.get("vad_energy_threshold", 0.01)),
        asr_model=str(raw.get("asr_model", "iic/SenseVoiceSmall")),
        pyannote_model=str(
            raw.get("pyannote_model", "pyannote/speaker-diarization-community-1")
        ),
        pyannote_min_speakers=int(raw.get("pyannote_min_speakers", 2)),
        pyannote_max_speakers=int(raw.get("pyannote_max_speakers", 2)),
        pyannote_hf_token_env=str(raw.get("pyannote_hf_token_env", "HF_TOKEN")),
        secrets_file=str(raw.get("secrets_file", "secrets.meeting.yaml")),
        output_formats=output_formats,
        max_audio_duration_sec=int(raw.get("max_audio_duration_sec", 14400)),
        batch_align_min_overlap_ms=int(raw.get("batch_align_min_overlap_ms", 300)),
        batch_max_segment_ms=int(raw.get("batch_max_segment_ms", 30000)),
        batch_diar_merge_gap_ms=int(raw.get("batch_diar_merge_gap_ms", 500)),
        batch_diar_min_segment_ms=int(raw.get("batch_diar_min_segment_ms", 300)),
        batch_asr_parallel_workers=int(raw.get("batch_asr_parallel_workers", 4)),
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
