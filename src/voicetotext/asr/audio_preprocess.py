"""Load and normalize audio for batch processing (16 kHz mono float32)."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np

from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


class AudioPreprocessError(ValueError):
    pass


# 复制路径时常见的不可见字符（如 U+202A）
_INVISIBLE = "".join(
    chr(c)
    for c in (
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0xFEFF,
    )
)


def sanitize_audio_path(raw: str | Path) -> Path:
    """Strip invisible bidi marks and quotes from pasted Windows paths."""
    s = str(raw).strip().strip('"').strip("'")
    for ch in _INVISIBLE:
        s = s.replace(ch, "")
    return Path(s).expanduser().resolve()


def load_audio_file(path: Path | str, config: AppConfig) -> tuple[np.ndarray, int]:
    """Return (float32 mono samples, duration_ms)."""
    path = sanitize_audio_path(path)
    if not path.is_file():
        raise AudioPreprocessError(f"Audio file not found: {path}")

    mono, sr = _read_audio(path)
    if mono.size == 0:
        raise AudioPreprocessError("Audio file is empty")
    target_sr = config.sample_rate
    if sr != target_sr:
        mono = _resample(mono, int(sr), target_sr)

    duration_ms = int(len(mono) / target_sr * 1000)
    max_ms = config.max_audio_duration_sec * 1000
    if duration_ms > max_ms:
        raise AudioPreprocessError(
            f"Audio duration {duration_ms}ms exceeds max {max_ms}ms"
        )
    if duration_ms < 100:
        raise AudioPreprocessError("Audio too short (<100ms)")

    logger.info("Loaded audio %s duration_ms=%d sr=%d", path.name, duration_ms, target_sr)
    return mono.astype(np.float32), duration_ms


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read audio via soundfile; fall back to ffmpeg/torchaudio for m4a/mp3/aac."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.ndim == 1:
            mono = data
        else:
            mono = data.mean(axis=1)
        return mono.astype(np.float32), int(sr)
    except Exception as sf_exc:
        logger.debug("soundfile failed for %s: %s", path.name, sf_exc)

    try:
        return _read_via_ffmpeg(path)
    except Exception as ff_exc:
        logger.debug("ffmpeg failed for %s: %s", path.name, ff_exc)

    try:
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(path))
        mono = waveform.mean(dim=0).numpy().astype(np.float32)
        return mono, int(sr)
    except Exception as ta_exc:
        raise AudioPreprocessError(
            f"Cannot read audio {path.name}. Install ffmpeg (or pip install imageio-ffmpeg), "
            f"or convert to wav. ({ta_exc})"
        ) from ta_exc


def _ffmpeg_executable() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _read_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        raise AudioPreprocessError("ffmpeg not found")

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioPreprocessError(err or f"ffmpeg exit {proc.returncode}")

    import soundfile as sf

    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
    mono = data if data.ndim == 1 else data.mean(axis=1)
    return mono.astype(np.float32), int(sr)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    try:
        import torch
        import torchaudio

        t = torch.from_numpy(audio).unsqueeze(0)
        out = torchaudio.functional.resample(t, src_sr, dst_sr)
        return out.squeeze(0).numpy()
    except Exception:
        ratio = dst_sr / src_sr
        new_len = int(len(audio) * ratio)
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)
