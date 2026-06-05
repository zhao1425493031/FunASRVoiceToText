"""Sliding-window Pyannote Community-1 diarization worker (enterprise)."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from voicetotext.asr.speaker_timeline import DiarizationSegment
from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

if TYPE_CHECKING:
    from voicetotext.asr.speaker_session import SpeakerSessionContext

logger = get_logger(__name__)


class AudioRingBuffer:
    """Thread-safe PCM s16le ring buffer with session-relative timestamps."""

    def __init__(self, sample_rate: int, max_seconds: int) -> None:
        self.sample_rate = sample_rate
        self._max_bytes = sample_rate * 2 * max_seconds
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._total_samples_written = 0

    def append(self, pcm_bytes: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm_bytes)
            self._total_samples_written += len(pcm_bytes) // 2
            if len(self._buf) > self._max_bytes:
                drop = len(self._buf) - self._max_bytes
                del self._buf[:drop]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._total_samples_written = 0

    def snapshot_context(self, context_sec: float) -> tuple[np.ndarray, int]:
        """Return up to context_sec of trailing audio and its session start_ms."""
        need_bytes = int(self.sample_rate * context_sec) * 2
        with self._lock:
            chunk = bytes(self._buf[-need_bytes:]) if self._buf else b""
            end_sample = self._total_samples_written
        if not chunk:
            return np.array([], dtype=np.float32), 0
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        start_sample = max(0, end_sample - len(audio))
        start_ms = int(start_sample / self.sample_rate * 1000)
        return audio, start_ms

    @property
    def duration_ms(self) -> int:
        with self._lock:
            samples = len(self._buf) // 2
        return int(samples / self.sample_rate * 1000)


def _parse_pyannote_output(output: Any, window_start_ms: int) -> list[DiarizationSegment]:
    segments: list[DiarizationSegment] = []
    try:
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive is None and hasattr(output, "speaker_diarization"):
            ann = output.speaker_diarization
        else:
            ann = exclusive
        if ann is None:
            return segments
        for turn, _, speaker in ann.itertracks(yield_label=True):
            start_ms = window_start_ms + int(turn.start * 1000)
            end_ms = window_start_ms + int(turn.end * 1000)
            segments.append(
                DiarizationSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_label=str(speaker),
                )
            )
    except Exception as exc:
        logger.warning("Parse pyannote output failed: %s", exc)
    return segments


class PyannoteWorker:
    def __init__(self, config: AppConfig, *, device: str) -> None:
        self.config = config
        self.device = device
        self._pipeline: Any = None
        self._ready = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._bound: SpeakerSessionContext | None = None

    @property
    def is_loaded(self) -> bool:
        return self._ready and self._pipeline is not None

    def hf_token(self) -> str | None:
        env_name = self.config.pyannote_hf_token_env
        return os.environ.get(env_name) or None

    def _pipeline_kwargs(self) -> dict[str, int]:
        max_spk = self.config.pyannote_max_speakers
        if max_spk <= 0:
            max_spk = self.config.meeting_max_speakers
        return {
            "min_speakers": max(1, self.config.pyannote_min_speakers),
            "max_speakers": max(1, max_spk),
        }

    def load(self) -> None:
        token = self.hf_token()
        if not token:
            raise RuntimeError(
                f"Missing HuggingFace token in env {self.config.pyannote_hf_token_env}"
            )
        from pyannote.audio import Pipeline

        logger.info(
            "Loading Pyannote model=%s device=%s kwargs=%s",
            self.config.pyannote_model,
            self.device,
            self._pipeline_kwargs(),
        )
        try:
            self._pipeline = Pipeline.from_pretrained(
                self.config.pyannote_model,
                token=token,
            )
        except Exception as exc:
            err = str(exc).lower()
            if "gated" in err or "403" in err or "authorized list" in err:
                raise RuntimeError(
                    "Pyannote 模型未授权（403）：Token 已有，但当前 HF 账号尚未同意模型许可。\n"
                    "请用【创建 Token 的同一账号】登录并打开：\n"
                    "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                    "在页面点击 Agree / Accept user conditions（有时需先验证邮箱）。\n"
                    "同意后再运行 python scripts/run_meeting.py"
                ) from exc
            raise
        if self.device.startswith("cuda"):
            import torch

            self._pipeline.to(torch.device(self.device))
        self._ready = True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def bind_session(self, ctx: SpeakerSessionContext) -> None:
        with self._lock:
            if self._bound is not None and self._bound.session_id != ctx.session_id:
                logger.warning(
                    "Pyannote rebind %s -> %s",
                    self._bound.session_id,
                    ctx.session_id,
                )
            self._bound = ctx
        logger.info("Pyannote bound to session %s", ctx.session_id)

    def unbind_session(self, ctx: SpeakerSessionContext) -> None:
        with self._lock:
            if self._bound is ctx:
                self._bound = None
        ctx.ring.clear()
        ctx.pyannote_segments.clear()
        logger.info("Pyannote unbound from session %s", ctx.session_id)

    def append_pcm(self, ctx: SpeakerSessionContext, pcm_bytes: bytes) -> None:
        ctx.ring.append(pcm_bytes)

    def _run_loop(self) -> None:
        step = self.config.pyannote_step_sec
        min_window = self.config.pyannote_window_sec
        context = max(min_window, self.config.pyannote_context_sec)
        while not self._stop.is_set():
            time.sleep(step)
            if self._pipeline is None:
                continue
            with self._lock:
                bound = self._bound
            if bound is None:
                continue
            ring = bound.ring
            if ring.duration_ms < int(min_window * 500):
                continue
            try:
                duration_ms = ring.duration_ms
                use_sec = min(context, duration_ms / 1000.0)
                use_sec = max(use_sec, min_window)
                audio, start_ms = ring.snapshot_context(use_sec)
                if audio.size < self.config.sample_rate:
                    continue
                import torch

                waveform = torch.from_numpy(audio).unsqueeze(0)
                sample = {
                    "waveform": waveform,
                    "sample_rate": self.config.sample_rate,
                }
                output = self._pipeline(sample, **self._pipeline_kwargs())
                new_segs = _parse_pyannote_output(output, start_ms)
                window_end_ms = start_ms + int(use_sec * 1000)
                from voicetotext.asr.speaker_timeline import merge_diarization_windows

                bound.pyannote_segments = merge_diarization_windows(
                    bound.pyannote_segments,
                    new_segs,
                    start_ms,
                    window_end_ms,
                )
                labels = {s.speaker_label for s in bound.pyannote_segments}
                logger.info(
                    "Pyannote [%s] context [%.1fs,%d,%d] new=%d total=%d speakers=%d",
                    bound.session_id,
                    use_sec,
                    start_ms,
                    window_end_ms,
                    len(new_segs),
                    len(bound.pyannote_segments),
                    len(labels),
                )
            except Exception as exc:
                logger.warning("Pyannote window failed: %s", exc)
