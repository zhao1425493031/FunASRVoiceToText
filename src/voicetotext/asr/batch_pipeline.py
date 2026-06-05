"""Offline batch orchestration: preprocess → ASR + diarization → align → export."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from voicetotext.asr.audio_preprocess import load_audio_file
from voicetotext.asr.batch_align import AsrSegment, align_batch_segments
from voicetotext.asr.funasr_vad import FunASRVAD
from voicetotext.asr.pyannote_offline import PyannoteOfflineDiarizer
from voicetotext.asr.sensevoice_funasr_engine import SenseVoiceFunASREngine
from voicetotext.config import AppConfig
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class BatchTranscript:
    job_id: str
    language: str
    duration_ms: int
    segments: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    summary: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "segments": self.segments,
            "meta": self.meta,
            "summary": self.summary,
        }


def transcript_to_plain(data: dict[str, Any] | BatchTranscript) -> str:
    """Human-readable lines for copy/paste."""
    if isinstance(data, BatchTranscript):
        segments = data.segments
    else:
        segments = data.get("segments") or []
    lines: list[str] = []
    for seg in segments:
        ms = int(seg.get("start_ms", 0))
        t = ms // 1000
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)
        ts = f"{h:02d}:{m:02d}:{s:02d}"
        lines.append(f"[{ts}] {seg.get('speaker_id', '')}: {seg.get('text', '')}")
    return "\n".join(lines)


def _ms_to_srt(ms: int) -> str:
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _split_long_vad(
    segments: list[tuple[int, int]],
    max_ms: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in segments:
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + max_ms)
            out.append((cursor, chunk_end))
            cursor = chunk_end
    return out or [(0, max_ms)]


def _transcribe_vad_segments(
    audio: np.ndarray,
    vad_segments: list[tuple[int, int]],
    asr: SenseVoiceFunASREngine,
    sample_rate: int,
    language: str,
) -> list[AsrSegment]:
    results: list[AsrSegment] = []
    cache: dict = {}
    for start_ms, end_ms in vad_segments:
        s0 = int(start_ms * sample_rate / 1000)
        s1 = int(end_ms * sample_rate / 1000)
        chunk = audio[s0:s1]
        if chunk.size == 0:
            continue
        text = asr.transcribe(chunk, cache, is_final=True, language=language)
        if text.strip():
            results.append(AsrSegment(start_ms=start_ms, end_ms=end_ms, text=text))
    return results


class BatchPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._asr = SenseVoiceFunASREngine(config)
        self._vad = FunASRVAD(config)
        self._diarizer = PyannoteOfflineDiarizer(config)
        self._loaded = False

    def load(self) -> None:
        self._asr.load()
        self._vad.load()
        self._diarizer.load()
        self._loaded = True

    def is_ready(self) -> bool:
        import os

        token_ok = bool(os.environ.get(self.config.pyannote_hf_token_env, "").strip())
        return self._loaded and self._asr.is_loaded and token_ok and self._diarizer.is_ready

    def readiness_detail(self) -> dict[str, Any]:
        import os

        return {
            "asr_loaded": self._asr.is_loaded,
            "vad_loaded": self._vad.is_loaded,
            "pyannote_ready": self._diarizer.is_ready,
            "hf_token_set": bool(
                os.environ.get(self.config.pyannote_hf_token_env, "").strip()
            ),
            "asr_model": self.config.asr_model,
        }

    def process_file(
        self,
        input_path: Path,
        output_dir: Path | None = None,
        *,
        job_id: str | None = None,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> BatchTranscript:
        if not self._loaded:
            self.load()

        def report(stage: str, pct: int) -> None:
            if on_progress is not None:
                on_progress(stage, pct)

        job_id = job_id or str(uuid.uuid4())
        report("preprocess", 10)
        audio, duration_ms = load_audio_file(input_path, self.config)
        sr = self.config.sample_rate
        report("preprocess", 20)

        def run_asr() -> list[AsrSegment]:
            report("asr", 30)
            vad_raw = self._vad.segment_utterances(audio)
            vad_segments = _split_long_vad(
                vad_raw or [(0, duration_ms)],
                self.config.batch_max_segment_ms,
            )
            segs = _transcribe_vad_segments(
                audio,
                vad_segments,
                self._asr,
                sr,
                self.config.language,
            )
            report("asr", 55)
            return segs

        def run_diar() -> list:
            report("diarization", 35)
            segs = self._diarizer.diarize(audio, sr)
            report("diarization", 55)
            return segs

        with ThreadPoolExecutor(max_workers=2) as pool:
            asr_future = pool.submit(run_asr)
            diar_future = pool.submit(run_diar)
            asr_segments = asr_future.result()
            diar_segments = diar_future.result()

        report("align", 70)
        aligned = align_batch_segments(asr_segments, diar_segments, self.config)
        report("align", 85)
        segment_dicts = [
            {
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "speaker_id": s.speaker_id,
                "text": s.text,
            }
            for s in aligned
        ]

        transcript = BatchTranscript(
            job_id=job_id,
            language=self.config.language,
            duration_ms=duration_ms,
            segments=segment_dicts,
            meta={
                "asr_model": self.config.asr_model,
                "diarization": "community-1",
                "source_file": input_path.name,
            },
            summary=None,
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            report("export", 92)
            self.export(transcript, output_dir)

        report("done", 100)
        return transcript

    def export(self, transcript: BatchTranscript, output_dir: Path) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        base = output_dir / transcript.job_id
        formats = {f.lower() for f in self.config.output_formats}

        if "json" in formats:
            p = base.with_suffix(".json")
            p.write_text(
                json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths["json"] = p

        if "md" in formats:
            p = base.with_suffix(".md")
            lines = [f"# Transcript {transcript.job_id}", ""]
            for seg in transcript.segments:
                t = seg["start_ms"] // 1000
                m, s = divmod(t, 60)
                h, m = divmod(m, 60)
                ts = f"{h:02d}:{m:02d}:{s:02d}"
                lines.append(
                    f"[{ts}] {seg['speaker_id']}: {seg['text']}"
                )
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            paths["md"] = p

        if "srt" in formats:
            p = base.with_suffix(".srt")
            blocks: list[str] = []
            for i, seg in enumerate(transcript.segments, 1):
                blocks.append(str(i))
                blocks.append(
                    f"{_ms_to_srt(seg['start_ms'])} --> {_ms_to_srt(seg['end_ms'])}"
                )
                blocks.append(f"{seg['speaker_id']}: {seg['text']}")
                blocks.append("")
            p.write_text("\n".join(blocks), encoding="utf-8")
            paths["srt"] = p

        return paths
