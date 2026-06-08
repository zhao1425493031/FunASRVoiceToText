"""Offline batch orchestration: preprocess → diarization → ASR per window → export."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from voicetotext.asr.audio_preprocess import load_audio_file
from voicetotext.asr.batch_align import build_aligned_from_diar
from voicetotext.asr.batch_segment_polish import polish_aligned_segments
from voicetotext.asr.batch_speaker_refine import refine_aligned_segments
from voicetotext.asr.batch_diar_segments import (
    DiarizationEmptyError,
    normalize_diar_segments,
)
from voicetotext.asr.batch_transcribe import transcribe_diar_windows
from voicetotext.asr.pyannote_offline import PyannoteOfflineDiarizer
from voicetotext.asr.sensevoice_funasr_engine import SenseVoiceFunASREngine
from voicetotext.config import AppConfig
from voicetotext.llm.factory import create_summary_provider
from voicetotext.llm.schemas import SummaryRequest, SummarySegment
from voicetotext.llm.summary import SummaryNotImplementedError, SummaryProvider
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class BatchTranscript:
    job_id: str
    language: str
    duration_ms: int
    segments: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] | None = None

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


def _build_summary_request(
    job_id: str,
    language: str,
    segments: list[dict[str, Any]],
) -> SummaryRequest:
    return SummaryRequest(
        job_id=job_id,
        language=language,
        segments=[
            SummarySegment(
                start_ms=int(s["start_ms"]),
                end_ms=int(s["end_ms"]),
                speaker_id=str(s["speaker_id"]),
                text=str(s["text"]),
            )
            for s in segments
        ],
    )


def _render_summary_md(summary: dict[str, Any]) -> str:
    markdown = str(summary.get("markdown", "")).strip()
    if markdown:
        return markdown + "\n"
    title = str(summary.get("title", "会议纪要"))
    lines = [f"# {title}", ""]
    overview = str(summary.get("overview", "")).strip()
    if overview:
        lines.extend(["## 概要", overview, ""])
    for topic in summary.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        lines.append(
            f"- **{topic.get('subject', '')}**: "
            f"{topic.get('discussion', '')} ({topic.get('conclusion', '')})"
        )
    action_items = summary.get("action_items") or []
    if action_items:
        lines.extend(["", "## 待办"])
        for item in action_items:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- [{item.get('owner', '待定')}] "
                f"{item.get('task', '')} (期限: {item.get('due', '待定')})"
            )
    return "\n".join(lines).strip() + "\n"


class BatchPipeline:
    def __init__(self, config: AppConfig, *, skip_summary: bool = False) -> None:
        self.config = config
        self._skip_summary = skip_summary
        self._summary_provider: SummaryProvider = create_summary_provider(config)
        self._asr = SenseVoiceFunASREngine(config)
        self._diarizer = PyannoteOfflineDiarizer(config)
        self._loaded = False

    def load(self) -> None:
        self._asr.load()
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
            "pyannote_ready": self._diarizer.is_ready,
            "hf_token_set": bool(
                os.environ.get(self.config.pyannote_hf_token_env, "").strip()
            ),
            "asr_model": self.config.asr_model,
            "pipeline": "diar_first",
            "llm": {
                "enabled": self.config.llm_enabled,
                "ready": self.config.llm_ready,
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
            },
        }

    def _apply_summary(self, transcript: BatchTranscript, report: Callable[[str, int], None]) -> None:
        if not self.config.llm_enabled or self._skip_summary:
            return

        report("summary", 95)
        llm_meta: dict[str, Any] = {
            "enabled": True,
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
        }
        try:
            req = _build_summary_request(
                transcript.job_id,
                transcript.language,
                transcript.segments,
            )
            resp = self._summary_provider.summarize(req)
            if resp.status == "ok" and resp.summary is not None:
                transcript.summary = resp.summary.to_dict()
                llm_meta.update({"status": "ok", **resp.meta})
            else:
                transcript.summary = None
                llm_meta.update(
                    {
                        "status": "error",
                        "error": resp.error or "summary generation failed",
                    }
                )
        except SummaryNotImplementedError:
            transcript.summary = None
            llm_meta = {"enabled": False, "status": "disabled"}
        except Exception as exc:
            logger.exception("LLM summary failed for job %s", transcript.job_id)
            if self.config.llm_on_failure == "fail":
                raise
            transcript.summary = None
            llm_meta.update({"status": "error", "error": str(exc)})
        transcript.meta["llm"] = llm_meta

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

        report("diarization", 30)
        raw_diar = self._diarizer.diarize(audio, sr)
        if not raw_diar:
            logger.error("Pyannote returned no diarization segments for %s", input_path.name)
            raise DiarizationEmptyError(
                f"No speaker diarization segments for {input_path.name}. "
                "Check HF token and audio content."
            )
        report("diarization", 45)

        diar_windows = normalize_diar_segments(raw_diar, self.config)
        if not diar_windows:
            raise DiarizationEmptyError(
                f"Diarization normalization produced no segments for {input_path.name}"
            )
        logger.info(
            "Diar windows: raw=%d normalized=%d",
            len(raw_diar),
            len(diar_windows),
        )
        report("diarization", 55)

        report("asr", 60)
        transcribed = transcribe_diar_windows(
            audio,
            diar_windows,
            self._asr,
            sr,
            self.config.language,
            self.config.batch_asr_parallel_workers,
        )
        report("asr", 80)

        report("align", 85)
        aligned = polish_aligned_segments(
            refine_aligned_segments(build_aligned_from_diar(transcribed))
        )
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
                "pipeline": "diar_first",
                "source_file": input_path.name,
                "diar_windows": len(diar_windows),
                "speaker_refined": True,
                "segment_polished": True,
                "text_beautified": True,
                "stamp_sents_available": False,
            },
            summary=None,
        )

        self._apply_summary(transcript, report)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            report("export", 98)
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

        if self.config.llm_output_summary_md and transcript.summary:
            p = base.with_suffix(".summary.md")
            p.write_text(_render_summary_md(transcript.summary), encoding="utf-8")
            paths["summary_md"] = p

        return paths
