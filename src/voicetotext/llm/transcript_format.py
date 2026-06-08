"""Format transcript segments for LLM input."""

from __future__ import annotations

from typing import Any


def format_timestamp(ms: int) -> str:
    t = max(0, int(ms)) // 1000
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_segment_line(seg: dict[str, Any]) -> str:
    text = str(seg.get("text", "")).strip()
    if not text:
        return ""
    ts = format_timestamp(int(seg.get("start_ms", 0)))
    speaker = str(seg.get("speaker_id", ""))
    return f"[{ts}] {speaker}: {text}"


def format_transcript(segments: list[dict[str, Any]]) -> str:
    lines = [line for seg in segments if (line := format_segment_line(seg))]
    return "\n".join(lines)


def chunk_segments(
    segments: list[dict[str, Any]],
    *,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    if max_chars <= 0:
        return [segments]

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_len = 0

    for seg in segments:
        line = format_segment_line(seg)
        if not line:
            continue
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append(current)
            current = [seg]
            current_len = line_len
        else:
            current.append(seg)
            current_len += line_len

    if current:
        chunks.append(current)
    return chunks if chunks else [segments]
