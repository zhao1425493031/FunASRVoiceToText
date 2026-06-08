"""Transcript formatting tests."""

from __future__ import annotations

from voicetotext.llm.transcript_format import (
    chunk_segments,
    format_segment_line,
    format_timestamp,
    format_transcript,
)


def test_format_timestamp() -> None:
    assert format_timestamp(65000) == "00:01:05"


def test_format_segment_line_skips_empty() -> None:
    assert format_segment_line({"start_ms": 0, "speaker_id": "SPEAKER_00", "text": "  "}) == ""


def test_format_transcript() -> None:
    segments = [
        {"start_ms": 1000, "end_ms": 2000, "speaker_id": "SPEAKER_00", "text": "hello"},
        {"start_ms": 2000, "end_ms": 3000, "speaker_id": "SPEAKER_01", "text": ""},
        {"start_ms": 3000, "end_ms": 4000, "speaker_id": "SPEAKER_01", "text": "world"},
    ]
    text = format_transcript(segments)
    assert "[00:00:01] SPEAKER_00: hello" in text
    assert "world" in text
    assert text.count("\n") == 1


def test_chunk_segments_respects_boundaries() -> None:
    segments = [
        {"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "speaker_id": "S", "text": f"line{i}"}
        for i in range(5)
    ]
    chunks = chunk_segments(segments, max_chars=30)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk
