"""MeetingTranscriptBuffer unit tests."""

from __future__ import annotations

from voicetotext.llm.meeting_transcript_buffer import MeetingTranscriptBuffer, map_speaker_id


def test_buffer_accumulates_final_segments() -> None:
    buf = MeetingTranscriptBuffer("sess-1", "zh")
    buf.append_final({"type": "partial", "text": "ignored"})
    buf.append_final({"type": "final", "text": "  你好  ", "seg_id": "s1", "t_start_ms": 0, "t_end_ms": 1000})
    buf.append_final({"type": "final", "text": "世界", "seg_id": "s2", "speaker_id": 1, "t_start_ms": 1000, "t_end_ms": 2000})
    assert len(buf.segments) == 2
    assert buf.segments[0]["text"] == "你好"
    assert buf.segments[1]["speaker_id"] == "SPEAKER_01"


def test_buffer_filters_empty_text_and_duplicate_seg_id() -> None:
    buf = MeetingTranscriptBuffer("sess-2", "ja")
    buf.append_final({"type": "final", "text": "", "seg_id": "x1"})
    buf.append_final({"type": "final", "text": "A", "seg_id": "x1"})
    buf.append_final({"type": "final", "text": "B", "seg_id": "x1"})
    assert len(buf.segments) == 1
    assert buf.segments[0]["text"] == "A"


def test_single_speaker_mode_maps_to_speaker_00() -> None:
    buf = MeetingTranscriptBuffer("sess-3", "zh", single_speaker_mode=True)
    buf.append_final({"type": "final", "text": "测试", "speaker_id": 3, "seg_id": "a"})
    assert buf.segments[0]["speaker_id"] == "SPEAKER_00"
    assert map_speaker_id(5, single_mode=True) == "SPEAKER_00"


def test_duration_ms_from_segments() -> None:
    buf = MeetingTranscriptBuffer("sess-4", "zh")
    assert buf.duration_ms() == 0
    buf.append_final({"type": "final", "text": "x", "t_start_ms": 100, "t_end_ms": 5000, "seg_id": "1"})
    assert buf.duration_ms() == 5000
