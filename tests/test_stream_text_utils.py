"""Tests for stream session text helpers."""

from voicetotext.stream_session import StreamSession, is_meaningful_text


def test_merge_segment_extends() -> None:
    assert StreamSession._merge_segment("こんにちは", "こんにちは世界") == "こんにちは世界"
    assert StreamSession._merge_segment("こんにちは", "すごい") == "こんにちはすごい"
