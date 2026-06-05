"""Speaker timeline tests (v2 replaces cam++ assigner)."""

from __future__ import annotations

from voicetotext.asr.speaker_timeline import (
    DiarizationSegment,
    SpeakerTimelineMerger,
    parse_runtime_timestamps,
)


def test_parse_timestamps_from_runtime_msg() -> None:
    msg = {"stamp_sents": [{"start": 10, "end": 20}]}
    assert parse_runtime_timestamps(msg) == (10, 20)


def test_merger_two_speakers_alternating() -> None:
    merger = SpeakerTimelineMerger(max_speakers=8)
    merger.update_segments(
        [
            DiarizationSegment(0, 3000, "S0"),
            DiarizationSegment(3000, 6000, "S1"),
        ]
    )
    a, _ = merger.assign_speaker(100, 2000)
    b, changed = merger.assign_speaker(3500, 5500)
    assert a == 0
    assert b == 1
    assert changed is True
