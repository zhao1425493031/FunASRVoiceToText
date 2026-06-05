"""SpeakerTimelineMerger unit tests."""

from __future__ import annotations

from voicetotext.asr.speaker_timeline import (
    DiarizationSegment,
    SpeakerTimelineMerger,
    merge_diarization_windows,
    parse_runtime_timestamps,
)


def test_parse_runtime_timestamps() -> None:
    msg = {"stamp_sents": [{"start": 100, "end": 500}]}
    assert parse_runtime_timestamps(msg) == (100, 500)


def test_assign_speaker_by_overlap() -> None:
    merger = SpeakerTimelineMerger(max_speakers=8)
    merger.update_segments(
        [
            DiarizationSegment(0, 2000, "A"),
            DiarizationSegment(2000, 5000, "B"),
        ]
    )
    spk, changed = merger.assign_speaker(500, 1500)
    assert spk == 0
    assert changed is False
    spk2, changed2 = merger.assign_speaker(2500, 4000)
    assert spk2 == 1
    assert changed2 is True


def test_single_speaker_mode() -> None:
    merger = SpeakerTimelineMerger(max_speakers=8)
    merger.update_segments(
        [DiarizationSegment(0, 5000, "A")]
    )
    spk, _ = merger.assign_speaker(100, 900, single_speaker_mode=True)
    assert spk == 0


def test_no_segments_returns_last() -> None:
    merger = SpeakerTimelineMerger(max_speakers=8)
    merger._last_speaker_id = 2
    spk, changed = merger.assign_speaker(0, 1000)
    assert spk == 2
    assert changed is False


def test_merge_diarization_windows_replaces_overlap() -> None:
    existing = [
        DiarizationSegment(0, 5000, "A"),
        DiarizationSegment(5000, 10000, "B"),
    ]
    incoming = [DiarizationSegment(8000, 12000, "C")]
    merged = merge_diarization_windows(existing, incoming, 8000, 12000)
    labels = [(s.start_ms, s.speaker_label) for s in merged]
    assert (0, "A") in labels
    assert (8000, "C") in labels
    assert not any(s.start_ms == 5000 and s.speaker_label == "B" for s in merged)


def test_max_speakers_cap() -> None:
    merger = SpeakerTimelineMerger(max_speakers=2)
    merger.update_segments(
        [
            DiarizationSegment(0, 1000, "A"),
            DiarizationSegment(1000, 2000, "B"),
            DiarizationSegment(2000, 3000, "C"),
        ]
    )
    merger.assign_speaker(0, 500)
    merger.assign_speaker(1000, 1500)
    spk_c, _ = merger.assign_speaker(2500, 2900)
    assert spk_c in (0, 1)
