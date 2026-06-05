"""Deprecated v1 cam++ module — v2 uses speaker_timeline + pyannote_worker."""

from voicetotext.asr.speaker_timeline import (  # noqa: F401
    DiarizationSegment,
    SpeakerTimelineMerger,
    parse_runtime_timestamps,
)

__all__ = [
    "DiarizationSegment",
    "SpeakerTimelineMerger",
    "parse_runtime_timestamps",
]
