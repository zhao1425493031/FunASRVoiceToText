"""Per-WebSocket speaker diarization state (isolated from global engine)."""

from __future__ import annotations

from dataclasses import dataclass, field

from voicetotext.asr.pyannote_worker import AudioRingBuffer
from voicetotext.asr.speaker_timeline import DiarizationSegment, SpeakerTimelineMerger
from voicetotext.asr.utterance_speaker import UtteranceSpeakerRegistry
from voicetotext.config import AppConfig


@dataclass
class SpeakerSessionContext:
    """Isolated merger + embedding registry + Pyannote ring for one WS meeting."""

    session_id: str
    merger: SpeakerTimelineMerger
    registry: UtteranceSpeakerRegistry
    ring: AudioRingBuffer
    pyannote_segments: list[DiarizationSegment] = field(default_factory=list)

    @classmethod
    def create(cls, config: AppConfig, session_id: str) -> SpeakerSessionContext:
        ring_sec = max(
            120,
            int(config.pyannote_context_sec) + int(config.pyannote_window_sec),
        )
        return cls(
            session_id=session_id,
            merger=SpeakerTimelineMerger(max_speakers=config.meeting_max_speakers),
            registry=UtteranceSpeakerRegistry(
                max_speakers=config.meeting_max_speakers,
                threshold=config.meeting_spk_embedding_threshold,
            ),
            ring=AudioRingBuffer(config.sample_rate, max_seconds=ring_sec),
        )

    def reset(self) -> None:
        self.merger.clear()
        self.registry.reset()
        self.ring.clear()
        self.pyannote_segments.clear()

    def pyannote_label_count(self) -> int:
        return len({s.speaker_label for s in self.pyannote_segments})
