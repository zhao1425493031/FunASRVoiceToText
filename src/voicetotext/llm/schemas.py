"""LLM summary request/response schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SummarySegment:
    start_ms: int
    end_ms: int
    speaker_id: str
    text: str


@dataclass
class SummaryRequest:
    job_id: str
    language: str
    segments: list[SummarySegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MeetingSummary:
    title: str
    overview: str
    topics: list[dict[str, str]]
    decisions: list[str]
    action_items: list[dict[str, str]]
    open_questions: list[str]
    markdown: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryResponse:
    job_id: str
    summary: MeetingSummary | None
    status: str
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "status": self.status,
            "error": self.error,
            "meta": self.meta,
        }
        if self.summary is not None:
            data["summary"] = self.summary.to_dict()
        else:
            data["summary"] = None
        return data


@dataclass
class MeetingTranscript:
    session_id: str
    language: str
    duration_ms: int
    segments: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "segments": self.segments,
            "meta": self.meta,
            "summary": self.summary,
        }
