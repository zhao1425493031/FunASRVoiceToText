"""LLM summary request/response schemas (reserved for future implementation)."""

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
class SummaryResponse:
    job_id: str
    summary: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
