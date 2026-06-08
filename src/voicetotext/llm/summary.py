"""Summary provider protocol and stub."""

from __future__ import annotations

from typing import Protocol

from voicetotext.llm.schemas import SummaryRequest, SummaryResponse


class SummaryNotImplementedError(RuntimeError):
    pass


class SummaryProvider(Protocol):
    def summarize(self, request: SummaryRequest) -> SummaryResponse: ...


class StubSummaryProvider:
    def summarize(self, request: SummaryRequest) -> SummaryResponse:
        raise SummaryNotImplementedError("LLM summary is disabled")
