"""Summary provider protocol and stub (LLM not implemented in this release)."""

from __future__ import annotations

from typing import Protocol

from voicetotext.llm.schemas import SummaryRequest, SummaryResponse


class SummaryNotImplementedError(RuntimeError):
    pass


class SummaryProvider(Protocol):
    def summarize(self, request: SummaryRequest) -> SummaryResponse: ...


class StubSummaryProvider:
    """Placeholder; raises until LLM backend is wired."""

    def summarize(self, request: SummaryRequest) -> SummaryResponse:
        raise SummaryNotImplementedError("LLM summary is not implemented yet")
