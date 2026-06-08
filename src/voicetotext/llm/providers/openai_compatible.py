"""OpenAI-compatible summary provider."""

from __future__ import annotations

from voicetotext.config import AppConfig
from voicetotext.llm.schemas import SummaryRequest, SummaryResponse
from voicetotext.llm.summary_service import SummaryService


class OpenAICompatibleSummaryProvider:
    def __init__(self, config: AppConfig) -> None:
        self._service = SummaryService(config)

    def summarize(self, request: SummaryRequest) -> SummaryResponse:
        return self._service.summarize(request)
