"""Factory for summary providers."""

from __future__ import annotations

from voicetotext.config import AppConfig
from voicetotext.llm.providers.azure import AzureOpenAISummaryProvider
from voicetotext.llm.providers.openai_compatible import OpenAICompatibleSummaryProvider
from voicetotext.llm.summary import StubSummaryProvider, SummaryProvider


def create_summary_provider(config: AppConfig) -> SummaryProvider:
    if not config.llm_enabled or config.llm_provider == "stub":
        return StubSummaryProvider()
    if config.llm_provider == "azure":
        return AzureOpenAISummaryProvider(config)
    return OpenAICompatibleSummaryProvider(config)
