"""Orchestrate meeting summary generation (single + map-reduce)."""

from __future__ import annotations

import json
import time
from typing import Any

from voicetotext.config import AppConfig
from voicetotext.llm.client import ChatMessage, OpenAIChatClient
from voicetotext.llm.parse import parse_meeting_summary
from voicetotext.llm.prompts import (
    REPAIR_SYSTEM_PROMPT,
    get_reduce_user_prompt,
    get_system_prompt,
    get_user_prompt,
)
from voicetotext.llm.schemas import MeetingSummary, SummaryRequest, SummaryResponse
from voicetotext.llm.transcript_format import chunk_segments, format_transcript
from voicetotext.logging_setup import get_logger

logger = get_logger(__name__)


def _segments_to_dicts(request: SummaryRequest) -> list[dict[str, Any]]:
    return [
        {
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "speaker_id": s.speaker_id,
            "text": s.text,
        }
        for s in request.segments
    ]


def _build_client(config: AppConfig) -> OpenAIChatClient:
    assert config.llm_api_url and config.llm_api_key and config.llm_model
    return OpenAIChatClient(
        api_url=config.llm_api_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        timeout_sec=config.llm_timeout_sec,
        retry_max=config.llm_retry_max,
        retry_backoff_sec=config.llm_retry_backoff_sec,
        deployment=config.llm_deployment,
        api_version=config.llm_api_version,
        provider=config.llm_provider,
    )


class SummaryService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = _build_client(config)

    def _call_llm(
        self,
        *,
        language: str,
        system: str,
        user: str,
        mode: str = "full",
    ) -> tuple[str, int]:
        if mode != "full":
            system = get_system_prompt(language, mode=mode)
        result = self._client.complete(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ]
        )
        return result.content, result.latency_ms

    def _parse_with_repair(self, text: str, *, language: str) -> MeetingSummary:
        try:
            return parse_meeting_summary(text)
        except (json.JSONDecodeError, ValueError) as first_exc:
            logger.warning("LLM JSON parse failed, attempting repair: %s", first_exc)
            repaired, _ = self._call_llm(
                language=language,
                system=REPAIR_SYSTEM_PROMPT,
                user=text,
                mode="full",
            )
            return parse_meeting_summary(repaired)

    def _summarize_text(self, *, language: str, transcript: str, mode: str) -> MeetingSummary:
        system = get_system_prompt(language, mode=mode)
        user = get_user_prompt(language, transcript) if mode == "full" else transcript
        content, _ = self._call_llm(language=language, system=system, user=user, mode=mode)
        return self._parse_with_repair(content, language=language)

    def summarize(self, request: SummaryRequest) -> SummaryResponse:
        start = time.perf_counter()
        segments = _segments_to_dicts(request)
        transcript = format_transcript(segments)
        if not transcript.strip():
            return SummaryResponse(
                job_id=request.job_id,
                summary=None,
                status="error",
                error="empty transcript",
            )

        cfg = self._config
        chunks = 1
        total_latency = 0

        if len(transcript) <= cfg.llm_max_input_chars:
            content, latency = self._call_llm(
                language=request.language,
                system=get_system_prompt(request.language),
                user=get_user_prompt(request.language, transcript),
            )
            total_latency += latency
            summary = self._parse_with_repair(content, language=request.language)
        else:
            seg_chunks = chunk_segments(segments, max_chars=cfg.llm_chunk_chars)
            chunks = len(seg_chunks)
            partials: list[str] = []
            for i, seg_chunk in enumerate(seg_chunks, 1):
                chunk_text = format_transcript(seg_chunk)
                content, latency = self._call_llm(
                    language=request.language,
                    system=get_system_prompt(request.language, mode="map"),
                    user=get_user_prompt(request.language, chunk_text),
                    mode="map",
                )
                total_latency += latency
                partials.append(f"--- Part {i} ---\n{content}")

            reduce_user = get_reduce_user_prompt("\n\n".join(partials))
            content, latency = self._call_llm(
                language=request.language,
                system=get_system_prompt(request.language, mode="reduce"),
                user=reduce_user,
                mode="reduce",
            )
            total_latency += latency
            summary = self._parse_with_repair(content, language=request.language)

        elapsed = int((time.perf_counter() - start) * 1000)
        meta = {
            "provider": cfg.llm_provider,
            "model": cfg.llm_model,
            "latency_ms": elapsed,
            "llm_latency_ms": total_latency,
            "chunks": chunks,
            "input_chars": len(transcript),
        }
        return SummaryResponse(
            job_id=request.job_id,
            summary=summary,
            status="ok",
            meta=meta,
        )
