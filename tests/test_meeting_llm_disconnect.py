"""Disconnect without end still triggers summary on cleanup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import load_config
from voicetotext.server.meeting_ws_protocol import MeetingWSProtocolHandler

from tests.conftest import LLM_TEST_CONFIG


@pytest.mark.asyncio
async def test_cleanup_runs_summary_when_not_ended_normally() -> None:
    cfg = load_config(LLM_TEST_CONFIG)
    ws = MagicMock()
    ws.send_text = AsyncMock()
    engine = create_asr_backend(cfg)
    handler = MeetingWSProtocolHandler(ws, engine, cfg)

    mock_session = MagicMock()
    mock_session.end = AsyncMock(return_value=True)
    mock_session.run_meeting_summary = AsyncMock(return_value=True)
    mock_session.close = AsyncMock()
    handler.asr_session = mock_session
    handler._ended_normally = False

    await handler.cleanup()
    mock_session.end.assert_awaited_once()
    mock_session.run_meeting_summary.assert_awaited_once_with(skip_summary=False)
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_skips_summary_after_normal_end() -> None:
    cfg = load_config(LLM_TEST_CONFIG)
    ws = MagicMock()
    engine = create_asr_backend(cfg)
    handler = MeetingWSProtocolHandler(ws, engine, cfg)

    mock_session = MagicMock()
    mock_session.end = AsyncMock()
    mock_session.run_meeting_summary = AsyncMock()
    mock_session.close = AsyncMock()
    handler.asr_session = mock_session
    handler._ended_normally = True

    await handler.cleanup()
    mock_session.end.assert_not_awaited()
    mock_session.run_meeting_summary.assert_not_awaited()
    mock_session.close.assert_awaited_once()
