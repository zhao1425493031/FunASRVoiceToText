"""Shared pytest fixtures for meeting LLM tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voicetotext.config import load_config
from voicetotext.llm.schemas import MeetingSummary, SummaryResponse

ROOT = Path(__file__).resolve().parents[1]
LLM_TEST_CONFIG = ROOT / "tests" / "fixtures" / "config_meeting_llm_test.yaml"


def mock_summary_dict() -> dict[str, Any]:
    return {
        "title": "测试会议",
        "overview": "会议概要",
        "topics": [
            {
                "subject": "主题A",
                "discussion": "讨论了方案",
                "conclusion": "达成一致",
            }
        ],
        "decisions": ["采用方案A"],
        "action_items": [{"owner": "张三", "task": "整理文档", "due": "周五"}],
        "open_questions": ["预算是否充足"],
        "markdown": "## 概要\n会议概要\n\n## 待办\n- 张三: 整理文档",
    }


def mock_summary_response(job_id: str = "test-session") -> SummaryResponse:
    data = mock_summary_dict()
    return SummaryResponse(
        job_id=job_id,
        summary=MeetingSummary(**data),
        status="ok",
        meta={"provider": "openai_compatible", "model": "qwen-plus-test"},
    )


def mock_llm_http_body() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(mock_summary_dict(), ensure_ascii=False),
                }
            }
        ]
    }


@pytest.fixture
def llm_test_config():
    return load_config(LLM_TEST_CONFIG)
