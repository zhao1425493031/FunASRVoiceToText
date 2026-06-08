"""LLM JSON parse tests."""

from __future__ import annotations

from tests.conftest_llm import sample_summary_json
from voicetotext.llm.parse import parse_meeting_summary


def test_parse_valid_json() -> None:
    summary = parse_meeting_summary(sample_summary_json())
    assert summary.title == "案件進捗確認ミーティング"
    assert summary.overview
    assert summary.topics
    assert summary.action_items[0]["owner"] == "SPEAKER_01"
    assert summary.markdown


def test_parse_json_in_code_fence() -> None:
    wrapped = f"```json\n{sample_summary_json()}\n```"
    summary = parse_meeting_summary(wrapped)
    assert summary.title == "案件進捗確認ミーティング"


def test_parse_missing_markdown_builds_fallback() -> None:
    raw = (
        '{"title":"T","overview":"O","topics":[],"decisions":[],"action_items":[],'
        '"open_questions":[]}'
    )
    summary = parse_meeting_summary(raw)
    assert summary.markdown.startswith("# T")
