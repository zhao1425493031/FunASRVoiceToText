"""Meeting transcript export to json and summary.md."""

from __future__ import annotations

import json

from voicetotext.llm.meeting_export import render_summary_md, write_meeting_transcript
from voicetotext.llm.schemas import MeetingTranscript

from tests.conftest import mock_summary_dict


def test_write_meeting_transcript_creates_json_and_md(tmp_path) -> None:
    summary = mock_summary_dict()
    transcript = MeetingTranscript(
        session_id="export-test-1",
        language="zh",
        duration_ms=5000,
        segments=[
            {
                "start_ms": 0,
                "end_ms": 1000,
                "speaker_id": "SPEAKER_00",
                "text": "你好",
            }
        ],
        meta={"service": "meeting"},
        summary=summary,
    )
    paths = write_meeting_transcript(tmp_path, transcript, write_summary_md=True)
    assert paths["json"].is_file()
    assert paths["summary_md"].is_file()

    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["session_id"] == "export-test-1"
    assert data["summary"]["title"] == summary["title"]
    md = paths["summary_md"].read_text(encoding="utf-8")
    assert "## 概要" in md or summary["markdown"] in md


def test_render_summary_md_fallback_without_markdown() -> None:
    md = render_summary_md({"title": "T", "overview": "O", "topics": [], "action_items": []})
    assert "# T" in md
    assert "O" in md
