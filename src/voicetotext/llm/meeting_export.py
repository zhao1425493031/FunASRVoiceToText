"""Export meeting transcript and summary to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voicetotext.llm.schemas import MeetingTranscript


def render_summary_md(summary: dict[str, Any]) -> str:
    markdown = str(summary.get("markdown", "")).strip()
    if markdown:
        return markdown + "\n"
    title = str(summary.get("title", "会议纪要"))
    lines = [f"# {title}", ""]
    overview = str(summary.get("overview", "")).strip()
    if overview:
        lines.extend(["## 概要", overview, ""])
    for topic in summary.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        lines.append(
            f"- **{topic.get('subject', '')}**: "
            f"{topic.get('discussion', '')} ({topic.get('conclusion', '')})"
        )
    action_items = summary.get("action_items") or []
    if action_items:
        lines.extend(["", "## 待办"])
        for item in action_items:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- [{item.get('owner', '待定')}] "
                f"{item.get('task', '')} (期限: {item.get('due', '待定')})"
            )
    return "\n".join(lines).strip() + "\n"


def write_meeting_transcript(
    output_dir: Path,
    transcript: MeetingTranscript,
    *,
    write_summary_md: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / transcript.session_id
    paths: dict[str, Path] = {}

    json_path = base.with_suffix(".json")
    json_path.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["json"] = json_path

    if write_summary_md and transcript.summary:
        md_path = base.with_suffix(".summary.md")
        md_path.write_text(render_summary_md(transcript.summary), encoding="utf-8")
        paths["summary_md"] = md_path

    return paths
