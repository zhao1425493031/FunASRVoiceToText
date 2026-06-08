"""Parse LLM JSON output into MeetingSummary."""

from __future__ import annotations

import json
import re
from typing import Any

from voicetotext.llm.schemas import MeetingSummary


def _strip_code_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _as_topic_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    topics: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        topics.append(
            {
                "subject": str(item.get("subject", "")).strip(),
                "discussion": str(item.get("discussion", "")).strip(),
                "conclusion": str(item.get("conclusion", "")).strip(),
            }
        )
    return topics


def _as_action_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "owner": str(item.get("owner", "待定")).strip() or "待定",
                "task": str(item.get("task", "")).strip(),
                "due": str(item.get("due", "待定")).strip() or "待定",
            }
        )
    return items


def parse_meeting_summary(text: str) -> MeetingSummary:
    raw = _strip_code_fence(text)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM summary must be a JSON object")

    title = str(data.get("title", "会议纪要")).strip() or "会议纪要"
    overview = str(data.get("overview", "")).strip()
    topics = _as_topic_list(data.get("topics"))
    decisions = _as_str_list(data.get("decisions"))
    action_items = _as_action_items(data.get("action_items"))
    open_questions = _as_str_list(data.get("open_questions"))
    markdown = str(data.get("markdown", "")).strip()

    if not markdown:
        lines = [f"# {title}", "", "## 概要", overview]
        if topics:
            lines.extend(["", "## 议题"])
            for t in topics:
                lines.append(f"- **{t['subject']}**: {t['discussion']} ({t['conclusion']})")
        if decisions:
            lines.extend(["", "## 决议"])
            lines.extend(f"- {d}" for d in decisions)
        if action_items:
            lines.extend(["", "## 待办"])
            for a in action_items:
                lines.append(f"- [{a['owner']}] {a['task']} (期限: {a['due']})")
        if open_questions:
            lines.extend(["", "## 未决问题"])
            lines.extend(f"- {q}" for q in open_questions)
        markdown = "\n".join(lines)

    return MeetingSummary(
        title=title,
        overview=overview,
        topics=topics,
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        markdown=markdown,
    )
