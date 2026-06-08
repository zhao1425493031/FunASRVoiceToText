"""Shared LLM test helpers."""

from __future__ import annotations

import json


def sample_summary_json() -> str:
    return json.dumps(
        {
            "title": "案件進捗確認ミーティング",
            "overview": "案件の進捗と課題を確認した。",
            "topics": [
                {
                    "subject": "進捗",
                    "discussion": "開発は75%完了",
                    "conclusion": "順調",
                }
            ],
            "decisions": ["レビュー会議を来週火曜午後に実施"],
            "action_items": [
                {
                    "owner": "SPEAKER_01",
                    "task": "最新スケジュールと課題一覧を共有",
                    "due": "金曜夕方まで",
                }
            ],
            "open_questions": [],
            "markdown": "## 概要\n案件進捗を確認。",
        },
        ensure_ascii=False,
    )


def mock_chat_response(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"total_tokens": 100},
    }
