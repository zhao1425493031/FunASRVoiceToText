"""Prompt templates for meeting summary generation."""

from __future__ import annotations

SYSTEM_PROMPTS: dict[str, str] = {
    "ja": (
        "あなたは会議議事録作成の専門家です。"
        "与えられた発言記録のみに基づき、事実を捏造せず議事録を作成してください。"
        "確認できない情報は「未定」と記載してください。"
        "action_items の owner は SPEAKER_xx または「未定」のみ使用してください。"
        "出力は必ず有効な JSON のみとし、コードブロックや説明文は含めないでください。"
        "JSON スキーマ: "
        '{"title":"str","overview":"str","topics":[{"subject":"","discussion":"","conclusion":""}],'
        '"decisions":["str"],"action_items":[{"owner":"","task":"","due":""}],'
        '"open_questions":["str"],"markdown":"str"}'
    ),
    "zh": (
        "你是会议纪要撰写专家。"
        "仅根据提供的发言记录生成纪要，不得编造未出现的事实。"
        "无法确认的信息标注为「待定」。"
        "action_items 的 owner 只能使用 SPEAKER_xx 或「待定」。"
        "输出必须是合法 JSON，不要包含代码块或额外说明。"
        "JSON 结构: "
        '{"title":"str","overview":"str","topics":[{"subject":"","discussion":"","conclusion":""}],'
        '"decisions":["str"],"action_items":[{"owner":"","task":"","due":""}],'
        '"open_questions":["str"],"markdown":"str"}'
    ),
}

USER_TEMPLATES: dict[str, str] = {
    "ja": "以下は会議の発言記録です。議事録 JSON を作成してください。\n\n{transcript}",
    "zh": "以下是会议发言记录。请生成会议纪要 JSON。\n\n{transcript}",
}

MAP_SYSTEM_PROMPTS: dict[str, str] = {
    "ja": (
        "あなたは会議議事録の部分要約担当です。"
        "与えられた発言記録の一部のみを要約し、事実を捏造しないでください。"
        "出力は有効な JSON のみ。"
        '{"title":"str","overview":"str","topics":[{"subject":"","discussion":"","conclusion":""}],'
        '"decisions":["str"],"action_items":[{"owner":"","task":"","due":""}],'
        '"open_questions":["str"],"markdown":"str"}'
    ),
    "zh": (
        "你是会议纪要分段摘要助手。"
        "仅摘要所给发言片段，不得编造事实。"
        "输出必须是合法 JSON。"
        '{"title":"str","overview":"str","topics":[{"subject":"","discussion":"","conclusion":""}],'
        '"decisions":["str"],"action_items":[{"owner":"","task":"","due":""}],'
        '"open_questions":["str"],"markdown":"str"}'
    ),
}

REDUCE_SYSTEM_PROMPTS: dict[str, str] = {
    "ja": (
        "複数の部分要約 JSON を統合し、最終的な会議議事録 JSON を作成してください。"
        "重複を整理し、事実を捏造しないでください。出力は有効な JSON のみ。"
        '{"title":"str","overview":"str","topics":[{"subject":"","discussion":"","conclusion":""}],'
        '"decisions":["str"],"action_items":[{"owner":"","task":"","due":""}],'
        '"open_questions":["str"],"markdown":"str"}'
    ),
    "zh": (
        "将多段部分摘要 JSON 合并为最终会议纪要 JSON。"
        "去重整理，不得编造事实。输出必须是合法 JSON。"
        '{"title":"str","overview":"str","topics":[{"subject":"","discussion":"","conclusion":""}],'
        '"decisions":["str"],"action_items":[{"owner":"","task":"","due":""}],'
        '"open_questions":["str"],"markdown":"str"}'
    ),
}

REPAIR_SYSTEM_PROMPT = (
    "Fix the following text into valid JSON matching the meeting summary schema. "
    "Output JSON only, no markdown fences."
)


def get_system_prompt(language: str, *, mode: str = "full") -> str:
    lang = language if language in ("ja", "zh") else "ja"
    if mode == "map":
        return MAP_SYSTEM_PROMPTS[lang]
    if mode == "reduce":
        return REDUCE_SYSTEM_PROMPTS[lang]
    return SYSTEM_PROMPTS[lang]


def get_user_prompt(language: str, transcript: str) -> str:
    lang = language if language in ("ja", "zh") else "ja"
    return USER_TEMPLATES[lang].format(transcript=transcript)


def get_reduce_user_prompt(partials: str) -> str:
    return f"部分要約一覧:\n\n{partials}"
