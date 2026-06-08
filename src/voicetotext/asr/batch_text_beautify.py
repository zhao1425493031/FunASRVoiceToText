"""Beautify Japanese ASR text for readable transcripts."""

from __future__ import annotations

import re

# Hiragana, Katakana, Kanji, prolongation mark
_JA_CHAR = r"[ぁ-んァ-ヶ一-龯々ー]"
_JA_SPACE_JA = re.compile(rf"({_JA_CHAR})\s+({_JA_CHAR})")

# Ordered replacements for common SenseVoice split errors
_PHRASE_FIXES: tuple[tuple[str, str], ...] = (
    (r"でしょ、うか", "でしょうか"),
    (r"でしょ\s*うか", "でしょうか"),
    (r"ござ。\s*います", "ございます"),
    (r"ござ\s*います", "ございます"),
    (r"で、すね", "ですね"),
    (r"で\s*すね", "ですね"),
    (r"進んで、います", "進んでいます"),
    (r"進んで\s*います", "進んでいます"),
    (r"考えて、います", "考えています"),
    (r"考えて\s*います", "考えています"),
    (r"2\s*秒\s*から\s*06\s*秒", "2秒から0.6秒"),
    (r"06\s*秒", "0.6秒"),
    (r"2\s*秒", "2秒"),
    (r"それ\s*で\s*は", "それでは"),
    (r"わかり\s*ま\s*した", "わかりました"),
    (r"承知\s*しま\s*した", "承知しました"),
    (r"よろし\s*く", "よろしく"),
    (r"お\s*願い", "お願い"),
    (r"こちら\s*こそ", "こちらこそ"),
    (r"お疲れ\s*様\s*です", "お疲れ様です"),
)

# Insert 。 before a new clause when ASR omits punctuation
_CLAUSE_BREAKS: tuple[tuple[str, str], ...] = (
    (r"(承知しました)(?=前回)", r"\1。"),
    (r"(ございます)(?=ところ)", r"\1。"),
    (r"(ですね)(?=お客)", r"\1。"),
    (r"(ですね)(?=お客様)", r"\1。"),
    (r"(ました)(?=では)", r"\1。"),
    (r"(ました)(?=金曜)", r"\1。"),
    (r"(ます)(?=本日)", r"\1。"),
    (r"(ます)(?=引き続き)", r"\1。"),
)


def _collapse_japanese_spaces(text: str) -> str:
    """Remove spaces between Japanese characters."""
    prev = None
    out = text
    while prev != out:
        prev = out
        out = _JA_SPACE_JA.sub(r"\1\2", out)
    return out


def beautify_japanese_text(text: str) -> str:
    """Normalize spacing, fix split words, add missing sentence punctuation."""
    t = re.sub(r"[\s\u3000]+", " ", text.strip())
    t = re.sub(r"\s+([、。．，！？])", r"\1", t)

    for pattern, repl in _PHRASE_FIXES:
        t = re.sub(pattern, repl, t)

    t = _collapse_japanese_spaces(t)

    for pattern, repl in _CLAUSE_BREAKS:
        t = re.sub(pattern, repl, t)

    # Comma before います when attached to verb (secondary pass)
    t = re.sub(r"(?<=[てで])(、)(?=います)", "", t)
    t = re.sub(r"[。．][、,]\s*", "。", t)
    t = re.sub(r"、ところで", "。ところで", t)

    t = _ensure_terminal_punctuation(t)
    return t.strip()


def _ensure_terminal_punctuation(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    if t[-1] not in "。．！？!?":
        if t.endswith("ます") or t.endswith("です") or t.endswith("した") or t.endswith("か"):
            return t + "。"
        if t.endswith("ね") or t.endswith("よ"):
            return t + "。"
    return t
