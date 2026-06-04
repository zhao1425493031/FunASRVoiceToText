"""Text cleanup for streaming utterance merge and SenseVoice output."""

from __future__ import annotations

import re

import numpy as np

_MODEL_TAG = re.compile(r"<\|[^|]+\|>")
_TRAILING_PUNCT = re.compile(r"[。．.,，,、!?！？…・]+$")
_LEADING_PUNCT = re.compile(r"^[。．.,，,、!?！？…・]+$")
_PUNCT_ONLY = re.compile(r"^[\s。．.,、!?！？…・\-_'\"]+$")


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def is_meaningful_text(text: str, min_chars: int = 2) -> bool:
    """Return True if text has enough non-punctuation content to show users."""
    stripped = text.strip()
    if not stripped:
        return False
    if _PUNCT_ONLY.match(stripped):
        return False
    core = re.sub(r"[\s。．.,、!?！？…・\-_'\"]", "", stripped)
    return len(core) >= min_chars


def strip_model_tags(text: str) -> str:
    return _MODEL_TAG.sub("", text).strip()


def strip_trailing_punct(text: str) -> str:
    return _TRAILING_PUNCT.sub("", text.strip())


def strip_leading_punct(text: str) -> str:
    return _LEADING_PUNCT.sub("", text.strip())


_JA_PUNCT_CHARS = "、。．.!?！？…"
_JA_SPACE_RUN = re.compile(r"[\s\u3000]+")
# Clause/sentence endings before a space → 。；otherwise → 、
_JA_SENTENCE_END_BEFORE_SPACE = re.compile(
    r"(?:です|ます|でした|ません|でしょう|ましょう|ございます|だ|った|ない|れます|けます|せます|てます)$"
)


def has_japanese_punctuation(text: str) -> bool:
    return any(ch in text for ch in _JA_PUNCT_CHARS)


def _punctuation_for_space_break(left: str) -> str:
    """Choose 。 after complete clauses (です/ます…), else 、 for phrase breaks."""
    if _JA_SENTENCE_END_BEFORE_SPACE.search(left):
        return "。"
    return "、"


def apply_japanese_punctuation(text: str) -> str:
    """Turn SenseVoice JA ITN phrase spaces into 、/。 for readable finals."""
    s = text.strip()
    if not s:
        return s
    if has_japanese_punctuation(s) and " " not in s and "\u3000" not in s:
        return normalize_trailing_punctuation(s)
    if not _JA_SPACE_RUN.search(s):
        if s and s[-1] not in _JA_PUNCT_CHARS:
            s = s + "。"
        return normalize_trailing_punctuation(s)

    parts = [p for p in _JA_SPACE_RUN.split(s) if p]
    if not parts:
        return s
    out = parts[0]
    for part in parts[1:]:
        out += _punctuation_for_space_break(out) + part
    out = re.sub(r"、+", "、", out)
    out = re.sub(r"。+", "。", out)
    out = re.sub(r"、。", "。", out)
    if out and out[-1] not in _JA_PUNCT_CHARS:
        out += "。"
    return normalize_trailing_punctuation(out)


def normalize_trailing_punctuation(text: str) -> str:
    """Remove weak pause marks before sentence endings (e.g. 需求，。 -> 需求。)."""
    s = text.strip()
    if not s:
        return s
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"[，,、；;：:]+([。．.!?！？]+)$", r"\1", s)
    s = re.sub(r"([。．.!?！？])\1+$", r"\1", s)
    return s


def merge_utterance_segments(draft: str, segment: str) -> str:
    """Merge window ASR into one growing utterance without mid-sentence periods."""
    segment = segment.strip()
    if not segment:
        return draft
    if not draft:
        return strip_leading_punct(segment)
    if segment == draft:
        return draft
    if draft in segment and len(segment) > len(draft):
        return segment
    if segment in draft:
        return draft
    left = strip_trailing_punct(draft)
    right = strip_leading_punct(segment)
    if not right:
        return left or draft
    if not left:
        return right
    return left + right
