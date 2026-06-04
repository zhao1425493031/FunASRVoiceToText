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
