"""Post-ASR speaker label refinement for diar-first batch output."""

from __future__ import annotations

import re

from voicetotext.asr.batch_align import AlignedSegment

_SHORT_REPLY_RE = re.compile(
    r"^(?:はい|ええ|うん|そうです(?:ね)?|承知(?:しました|いたします)?|"
    r"わかり(?:ました|いたします)?)[。．、!！?？\s]*$",
    re.IGNORECASE,
)
_QUESTION_HINTS = ("でしょうか", "ですか", "ますか", "ございますか", "いかが")
_RESPONSE_HINTS = ("わかり", "承知", "了解", "かしこまり", "こちらこそ", "採用")
_LEADING_JUNK_RE = re.compile(r"^[ぁ-んァ-ヶーa-zA-Z]{1,6}[。．]?$")


def _duration_ms(seg: AlignedSegment) -> int:
    return seg.end_ms - seg.start_ms


def _other_speaker(speaker_id: str, pool: set[str]) -> str:
    for sid in sorted(pool):
        if sid != speaker_id:
            return sid
    if speaker_id == "SPEAKER_00":
        return "SPEAKER_01"
    if speaker_id == "SPEAKER_01":
        return "SPEAKER_00"
    return speaker_id


def _compact(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", text.strip())


def _contains_hint(text: str, hints: tuple[str, ...]) -> bool:
    compact = _compact(text)
    return any(h in compact or h in text for h in hints)


def _is_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _contains_hint(t, _QUESTION_HINTS):
        return True
    c = _compact(t)
    return c.endswith("か") or c.endswith("か。") or "？" in t or "?" in t


def _is_short_reply(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    return bool(_SHORT_REPLY_RE.match(compact))


def _is_response_utterance(text: str) -> bool:
    return _contains_hint(text, _RESPONSE_HINTS)


def _replace_speaker(seg: AlignedSegment, speaker_id: str) -> AlignedSegment:
    return AlignedSegment(
        start_ms=seg.start_ms,
        end_ms=seg.end_ms,
        speaker_id=speaker_id,
        text=seg.text,
    )


def drop_leading_junk(
    segments: list[AlignedSegment],
    *,
    max_start_ms: int = 3000,
    max_duration_ms: int = 2500,
) -> list[AlignedSegment]:
    """Remove spurious opening segment (e.g. mis-diar 'すはい')."""
    if not segments:
        return []
    first = segments[0]
    if first.start_ms > max_start_ms:
        return segments
    if _duration_ms(first) > max_duration_ms:
        return segments
    compact = re.sub(r"\s+", "", first.text.strip())
    if len(compact) <= 8 and (
        _LEADING_JUNK_RE.match(compact)
        or _is_short_reply(first.text)
        or compact in ("すはい", "はい")
    ):
        return segments[1:]
    return segments


def fix_short_replies_after_questions(
    segments: list[AlignedSegment],
    *,
    max_duration_ms: int = 1500,
) -> list[AlignedSegment]:
    """Flip short 'はい' etc. when diar glued it to the questioner."""
    if len(segments) < 2:
        return segments
    pool = {s.speaker_id for s in segments}
    out: list[AlignedSegment] = []
    for i, seg in enumerate(segments):
        if (
            i > 0
            and _duration_ms(seg) <= max_duration_ms
            and _is_short_reply(seg.text)
            and _is_question(segments[i - 1].text)
            and seg.speaker_id == segments[i - 1].speaker_id
        ):
            other = _other_speaker(seg.speaker_id, pool)
            out.append(_replace_speaker(seg, other))
            continue
        out.append(seg)
    return out


def fix_response_after_request(segments: list[AlignedSegment]) -> list[AlignedSegment]:
    """Flip 'わかりました/採用' when labeled same as preceding request."""
    if len(segments) < 2:
        return segments
    pool = {s.speaker_id for s in segments}
    request_hints = ("準備", "共有", "採用", "確認", "しましょう", "ください")
    out: list[AlignedSegment] = []
    for i, seg in enumerate(segments):
        if i > 0 and _is_response_utterance(seg.text):
            prev = segments[i - 1]
            prev_is_request = _contains_hint(prev.text, request_hints)
            if prev_is_request and seg.speaker_id == prev.speaker_id:
                if "こちらこそ" not in _compact(seg.text):
                    out.append(_replace_speaker(seg, _other_speaker(seg.speaker_id, pool)))
                    continue
        out.append(seg)
    return out


def fix_closing_counter_reply(segments: list[AlignedSegment]) -> list[AlignedSegment]:
    """Flip 'こちらこそ…' when chained to same speaker as sign-off."""
    if len(segments) < 2:
        return segments
    pool = {s.speaker_id for s in segments}
    out = list(segments)
    for i, seg in enumerate(out):
        if "こちらこそ" not in _compact(seg.text):
            continue
        for j in range(max(0, i - 3), i):
            prev = out[j]
            if _contains_hint(prev.text, ("よろしく", "よろし")) and seg.speaker_id == prev.speaker_id:
                out[i] = _replace_speaker(seg, _other_speaker(seg.speaker_id, pool))
                break
    return out


def refine_aligned_segments(segments: list[AlignedSegment]) -> list[AlignedSegment]:
    """Apply all speaker refinement passes in order."""
    segs = drop_leading_junk(segments)
    segs = fix_short_replies_after_questions(segs)
    segs = fix_response_after_request(segs)
    segs = fix_closing_counter_reply(segs)
    return segs
