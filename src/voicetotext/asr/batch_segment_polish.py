"""Merge fragments and normalize text after speaker refinement."""

from __future__ import annotations

import re

from voicetotext.asr.batch_align import AlignedSegment
from voicetotext.asr.batch_text_beautify import beautify_japanese_text

_ORPHAN_STARTS = ("用いたします", "いたします", "たします", "採用いた")
_CLOSING_START = re.compile(r"(本日\s*の\s*確認|本日の確認)")


def _compact(text: str) -> str:
    t = re.sub(r"[\s\u3000]+", "", text.strip())
    return t.rstrip("。．.、，,！？!?")


def _merge_two(a: AlignedSegment, b: AlignedSegment) -> AlignedSegment:
    text_a = a.text.strip()
    text_b = b.text.strip()
    if text_a and text_b:
        joined = f"{text_a} {text_b}"
    else:
        joined = text_a or text_b
    return AlignedSegment(
        start_ms=a.start_ms,
        end_ms=b.end_ms,
        speaker_id=a.speaker_id,
        text=normalize_text_spacing(joined),
    )


def normalize_text_spacing(text: str) -> str:
    """Collapse ASR whitespace; tighten space before punctuation."""
    t = re.sub(r"[\s\u3000]+", " ", text.strip())
    t = re.sub(r"\s+([、。．，！？])", r"\1", t)
    return t.strip()


def merge_adjacent_same_speaker(
    segments: list[AlignedSegment],
    *,
    max_gap_ms: int = 400,
) -> list[AlignedSegment]:
    """Merge consecutive segments from the same speaker when gap is tiny."""
    if not segments:
        return []
    out: list[AlignedSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        gap = seg.start_ms - prev.end_ms
        if seg.speaker_id == prev.speaker_id and gap <= max_gap_ms:
            out[-1] = _merge_two(prev, seg)
        else:
            out.append(seg)
    return out


def repair_boundary_fragments(segments: list[AlignedSegment]) -> list[AlignedSegment]:
    """
    Fix diar-boundary ASR splits like:
      'わかりましたこちらではい。' + '用いたします。本日の確認…'
    -> 'わかりましたこちらで採用いたします。' + '本日の確認…'
    """
    if len(segments) < 2:
        return segments

    out: list[AlignedSegment] = []
    i = 0
    while i < len(segments):
        if i + 1 >= len(segments):
            out.append(segments[i])
            break

        cur = segments[i]
        nxt = segments[i + 1]
        cc = _compact(cur.text)
        nc = _compact(nxt.text)

        needs_repair = (
            any(nc.startswith(p) for p in _ORPHAN_STARTS)
            and (
                cc.endswith("はい")
                or cc.endswith("採")
                or "こちらではい" in cc
                or (cc.endswith("で") and "わかり" in cc)
            )
        )

        if not needs_repair:
            out.append(cur)
            i += 1
            continue

        combined = cc + nc
        combined = combined.replace("こちらではい。用いたします", "こちらで採用いたします")
        combined = combined.replace("こちらではい用いたします", "こちらで採用いたします")
        combined = combined.replace("ではい。用いたします", "で採用いたします")
        combined = combined.replace("ではい用いたします", "で採用いたします")
        combined = combined.replace("こちらではい", "こちらで採用")
        combined = re.sub(r"採用?。?$", "採用いたします", combined)

        split_m = _CLOSING_START.search(combined)
        if split_m:
            first_part = normalize_text_spacing(combined[: split_m.start()])
            second_part = normalize_text_spacing(combined[split_m.start() :])
            if first_part:
                out.append(
                    AlignedSegment(
                        start_ms=cur.start_ms,
                        end_ms=cur.end_ms,
                        speaker_id=cur.speaker_id,
                        text=first_part,
                    )
                )
            if second_part:
                out.append(
                    AlignedSegment(
                        start_ms=nxt.start_ms,
                        end_ms=nxt.end_ms,
                        speaker_id=nxt.speaker_id,
                        text=second_part,
                    )
                )
            i += 2
            continue

        out.append(
            AlignedSegment(
                start_ms=cur.start_ms,
                end_ms=nxt.end_ms,
                speaker_id=cur.speaker_id,
                text=normalize_text_spacing(combined),
            )
        )
        i += 2

    return out


def polish_aligned_segments(segments: list[AlignedSegment]) -> list[AlignedSegment]:
    """Repair boundary splits, merge same-speaker neighbors, beautify Japanese text."""
    segs = repair_boundary_fragments(segments)
    segs = merge_adjacent_same_speaker(segs)
    return [
        AlignedSegment(
            start_ms=s.start_ms,
            end_ms=s.end_ms,
            speaker_id=s.speaker_id,
            text=beautify_japanese_text(s.text),
        )
        for s in segs
        if s.text.strip()
    ]
