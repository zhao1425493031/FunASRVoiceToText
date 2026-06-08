"""Segment polish: merge fragments and normalize text."""

from __future__ import annotations

import json
from pathlib import Path

from voicetotext.asr.batch_align import AlignedSegment
from voicetotext.asr.batch_segment_polish import (
    merge_adjacent_same_speaker,
    normalize_text_spacing,
    polish_aligned_segments,
    repair_boundary_fragments,
)
from voicetotext.asr.batch_speaker_refine import refine_aligned_segments

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_E54 = ROOT / "out" / "e54b17db-c32d-4a5b-98ff-3d5600ab07fb.json"


def _seg(start: int, end: int, spk: str, text: str) -> AlignedSegment:
    return AlignedSegment(start_ms=start, end_ms=end, speaker_id=spk, text=text)


def test_normalize_text_spacing() -> None:
    assert normalize_text_spacing("お疲れ  様 です") == "お疲れ 様 です"


def test_merge_adjacent_same_speaker_zero_gap() -> None:
    segs = [
        _seg(77689, 78584, "SPEAKER_01", "はい。"),
        _seg(78584, 82718, "SPEAKER_01", "来週火曜日の午後二時から"),
    ]
    out = merge_adjacent_same_speaker(segs)
    assert len(out) == 1
    assert "はい" in out[0].text and "来週" in out[0].text


def test_repair_boundary_adoption_phrase() -> None:
    segs = [
        _seg(87764, 89114, "SPEAKER_01", "わかりましたこちらではい。"),
        _seg(89114, 95324, "SPEAKER_00", "用いたします。本日の確認事項は以上です。引き続きよろしくお願いいたします。"),
    ]
    out = repair_boundary_fragments(segs)
    assert len(out) == 2
    assert "採用いたします" in _compact(out[0].text)
    assert "こちらではい" not in _compact(out[0].text)
    assert out[1].text.startswith("本日")
    assert out[0].speaker_id == "SPEAKER_01"
    assert out[1].speaker_id == "SPEAKER_00"


def _compact(text: str) -> str:
    import re

    return re.sub(r"[\s\u3000]+", "", text.strip())


def test_polish_e54_sample_closing() -> None:
    if not SAMPLE_E54.is_file():
        return
    data = json.loads(SAMPLE_E54.read_text(encoding="utf-8"))
    raw = [
        AlignedSegment(
            start_ms=s["start_ms"],
            end_ms=s["end_ms"],
            speaker_id=s["speaker_id"],
            text=s["text"],
        )
        for s in data["segments"]
    ]
    polished = polish_aligned_segments(refine_aligned_segments(raw))
    texts = [s.text for s in polished]
    assert any("採用いたします" in _compact(t) for t in texts)
    assert not any("用いたします" in _compact(t) and "本日" in _compact(t) for t in texts)
    assert any("こちらこそ" in _compact(t) for t in texts)
    assert len(polished) < len(raw)
