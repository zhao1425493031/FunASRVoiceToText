"""Japanese transcript beautification tests."""

from __future__ import annotations

import json
from pathlib import Path

from voicetotext.asr.batch_align import AlignedSegment
from voicetotext.asr.batch_segment_polish import polish_aligned_segments
from voicetotext.asr.batch_speaker_refine import refine_aligned_segments
from voicetotext.asr.batch_text_beautify import beautify_japanese_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "out" / "df8e397c-c177-4a1e-bddf-ad82a17c4e56.json"


def test_remove_intraword_spaces() -> None:
    out = beautify_japanese_text("お疲れ 様 です 先週ご相談 した 案件")
    assert " " not in out or "お疲れ様です" in out
    assert "お疲れ 様" not in out


def test_fix_broken_punctuation() -> None:
    assert "でしょうか" in beautify_japanese_text("テスト環境の準備状況はいかがでしょ、うか")
    assert "ございます" in beautify_japanese_text("ありがとうござ。います、ところで")
    assert "ですね" in beautify_japanese_text("それはいいで、すね お客様")


def test_insert_clause_periods() -> None:
    out = beautify_japanese_text("承知しました前回課題になっていた")
    assert "承知しました。" in out
    assert "承知しました前" not in out


def test_ensure_terminal_punctuation() -> None:
    assert beautify_japanese_text("引き続きよろしくお願いたします").endswith("。")


def test_beautify_full_sample_segments() -> None:
    if not SAMPLE.is_file():
        return
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
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
    texts = " ".join(s.text for s in polished)
    assert "でしょ、うか" not in texts
    assert "ござ。い" not in texts
    assert "お疲れ 様" not in texts
    for seg in polished:
        if len(seg.text) <= 10:
            continue
        assert seg.text.endswith(("。", "？", "！")) or "か。" in seg.text
