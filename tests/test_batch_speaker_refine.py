"""Speaker refinement post-processing tests."""

from __future__ import annotations

import json
from pathlib import Path

from voicetotext.asr.batch_align import AlignedSegment
from voicetotext.asr.batch_speaker_refine import (
    drop_leading_junk,
    fix_closing_counter_reply,
    fix_response_after_request,
    fix_short_replies_after_questions,
    refine_aligned_segments,
)
from tests.test_regression_acceptance import check_acceptance_a1_a6

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSON = ROOT / "out" / "1cb58524-4050-4efd-92bc-7ea9b8b3838c.json"


def _seg(start: int, end: int, spk: str, text: str) -> AlignedSegment:
    return AlignedSegment(start_ms=start, end_ms=end, speaker_id=spk, text=text)


def test_drop_leading_junk() -> None:
    segs = [
        _seg(30, 1026, "SPEAKER_01", "すはい。"),
        _seg(1026, 6308, "SPEAKER_00", "お疲れ様です"),
    ]
    out = drop_leading_junk(segs)
    assert len(out) == 1
    assert out[0].text.startswith("お疲れ")


def test_fix_short_reply_after_question() -> None:
    segs = [
        _seg(71074, 76862, "SPEAKER_00", "来週のレビュー会議の日程は確定していますか。"),
        _seg(77689, 78584, "SPEAKER_00", "はい。"),
        _seg(78584, 82718, "SPEAKER_01", "来週火曜日の午後二時から"),
    ]
    out = fix_short_replies_after_questions(segs)
    assert out[1].speaker_id == "SPEAKER_01"


def test_fix_response_wakarimashita() -> None:
    segs = [
        _seg(83157, 86970, "SPEAKER_00", "それではレビュー用の資料を事前に準備しておきましょう。"),
        _seg(87764, 89316, "SPEAKER_00", "わかりましたこちらで採用いたします。"),
    ]
    out = fix_response_after_request(segs)
    assert out[1].speaker_id == "SPEAKER_01"


def test_fix_closing_kochirakoso() -> None:
    segs = [
        _seg(89316, 95324, "SPEAKER_00", "本日の確認事項は以上です引き続きよろしくお願いいたします。"),
        _seg(95948, 97939, "SPEAKER_00", "こちらこそよろしくお願いいたします。"),
    ]
    out = fix_closing_counter_reply(segs)
    assert out[1].speaker_id == "SPEAKER_01"


def test_fix_closing_with_spaced_asr_text() -> None:
    segs = [
        _seg(89316, 95324, "SPEAKER_00", "たします本日の確認 事項は以上 です 引き続き よろし くお 願いいたし ます。"),
        _seg(95948, 97939, "SPEAKER_00", "こちら こそ よろし くお 願い いたし ます。"),
    ]
    out = fix_closing_counter_reply(segs)
    assert out[1].speaker_id == "SPEAKER_01"


def test_refine_real_output_json_passes_a3_a4() -> None:
    if not SAMPLE_JSON.is_file():
        return
    data = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
    raw = [
        AlignedSegment(
            start_ms=s["start_ms"],
            end_ms=s["end_ms"],
            speaker_id=s["speaker_id"],
            text=s["text"],
        )
        for s in data["segments"]
    ]
    refined = refine_aligned_segments(raw)
    payload = {**data, "segments": [s.__dict__ for s in refined]}
    results = check_acceptance_a1_a6(payload)
    assert results["A3"] is True
    assert results["A4"] is True
