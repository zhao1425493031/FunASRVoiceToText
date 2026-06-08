"""Acceptance criteria A1–A6 for diar-first batch output."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_JSON = ROOT / "out" / "e7fb0c80-821a-41ab-9638-ea17a87cecbc.json"


def check_acceptance_a1_a6(data: dict) -> dict[str, bool]:
    segments = data.get("segments") or []
    meta = data.get("meta") or {}

    a1 = len(segments) > 13

    a2 = True
    for seg in segments:
        dur = seg["end_ms"] - seg["start_ms"]
        text = seg.get("text", "")
        if dur >= 25000 and "でしょうか" in text and "完了しました" in text:
            a2 = False
            break

    a3 = True
    for i, seg in enumerate(segments):
        if "来週のレビュー会議の日程は確定していますか" in seg.get("text", ""):
            q_speaker = seg.get("speaker_id")
            for j in range(i + 1, min(i + 4, len(segments))):
                nxt = segments[j]
                if nxt.get("text", "").strip() == "はい":
                    # 「はい」 must be answered by someone other than the questioner
                    if nxt.get("speaker_id") == q_speaker:
                        a3 = False
                    break

    a4 = True
    for i, seg in enumerate(segments):
        if "こちら こそ よろし くお 願い いたし ます" in seg.get("text", "") or (
            "こちらこそ" in seg.get("text", "")
        ):
            same_run = 0
            for k in range(max(0, i - 4), i + 1):
                if segments[k].get("speaker_id") == seg.get("speaker_id"):
                    same_run += 1
            if same_run >= 4:
                a4 = False
            break

    a5 = all(
        k in data
        for k in ("job_id", "language", "duration_ms", "segments", "meta", "summary")
    )
    a6 = meta.get("pipeline") == "diar_first"

    return {
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "A5": a5,
        "A6": a6,
    }


def test_old_vad_first_output_fails_acceptance() -> None:
    if not OLD_JSON.is_file():
        return
    data = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    results = check_acceptance_a1_a6({**data, "meta": {**data.get("meta", {}), "pipeline": "vad_first"}})
    assert results["A1"] is False
    assert results["A2"] is False


def test_mock_diar_first_output_passes_acceptance() -> None:
    data = {
        "job_id": "test",
        "language": "ja",
        "duration_ms": 100969,
        "segments": [
            {"start_ms": 1040, "end_ms": 6510, "speaker_id": "SPEAKER_00", "text": "お疲れ様です 先週ご相談した 案件 の進捗 について 確認させて ください。"},
            {"start_ms": 8020, "end_ms": 12000, "speaker_id": "SPEAKER_01", "text": "お疲れ様です現在基本設計は完了しており開発作業は約75%もり進んでいます"},
            {"start_ms": 12000, "end_ms": 15000, "speaker_id": "SPEAKER_00", "text": "テスト環境の準備状況はいかがでしょうか"},
            {"start_ms": 15000, "end_ms": 20000, "speaker_id": "SPEAKER_01", "text": "サーバの構築は完了しましたので 明日から内部テストを開始する予定です"},
            {"start_ms": 20000, "end_ms": 22000, "speaker_id": "SPEAKER_00", "text": "承知しました"},
            {"start_ms": 22000, "end_ms": 26000, "speaker_id": "SPEAKER_00", "text": "前回課題になっていた処理速度の問題は改善されていますか"},
            {"start_ms": 26000, "end_ms": 30000, "speaker_id": "SPEAKER_01", "text": "プログラムを見直した結果、平均処理時間を約2秒から06秒まで短縮できました"},
            {"start_ms": 30000, "end_ms": 32000, "speaker_id": "SPEAKER_00", "text": "それはいいですね"},
            {"start_ms": 32000, "end_ms": 36000, "speaker_id": "SPEAKER_00", "text": "お客様から追加要望が出ていた帳票機能についてはどうでしょうか"},
            {"start_ms": 36000, "end_ms": 40000, "speaker_id": "SPEAKER_01", "text": "開発工数を見積もったところ、3日から5日程度の追加作業で対応可能と考えています"},
            {"start_ms": 40000, "end_ms": 42000, "speaker_id": "SPEAKER_00", "text": "わかりましたでは今週中に最新のスケジュールと課題一覧を共有してください。"},
            {"start_ms": 65950, "end_ms": 70680, "speaker_id": "SPEAKER_01", "text": "承知 しま した 金曜日 の 夕方 まで に 関係 者 へ 展開 いたし ます。"},
            {"start_ms": 73170, "end_ms": 77040, "speaker_id": "SPEAKER_00", "text": "来週のレビュー会議の日程は確定していますか"},
            {"start_ms": 77450, "end_ms": 78280, "speaker_id": "SPEAKER_01", "text": "はい"},
            {"start_ms": 78340, "end_ms": 82700, "speaker_id": "SPEAKER_01", "text": "来週火 曜日 の 午後 2 時から 1 時間半 を 予定 して おり ます。"},
            {"start_ms": 87590, "end_ms": 90000, "speaker_id": "SPEAKER_01", "text": "わかりましたこちらで採用いたします。"},
            {"start_ms": 90170, "end_ms": 92980, "speaker_id": "SPEAKER_00", "text": "本日 の 確認 事項 は 以上 です。"},
            {"start_ms": 92980, "end_ms": 95310, "speaker_id": "SPEAKER_00", "text": "引き続き よろし くお 願い いたし ます。"},
            {"start_ms": 95740, "end_ms": 98040, "speaker_id": "SPEAKER_01", "text": "こちら こそ よろし くお 願い いたし ます。"},
        ],
        "meta": {"pipeline": "diar_first", "asr_model": "iic/SenseVoiceSmall"},
        "summary": None,
    }
    results = check_acceptance_a1_a6(data)
    assert all(results.values()), results
