"""Accumulate meeting final segments for LLM summary."""

from __future__ import annotations

from typing import Any


def map_speaker_id(speaker_id: int | str | None, *, single_mode: bool) -> str:
    if single_mode:
        return "SPEAKER_00"
    try:
        idx = int(speaker_id if speaker_id is not None else 0)
    except (TypeError, ValueError):
        idx = 0
    return f"SPEAKER_{idx:02d}"


class MeetingTranscriptBuffer:
    def __init__(
        self,
        session_id: str,
        language: str,
        *,
        single_speaker_mode: bool = False,
    ) -> None:
        self.session_id = session_id
        self.language = language
        self._single_speaker_mode = single_speaker_mode
        self._segments: list[dict[str, Any]] = []
        self._seen_seg_ids: set[str] = set()

    def append_final(self, msg: dict[str, Any]) -> None:
        if msg.get("type") != "final":
            return
        text = str(msg.get("text", "")).strip()
        if not text:
            return
        seg_id = str(msg.get("seg_id", ""))
        if seg_id and seg_id in self._seen_seg_ids:
            return
        if seg_id:
            self._seen_seg_ids.add(seg_id)

        start_ms = int(msg.get("t_start_ms", 0))
        end_ms = int(msg.get("t_end_ms", start_ms))
        speaker = map_speaker_id(
            msg.get("speaker_id"),
            single_mode=self._single_speaker_mode,
        )
        self._segments.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_id": speaker,
                "text": text,
                "seg_id": seg_id,
            }
        )

    @property
    def segments(self) -> list[dict[str, Any]]:
        return list(self._segments)

    def duration_ms(self) -> int:
        if not self._segments:
            return 0
        return max(int(s.get("end_ms", 0)) for s in self._segments)

    def __bool__(self) -> bool:
        return bool(self._segments)
