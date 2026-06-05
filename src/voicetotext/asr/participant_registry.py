"""Stable participant_id -> speaker_id mapping for multi-client meetings."""

from __future__ import annotations

import threading


class ParticipantSpeakerRegistry:
    """Thread-safe registry: one browser client = one speaker slot."""

    def __init__(self, max_speakers: int) -> None:
        self.max_speakers = max(1, max_speakers)
        self._lock = threading.Lock()
        self._participant_to_id: dict[str, int] = {}
        self._id_to_participant: dict[int, str] = {}

    def speaker_id_for(self, participant_id: str) -> int:
        pid = participant_id.strip()
        if not pid:
            return 0
        with self._lock:
            if pid in self._participant_to_id:
                return self._participant_to_id[pid]
            if len(self._participant_to_id) >= self.max_speakers:
                return min(self._participant_to_id.values())
            new_id = len(self._participant_to_id)
            self._participant_to_id[pid] = new_id
            self._id_to_participant[new_id] = pid
            return new_id

    def release(self, participant_id: str) -> None:
        pid = participant_id.strip()
        if not pid:
            return
        with self._lock:
            self._participant_to_id.pop(pid, None)
            self._id_to_participant = {
                sid: p for p, sid in self._participant_to_id.items()
            }

    def active_count(self) -> int:
        with self._lock:
            return len(self._participant_to_id)
