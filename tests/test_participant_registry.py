"""Participant speaker registry tests."""

from __future__ import annotations

from voicetotext.asr.participant_registry import ParticipantSpeakerRegistry


def test_participant_stable_ids() -> None:
    reg = ParticipantSpeakerRegistry(max_speakers=8)
    a = reg.speaker_id_for("alice")
    b = reg.speaker_id_for("bob")
    assert a == 0
    assert b == 1
    assert reg.speaker_id_for("alice") == 0


def test_participant_release() -> None:
    reg = ParticipantSpeakerRegistry(max_speakers=8)
    reg.speaker_id_for("alice")
    reg.release("alice")
    assert reg.active_count() == 0
