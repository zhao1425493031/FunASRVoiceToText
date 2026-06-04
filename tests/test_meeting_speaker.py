"""Enterprise speaker registry tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from voicetotext.asr.meeting_speaker import (
    MeetingSpeakerAssigner,
    _SpeakerRegistry,
)
from voicetotext.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _norm_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(192).astype(np.float32)
    return v / np.linalg.norm(v)


def test_registry_same_embedding_same_id() -> None:
    reg = _SpeakerRegistry(
        max_speakers=8,
        similarity_threshold=0.68,
        new_speaker_max_sim=0.42,
    )
    emb = _norm_vec(42)
    assert reg.assign(emb) == 0
    assert reg.assign(emb) == 0


def test_registry_different_embeddings_new_id() -> None:
    reg = _SpeakerRegistry(
        max_speakers=8,
        similarity_threshold=0.95,
        new_speaker_max_sim=0.42,
    )
    assert reg.assign(_norm_vec(1)) == 0
    assert reg.assign(_norm_vec(2)) == 1


def test_single_mode_fixed_speaker() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    assert cfg.meeting_spk_mode == "single"
    assigner = MeetingSpeakerAssigner(cfg)
    msg = {"text": "a", "stamp_sents": [{"start": 0, "end": 1000}]}
    spk, _, _, _ = assigner.assign(msg, pcm_utterance=b"\x00" * 32000)
    assert spk == 0


def test_legacy_no_diarization() -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    cfg = replace(cfg, meeting_spk_mode="multi", meeting_use_diarization=False)
    assigner = MeetingSpeakerAssigner(cfg)
    msg1 = {"text": "a", "stamp_sents": [{"start": 0, "end": 1000}]}
    msg2 = {"text": "b", "stamp_sents": [{"start": 8000, "end": 9000}]}
    spk1, _, _, _ = assigner.assign(msg1)
    spk2, _, _, changed = assigner.assign(msg2)
    assert spk1 == 0
    assert spk2 == 0
    assert changed is False


def test_assigner_clustering_stable_across_utterances(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(ROOT / "config.meeting.yaml")
    cfg = replace(
        cfg,
        meeting_spk_mode="multi",
        meeting_use_diarization=True,
        meeting_spk_similarity_threshold=0.5,
        meeting_spk_new_speaker_max_sim=0.42,
    )
    vec = _norm_vec(7)

    def _fake_extract(_config: object, _pcm: bytes) -> np.ndarray | None:
        return vec

    monkeypatch.setattr(
        "voicetotext.asr.meeting_speaker._extract_spk_embedding",
        _fake_extract,
    )
    assigner = MeetingSpeakerAssigner(cfg)
    pcm = b"\x01" * 32000
    msg = {"text": "x", "stamp_sents": [{"start": 0, "end": 1000}]}
    spk1, _, _, _ = assigner.assign(msg, pcm_utterance=pcm)
    msg2 = {"text": "y", "stamp_sents": [{"start": 10000, "end": 11000}]}
    spk2, _, _, changed = assigner.assign(msg2, pcm_utterance=pcm)
    assert spk1 == 0
    assert spk2 == 0
    assert changed is False
