"""Utterance embedding speaker registry tests."""

from __future__ import annotations

import numpy as np

from voicetotext.asr.utterance_speaker import UtteranceSpeakerRegistry


def _vec(seed: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000))
    return rng.standard_normal(192).astype(np.float32)


def test_first_embedding_creates_speaker_zero() -> None:
    reg = UtteranceSpeakerRegistry(max_speakers=8, threshold=0.72)
    assert reg.assign(_vec(1.0)) == 0
    assert reg.speaker_count == 1


def test_similar_embedding_matches_existing() -> None:
    reg = UtteranceSpeakerRegistry(max_speakers=8, threshold=0.72)
    base = _vec(2.0)
    reg.assign(base)
    similar = base + np.random.default_rng(0).standard_normal(192).astype(np.float32) * 0.01
    spk = reg.assign(similar)
    assert spk == 0
    assert reg.speaker_count == 1


def test_dissimilar_embedding_creates_new_speaker() -> None:
    reg = UtteranceSpeakerRegistry(max_speakers=8, threshold=0.99)
    reg.assign(_vec(3.0))
    spk = reg.assign(_vec(99.0))
    assert spk == 1
    assert reg.speaker_count == 2


def test_max_speakers_caps_new_centroids() -> None:
    reg = UtteranceSpeakerRegistry(max_speakers=2, threshold=0.99)
    reg.assign(_vec(6.0))
    reg.assign(_vec(7.0))
    assert reg.speaker_count == 2
    # Dissimilar but cap reached -> assign to nearest existing id
    spk = reg.assign(_vec(100.0))
    assert spk in (0, 1)
    assert reg.speaker_count == 2


def test_reset_clears_centroids() -> None:
    reg = UtteranceSpeakerRegistry(max_speakers=8, threshold=0.72)
    reg.assign(_vec(4.0))
    reg.reset()
    assert reg.speaker_count == 0
    assert reg.assign(_vec(5.0)) == 0
