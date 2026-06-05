"""Utterance embedding extraction helpers."""

from __future__ import annotations

import numpy as np

from voicetotext.asr.utterance_speaker import _extract_embedding, _to_numpy_vector


def test_to_numpy_vector_from_nested_list() -> None:
    vec = _to_numpy_vector([[[0.1, 0.2, 0.3]]])
    assert vec is not None
    assert vec.size == 3


def test_extract_embedding_from_dict() -> None:
    emb = _extract_embedding({"spk_embedding": np.array([1.0, 2.0], dtype=np.float32)})
    assert emb is not None
    assert emb.size == 2


def test_extract_embedding_from_funasr_list_shape() -> None:
    emb = _extract_embedding(
        [{"key": "utt", "spk_embedding": [np.array([0.5, -0.5], dtype=np.float32)]}]
    )
    assert emb is not None
    assert emb.size == 2
