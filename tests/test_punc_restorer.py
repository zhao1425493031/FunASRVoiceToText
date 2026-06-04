"""Tests for PuncRestorer (mocked FunASR)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from voicetotext.asr.punc_restorer import PuncRestorer
from voicetotext.config import load_config


def test_restore_calls_punc_task() -> None:
    cfg = replace(load_config(), punc_model="ct-punc")
    restorer = PuncRestorer(cfg)
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "你好，世界。"}]

    with patch("funasr.AutoModel", return_value=mock_model):
        text = restorer.restore("你好世界")

    mock_model.generate.assert_called_once()
    assert mock_model.generate.call_args.kwargs.get("task") == "punc"
    assert text == "你好，世界。"


def test_restore_empty_without_model() -> None:
    cfg = replace(load_config(), punc_model="")
    restorer = PuncRestorer(cfg)
    assert restorer.restore("") == ""
