"""Tests for stream text helpers."""

from voicetotext.text_utils import (
    apply_japanese_punctuation,
    merge_utterance_segments,
    normalize_trailing_punctuation,
    strip_model_tags,
    strip_trailing_punct,
)


def test_strip_model_tags() -> None:
    assert strip_model_tags("<|zh|>你好") == "你好"


def test_merge_strips_mid_sentence_punct() -> None:
    assert merge_utterance_segments("今天。", "我来测试") == "今天我来测试"
    assert merge_utterance_segments("大家好", "今天我来") == "大家好今天我来"


def test_merge_segment_extends() -> None:
    assert merge_utterance_segments("こんにちは", "こんにちは世界") == "こんにちは世界"
    assert merge_utterance_segments("こんにちは", "すごい") == "こんにちはすごい"


def test_strip_trailing_punct() -> None:
    assert strip_trailing_punct("测试。") == "测试"


def test_normalize_trailing_punctuation() -> None:
    assert normalize_trailing_punctuation("场景需求，。") == "场景需求。"
    assert normalize_trailing_punctuation("你好。。") == "你好。"
    assert normalize_trailing_punctuation("结束") == "结束"


def test_apply_japanese_punctuation_from_spaces() -> None:
    raw = "皆さんこんにちは 今日はテストです"
    out = apply_japanese_punctuation(raw)
    assert "こんにちは、今日" in out
    assert out.endswith("テストです。")
    assert " " not in out


def test_apply_japanese_punctuation_masu_desu_use_period() -> None:
    raw = "テストを行います 現在の時刻は午前10時30分です 私は話します"
    out = apply_japanese_punctuation(raw)
    assert "行います。現在" in out
    assert "30分です。私" in out
    assert "話します。" in out or out.endswith("。")
