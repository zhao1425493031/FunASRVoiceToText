"""Backward-compatible re-export; use voicetotext.asr.factory instead."""

from voicetotext.asr.paraformer_engine import ParaformerEngine as ASREngine

__all__ = ["ASREngine"]
