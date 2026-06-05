#!/usr/bin/env python3
"""Diagnose HuggingFace access for Pyannote Community-1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicetotext.config import DEFAULT_CONFIG_PATH, apply_secrets

MODEL = "pyannote/speaker-diarization-community-1"
# Community-1 可能还依赖这些 gated 子模型
EXTRA_MODELS = [
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-3.1",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
]


def main() -> int:
    config_path = DEFAULT_CONFIG_PATH
    if not apply_secrets(config_path):
        print("FAIL: secrets.meeting.yaml 中无有效 hf_token")
        return 1

    import os

    token = os.environ.get("HF_TOKEN", "")
    print(f"Token loaded: yes (len={len(token)}, starts with hf_)")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    try:
        who = api.whoami()
        print(f"whoami: {who.get('name', who)}")
    except Exception as exc:
        print(f"FAIL whoami: {exc}")
        return 1

    def try_file(repo: str) -> str:
        try:
            path = hf_hub_download(repo, "config.yaml", token=token)
            return f"OK  {repo} -> {path}"
        except Exception as exc:
            short = str(exc).split("\n")[0][:120]
            return f"FAIL {repo}: {short}"

    print(f"\n--- {MODEL} ---")
    print(try_file(MODEL))

    print("\n--- extra gated models ---")
    for repo in EXTRA_MODELS:
        print(try_file(repo))

    print("\nIf community-1 FAIL: accept license on model page (same account as token).")
    print("URL: https://huggingface.co/pyannote/speaker-diarization-community-1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
