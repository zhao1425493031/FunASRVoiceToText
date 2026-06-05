#!/usr/bin/env python3
"""Start meeting-only ASR gateway (no single-user routes)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEETING_CONFIG = ROOT / "config.meeting.yaml"
sys.path.insert(0, str(ROOT / "src"))
os.environ["VOICETOTEXT_CONFIG"] = str(MEETING_CONFIG.resolve())

from voicetotext.config import load_config, require_hf_token_for_meeting

require_hf_token_for_meeting(MEETING_CONFIG)

import uvicorn

from voicetotext.server import meeting_app as meeting_app_module


def main() -> None:
    meeting_app_module.init_app(MEETING_CONFIG)
    config = load_config(MEETING_CONFIG)

    ssl_paths = None
    if config.ssl_enabled:
        try:
            ssl_paths = config.resolve_ssl_paths()
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            print("Falling back to HTTP.")
    scheme = "https" if ssl_paths else "http"

    print(f"Config: {MEETING_CONFIG}")
    print(f"Meeting ASR: {config.asr_backend} language={config.language}")
    print(f"Starting at {scheme}://{config.host}:{config.port}")
    print(f"Demo: {scheme}://127.0.0.1:{config.port}/meeting")
    if config.auth_enabled:
        print("(访问 /meeting 会自动重定向并带上 ?key=)")

    kwargs: dict = {
        "app": "voicetotext.server.meeting_app:app",
        "host": config.host,
        "port": config.port,
        "reload": False,
        "app_dir": str(ROOT / "src"),
    }
    if ssl_paths:
        kwargs["ssl_certfile"] = str(ssl_paths[0])
        kwargs["ssl_keyfile"] = str(ssl_paths[1])

    uvicorn.run(**kwargs)


if __name__ == "__main__":
    main()
