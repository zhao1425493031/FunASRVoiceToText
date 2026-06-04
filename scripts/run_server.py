#!/usr/bin/env python3
"""Start the VoiceToText FastAPI server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn

from voicetotext.config import load_config
from voicetotext.server import app as app_module


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceToText ASR server")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config.yaml or VOICETOTEXT_CONFIG)",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is not None:
        config_path = config_path if config_path.is_absolute() else ROOT / config_path
        os.environ["VOICETOTEXT_CONFIG"] = str(config_path)
        app_module.init_app(config_path)

    config = load_config(config_path)
    ssl_paths = None
    if config.ssl_enabled:
        try:
            ssl_paths = config.resolve_ssl_paths()
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            print("Falling back to HTTP. For iPhone mic, run:")
            print("  python scripts/generate_cert.py --ip <your-lan-ip>")
    scheme = "https" if ssl_paths else "http"

    print(f"Config: {config_path or 'default'}")
    print(f"ASR backend: {config.asr_backend} language={config.language}")
    print(f"Starting server at {scheme}://{config.host}:{config.port}")
    print(f"Open on PC: {scheme}://127.0.0.1:{config.port}")
    print(f"Open on phone (same WiFi): {scheme}://<your-lan-ip>:{config.port}")
    if config.auth_enabled:
        print("API Key auth enabled — use ?key=<api_key> in browser URL for Web UI")
    if not ssl_paths:
        print("WARNING: HTTP mode — iPhone microphone may not work. Run:")
        print("  python scripts/generate_cert.py --ip <your-lan-ip>")
        print("Then set ssl_certfile/ssl_keyfile in config.yaml and restart.")

    kwargs: dict = {
        "app": "voicetotext.server.app:app",
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
