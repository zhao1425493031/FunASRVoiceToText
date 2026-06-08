#!/usr/bin/env python3
"""CLI / HTTP server for offline batch transcription."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicetotext.asr.audio_preprocess import sanitize_audio_path  # noqa: E402
from voicetotext.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    apply_secrets,
    load_config,
    require_hf_token,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceToText batch ASR")
    parser.add_argument("input", nargs="?", help="Input audio file (wav/mp3/flac)")
    parser.add_argument("-o", "--output", default="out", help="Output directory")
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start batch HTTP API instead of CLI",
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    apply_secrets(config_path)
    if not args.serve:
        require_hf_token(config_path)

    config = load_config(config_path)

    if args.serve:
        import uvicorn

        from voicetotext.server.batch_app import app, init_app

        init_app(config_path)
        ssl_paths = None
        if config.ssl_enabled:
            try:
                ssl_paths = config.resolve_ssl_paths()
            except FileNotFoundError as exc:
                print(f"WARNING: {exc}")
                print("Falling back to HTTP.")
        scheme = "https" if ssl_paths else "http"
        print(f"Batch web UI: {scheme}://127.0.0.1:{config.port}/batch")
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            ssl_certfile=str(ssl_paths[0]) if ssl_paths else None,
            ssl_keyfile=str(ssl_paths[1]) if ssl_paths else None,
        )
        return

    if not args.input:
        parser.error("input audio file required unless --serve")

    from voicetotext.asr.batch_pipeline import BatchPipeline

    pipeline = BatchPipeline(config)
    pipeline.load()
    out = Path(args.output)
    input_path = sanitize_audio_path(args.input)
    transcript = pipeline.process_file(input_path, out)
    base = out / transcript.job_id
    print(f"完成: {input_path.name}")
    print(f"job_id={transcript.job_id}  segments={len(transcript.segments)}")
    for fmt in config.output_formats:
        print(f"  {fmt}: {base.with_suffix('.' + fmt.lower())}")
    if "md" in config.output_formats:
        print(f"\nmd (copy-friendly): {base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
