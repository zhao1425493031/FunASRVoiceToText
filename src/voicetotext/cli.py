"""CLI for microphone and WAV streaming recognition."""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

from voicetotext.asr.factory import create_asr_backend
from voicetotext.config import load_config, resolve_config_path
from voicetotext.logging_setup import get_logger, setup_logging
from voicetotext.stream_session import StreamSession


def _float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _load_wav_chunks(path: str, stride_samples: int) -> list[bytes]:
    import soundfile as sf

    speech, sr = sf.read(path, dtype="float32")
    if speech.ndim > 1:
        speech = speech[:, 0]
    if sr != 16000:
        duration = len(speech) / sr
        target_len = int(duration * 16000)
        x_old = np.linspace(0, duration, num=len(speech), endpoint=False)
        x_new = np.linspace(0, duration, num=target_len, endpoint=False)
        speech = np.interp(x_new, x_old, speech).astype(np.float32)

    chunks: list[bytes] = []
    for offset in range(0, len(speech), stride_samples):
        segment = speech[offset : offset + stride_samples]
        if len(segment) < stride_samples:
            pad = np.zeros(stride_samples - len(segment), dtype=np.float32)
            segment = np.concatenate([segment, pad])
        chunks.append(_float32_to_pcm_bytes(segment))
    return chunks


def run_wav(path: str, config_path: Path | None = None) -> int:
    config = load_config(config_path)
    setup_logging(config)
    logger = get_logger(__name__)
    engine = create_asr_backend(config)
    engine.load()
    session = StreamSession(engine=engine, config=config)
    chunks = _load_wav_chunks(path, config.chunk_stride_samples)
    logger.info("Processing %s chunks from %s", len(chunks), path)

    def on_partial(text: str) -> None:
        print(text, end="", flush=True)

    def on_final(text: str) -> None:
        print(f"\n[final] {text}", flush=True)

    for chunk in chunks:
        result = session.feed_pcm(chunk)
        if result.partial:
            on_partial(result.partial)
        if result.final:
            on_final(result.final)
    final = session.finalize()
    if final.final:
        on_final(final.final)
    print(f"\n[confirmed] {session.confirmed}")
    return 0


def run_mic(config_path: Path | None = None) -> int:
    import sounddevice as sd

    config = load_config(config_path)
    setup_logging(config)
    logger = get_logger(__name__)
    engine = create_asr_backend(config)
    engine.load()
    session = StreamSession(engine=engine, config=config)

    audio_q: queue.Queue[bytes] = queue.Queue()
    stride = config.chunk_stride_samples

    def callback(indata, _frames, _time, status) -> None:
        if status:
            logger.warning("Audio status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio_q.put(_float32_to_pcm_bytes(mono.astype(np.float32)))

    logger.info(
        "Microphone streaming (backend=%s language=%s). Press Ctrl+C to stop.",
        config.asr_backend,
        config.language,
    )
    print("Listening...", flush=True)

    try:
        with sd.InputStream(
            samplerate=config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=stride,
            callback=callback,
        ):
            buffer = bytearray()
            while True:
                try:
                    block = audio_q.get(timeout=0.1)
                except queue.Empty:
                    result = session.feed_pcm(b"")
                    if result.partial:
                        print(result.partial, end="", flush=True)
                    if result.final:
                        print(f"\n[final] {result.final}", flush=True)
                    continue
                buffer.extend(block)
                while len(buffer) >= config.chunk_stride_bytes:
                    chunk = bytes(buffer[: config.chunk_stride_bytes])
                    del buffer[: config.chunk_stride_bytes]
                    result = session.feed_pcm(chunk)
                    if result.partial:
                        print(result.partial, end="", flush=True)
                    if result.final:
                        print(f"\n[final] {result.final}", flush=True)
                time.sleep(0.01)
    except KeyboardInterrupt:
        final = session.finalize()
        if final.final:
            print(f"\n[final] {final.final}", flush=True)
        print("\nStopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FunASR streaming CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("mic", help="Recognize from microphone")
    wav_parser = sub.add_parser("wav", help="Recognize from WAV file (streaming chunks)")
    wav_parser.add_argument("--path", required=True, help="Path to WAV file")

    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config) if args.config else None

    if args.command == "mic":
        return run_mic(config_path)
    if args.command == "wav":
        return run_wav(args.path, config_path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
