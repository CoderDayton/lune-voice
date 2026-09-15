#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: GPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = ["sounddevice", "numpy"]
# ///
"""Record default mic to a 16kHz mono float32 WAV file.

Usage: uv run scripts/dev/record_mic.py [--output-file FILE] [--seconds N]
  Press Ctrl+C to stop early.
"""
import argparse
import signal
import sys
import wave
import struct

import sounddevice as sd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Record mic to WAV")
    default_out = f"recording_{int(__import__('time').time())}.wav"
    parser.add_argument("--output-file", "-o", default=default_out)
    parser.add_argument("--seconds", "-s", type=float, default=120,
                        help="Max recording length (default: 60s, Ctrl+C to stop early)")
    args = parser.parse_args()

    sr = 16000
    channels = 1
    print(f"Recording to {args.output_file} (16kHz mono float32, max {args.seconds}s)")
    print("Press Ctrl+C to stop.\n")

    frames = []
    stopped = False

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"  [warn] {status}", file=sys.stderr)
        frames.append(indata[:, 0].copy())

    def on_sigint(sig, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, on_sigint)

    stream = sd.InputStream(samplerate=sr, channels=channels, dtype="float32",
                            blocksize=1600, callback=callback)
    stream.start()

    elapsed = 0.0
    try:
        while not stopped and elapsed < args.seconds:
            sd.sleep(100)
            elapsed += 0.1
            secs = int(elapsed)
            print(f"\r  {secs}s recorded...", end="", flush=True)
    finally:
        stream.stop()
        stream.close()

    print(f"\n\nStopped. Writing {elapsed:.1f}s to {args.output_file}")

    audio = np.concatenate(frames) if frames else np.array([], dtype=np.float32)

    # Write float32 WAV (format code 3 = IEEE float)
    with wave.open(args.output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(sr)
        # wave module writes PCM by default; we need to patch the format tag
        wf.writeframes(b"")  # init headers

    # Rewrite with correct float32 format tag
    with open(args.output_file, "rb") as f:
        data = bytearray(f.read())

    raw = audio.astype(np.float32).tobytes()
    total_size = 36 + len(raw)

    out = bytearray()
    out += b"RIFF"
    out += struct.pack("<I", total_size)
    out += b"WAVE"
    out += b"fmt "
    out += struct.pack("<I", 16)        # chunk size
    out += struct.pack("<H", 3)         # format = IEEE float
    out += struct.pack("<H", 1)         # channels
    out += struct.pack("<I", sr)        # sample rate
    out += struct.pack("<I", sr * 4)    # byte rate
    out += struct.pack("<H", 4)         # block align
    out += struct.pack("<H", 32)        # bits per sample
    out += b"data"
    out += struct.pack("<I", len(raw))
    out += raw

    with open(args.output_file, "wb") as f:
        f.write(out)

    print(f"Done: {len(audio)} samples, {len(audio)/sr:.1f}s")


if __name__ == "__main__":
    main()
