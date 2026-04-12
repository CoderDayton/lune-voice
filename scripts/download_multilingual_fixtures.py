#!/usr/bin/env python3
"""Download multilingual speech fixtures from VoxPopuli (EU Parliament).

Creates test_fixtures/voxpopuli_{lang}.wav + .txt pairs.
Audio is already 16kHz mono from VoxPopuli. Picks samples in the
10-30s range for realistic dictation length.

Usage:
    uv run python scripts/download_multilingual_fixtures.py
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


def write_wav_16k(path: Path, samples: np.ndarray) -> None:
    """Write 16-bit PCM WAV at 16kHz mono."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    n = len(pcm)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        f.write(pcm.tobytes())


def main() -> None:
    out_dir = Path("test_fixtures")
    out_dir.mkdir(exist_ok=True)

    langs = {
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "it": "italian",
        "pt": "portuguese",
    }

    for code, name in langs.items():
        print(f"\n── {name} ({code}) ──")
        ds = load_dataset(
            "facebook/voxpopuli", code, split="test", streaming=True
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        best = None
        best_dur = 0.0
        for i, sample in enumerate(ds):
            raw = sample["audio"]["bytes"]
            data, sr = sf.read(io.BytesIO(raw))
            dur = len(data) / sr
            if 12.0 <= dur <= 30.0 and sample.get("is_gold_transcript", True):
                best = (data, sr, sample["raw_text"].strip(), dur)
                break
            if dur > best_dur and dur <= 40.0:
                best = (data, sr, sample["raw_text"].strip(), dur)
                best_dur = dur
            if i > 300:
                break

        if best is None:
            print(f"  [skip] no suitable sample")
            continue

        data, sr, text, dur = best
        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Resample if needed (VoxPopuli is already 16kHz)
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(data) * ratio)
            idx = np.arange(new_len) / ratio
            idx = np.clip(idx, 0, len(data) - 1).astype(int)
            data = data[idx]

        wav_path = out_dir / f"voxpopuli_{name}.wav"
        txt_path = out_dir / f"voxpopuli_{name}.txt"

        write_wav_16k(wav_path, data.astype(np.float32))
        txt_path.write_text(text.upper(), encoding="utf-8")

        print(f"  {wav_path}: {dur:.1f}s")
        print(f"  transcript: {text[:100]}...")
        print(f"  ✓ done")


if __name__ == "__main__":
    main()
