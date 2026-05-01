#!/usr/bin/env python3
"""Extract specific MLS samples from downloaded parquet files."""
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import io

CACHE = Path.home() / ".cache/huggingface/hub/datasets--facebook--multilingual_librispeech/snapshots/2e83e61823b4c47dcbcb1980bb88601274127609"
OUT = Path("test_fixtures")
OUT.mkdir(exist_ok=True)

TARGETS = {
    "french": ("french/train-00001-of-00034.parquet", "12541_10527_000152.opus"),
    "german": ("german/test-00000-of-00001.parquet", "4705_13109_000000.opus"),
    "spanish": ("spanish/train-00000-of-00030.parquet", "8882_10139_000005.opus"),
    "italian": ("italian/train-00000-of-00008.parquet", "6001_10011_000002.opus"),
    "portuguese": ("portuguese/test-00000-of-00001.parquet", "11247_10229_000001.opus"),
    "dutch": ("dutch/train-00002-of-00048.parquet", "2450_10565_001453.opus"),
}


def write_wav_16k(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    n = len(pcm16)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        f.write(pcm16.tobytes())


for lang, (parquet_rel, target_file) in TARGETS.items():
    print(f"\n── {lang} ──")
    pq_path = CACHE / parquet_rel
    if not pq_path.exists():
        print(f"  [skip] {pq_path} not found")
        continue

    df = pd.read_parquet(pq_path)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Find the target row by filename
    match = None
    for idx, row in df.iterrows():
        audio = row.get("audio", {})
        if isinstance(audio, dict):
            fname = audio.get("path", "")
            if target_file in fname:
                match = (idx, row)
                break

    if match is None:
        print(f"  [skip] {target_file} not found in parquet")
        # Show first few filenames for debugging
        for idx, row in df.head(3).iterrows():
            audio = row.get("audio", {})
            if isinstance(audio, dict):
                print(f"    sample: {audio.get('path', 'N/A')}")
        continue

    idx, row = match
    audio = row["audio"]
    transcript = row.get("transcript", row.get("text", ""))

    # Decode audio bytes
    audio_bytes = audio.get("bytes", None)
    if audio_bytes is None:
        print(f"  [skip] no audio bytes")
        continue

    pcm, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import scipy.signal
        pcm = scipy.signal.resample(pcm, int(len(pcm) * 16000 / sr)).astype(np.float32)

    dur = len(pcm) / 16000

    # Remove old voxpopuli fixtures
    for prefix in ["voxpopuli_", "mls_"]:
        old = OUT / f"{prefix}{lang}.wav"
        old_txt = OUT / f"{prefix}{lang}.txt"
        if old.exists():
            old.unlink()
            print(f"  removed {old}")
        if old_txt.exists():
            old_txt.unlink()
            print(f"  removed {old_txt}")

    wav_path = OUT / f"mls_{lang}.wav"
    txt_path = OUT / f"mls_{lang}.txt"
    write_wav_16k(wav_path, pcm)
    txt_path.write_text(transcript.strip())
    print(f"  {wav_path}: {dur:.1f}s, {len(pcm)} samples")
    print(f"  text: {transcript.strip()}")
