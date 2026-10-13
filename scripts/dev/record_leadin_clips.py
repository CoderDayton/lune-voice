#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: GPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = ["sounddevice", "numpy"]
# ///
"""Record a list of short phrases in one take and split it into clips.

Read the phrases in PHRASES top to bottom, with a pause of about two seconds
between them. The take is cut at the pauses into NN.wav (16 kHz mono) + NN.txt
pairs for `cgs bench --env TYPE=leadin --env LUNE_BENCH_DIR=<out-dir>`.

Usage: uv run scripts/dev/record_leadin_clips.py <out-dir>
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

PHRASES = [
    "Hey can you send me that file",
    "Thanks for getting back to me",
    "Open the settings page",
    "I think we should ship it today",
    "Please rename this function",
    "Fix the sidebar spacing",
    "What time is the meeting",
    "Add a test for the parser",
    "Sure that works for me",
    "Delete the last paragraph",
    "Yes go ahead and commit",
    "The build is failing again",
    "Could you check the logs",
    "Make the button a bit bigger",
    "No keep the old version",
    "Let me know when you are done",
    "Run the benchmark one more time",
    "Why is this so slow",
    "Change the title to something shorter",
    "Okay that looks good to me",
]

RATE = 16_000
FRAME = 320  # 20 ms
CALIBRATION_S = 1.5
MIN_GAP_S = 0.9  # a shorter silence is a pause inside a phrase
MIN_PHRASE_S = 0.4
MARGIN_S = 0.8  # audio kept on each side of a phrase
END_SILENCE_S = 4.0
MAX_TAKE_S = 240.0


def notify(text: str) -> None:
    subprocess.run(["notify-send", "Lune lead-in test", text], check=False)


def phrase_spans(levels: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Frame spans of speech, joined across gaps shorter than MIN_GAP_S."""
    min_gap = int(MIN_GAP_S * RATE / FRAME)
    min_len = int(MIN_PHRASE_S * RATE / FRAME)
    spans: list[list[int]] = []
    for frame in np.flatnonzero(levels > threshold):
        if spans and frame - spans[-1][1] <= min_gap:
            spans[-1][1] = int(frame)
        else:
            spans.append([int(frame), int(frame)])
    return [(a, b) for a, b in spans if b - a >= min_len]


def frame_levels(audio: np.ndarray) -> np.ndarray:
    whole = audio[: len(audio) // FRAME * FRAME].reshape(-1, FRAME)
    return np.sqrt((whole**2).mean(axis=1))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []
    stream = sd.InputStream(
        samplerate=RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME * 5,
        callback=lambda data, *_: chunks.append(data[:, 0].copy()),
    )
    stream.start()
    sd.sleep(int(CALIBRATION_S * 1000))
    floor = float(np.median(frame_levels(np.concatenate(chunks))))
    threshold = max(floor * 5.0, 0.008)
    notify("Recording. Read the list now, two seconds between phrases.")

    elapsed = CALIBRATION_S
    spans: list[tuple[int, int]] = []
    while elapsed < MAX_TAKE_S:
        sd.sleep(250)
        elapsed += 0.25
        levels = frame_levels(np.concatenate(chunks))
        spans = phrase_spans(levels, threshold)
        quiet_for = (len(levels) - spans[-1][1]) * FRAME / RATE if spans else 0.0
        if len(spans) >= len(PHRASES) and quiet_for >= END_SILENCE_S:
            break
    stream.stop()
    stream.close()
    audio = np.concatenate(chunks)
    notify(f"Done. Heard {len(spans)} of {len(PHRASES)} phrases.")

    if len(spans) != len(PHRASES):
        print(f"heard {len(spans)} phrases, expected {len(PHRASES)}; nothing written")
        return 1
    margin = int(MARGIN_S * RATE)
    for i, ((a, b), phrase) in enumerate(zip(spans, PHRASES), start=1):
        start = max(a * FRAME - margin, 0)
        end = min((b + 1) * FRAME + margin, len(audio))
        pcm = (np.clip(audio[start:end], -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(out / f"{i:02d}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm.tobytes())
        (out / f"{i:02d}.txt").write_text(phrase + "\n")
    print(f"wrote {len(spans)} clips to {out} (threshold {threshold:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
