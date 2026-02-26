<div align="center">

# 🌙 Lune Voice

**Privacy-first voice dictation. Everything runs on your machine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-cross--platform-lightgrey.svg)](#building)

*Speak naturally → corrected text appears in your focused window — no cloud, no API keys, no telemetry.*

</div>

---

## Table of Contents

- [Why Lune Voice?](#why-lune-voice)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Building](#building)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## Why Lune Voice?

Most voice dictation tools either require a cloud subscription or produce raw transcription with no cleanup. Lune Voice does both locally:

- **100% offline** — Qwen3-ASR 0.6B transcribes audio; Qwen3 0.6B corrects it. Both run via ONNX Runtime. Neither leaves your device.
- **LLM-polished output** — filler words, grammar errors, and punctuation issues are fixed before text reaches you. You only ever see corrected output.
- **Smart batching** — low-confidence fragments accumulate until there's enough context for a quality correction. Short phrases return in ~300ms; paragraphs scale proportionally.
- **Zero configuration** — models download automatically on first run from HuggingFace. Bind a hotkey and you're done.
- **Style-aware** — switch between General, Email, Slack, Meeting Notes, and Code Comments presets without restarting.

---

## How It Works

```
Microphone
    │
    ▼
Audio Capture (cpal, 16kHz mono)
    │
    ▼
Qwen3-ASR 0.6B ──── mel-spectrogram (128-mel, Slaney) ──► ONNX Runtime
    │                                                               │
    │                                                   streaming decoder
    │                                                     + KV cache
    ▼
Confidence Router ── high (≥0.82): queue direct ───────────────────┐
    │                                                                │
    ├── mid (0.50–0.82): accumulate → 12 words → batch to LLM      │
    │                                                                │
    └── low (<0.50):     accumulate →  8 words → batch to LLM      │
                                                    │                │
                                                    ▼                ▼
                                              Qwen3 0.6B (ONNX, ORT)
                                                    │
                                             ChatML + few-shot
                                              greedy decoding
                                                    │
                                                    ▼
                                             Corrected Text
                                                    │
                                       ┌────────────┴───────────┐
                                       ▼                        ▼
                                 TypeInPlace                 Clipboard
```

All text is **held** until the LLM processes it — no raw ASR output ever reaches the user.

> **Deep dive:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate graph, session lifecycle, hold-delivery model, threading model, data flow diagrams.

---

## Quick Start

```bash
# 1. Build
cargo build --release -p lune-app

# 2. Start daemon (downloads ~500MB of models on first launch)
./target/release/lune-voice daemon

# 3. Bind a hotkey in your OS settings, then dictate
lune-voice toggle
```

---

## Building

**Prerequisites:** Rust (stable, 2024 edition)

```bash
# CPU-only
cargo build --release -p lune-app

# With CUDA GPU acceleration
cargo build --release -p lune-app --features cuda

# Run tests
cargo test --workspace
```

**Platform notes:** Audio capture uses `cpal` (ALSA/PipeWire on Linux, CoreAudio on macOS, WASAPI on Windows). D-Bus integration is Linux-only; other platforms use direct CLI subcommands.

---

## Usage

```
lune-voice [SUBCOMMAND]
```

| Subcommand | Description |
|------------|-------------|
| `daemon`   | Start the background daemon *(default)* |
| `toggle`   | Start if idle, stop if active |
| `start`    | Begin dictation (hold-to-talk press) |
| `stop`     | End dictation and deliver text |
| `cancel`   | Discard current session |
| `status`   | Print daemon state |
| `mode`     | Switch dictation/activation mode |
| `settings` | View or change runtime settings |
| `quit`     | Shut down daemon |

**Recommended hotkey setup:**

```
Ctrl+Space  →  lune-voice toggle
Escape      →  lune-voice cancel
```

---

## Configuration

Settings are stored as TOML at your platform's config directory.

```toml
[asr_backend]
type = "OrtQwenAsr"
use_gpu = true

[llm_backend]
type = "OrtLocal"
use_gpu = true
n_ctx = 2048
max_output_tokens = 256

dictation_mode = "TypeInPlace"   # TypeInPlace | Clipboard | Both
activation_mode = "Toggle"       # Toggle | Hold
default_style = "general"        # general | email | slack | meeting_notes | code_comments
privacy_mode = "LocalOnly"       # LocalOnly | LocalPreferred | AllowRemote
```

All settings are mutable at runtime via D-Bus (no daemon restart required for most changes):

```bash
lune-voice settings                     # dump all settings as TOML
lune-voice settings style email         # switch style preset
lune-voice settings language en-GB      # change language (BCP-47 validated)
lune-voice mode dictation Clipboard     # switch delivery mode
```

ASR/LLM backend changes are persisted immediately but take effect on daemon restart (model loading is a startup operation).

> **Full reference:** [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — all settings, remote backends, style presets, dictionary, session history, confidence router tuning, model management.

---

## Architecture

Nine crates with single responsibilities:

| Crate | Role |
|-------|------|
| `lune-app` | Session orchestration, confidence routing, LLM worker, D-Bus daemon |
| `lune-asr` | Qwen3-ASR backend (ONNX Runtime), Parakeet TDT backend, mel-spectrogram extraction |
| `lune-llm` | Qwen3 LLM backend (ONNX Runtime), ChatML prompting, stop-sequence detection |
| `lune-audio` | Microphone capture (cpal) |
| `lune-core` | Shared traits, config, error types |
| `lune-commands` | Voice command detection |
| `lune-dict` | User dictionary with substitution rules |
| `lune-output` | Text delivery with incremental diff |
| `lune-persist` | SQLite history, settings, HuggingFace model downloads |

Models downloaded automatically on first run:

| Model | Format | Size |
|-------|--------|------|
| [Qwen3-ASR 0.6B](https://huggingface.co/lune-voice/qwen3-asr-0.6b-onnx) | ONNX | ~1.2 GB |
| [Qwen3 0.6B](https://huggingface.co/lune-voice/qwen3-0.6b-onnx) | ONNX | ~600 MB |

> **Deep dive:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline walkthrough, hold-delivery model, session lifecycle, D-Bus interface, threading model, data flow diagrams.

---

## Performance

Mel-spectrogram extraction runs at **5000× real-time** — the bottleneck is TDT inference (~100ms) and LLM correction (500ms–2s depending on length).

```bash
# Pipeline latency benchmark
cargo run --example pipeline_latency -p lune-app --release

# Mel-spectrogram throughput benchmark
cargo run --example mel_throughput -p lune-asr --release
```

| Component | Latency | Notes |
|-----------|---------|-------|
| Mel extraction | 0.06 ms | 0.02% of inference budget |
| ASR inference | ~100 ms | Parakeet TDT 0.6B, INT8, CPU |
| Silence timeout | 800 ms | Configurable (biggest tuning knob) |
| LLM correction | 500–2000 ms | Proportional to input length |

> **Deep dive:** [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — latency budgets, benchmark walkthroughs, optimization techniques (memory, compute, concurrency), tuning profiles, profiling guide.

---

## Contributing

1. Fork and create a feature branch
2. Make changes with tests (`cargo test --workspace`)
3. Ensure `cargo fmt --all` and `cargo clippy --workspace` pass
4. Open a pull request

Pre-commit hooks run `fmt`, `clippy`, and `check` automatically via [lefthook](https://github.com/evilmartians/lefthook).

---

## License

MIT — see [LICENSE](LICENSE).
