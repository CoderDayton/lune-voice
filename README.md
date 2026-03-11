<div align="center">

<img src="assets/lune-audio-logo-s-512.png" width="180" alt="Lune Voice" />

# Lune Voice

**4× faster than cloud voice dictation. Fully offline. Built in Rust.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-cross--platform-lightgrey.svg)](#building)

*Speak → corrected text appears in your focused window. No cloud. No API keys. No telemetry.*

</div>

---

## Why Lune Voice?

Most voice tools either phone home or dump raw transcription with no cleanup. Lune gives you neither compromise:

- **Fully offline** — audio stays on your machine.
- **Fast local transcription** — optimized Rust pipeline with ONNX Runtime backends.
- **Cleaner final text** — optional local post-processing and dictionary fixes before delivery.
- **Desktop-first workflow** — CLI daemon plus a Tauri desktop UI.

---

## How It Works

- Capture microphone audio
- Run local ASR with Qwen3-ASR or Parakeet
- Apply dictionary and optional cleanup
- Deliver the final text to the active app

Technical details live in:

- [Architecture](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [Performance](docs/PERFORMANCE.md)

---

## Quick Start

```bash
# 1. Build
cargo build --release -p lune-app

# 2. Start daemon (downloads ~1.8GB of models on first launch)
./target/release/lune-voice daemon

# 3. Bind a hotkey, then dictate
lune-voice toggle
```

## Desktop UI

The repository also includes a Tauri desktop shell in [`desktop-ui/`](desktop-ui).

```bash
cd desktop-ui
bun install
bun run dev
```

Frontend verification:

```bash
bun run test
bun run test:visual
```

---

## Building

**Prerequisites:** Rust stable (2024 edition)

```bash
# CPU-only
cargo build --release -p lune-app

# With CUDA GPU acceleration
cargo build --release -p lune-app --features cuda

# Run tests
cargo test --workspace
```

Audio capture uses `cpal` — ALSA/PipeWire on Linux, CoreAudio on macOS, WASAPI on Windows.

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md): pipeline, session lifecycle, D-Bus surface, backend layout
- [Configuration](docs/CONFIGURATION.md): settings, backends, hotkeys, style presets, model files
- [Performance](docs/PERFORMANCE.md): benchmark workflow, latency breakdowns, tuning notes

---

## Usage

| Subcommand | Description |
|------------|-------------|
| `daemon`   | Start the background daemon |
| `toggle`   | Start if idle, stop if active |
| `start`    | Begin dictation |
| `stop`     | End and deliver text |
| `cancel`   | Discard current session |
| `status`   | Print daemon state |
| `settings` | View or change runtime settings |
| `quit`     | Shut down daemon |

**Recommended:**
```
Ctrl+Space  →  lune-voice toggle
Escape      →  lune-voice cancel
```

---

## Configuration

```bash
lune-voice settings style email
lune-voice settings language en-GB
lune-voice mode dictation Clipboard
```

For the full settings reference, backend options, and model file layout, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Architecture

The core system is a small Rust workspace with separate crates for audio capture, ASR, LLM cleanup, output delivery, persistence, and orchestration.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Performance

Benchmarks and profiling notes are documented in [docs/PERFORMANCE.md](docs/PERFORMANCE.md), including the example benchmark commands and backend-forcing options for Qwen and Parakeet.

---

## Contributing

1. Fork → feature branch → `cargo test --workspace`
2. Ensure `cargo fmt --all` and `cargo clippy --workspace` pass
3. Open a PR

Lefthook runs a stricter local gate automatically: pre-commit applies Rust formatting, then runs workspace `cargo check`, `cargo clippy`, Rust unit/bin tests, and the Bun desktop UI unit tests. Pre-push runs full-workspace `cargo clippy --all-features`, `cargo test --workspace`, the desktop UI production build, and optional real-model E2E tests when `LUNE_E2E_MODELS=1` is set.

---

## License

MIT — see [LICENSE](LICENSE).
