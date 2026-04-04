<div align="center">

<img src="assets/lune-voice-logo-512.png" width="160" alt="Lune Voice" />

# Lune Voice

**Offline voice dictation for Linux. Speak, and clean text appears in your app.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)
[![Tauri](https://img.shields.io/badge/tauri-v2-blue.svg)](https://tauri.app)

</div>

---

Lune Voice runs a full speech-to-text pipeline on your machine — no cloud, no API keys, no telemetry. Audio goes in, corrected text comes out, delivered straight to whatever app you're focused on.

The pipeline: mic capture → VAD → ASR (Qwen3 or Parakeet, via ONNX Runtime) → confidence gating → optional LLM cleanup → dictionary fixes → paste into focused window. Everything runs locally. GPU optional but fast.

## Desktop App

A Tauri desktop shell with a floating dictation pill, session history, custom dictionary, and hotkey configuration. Built with React, Tailwind, and DaisyUI.

```bash
cd desktop-ui
bun install && bun run dev
```

The pill overlay sits at the bottom of your screen — click to dictate, watch the audio dots react, stop or cancel with one tap. It hides when you're not using it.

## CLI Daemon

For a headless setup or scripting:

```bash
cargo build --release -p lune-app
./target/release/lune-voice daemon      # downloads ~1.8 GB of models on first run
```

| Command | |
|---|---|
| `lune-voice toggle` | Start if idle, stop if active |
| `lune-voice start` | Begin dictation |
| `lune-voice stop` | End and deliver text |
| `lune-voice cancel` | Discard current session |
| `lune-voice status` | Print daemon state |
| `lune-voice quit` | Shut down |

Bind `Ctrl+Space` to `lune-voice toggle` and `Escape` to `lune-voice cancel` for a hands-free workflow. The desktop app does this automatically via evdev.

## Building

Rust stable (2024 edition), Bun for the frontend.

```bash
# CPU-only
cargo build --release -p lune-app

# With CUDA
cargo build --release -p lune-app --features cuda

# Desktop UI
cd desktop-ui && bun install && bun run build

# Tests
cargo test --workspace
bun test                    # frontend unit tests
```

Audio capture uses `cpal` — PipeWire/ALSA on Linux, CoreAudio on macOS, WASAPI on Windows.

## Project Layout

```
crates/
  lune-app/       Session orchestration, delivery, hotkey listener
  lune-asr/       ASR backends (Qwen3, Parakeet, OpenAI-compatible)
  lune-audio/     Mic capture, VAD, segmenter
  lune-core/      Shared types, config, session model
  lune-llm/       LLM cleanup (local GGUF or OpenAI-compatible)
  lune-commands/  Voice command router
  lune-dict/      Dictionary manager (fuzzy match + priority)
  lune-output/    Text delivery (wtype/wl-copy)
  lune-persist/   SQLite DB, settings store, model hub
desktop-ui/       Tauri + React desktop shell
```

## Docs

- [Architecture](docs/ARCHITECTURE.md) — pipeline, session lifecycle, D-Bus, backend layout
- [Configuration](docs/CONFIGURATION.md) — settings, backends, hotkeys, styles, models
- [Performance](docs/PERFORMANCE.md) — benchmarks, latency breakdowns, tuning

## License

MIT
