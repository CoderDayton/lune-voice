<div align="center">

<img src="https://raw.githubusercontent.com/CoderDayton/lune-voice/main/assets/lune-icon-512.png" alt="" width="75">

# Lune Voice

**Open source dictation for Linux that runs on your own hardware.**

[![Release](https://img.shields.io/github/v/release/CoderDayton/lune-voice?style=flat&label=release)](https://github.com/CoderDayton/lune-voice/releases/latest)
[![CI](https://img.shields.io/github/actions/workflow/status/CoderDayton/lune-voice/ci.yml?branch=main&style=flat&label=CI)](https://github.com/CoderDayton/lune-voice/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/github/downloads/CoderDayton/lune-voice/total?style=flat&label=downloads)](https://github.com/CoderDayton/lune-voice/releases)
[![License](https://img.shields.io/badge/license-GPLv3-blue?style=flat)](LICENSE)

</div>

---

Lune Voice converts speech to text without sending your audio anywhere. You hold a hotkey (or toggle one), speak, and the transcribed text is inserted into whatever window has focus: an editor, an email draft, a chat box.

Transcription and correction run locally, so the app keeps working after you disconnect from the internet. It is free software under the GPLv3: read the code, change it, redistribute it, and anyone you pass it to gets those same freedoms.

## How it works

```
Mic → Whisper (speech-to-text + voice-activity trim) → Dictionary + formatting → [optional styling] → Paste
```

Whisper handles transcription on your machine. Your dictionary corrects the names, acronyms, and terminology it would otherwise get wrong, and the formatter applies capitalization, punctuation, and filler removal.

Styling is optional and off by default. When you enable it, an LLM rewrites the transcript in a tone you select: casual, professional, technical, or one you write yourself. Point it at a local Ollama or llama-server instance to keep transcripts on your machine, or at a hosted API if you would rather not run a model.

First launch downloads about 540 MB of Whisper models. After that, dictation needs no network access.

## Features

**Dictation pill.** A small floating overlay showing audio levels and session state. You can stop or cancel a session without switching away from the application you are working in.

**Style presets.** "Professional" tightens up loose phrasing. "Technical" preserves jargon and code formatting. You can define your own with custom prompts and few-shot examples, or leave styling off and keep only the dictionary and punctuation passes.

**Custom dictionary.** Add the names, acronyms, and terms specific to your work. Exact matches always apply, close single-word misses are caught when the recognizer was unsure of the word, and priority ranking decides between competing entries.

**Hold or toggle.** Hold the hotkey for short corrections, or toggle it for longer passages. `Ctrl+Space` by default, global, and works in any application.

**Searchable history.** Every session is recorded with its transcript, timing, and the style applied.

## Privacy

On its own the app makes three kinds of network request: the Whisper model download on first launch, the CUDA plugin download if you ask for GPU acceleration, and the update check. The update check runs only when you ask for it — from the tray menu or Settings — unless you turn on "Check for updates on launch", which is off by default. Local transcription, dictionary correction, and formatting never reach the network.

Two settings can send your words out, and only once you pick them: remote styling, and a remote transcription backend in place of local Whisper. Pointing either at a local endpoint keeps transcripts on your machine.

The app collects no telemetry and no analytics. Since the source is public, you do not have to take that on faith.

## Install

Download the `.deb`, `.rpm`, or `.AppImage` from the [releases page](https://github.com/CoderDayton/lune-voice/releases).

```bash
sudo apt install ./lune-voice_*_amd64.deb
```

## Building from source

You need Rust, [Bun](https://bun.sh), CMake, LLVM, the Tauri WebView dependencies, and the `cgs` task runner that drives the build scripts. A CUDA toolkit is only needed if you want to build the GPU plugin yourself.

```bash
cargo install cargo-run   # provides the `cgs` command
git clone https://github.com/CoderDayton/lune-voice
cd lune-voice
cgs dev      # run the app with hot reload
cgs build    # build a desktop bundle
```

Both commands detect CUDA automatically and fall back to CPU. You can force either path with `--env BACKEND=cpu` or `--env BACKEND=cuda`. See [docs/RELEASE.md](docs/RELEASE.md) for the full build matrix, and run `cgs show` to list every available script.

| | |
|---|---|
| **OS** | Linux (X11 or Wayland) |
| **Typing** | IBus when it is your input method, otherwise `/dev/uinput` — the first run asks for a polkit password to install a udev rule. `wl-copy` or `xclip` for the paste path. |
| **GPU** | Optional. If you have an NVIDIA card, the app offers a one-time download that moves transcription onto it. CPU-only works, just slower. |
| **Disk** | ~540 MB for the default model, ~1.6 GB for all three |
| **Internet** | Once, for the model download |

### GPU acceleration

There is no separate GPU build to pick. The app detects an NVIDIA driver and offers to download the CUDA backend, which installs alongside your settings rather than into the application. You need a driver new enough for CUDA 13 (r580 or later); the toolkit itself is not required. See [docs/GPU_PLUGIN.md](docs/GPU_PLUGIN.md).

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: pipeline internals, session lifecycle, threading model
- **[Configuration](docs/CONFIGURATION.md)**: settings, backends, hotkeys, style presets
- **[Performance](docs/PERFORMANCE.md)**: benchmarks, latency breakdowns, tuning
- **[GPU plugin](docs/GPU_PLUGIN.md)**: how CUDA support is detected, downloaded, and published
- **[Release](docs/RELEASE.md)**: build matrix, packaging, rollback, third-party notices
- **[Runbook](docs/RUNBOOK.md)**: common failure modes and how to recover

## License

GNU General Public License, version 3 or later. See [LICENSE](LICENSE) for the full text, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream projects bundled into the binary.

If you distribute Lune Voice, modified or not, you have to make the source available to whoever you give it to, under these same terms.
