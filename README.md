<div align="center">

<img src="assets/lune-icon-512.png" alt="" width="75">

# Lune Voice

**Open source dictation for Linux and Windows that runs on your own hardware.**

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

Two settings can send your words out, and only once you pick them: remote styling, and a remote transcription backend in place of local Whisper. Neither runs until you also turn on "Allow remote services", which is off by default. Pointing either at a local endpoint keeps transcripts on your machine.

The app collects no telemetry and no analytics. Since the source is public, you do not have to take that on faith.

## Install

Grab the build for your platform from the [releases page](https://github.com/CoderDayton/lune-voice/releases).

**Linux** — `.deb`, `.rpm`, or `.AppImage`:

```bash
sudo apt install ./lune-voice_*_amd64.deb
```

**Windows** — run the `-setup.exe` installer. The in-app updater keeps it
current from there.

### Uninstall

Removing the app leaves your data behind, including the downloaded models (up
to ~1.6 GB). Delete these folders to remove it too:

- **Linux**: `~/.local/share/lune-voice/` (models, history) and `~/.config/lune-voice/` (settings)
- **Windows**: `%APPDATA%\lune-voice\`

The `.deb` and `.rpm` remove their udev rule themselves. The AppImage and the
portable archive install it to `/etc/udev/rules.d/99-lune-voice-uinput.rules`;
remove it from Settings → Text delivery before deleting the app, or with
`sudo rm` afterwards.

## Building from source

You need Rust, [Bun](https://bun.sh) 1.3.10 (pinned in `desktop-ui/package.json`), CMake, a C++ toolchain, LLVM with libclang, the Tauri WebView dependencies, and the `cgs` task runner that drives the build scripts. A CUDA toolkit is only needed if you want to build the GPU plugin yourself.

Some tasks need more:

- **Cold builds** need `LIBCLANG_PATH` set for bindgen, e.g. `export LIBCLANG_PATH=/lib/x86_64-linux-gnu` on Debian and Ubuntu.
- **`cgs package`** builds inside an `ubuntu:22.04` container, so it needs Docker (or another engine named in `CONTAINER_ENGINE`).
- **`cgs check-windows`** needs [zig](https://ziglang.org) and `binutils-mingw-w64-x86-64`.
- **`cgs licenses`** needs `cargo-about` 0.9.2 (`cargo install cargo-about --locked --features cli --version 0.9.2`) and an installed `desktop-ui/node_modules`. The version is pinned because the release build compares the generated file byte for byte.

```bash
cargo install cargo-script   # provides the `cgs` command
git clone https://github.com/CoderDayton/lune-voice
cd lune-voice
cgs dev      # run the app with hot reload
cgs build    # compile the desktop app
cgs package  # build the .deb / .rpm / .AppImage / .tar.gz installers
```

`cgs dev` and `cgs build` detect CUDA automatically and fall back to CPU. You can force either path with `--env BACKEND=cpu` or `--env BACKEND=cuda`. See [docs/RELEASE.md](docs/RELEASE.md) for the full build matrix, and run `cgs show` to list every available script.

| | |
|---|---|
| **OS** | Linux (X11 or Wayland) or Windows 10/11 |
| **Typing** | **Linux**: IBus when it is your input method, otherwise `/dev/uinput` — the first run asks for a polkit password to install a udev rule. `wl-copy` or `xclip` for the paste path. **Windows**: the clipboard plus a synthetic paste, with per-key typing as the fallback. Either way the text goes to the window you were in when you started speaking. |
| **Hotkey** | Toggle or push-to-talk on both platforms — Linux grabs the keyboard through evdev, Windows uses a low-level keyboard hook. |
| **GPU** | Optional, Linux only for now. If you have an NVIDIA card, the app offers a one-time download that moves transcription onto it. CPU-only works everywhere, just slower. |
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

GNU General Public License, version 3 or later. See [LICENSE](LICENSE) for the full text, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the native libraries, models, fonts and data it ships or downloads — including the modified `whisper-rs-sys` fork — and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the licences of every Rust crate and JavaScript package it includes.

If you distribute Lune Voice, modified or not, you have to make the source available to whoever you give it to, under these same terms.
