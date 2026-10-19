<div align="center">

<img src="assets/lune-icon-512.png" alt="" width="75">

# Lune Voice

**Open source dictation for Linux and Windows that runs on your own hardware.**

[![CI](https://img.shields.io/github/actions/workflow/status/CoderDayton/lune-voice/ci.yml?branch=main&style=flat&label=CI)](https://github.com/CoderDayton/lune-voice/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPLv3-blue?style=flat)](LICENSE)

</div>

---

Lune Voice turns speech into text without sending your audio anywhere. Hold a hotkey (or tap it to toggle), talk, and the text lands in whatever window you were using: an editor, an email, a chat box.

Transcription and correction run locally, so the app keeps working after you disconnect from the internet. It is free software under the GPLv3: read the code, change it, redistribute it, and anyone you pass it to gets those same freedoms.

## How it works

```
Mic → Whisper or Parakeet (speech-to-text + voice-activity trim) → Dictionary + formatting → [optional styling] → Paste
```

Whisper or Parakeet does the transcription on your machine. Whisper covers the most languages and comes in four sizes. Parakeet, NVIDIA's speech model, handles 25 European languages many times faster on a CPU and punctuates by itself. Your dictionary fixes the names, acronyms, and jargon they get wrong, and the formatter handles capitalization, punctuation, and filler words.

Styling is optional and off by default. Turn it on and an LLM rewrites the transcript in the tone you pick: casual, professional, technical, or one you write. Point it at a local Ollama or llama-server to keep everything on your machine, or at a hosted API if you don't want to run a model yourself.

First launch downloads the model you pick — about 540 MB for the default Whisper, 416 or 669 MB for Parakeet. After that, dictation needs no network access.

## Features

**Dictation pill.** A small floating overlay that shows your mic level and what the app is doing. You can stop or cancel without leaving the app you're typing into.

**Style presets.** "Professional" tightens loose phrasing. "Technical" keeps jargon and code formatting intact. You can write your own with a custom prompt and examples, or leave styling off and keep just the dictionary and punctuation.

**Custom dictionary.** Add the names, acronyms, and terms you use at work. Exact matches always apply. A near miss on a single word is fixed only when Whisper wasn't sure of it, and priority decides when two entries compete.

**Hold or toggle.** Hold the hotkey for a quick line, or toggle it for longer passages. The default is `Ctrl+Space`, and it works in any app.

**Searchable history.** Every session is saved with its transcript, timing, and the style it used.

## Privacy

On its own the app makes three kinds of network request: the speech model download on first launch, the CUDA plugin download if you ask for GPU acceleration, and the update check. The update check only runs when you ask for it from the tray menu or Settings, unless you turn on "Check for updates on launch" (off by default). Transcription, dictionary fixes, and formatting never touch the network.

Two settings can send your words out, and only if you choose them: remote styling, and a remote transcription backend instead of the on-device engines. Neither runs until you also turn on "Allow remote services", which is off by default. If you point either one at a local endpoint, your transcripts stay on your machine.

There is no telemetry and no analytics. The source is public, so you can check that yourself.

## Install

Builds go on the [releases page](https://github.com/CoderDayton/lune-voice/releases). Until the first release is out, [build from source](#building-from-source).

**Linux:** `.deb`, `.rpm`, or `.AppImage`:

```bash
sudo apt install ./LuneVoice_*_amd64.deb
```

**Windows:** run the `-setup.exe` installer. The built-in updater keeps it
current after that.

### Uninstall

Uninstalling leaves your data behind, including the downloaded models (up to
~2.8 GB). Delete these folders to remove that too:

- **Linux**: `~/.local/share/lune-voice/` (models, history) and `~/.config/lune-voice/` (settings)
- **Windows**: `%APPDATA%\lune-voice\`

The `.deb` and `.rpm` remove their udev rule themselves. The AppImage and the
portable archive install it to `/etc/udev/rules.d/70-lune-voice-input.rules`.
Remove it from Settings → Text delivery before you delete the app, or with
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
| **Typing** | **Linux**: IBus when it is your input method, otherwise `/dev/uinput`. The .deb and .rpm install the udev rule for typing and the hotkey; the AppImage and portable archive ask for a polkit password once to install it. `wl-copy` or `xclip` for the paste path. **Windows**: the clipboard plus a synthetic paste, with per-key typing as the fallback. Either way the text goes to the window you were in when you started speaking. |
| **Hotkey** | Toggle or push-to-talk on both platforms. Linux grabs the keyboard through evdev; Windows uses a low-level keyboard hook. |
| **GPU** | Optional, on Linux and Windows. If you have an NVIDIA card, the app offers a one-time download that moves transcription onto it. CPU-only works everywhere, just slower. |
| **Disk** | ~540 MB for the default Whisper model, ~1.7 GB for all four Whisper sizes, 416 or 669 MB for Parakeet |
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

GNU General Public License, version 3 or later. See [LICENSE](LICENSE) for the full text, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the native libraries, models, fonts and data it ships or downloads (including the modified `whisper-rs-sys` fork), and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the licences of every Rust crate and JavaScript package it includes.

If you distribute Lune Voice, modified or not, you have to make the source available to whoever you give it to, under these same terms.
