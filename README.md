<div align="center">

<img src="assets/lune-icon-512.png" alt="" width="75">

# Lune Voice

Local dictation for Linux and Windows.

[![CI](https://img.shields.io/github/actions/workflow/status/CoderDayton/lune-voice/ci.yml?branch=main&style=flat&label=CI)](https://github.com/CoderDayton/lune-voice/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPLv3-blue?style=flat)](LICENSE)

</div>

Speech-to-text runs on your machine with Whisper or Parakeet. A dictionary fixes names and jargon, and a formatter handles punctuation, capitals and filler words. You can also have an LLM restyle the text. That's off by default, and it works with a local model too.

```
mic → Whisper / Parakeet → dictionary + formatting → (styling) → paste
```

## Install

No release yet. For now, [build from source](#build).

Once there is one, grab the `.deb`, `.rpm`, `.AppImage` or Windows installer from [releases](https://github.com/CoderDayton/lune-voice/releases). The first launch downloads a speech model (about 540 MB for the default). After that it works offline.

## Privacy

Nothing leaves your machine unless you set up remote styling or a remote speech backend, and both stay off until you turn on "Allow remote services". Otherwise the only network calls are the model download, the optional GPU plugin download, and update checks you ask for. No telemetry.

## Build

You need Rust, [Bun](https://bun.sh), CMake, a C++ toolchain, libclang, the Tauri system deps and [`cgs`](https://crates.io/crates/cargo-script).

```bash
cargo install cargo-script
git clone https://github.com/CoderDayton/lune-voice
cd lune-voice
cgs dev       # run with hot reload
cgs package   # build the installers
```

The full requirements and build matrix are in [docs/RELEASE.md](docs/RELEASE.md). Run `cgs show` for every script.

## Docs

- [Architecture](docs/ARCHITECTURE.md): how the pipeline fits together
- [Configuration](docs/CONFIGURATION.md): every setting
- [GPU plugin](docs/GPU_PLUGIN.md): optional GPU acceleration (Vulkan: NVIDIA, AMD, Intel)
- [Performance](docs/PERFORMANCE.md): benchmarks
- [Runbook](docs/RUNBOOK.md): when something breaks, and where your data lives
- [Release](docs/RELEASE.md): packaging and releases

## License

GPL-3.0-or-later. Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
