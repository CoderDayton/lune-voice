# Changelog

All notable changes to Lune Voice are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Parakeet, a second on-device speech engine: NVIDIA Parakeet TDT 0.6B v3, run by whisper.cpp's own `parakeet` library. Many times faster than Whisper on a CPU, punctuates and capitalizes by itself, and detects which of its 25 European languages is spoken. Pick it in Settings → Transcription or during setup, in a 669 MB (q8_0, recommended) or a 416 MB (q4_k) size; it shares the GPU plugin and the voice-activity model with Whisper and adds about 0.15 MB to the app. Parakeet cannot report which language it heard, so it has no Auto-detect: the language selector lists only its languages, switching to it picks your system language (or English) in the same step, and a note next to the engine explains why. The language list itself now covers all 25 European languages for Whisper too, and each is formatted by its own rules instead of falling back to English

### Fixed

- A plugin build with a missing or damaged `backends/` folder used to close the instant a model loaded. It now reports that no CPU backend is available and asks for a reinstall

## [0.1.0]

First release, for Linux and Windows.

### Dictation

- Hotkey-bounded dictation, toggle or push-to-talk hold, `Ctrl+Space` by default. The hotkey is global and the text lands in whatever window has focus
- whisper.cpp transcription on your own machine, in three model sizes
- Silero endpoint trim before decode: silence is cut off both ends and internal pauses are kept, because whisper reads them as punctuation
- PCEN automatic gain control on the capture path, so a quiet microphone still reaches the recognizer at a usable level
- Correction pass over the transcript — user dictionary (exact matches, plus single-word near misses when the recognizer was unsure), bundled vocabulary for capitalization, and a curated table of common mishears
- Formatting pass — capitalization, terminal punctuation, filler-word removal, and the quotes whisper wraps around a read-aloud utterance. Spaced digit groups that are not thousands separators, such as phone numbers, are left intact
- Optional styling through any OpenAI-compatible endpoint, local or hosted: five built-in presets, or your own with custom prompts and few-shot examples
- Voice commands: new line, new paragraph, list item, scratch that, stop dictation — each recognized only as a whole utterance, so the same words inside a sentence are dictated normally
- Text delivery through IBus when it is the active input method, otherwise a `uinput` virtual keyboard, with a clipboard paste path for non-ASCII. With **Copy to clipboard** off, the previous clipboard — text or images — is restored byte-for-byte. Multi-line text is never typed key by key, so newlines cannot press Enter in the focused app
- Remote OpenAI-compatible transcription as an alternative to local whisper
- The idle whisper model can be unloaded to give back its memory while the microphone stays warm, so the next dictation still starts immediately. Off by default; set `model_idle_unload_minutes` to turn it on

### Platforms

- **Linux** (X11 or Wayland): evdev hotkey grab, IBus or `uinput` delivery, and focused-window detection on COSMIC, `wlr-foreign-toplevel` and the major compositors. Ships as `.deb`, `.rpm`, `.AppImage` and a portable `.tar.gz`. The `/dev/uinput` udev rule ships with the `.deb` and `.rpm`
- **Windows** 10/11: low-level keyboard hook, clipboard paste with per-key typing as the fallback, and an NSIS installer. Delivery targets the window that was focused when you started speaking, and puts focus back there first if one of the app's own windows took it
- In-app updates for the `.deb`, `.rpm`, `.AppImage` and Windows installer. The portable `.tar.gz` announces a new version and links to the download instead of asking for root

### Interface

- Dictation pill overlay with live audio levels and session state: choose its monitor, anchor, size, orientation, corner style and edge margin, drag it anywhere, and lock it in place. It never takes keyboard focus, so clicking it cannot divert your text. Keeping it across workspaces works on Linux and macOS; Windows virtual desktops cannot be addressed per-window. Runs under XWayland on compositors without `wlr-layer-shell`
- The pill places itself correctly across monitors running at different scale factors, so a laptop panel next to a scaled external display puts it where you dropped it
- Onboarding that can be cancelled and resumed. A model download survives the window closing, and a reopened window re-attaches to it
- Searchable session history with per-session transcript, timing, and the style applied. Retention is configurable, including forever, and sessions can be deleted in bulk
- Dictionary management with inline editing and priority ranking
- Custom style presets, with generated few-shot examples
- Sound cues on start, stop, and error, with a volume slider and an off switch
- Sidebar quick actions and an in-app feedback relay
- Start minimized to the system tray
- Licence, warranty and source-code notice in Settings → General
- Localized interface (Lingui), keyboard-navigable with visible focus rings and WCAG AA contrast

### Performance

- GPU acceleration on Linux and Windows as an optional download rather than a separate build: ggml's Vulkan backend as a `dlopen`-able plugin (about 17 MB) that runs on NVIDIA, AMD and Intel GPUs, verified against a pinned SHA-256 and installed beside your settings. It is offered during onboarding and in Settings → Transcription. One build serves every machine, GPU or not
- mimalloc as the global allocator; thin LTO release builds
- Steady-state recording allocates nothing on the audio path, and the whisper model is loaded once and shared across sessions
- History statistics read a maintained word count, so the overview does not slow down as history grows

### Security and privacy

- `privacy_mode` is enforced: under the default `LocalOnly`, a remote transcription backend is refused with an actionable message and remote styling degrades to passthrough
- The transcription and styling endpoints require HTTPS for anything but a local address (loopback, private LAN, `.local`/`.localhost`)
- Transcript text is redacted from logs by default; `LUNE_LOG_TRANSCRIPTS=1` opts in while debugging recognition
- API keys are kept in the OS keyring where one is available. `settings.toml` and the session-history database, with its WAL/SHM sidecars, are created owner-only
- Transcript export writes through a validated Rust command; the webview has no file-write capability
- Model downloads are checked against a per-model integrity manifest, so a truncated file is caught even when the metadata fetch fails
- Dictated text carries the clipboard-history exclusion hint, so it stays out of clipboard managers
- Content Security Policy on the Tauri webview
- GPL-3.0-or-later, with `cargo deny` gating licenses and advisories across the dependency tree
- Every bundle ships `THIRD_PARTY_NOTICES.md` for the native libraries, models, fonts and data, and a generated `THIRD_PARTY_LICENSES.md` carrying the full licence text of every Rust crate and JavaScript package. The release build fails if the generated file is out of date
