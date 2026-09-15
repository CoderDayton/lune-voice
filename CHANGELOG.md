# Changelog

All notable changes to Lune Voice are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
- The idle whisper model is unloaded to give back its memory while the microphone stays warm, so the next dictation still starts immediately

### Platforms

- **Linux** (X11 or Wayland): evdev hotkey grab, IBus or `uinput` delivery, and focused-window detection on COSMIC, `wlr-foreign-toplevel` and the major compositors. Ships as `.deb`, `.rpm`, `.AppImage` and a portable `.tar.gz`. The `/dev/uinput` udev rule ships with the `.deb` and `.rpm`
- **Windows** 10/11: low-level keyboard hook, clipboard paste with per-key typing as the fallback, and an NSIS installer
- In-app updates for the `.deb`, `.rpm`, `.AppImage` and Windows installer. The portable `.tar.gz` announces a new version and links to the download instead of asking for root

### Interface

- Dictation pill overlay with live audio levels and session state: choose its monitor, anchor, size, orientation, corner style and edge margin, drag it anywhere, lock it in place, and keep it across workspaces. Runs under XWayland on compositors without `wlr-layer-shell`
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

- GPU acceleration on Linux as an optional download rather than a separate build: the CUDA backend is a `dlopen`-able ggml plugin, verified against a pinned SHA-256 and installed beside your settings. It is offered during onboarding and in Settings → Models. One Linux build serves every machine, including those with no NVIDIA driver at all
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
