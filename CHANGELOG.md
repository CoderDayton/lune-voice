# Changelog

All notable changes to Lune Voice are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Windows support. The global hotkey, text delivery, focused-window detection and the pill overlay each have a Windows half behind the same interface as the Linux one, and CI builds and tests the whole workspace on `windows-latest` alongside Linux
- In-app updates, and setup that can be cancelled and resumed. A model download survives the window closing and a reopened window re-attaches to it
- History retention can be set to keep sessions forever, and sessions can be deleted in bulk
- The idle whisper model is unloaded to give back its memory while the microphone stays warm, so the next dictation still starts immediately
- GPU setup is offered during onboarding as well as in Settings → Models
- Sidebar quick actions and an in-app feedback relay
- The `/dev/uinput` udev rule ships with the `.deb` and `.rpm` instead of only being installable from the app

### Changed

- The overview is one instrument pane rather than a grid of cards, and the list pages (history, dictionary, snippets, styles) share one card and row chrome
- The snippet composer unfolds out of the New snippet bar instead of opening a separate form

### Security

- `privacy_mode` is enforced instead of merely stored: under the default `LocalOnly`, a remote transcription backend is refused with an actionable message and remote styling degrades to passthrough
- The ASR and styling LLM endpoints both enforce HTTPS for anything but a local address (loopback, private LAN, `.local`/`.localhost`) — a plaintext public endpoint would otherwise send the API key and dictated text in cleartext. A refused LLM backend degrades to passthrough styling rather than failing startup
- The session-history database (full dictated transcripts) is now created owner-only, with the same 0600 file permissions as `settings.toml`; the process umask is hardened so its WAL/SHM sidecars are protected too
- Transcript export now writes through a Rust command (absolute-path, no-`..` validated) instead of the webview `fs` plugin; the broad `fs:write-files` capability is removed
- The pre-formatted ASR `Authorization` header is held in a zeroizing `RedactedString` instead of a plain `String`
- Clipboard MIME types (remote-controlled data) are validated against a well-formed MIME shape before being passed to `wl-paste`/`wl-copy`
- The CUDA backend plugin archive's SHA-256 is now pinned for the `gpu-cuda13-v1` release, so the GPU install is actually enabled end to end

### Performance

- History statistics read a maintained `word_count` column instead of loading every transcript, so the overview no longer slows down as history grows
- Whisper decode is documented as `block_in_place` on a multi-thread runtime (the app's actual configuration), and a decode on a current-thread runtime now warns loudly instead of silently stalling the async runtime

### Fixed

- When the clipboard is unavailable, multi-line text is no longer typed key by key — the newlines pressed Enter into the focused app. The dictation is dropped and reported as undelivered instead
- A dictation that failed while the UI was busy is reported as failed rather than as a clean stop
- Transcript timestamps are absolute again when silence is trimmed from the start of an utterance
- Concurrent settings saves apply in order, so a fast second change is no longer reverted by the first
- The styling pass no longer collapses spaced digit groups that are not thousands separators — phone numbers and part/ID numbers like `555 123 4567`, and dialling prefixes like `0 800 123 456`, are left intact, while `1 000 000` still collapses to `1000000`
- `input:`/`output:` in the middle of a styled transcript are no longer misread as the model echoing the few-shot labels (only a line-starting label is treated as an echo), so legitimate text like "I said input: the password" survives styling
- Reasoning blocks in the middle of a styled answer (`<think>`/`<thinking>`/`<reasoning>`/`<thought>`) are now stripped, not just leading ones, and cutting one out no longer leaves a doubled space behind
- The Linux hotkey no longer strands keys down. The evdev grab releases whatever it was holding when it lets go, a modifier you are still holding is not reported as released, and the release sweep stands down when the kernel cannot say which keys are actually down
- A slow or chatty display-server probe can no longer stall or lose a dictation: every probe is time-bounded, both child pipes are drained for the whole wait (a `wl-paste` on an image used to be killed mid-answer), and the clipboard path's drain join is bounded too
- Focused-window detection works on multi-monitor Wayland sessions where one of the activated toplevels has no title
- The microphone picker lists one entry per device instead of one per ALSA route
- Switching whisper models releases the outgoing model before loading its replacement, so the two are never resident at once
- Turning idle model unload off is honoured, and an update check disabled in settings is shown as a notice rather than silently doing nothing
- Downloads are no longer offered for artifacts the release repository cannot serve
- The window is sized from the largest monitor on Wayland, where the compositor does not report the focused output
- Strings the interface still rendered in English are translated, including the Large v3 Turbo model label and the snippet screens

## [0.1.0]

First release.

### Dictation

- Hotkey-bounded dictation, toggle or push-to-talk hold, `Ctrl+Space` by default. The hotkey is global and the text lands in whatever window has focus
- whisper.cpp transcription on your own machine, in three model sizes
- Silero endpoint trim before decode: silence is cut off both ends and internal pauses are kept, because whisper reads them as punctuation
- PCEN automatic gain control on the capture path, so a quiet microphone still reaches the recognizer at a usable level
- Correction pass over the transcript — user dictionary (exact matches, plus single-word near misses when the recognizer was unsure), bundled vocabulary for capitalization, and a curated table of common mishears
- Formatting pass — capitalization, terminal punctuation, filler-word removal, and the quotes whisper wraps around a read-aloud utterance
- Optional styling through any OpenAI-compatible endpoint, local or hosted: five built-in presets, or your own with custom prompts and few-shot examples
- Voice commands: new line, new paragraph, list item, scratch that, stop dictation — each recognized only as a whole utterance, so the same words inside a sentence are dictated normally
- Text delivery through IBus when it is the active input method, otherwise a `uinput` virtual keyboard, with a clipboard paste path for non-ASCII. With **Copy to clipboard** off, the previous clipboard — text or images — is restored byte-for-byte
- Remote OpenAI-compatible transcription as an alternative to local whisper

### Interface

- Dictation pill overlay with live audio levels and session state: choose its monitor, anchor, size, orientation, corner style and edge margin, drag it anywhere, lock it in place, and keep it across workspaces. Runs under XWayland on compositors without `wlr-layer-shell`
- Searchable session history with per-session transcript, timing, and the style applied
- Dictionary management with inline editing and priority ranking
- Custom style presets, with generated few-shot examples
- Sound cues on start, stop, and error, with a volume slider and an off switch
- Wayland focused-window detection on COSMIC and `wlr-foreign-toplevel` compositors
- Start minimized to the system tray
- Localized interface (Lingui), keyboard-navigable with visible focus rings and WCAG AA contrast

### Performance

- GPU acceleration as an optional download rather than a separate build: the CUDA backend is a `dlopen`-able ggml plugin, verified against a pinned SHA-256 and installed beside your settings. One Linux build serves every machine, including those with no NVIDIA driver at all
- mimalloc as the global allocator; thin LTO release builds
- Steady-state recording allocates nothing on the audio path, and the whisper model is loaded once and shared across sessions

### Security and privacy

- Transcript text is redacted from logs by default; `LUNE_LOG_TRANSCRIPTS=1` opts in while debugging recognition
- API keys are kept in the OS keyring where one is available, and `settings.toml` is created owner-only for the fallback case
- Model downloads are checked against a per-model integrity manifest, so a truncated file is caught even when the metadata fetch fails
- Dictated text carries the clipboard-history exclusion hint, so it stays out of clipboard managers
- Content Security Policy on the Tauri webview
- GPL-3.0-or-later, with `cargo deny` gating licenses and advisories across the dependency tree
