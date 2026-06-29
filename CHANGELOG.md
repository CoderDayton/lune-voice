# Changelog

All notable changes to Lune Voice are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Sound cues on dictation start, stop, and error, played from the backend, with an enable toggle and volume slider in Settings
- Separate **Copy to clipboard** toggle in Output settings, independent of the Type / Paste delivery method
- mimalloc as the global allocator in the desktop shell
- `start_minimized` setting to launch the app hidden to the system tray
- Wayland focused-window detection on COSMIC and wlr-foreign-toplevel compositors
- Descriptive dropdown component for settings selects
- Dictionary inline editing and priority control in the UI
- 10-language corpus frequency maps for spell correction
- Categorized supplemental vocabulary with bincode embedding
- N-gram language model replacing the old bigram model
- Domain benchmark suite and RTF metrics
- `cargo-deny` configuration for license/advisory auditing
- Content Security Policy for Tauri webview
- `rustfmt.toml` and `clippy.toml` for consistent tooling

### Changed
- Shared a single softmax across HWCE entropy, Tsallis, and top-margin features
- Session statistics aggregated across the full database instead of capped history
- Benchmarks moved from `examples/` to `benches/` with `harness = false`
- License field updated from MIT to LicenseRef-Proprietary (LICENSE file already proprietary)
- Local LLM kept warm briefly after dictation to reduce cold-start latency
- Output settings split into a delivery method (`TypeInPlace` / `Clipboard`) and an independent `copy_to_clipboard` toggle; the combined `Both` mode was removed (legacy `Both` configs migrate to `TypeInPlace` with copy enabled)
- Text delivery prefers the IBus input method when it is active — committing text directly with no clipboard — and falls back to the uinput keyboard path otherwise

### Fixed
- Dict corrector risk lookup now O(1) per word (was O(words²) overall)
- Dictation startup latency, language detection, and hotkey state bugs
- Window lifecycle, button system, focus styles, and layout polish
- VAD init failure now propagates gracefully instead of panicking
- Mid-utterance pauses no longer fragment dictation into spurious sentences — the VAD caps the silence the recognizer hears, so a 0.5–1s breath isn't read as a sentence end
- Filler words ("um", "uh", and the like) are stripped from dictation output, including mid-sentence (English only)
- Long sentences now get a terminal period even when the LLM is skipped (minimal style); one-word dictations stay bare
- Hotkey toggle, session, and IPC pipeline hardened against races and silent failures
- Gate false positives via case-insensitive lang detection and proper noun bypass
- Words containing digits skipped in spell corrector
- Clippy warnings in examples and benchmarks
- Wayland clipboard no longer clobbered by dictation: with **Copy to clipboard** off, the `Clipboard` paste path restores the previous clipboard (text or images) byte-for-byte after pasting
- Keyboard focus rings are visible again across cards, list rows, tabs, and navigation — a global focus reset had erased every component indicator; muted UI text raised to WCAG AA contrast
- Hotkey capture grabs only real keyboards, ignoring virtual and non-keyboard input devices
- Quietly-spoken words are no longer dropped from the transcript
- Trailing "though" / "although" / "whereas" continuations are joined to their sentence instead of split into a fragment
- Clauses are no longer over-split into separate sentences

### Removed
- ONNX Runtime (`ort`) and every ONNX model.
- The local Qwen3-ASR backend (superseded by whisper.cpp).
- The standalone streaming Silero VAD gate (whisper's built-in VAD replaces it).
- The LLM grammar-correction pass, calibrated edit gate, and the token sampler / KV-cache code — the styling LLM no longer corrects text (Whisper output is already punctuated).
- The full benchmark suite, which exercised the removed ONNX/Qwen architecture (preserved on the `archive/onnx-qwen` branch).

### Security
- Updated rustls-webpki to 0.103.13
- Added SAFETY comments to all unsafe blocks
