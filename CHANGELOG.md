# Changelog

All notable changes to Lune Voice are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- mimalloc as the global allocator in the desktop shell
- `start_minimized` setting to launch the app hidden to the system tray
- Wayland focused-window detection on COSMIC and wlr-foreign-toplevel compositors
- Descriptive dropdown component for settings selects
- Dictionary inline editing and priority control in the UI
- Correction spans for targeted LLM text editing
- 10-language corpus frequency maps for spell correction
- Categorized supplemental vocabulary with bincode embedding
- N-gram language model replacing the old bigram model
- Domain benchmark suite and RTF metrics
- `cargo-deny` configuration for license/advisory auditing
- Content Security Policy for Tauri webview
- `rustfmt.toml` and `clippy.toml` for consistent tooling

### Changed
- Reused mel scratch buffers in `WhisperMelF16` to amortize per-call allocations
- Shared a single softmax across HWCE entropy, Tsallis, and top-margin features
- Vectorized the ASR decoder's argmax softmax (~2.2× faster argmax, ~8% lower dictation latency, identical output)
- Expanded the pipeline-stages benchmark with per-stage timings, decode-loop op breakdown, and a profiling loop mode
- 4-bit KV-cache quantization rewritten with row-major min/max for SIMD-friendly access
- ORT LLM input vectors built into preallocated buffers
- Session statistics aggregated across the full database instead of capped history
- Style LLM swapped to Qwen3-0.6B with dynamic model registry
- Corrector restructured as LLM intelligence feed with token-level validator
- Benchmarks moved from `examples/` to `benches/` with `harness = false`
- License field updated from MIT to LicenseRef-Proprietary (LICENSE file already proprietary)
- ORT session builder optimized with refactored build_instructions
- Local LLM kept warm briefly after dictation to reduce cold-start latency

### Fixed
- Dict corrector risk lookup now O(1) per word (was O(words²) overall)
- Dictation startup latency, language detection, and hotkey state bugs
- Window lifecycle, button system, focus styles, and layout polish
- VAD init failure now propagates gracefully instead of panicking
- Mid-utterance pauses no longer fragment dictation into spurious sentences — the VAD caps the silence the recognizer hears, so a 0.5–1s breath isn't read as a sentence end
- Hotkey toggle, session, and IPC pipeline hardened against races and silent failures
- Qwen3 ASR model readiness and language parsing
- Gate false positives via case-insensitive lang detection and proper noun bypass
- Words containing digits skipped in spell corrector
- Clippy warnings in examples and benchmarks

### Security
- Updated rustls-webpki to 0.103.13
- Added SAFETY comments to all unsafe blocks
