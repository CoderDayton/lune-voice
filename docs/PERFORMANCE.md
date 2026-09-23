# Performance

> Latency profile and optimization techniques for the whisper.cpp pipeline (the styling LLM runs against a remote OpenAI-compatible endpoint). For a quick overview, see the [README](../README.md).

---

## Table of Contents

- [Latency Budget](#latency-budget)
- [Benchmark Status](#benchmark-status)
- [Optimization Techniques](#optimization-techniques)
  - [Memory & Allocation](#memory--allocation)
  - [Compute](#compute)
  - [Concurrency](#concurrency)
- [Pipeline Tuning](#pipeline-tuning)
- [Profiling Guide](#profiling-guide)

---

## Latency Budget

The user-perceived latency is the **after-speech wait**: time from when the user releases the hotkey (or presses it to stop) to when finished text lands in the focused app.

```
After-speech wait = poll_final (Silero endpoint trim + whisper.cpp decode) + optional LLM styling + delivery
```

| Stage | Cost profile | Notes |
|-------|-------------|-------|
| Audio flush | Negligible | 30 ms tail flush after hotkey release |
| whisper.cpp decode (`poll_final`) | **Primary fixed cost** | Batch decode of the full utterance; scales with audio length and model size |
| Standalone Silero VAD | Separate endpoint-trim stage | Whisper's VAD trims leading/trailing silence before decode and keeps internal pauses (a punctuation cue); NOT the in-decode `enable_vad` |
| LLM styling | Variable; skipped when no style preset is selected | One completion call to the configured remote OpenAI-compatible endpoint; plain async HTTP, so it does not stall the runtime |
| Delivery | Flat for a paste; ~8 ms per character when typed | Text over 80 characters, non-ASCII text and multi-line text are pasted (flat, ~100–360 ms). Shorter ASCII text is typed per key, which is faster at that length and leaves the clipboard untouched. On Windows, add 30 ms whenever focus has to be restored to the captured window first |

### What Does NOT Add to After-Speech Wait

- **Audio accumulation** runs concurrently with the user speaking.
- **Model load** happens once at startup and the context is shared across sessions — unless `model_idle_unload_minutes` is set, which frees the model after idle and pays the reload on the next hotkey press: ~160 ms with a warm page cache, ~750 ms cold, on `Medium`. Off by default.
- **Dictionary seed** is a pure in-process string pass; negligible.

### Bottleneck Analysis

| Rank | Bottleneck | Why it matters |
|------|------------|----------------|
| 1 | whisper.cpp decode time | Scales with utterance length and model size; dominates on CPU |
| 2 | LLM styling call | A few hundred milliseconds per utterance; skipped entirely when no style preset is selected |
| 3 | Memory footprint | whisper.cpp holds its model weights in VRAM/RAM; the remote styling LLM adds no local memory footprint |

**GPU impact:** a registered CUDA device moves ASR inference to the GPU and substantially reduces decode time. On Linux that comes from the downloadable GPU plugin ([docs/GPU_PLUGIN.md](GPU_PLUGIN.md)); a `whisper-cuda` build does the same thing statically. Without one, whisper.cpp uses its AVX2-optimized GGML kernels (portable baseline; compile flags: `GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON`). Plugin builds instead use `GGML_CPU_ALL_VARIANTS=ON`, which compiles per-microarchitecture CPU backends and picks the best one at runtime.

**Model size trade-off:** `Base` (~78 MB) is fastest and the choice for low-end hardware; `Small` (~190 MB) is next; `LargeV3Turbo` (~874 MB) provides the best accuracy with whisper's turbo decoder keeping latency closer to `Medium` (~539 MB) than to the full `Large` model.

---

## Benchmark Status

`cgs bench --env TYPE=<type>` runs the suite under `benchmarks/`:

| Type | Measures |
|---|---|
| `agc` | PCEN AGC frontend |
| `transcribe` | Whole capture → AGC → VAD → whisper (+ styling) pipeline on a WAV/f32/dir (`--env WAV=`) |
| `pipeline` / `stages` / `latency` | Real pipeline, per-stage timings, session latency |
| `domain` / `punctuation` / `dict` / `snippets` | Output-quality A/Bs |
| `styling` / `rulecmp` | Remote styling capability and rule comparison (endpoint via `LUNE_BENCH_LLM_*`, see `benchmarks/bench_utils.rs`) |
| `vram` | GPU memory use per model (`--env DEVICE=`, `CYCLES=`) |
| `modelload` | Cold and warm model load + unload timings (`--env REPEATS=`, `WHISPER_MODEL=`) — the source of the idle-unload figures above |
| `live` / `livewer` | Live microphone transcription, and its word error rate |
| `confusion` / `noisify` | Recognition confusion pairs, and synthetic noise robustness |

`LUNE_BENCH_BASELINE` and `LUNE_BENCH_TOLERANCE` gate a run against a recorded
baseline. Model-backed targets compile whisper.cpp and need a model on disk. Numbers are not checked into the repository — decode time swings by an order of magnitude across CPU, GPU, model size and utterance length, so run the targets on the hardware you care about, or use the profiling guide below.

---

## Optimization Techniques

### Memory & Allocation

| Technique | Location | Impact |
|-----------|----------|--------|
| **Global mimalloc allocator** | `desktop-ui/src-tauri/src/main.rs` | Replaces glibc malloc; faster small-alloc paths and lower fragmentation across the desktop shell and pipeline crates |
| **whisper.cpp audio buffer reuse** | `crates/lune-asr/src/backend/whisper_cpp.rs` | The per-session `Vec<f32>` audio buffer is pre-allocated for 30 s of audio (`Vec::with_capacity(16_000 * 30)`) and cleared without freeing after each `poll_final` — zero heap allocation during steady-state recording |
| **Corrector single-buffer output** | `crates/lune-dict/src/corrector.rs` | Builds the corrected text directly into a sized `String` instead of a `Vec<String>` + `join(" ")` |
| **Output buffer preallocation** | `crates/lune-llm/src/prompt.rs` | `fix_all_number_spaces` sizes its output buffer to the input length (`String::with_capacity(input.len())`) before filling |

### Compute

| Technique | Location | Impact |
|-----------|----------|--------|
| **O(1) risk lookup in corrector** | `crates/lune-dict/src/corrector.rs` | Risk positions are indexed in a `HashMap`, so each word costs an O(1) lookup instead of a linear scan over the risk list |
| **PCEN AGC frontend** | `crates/lune-audio/src/pcen.rs`, `crates/lune-voice/src/capture.rs` | Adaptive gain normalization applied to every captured frame through `MicAgc`, gated by the `vad_pcen` setting (default on); lifts a quiet mic toward a target RMS before the recognizer sees it |

### Concurrency

| Technique | Location | Impact |
|-----------|----------|--------|
| **Blocking-pool inference** | `crates/lune-asr/src/backend/whisper_cpp.rs` | Whisper decode runs via `block_in_place` (the `!Send` session can't `spawn_blocking`), so model execution never starves the async runtime or the audio capture loop |
| **Shared model, per-session state (ASR)** | `crates/lune-asr/src/backend/whisper_cpp.rs` | `WhisperContext` (model weights) is `Arc`-shared; `WhisperState` is per-session and allocates its own decode and KV-cache buffers on creation |
| **Plain async HTTP (LLM)** | `crates/lune-llm/src/backend/openai.rs` | The remote styling call is a single `async-openai` request on the async runtime — no local model, no blocking pool |

---

## Pipeline Tuning

These live in the `[pipeline]` section of `settings.toml`. For the full list of fields, see [Configuration → Pipeline Tuning](CONFIGURATION.md#pipeline-tuning).

| Parameter | Default | Effect | Tradeoff |
|-----------|---------|--------|----------|
| `max_speech_secs` | 300 (local Whisper), 120 (remote) | Auto-end the session after this many seconds of audio; longer utterances accumulate more audio for whisper to decode | Higher allows longer single dictations at the cost of a proportionally longer decode |
| `llm_timeout_secs` | 30 | Fall back to unstyled (dictionary-seeded) text if the styling call exceeds this | Lower bounds worst-case latency; too low will skip styling on slow CPU runs |

### Recommended Profiles

**Quick commands:**
```toml
[pipeline]
max_speech_secs = 30
```

**Long dictation:**
```toml
[pipeline]
max_speech_secs = 120
```

---

## Profiling Guide

### CPU profiling with perf

```bash
# Build with debug symbols
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release -p lune-ui --features "whisper-cpp"

# Record
perf record -g --call-graph dwarf ./target/release/lune-voice

# Analyze
perf report --hierarchy
```

### Flamegraph (samply)

`samply` needs no root, but it does need `perf_event_paranoid <= 1`:

```bash
cargo install samply
echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid   # reverts on reboot

# Build with symbols + frame pointers
RUSTFLAGS="-C force-frame-pointers=yes" \
CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
  cargo build --release --features "whisper-cpp" -p lune-ui

samply record -- ./target/release/lune-voice
# then trigger a dictation session; Ctrl+C to stop recording
```

For a statically linked GPU build, substitute `--features "whisper-cuda"`. To benchmark the plugin path instead, install the GPU plugin and run the CPU build — the backend is chosen at runtime.

### Tracing

Set `RUST_LOG` to see per-component timing:

```bash
RUST_LOG=lune_asr=debug,lune_llm=debug,lune_voice=debug ./target/release/lune-voice
```

Key trace points:
- `lune_asr::backend::whisper_cpp` — `poll_final` with `poll_final_ms`
- `lune_llm::backend::openai` — completion call timing
- `lune_voice::capture` — `asr: poll_final complete` with `poll_final_ms`

### Memory profiling

```bash
# With heaptrack
heaptrack ./target/release/lune-voice
heaptrack_gui heaptrack.lune-voice.*.gz

# With valgrind/DHAT
cargo build --release -p lune-ui --features "whisper-cpp"
valgrind --tool=dhat ./target/release/lune-voice
```

### Running tests

```bash
# Unit tests (no real models needed)
cgs test

# Real-model integration tests (requires models on disk)
cgs test-e2e
```
