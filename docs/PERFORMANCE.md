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
After-speech wait = poll_final (whisper.cpp decode + built-in VAD) + optional LLM styling + delivery
```

| Stage | Cost profile | Notes |
|-------|-------------|-------|
| Audio flush | Negligible | 30 ms tail flush after hotkey release |
| whisper.cpp decode (`poll_final`) | **Primary fixed cost** | Batch decode of the full utterance; scales with audio length and model size |
| Standalone Silero VAD | Separate endpoint-trim stage | Whisper's VAD trims leading/trailing silence before decode and keeps internal pauses (a punctuation cue); NOT the in-decode `enable_vad` |
| LLM styling | Variable; skipped when no style preset is selected | One completion call to the configured remote OpenAI-compatible endpoint; plain async HTTP, so it does not stall the runtime |
| Delivery | Small | Dominated by decode/styling |

### What Does NOT Add to After-Speech Wait

- **Audio accumulation** runs concurrently with the user speaking.
- **Model load** happens once at startup; shared context is reused across sessions.
- **Dictionary seed** is a pure in-process string pass; negligible.

### Bottleneck Analysis

| Rank | Bottleneck | Why it matters |
|------|------------|----------------|
| 1 | whisper.cpp decode time | Scales with utterance length and model size; dominates on CPU |
| 2 | LLM styling call | A few hundred milliseconds per utterance; skipped entirely when no style preset is selected |
| 3 | Memory footprint | whisper.cpp holds its model weights in VRAM/RAM; the remote styling LLM adds no local memory footprint |

**GPU impact:** a CUDA build (`whisper-cuda`) moves ASR inference to the GPU and substantially reduces decode time. CPU builds use whisper.cpp's AVX2-optimized GGML kernels (portable baseline; compile flags: `GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON`).

**Model size trade-off:** `Small` (~264 MB) is fastest; `LargeV3Turbo` (~874 MB) provides the best accuracy with whisper's turbo decoder keeping latency closer to `Medium` (~823 MB) than to the full `Large` model.

---

## Benchmark Status

The benchmark suite for the whisper.cpp stack is **pending** (the styling-LLM benchmarks talk to a remote endpoint configured via `LUNE_BENCH_LLM_*` env vars — see `benchmarks/bench_utils.rs`). The prior suite (`pipeline_stages`, `pipeline_real`, and related examples) was written against the removed Qwen3-ASR ONNX backend and has been deleted from this branch. The old benchmarks and their numbers are preserved on `archive/onnx-qwen` for historical reference.

The only currently active benchmark target is the AGC frontend:

```bash
cgs bench --env TYPE=agc
```

Re-authoring a stage-latency and real-pipeline benchmark for the whisper.cpp stack is a planned follow-up. Until then, use the profiling guide below for ad-hoc measurement.

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
| **Shared softmax for HWCE** | `crates/lune-core/src/hwce/features.rs` | Computes softmax once per token and feeds the probability slice to entropy/Tsallis/margin instead of three independent passes |
| **O(1) risk lookup in corrector** | `crates/lune-dict/src/corrector.rs` | Risk positions indexed in a `HashMap`, replacing a per-word linear scan that was O(words²) overall |
| **PCEN AGC frontend** | `crates/lune-audio/src/pcen.rs` | Adaptive gain normalization available in `lune-audio`; not currently wired into the dictation pipeline (whisper.cpp handles input normalization internally) |

### Concurrency

| Technique | Location | Impact |
|-----------|----------|--------|
| **Blocking-pool inference** | `crates/lune-asr/src/backend/whisper_cpp.rs` | Whisper decode runs via `spawn_blocking`, so model execution never starves the async runtime or the audio capture loop |
| **Shared model, per-session state (ASR)** | `crates/lune-asr/src/backend/whisper_cpp.rs` | `WhisperContext` (model weights) is `Arc`-shared; `WhisperState` (decode scratch) is lightweight and per-session |
| **Plain async HTTP (LLM)** | `crates/lune-llm/src/backend/openai.rs` | The remote styling call is a single `async-openai` request on the async runtime — no local model, no blocking pool |

---

## Pipeline Tuning

These live in the `[pipeline]` section of `settings.toml`. For the full list of fields, see [Configuration → Pipeline Tuning](CONFIGURATION.md#pipeline-tuning).

| Parameter | Default | Effect | Tradeoff |
|-----------|---------|--------|----------|
| `max_speech_secs` | 60 | Auto-end the session after this many seconds of audio; longer utterances accumulate more audio for whisper to decode | Higher allows longer single dictations at the cost of a proportionally longer decode |
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
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release -p lune-app

# Record
perf record -g --call-graph dwarf ./target/release/lune-voice daemon

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
  cargo build --release --features "whisper-cpp" -p lune-app

samply record -- ./target/release/lune-voice daemon
# then trigger a dictation session; Ctrl+C to stop recording
```

For GPU builds, substitute `--features "whisper-cuda"`.

### Tracing

Set `RUST_LOG` to see per-component timing:

```bash
RUST_LOG=lune_asr=debug,lune_llm=debug,lune_app=debug ./target/release/lune-voice daemon
```

Key trace points:
- `lune_asr::backend::whisper_cpp` — `poll_final` with `poll_final_ms`
- `lune_llm::backend::openai` — completion call timing
- `lune_app::capture` — `asr: poll_final complete` with `poll_final_ms`

### Memory profiling

```bash
# With heaptrack
heaptrack ./target/release/lune-voice daemon
heaptrack_gui heaptrack.lune-voice.*.gz

# With valgrind/DHAT
cargo build --release -p lune-app
valgrind --tool=dhat ./target/release/lune-voice daemon
```

### Running tests

```bash
# Unit tests (no real models needed)
cgs test

# Real-model integration tests (requires models on disk)
cgs test-e2e
```
