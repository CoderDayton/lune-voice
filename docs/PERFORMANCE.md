# Performance

> Latency budgets, benchmark results, and optimization techniques used throughout the pipeline. For a quick overview, see the [README](../README.md).

---

## Table of Contents

- [Latency Budget](#latency-budget)
- [Benchmarks](#benchmarks)
  - [Pipeline Latency](#pipeline-latency)
  - [Full Pipeline](#full-pipeline)
  - [Recent Benchmark Snapshot](#recent-benchmark-snapshot)
- [Optimization Techniques](#optimization-techniques)
  - [Memory & Allocation](#memory--allocation)
  - [Compute](#compute)
  - [Concurrency](#concurrency)
- [Tuning Parameters](#tuning-parameters)
- [Profiling Guide](#profiling-guide)

---

## Latency Budget

The user-perceived latency is the **after-speech wait**: time from when the speaker stops talking to when finished text is ready for delivery.

```
After-speech wait = utterance finalization + ASR completion + optional LLM cleanup + delivery
```

Current benchmarked emphasis:

| Stage | Cost Profile | Notes |
|-------|--------------|-------|
| Mel extraction | Negligible | Single-digit milliseconds or less on real fixtures |
| ASR inference | Primary fixed cost | Depends on backend: Parakeet vs Qwen |
| LLM cleanup | Variable | Triggered only for lower-confidence text |
| Delivery | Small | Usually dominated by ASR/LLM, not paste/type output |
| Memory footprint | Important secondary constraint | Both examples now print `RSS` and `Peak` |

### What Does NOT Add to After-Speech Wait

- **Benchmark backend selection** only changes which ASR path is exercised, not the rest of the pipeline shape.
- **Audio accumulation** runs concurrently with speech.
- **Mel extraction** is not the limiting factor on the current local runs.

### Bottleneck Analysis

| Rank | Bottleneck | Why it matters |
|------|------------|----------------|
| 1 | ASR backend choice | The largest consistent latency/quality tradeoff is Parakeet INT8 vs Qwen INT4 |
| 2 | LLM routing threshold | Cleanup can add a few hundred milliseconds on benchmark fixtures |
| 3 | Utterance timing / VAD | Finalization timing changes perceived responsiveness materially |
| 4 | Memory movement | Several recent optimizations removed hot-path copies and queue churn |

---

## Benchmarks

### Pipeline Stages

Measures isolated ASR, LLM, and VAD stages on real WAV fixtures. This is the best benchmark for comparing backend speed, quality, and memory use in isolation, but it is not the full dictation pipeline.

```bash
cargo run --example pipeline_stages -p lune-app --release

# Force a backend
ASR=qwen cargo run --release --example pipeline_stages -p lune-app
ASR=parakeet cargo run --release --example pipeline_stages -p lune-app

# Equivalent CLI flag form
cargo run --release --example pipeline_stages -p lune-app -- --asr qwen
cargo run --release --example pipeline_stages -p lune-app -- --asr parakeet
```

#### What It Measures

| Stage | Description |
|-------|-------------|
| Stage 1 | Qwen mel extraction on real fixtures |
| Stage 2 | Real ASR inference on every available backend |
| Stage 3 | Confidence gate behavior |
| Stage 4 | Threshold sweep for LLM cleanup routing |
| Stage 5 | VAD timing sweep |
| Stage 6 | Adaptive VAD sweep |

#### Output Fields

| Field | Meaning |
|-------|---------|
| `RTF` | Real-time factor. Lower is faster. |
| `WER` / `CER` | Word / character error rate against fixture references |
| `RSS` | Current resident memory from `/proc/self/status` |
| `Peak` | Process high-water memory mark from `/proc/self/status` |

### Real Pipeline

Measures the real `SessionManager` path with VAD, ASR, LLM, and final delivery events. This is the closest benchmark to actual dictation behavior.

```bash
cargo run --release --example pipeline_real -p lune-app

# Force a backend
ASR=qwen cargo run --release --example pipeline_real -p lune-app
ASR=parakeet cargo run --release --example pipeline_real -p lune-app

# Equivalent CLI flag form
cargo run --release --example pipeline_real -p lune-app -- --asr qwen
cargo run --release --example pipeline_real -p lune-app -- --asr parakeet
```

#### What It Measures

| Stage | Description |
|-------|-------------|
| Full-session wall time | End-to-end time from session start to final text event |
| WER / CER | Quality after the real session path, not just raw ASR |
| Memory after ASR / LLM load | Per-process model footprint before fixtures run |
| Per-fixture memory | Current and peak memory while processing the fixture |

### Recent Benchmark Snapshot

Recent local runs on this repository produced the following comparisons.

#### `pipeline_stages` Aggregate Winner

| Backend | Avg ASR Time | Avg RTF | Peak Memory | Notes |
|---------|--------------|---------|-------------|-------|
| Parakeet INT8 | `1.308s` | `0.045x` | `3186 MB` | Best overall latency/quality balance |
| Qwen INT4 | `1.476s` | `0.051x` | `2690 MB` | Lower memory, slightly slower aggregate |

#### `pipeline_real` Aggregate Winner

| Backend | Avg Wall Time | Avg WER | Avg CER | Peak Memory | Notes |
|---------|---------------|---------|---------|-------------|-------|
| Parakeet INT8 | `1.520s` | `2.53%` | `2.2%` | `2789 MB` | Current default recommendation |
| Qwen INT4 | `2.379s` | `3.2%` | `2.5%` | `2737 MB` | Good fallback when memory matters more |

#### Qwen INT4 Stage-2 ASR After Export Fix

| Fixture | Total | RTF | WER | CER |
|---------|-------|-----|-----|-----|
| `librispeech_clean` | `1.091s` | `0.047x` | `1.6%` | `0.3%` |
| `librispeech_40744` | `1.755s` | `0.059x` | `4.8%` | `4.3%` |
| `custom_recording` | `1.581s` | `0.046x` | `2.5%` | `1.1%` |

#### Practical Takeaway

| Choice | Recommended When |
|--------|-------------------|
| Parakeet INT8 | You want the best default speed/quality tradeoff |
| Qwen INT4 | You want the Qwen path specifically or lower memory in the latency benchmark |

---

## Optimization Techniques

### Memory & Allocation

| Technique | Location | Impact |
|-----------|----------|--------|
| **Pre-allocated audio snapshot** | `parakeet_backend.rs` | `mem::take` swaps a pre-allocated `Vec` in/out of the session — zero heap allocation during steady-state inference |
| **Persistent partial buffers** | `session.rs` | `partial_buf` and `audio_batch` are allocated once and reused across all iterations of the main loop |
| **Pre-computed capacity** | `confidence_router.rs`, `prompt.rs` | String builders compute exact capacity before allocation (context joins, ChatML prompts) |
| **Segment tracker drain** | `segment.rs` | `mem::replace` with a pre-allocated `Vec::with_capacity(8)` avoids reallocation on the next burst |
| **Static caches** | `mel.rs` | `OnceLock<Array2<f32>>` for the mel filterbank (128×257) and Hann window (400) — computed once, shared immutably |
| **Batch vec reuse** | `llm_worker.rs` | The `batch: Vec<LlmRequest>` is allocated once and cleared each iteration |
| **Pending buffer** | `confidence_router.rs` | 128-byte pre-allocated `String` with incremental word counting |

### Compute

| Technique | Location | Impact |
|-----------|----------|--------|
| **Welford's online algorithm** | `mel.rs:normalize_per_feature` | Single pass for mean+variance instead of two full strided traversals |
| **Branchless triangle filters** | `mel.rs:create_mel_filterbank` | `max(0, min(lower, upper))` eliminates branches in the hot inner loop |
| **norm_sqr() vs norm()*norm()** | `mel.rs:stft` | Eliminates a `sqrt()` call per bin per frame in the STFT power spectrum |
| **Sequential-write transpose** | `mel.rs:extract_features` | Outer loop over time gives stride-1 writes (cache-friendly for L1) |
| **memchr stop detection** | `decode_utils.rs` | `memchr_iter(b'<')` + byte comparison instead of string search — all stop sequences verified ASCII at compile time |
| **Single-pass argmax + logit collection** | `tdt_decoder.rs` | Combined loop finds the max logit while collecting values, avoiding a second O(vocab_size) scan |
| **Incremental word counting** | `confidence_router.rs` | Counts only new words per accumulation, not the whole buffer |
| **Bit-shift length check** | `confidence_router.rs` | `(corr_len << 1) >= orig_len` for the default 0.5/2.0 thresholds |

### Concurrency

| Technique | Location | Impact |
|-----------|----------|--------|
| **Split-phase locking** | `parakeet_backend.rs:poll_results` | Session lock held only during snapshot copy; released before mel extraction and inference. Allows `push_audio` to make progress concurrently |
| **Audio frame coalescing** | `session.rs` | Initial frame + `try_recv()` batch reduces await-point count during microphone bursts |
| **spawn_blocking for LLM** | `ort_backend.rs` | Inference runs on the blocking thread pool, never starving the async runtime |
| **recv_many batch drain** | `llm_worker.rs` | Single lock acquisition to drain all queued LLM requests |
| **State write elision** | `main.rs:update_state` | Skips the write lock when state string is already correct (avoids cache-line dirty on every PartialText) |

---

## Tuning Parameters

| Parameter | Default | Effect | Tradeoff |
|-----------|---------|--------|----------|
| `silence_timeout` | 800ms | Time after last ASR update before finalizing | Lower = faster delivery, but may split mid-sentence pauses |
| `context_secs` | 30.0 | ASR sliding window size | Larger = better context for long dictation, more memory |
| `n_ctx` | 2048 | LLM context window (tokens) | Larger = handles longer prompts, more VRAM/RAM |
| `max_output_tokens` | 512 | Max generated tokens per LLM call | Lower = faster, but may truncate long corrections |
| `high_threshold` | 0.82 | Confidence above which text still goes to LLM | Higher = more text gets batched for LLM context |
| `mid_threshold` | 0.50 | Boundary between low/mid word thresholds | Adjusts accumulation aggressiveness |
| `low_word_threshold` | 8 | Words needed before dispatching low-confidence batch | Lower = faster dispatch, less context for LLM |
| `mid_word_threshold` | 12 | Words needed before dispatching mid-confidence batch | Lower = faster dispatch, less context for LLM |
| `finalize_after_ms` | 500 | Segment tracker silence threshold | Lower = faster segment finalization |
| `temperature` | 0.0 | LLM sampling temperature | 0.0 = greedy/deterministic, higher = creative variation |

### Recommended Profiles

**Low-latency (short commands):**
```toml
silence_timeout = 200    # ms
low_word_threshold = 4
mid_word_threshold = 6
```

**High-quality (long dictation):**
```toml
silence_timeout = 1000   # ms
low_word_threshold = 12
mid_word_threshold = 20
context_secs = 60.0
```

---

## Profiling Guide

### CPU profiling with perf

```bash
# Build with debug symbols
cargo build --release -p lune-app
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release -p lune-app

# Record
perf record -g --call-graph dwarf ./target/release/lune-voice daemon

# Analyze
perf report --hierarchy
```

### Flamegraph

```bash
cargo install flamegraph
flamegraph -- ./target/release/lune-voice daemon
```

### Tracing

Set `RUST_LOG` to see per-component timing:

```bash
RUST_LOG=lune_asr=debug,lune_llm=debug,lune_app=debug ./target/release/lune-voice daemon
```

Key trace points:
- `lune_asr::parakeet_backend` — `inference` with `elapsed_ms`
- `lune_llm::ort_backend` — `LLM post-process request` with `prompt_len`, `raw_text_len`
- `lune_llm::ort_backend` — `LLM post-process complete` with `output_len`
- `lune_app::llm_worker` — `LLM batch: combined N requests`

### Memory profiling

```bash
# With heaptrack
heaptrack ./target/release/lune-voice daemon
heaptrack_gui heaptrack.lune-voice.*.gz

# With valgrind/DHAT
cargo build --release -p lune-app
valgrind --tool=dhat ./target/release/lune-voice daemon
```

### Running benchmarks

```bash
# Stage-oriented latency benchmark
cargo run --release --example pipeline_stages -p lune-app

# Real end-to-end session benchmark
cargo run --release --example pipeline_real -p lune-app

# Optional backend forcing
ASR=qwen cargo run --release --example pipeline_stages -p lune-app
ASR=parakeet cargo run --release --example pipeline_real -p lune-app

# Quick regression pass
cargo test --workspace --release
```
