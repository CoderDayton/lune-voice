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
| ASR inference | Primary fixed cost | Qwen3-ASR 0.6B (INT4) is the only local backend; OpenAI-compatible remote ASR is also supported |
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
| 1 | ASR inference time | Qwen3-ASR 0.6B (INT4) dominates the after-speech wait on local runs |
| 2 | LLM routing threshold | Cleanup can add a few hundred milliseconds on benchmark fixtures |
| 3 | Utterance timing / VAD | Finalization timing changes perceived responsiveness materially |
| 4 | Memory movement | Several recent optimizations removed hot-path copies and queue churn |

---

## Benchmarks

### Pipeline Stages

Times every stage of the dictation pipeline on real WAV fixtures and breaks the ASR stage down into its sub-stages — the best benchmark for finding which stage dominates. Each fixture runs `WARMUP` discarded passes then `ITERS` timed passes, so the reported P50/P95 reflect steady state (CUDA arenas warm, `.onnx.opt` cached), not first-call cost.

```bash
cargo run --example pipeline_stages -p lune-app --release
# GPU:   cargo run --release --features cuda --example pipeline_stages -p lune-app
# Knobs: ITERS=N WARMUP=N STYLE=<preset> LUNE_LLM=off
# Flame: PROFILE=1 PROFILE_SECS=30 samply record -- \
#          target/release/examples/pipeline_stages
```

#### What It Measures

| Stage | Description |
|-------|-------------|
| Session create | `start_session` setup cost |
| Push audio | Buffer fill / ingest |
| ASR `poll_final` | Real Qwen3-ASR transcription, split into mel → encoder → prompt → prefill → decode |
| Decode-loop ops | Per-token split of the autoregressive loop: embed, decoder run, argmax, KV collect, bind |
| LLM correction | Real post-processing pass over the ASR output |

#### Output Fields

| Field | Meaning |
|-------|---------|
| `Sess` / `Push` / `Mel` / `Encoder` / `Decode` / `LLM` / `Total` | Per-stage wall time (ms) |
| `RTF` | Real-time factor (wall / audio). Lower is faster. |
| `tokens/s` | Decode steps per second (autoregressive throughput) |
| `WER` / `CER` | Word / character error rate against fixture references |
| Decode-loop op breakdown | Share and µs/token of each per-token decode op |
| Time distribution | Share of total wall clock per stage |
| `RSS` / `Peak` | Current / high-water resident memory from `/proc/self/status` |

### Real Pipeline

Measures the real `SessionManager` path with VAD, ASR, LLM, and final delivery events. This is the closest benchmark to actual dictation behavior.

```bash
cargo run --release --example pipeline_real -p lune-app
```

#### What It Measures

| Stage | Description |
|-------|-------------|
| Full-session wall time | End-to-end time from session start to final text event |
| WER / CER | Quality after the real session path, not just raw ASR |
| Memory after ASR / LLM load | Per-process model footprint before fixtures run |
| Per-fixture memory | Current and peak memory while processing the fixture |

### Recent Benchmark Snapshot

Qwen3-ASR 0.6B (INT4) is the only local backend. Representative
aggregate from `pipeline_stages` (24 fixtures, RTX 4090, P50):

| Metric | P50 | P95 |
|--------|-----|-----|
| Total wall | `0.80s` | `1.13s` |
| RTF | `0.034x` | `0.050x` |
| Decode tokens/s | `140` | `143` |
| WER | `9.3%` | `35.7%` |
| CER | `3.1%` | `21.7%` |

Clean-audio fixtures (`librispeech_clean`) land near `0.85s` / `0.037x`
with `0%` WER; the WER/CER P95 is driven by the noisy and multilingual
fixtures in the suite.

Where the wall clock goes: **ASR decode ~66%**, LLM correction ~24%,
prefill ~7%, mel + encoder ~1.5%. Within decode, the autoregressive
`decoder.run` is ~90% — a GPU launch/occupancy-bound forward pass at
batch=1/seq=1 (GPU at ~46% power, 33% memory bandwidth during decode),
the natural target for CUDA-graph capture. Peak resident memory sits
near 2.0 GB once the LLM correction model is loaded alongside ASR.

---

## Optimization Techniques

### Memory & Allocation

| Technique | Location | Impact |
|-----------|----------|--------|
| **Global mimalloc allocator** | `desktop-ui/src-tauri/src/main.rs` | Replaces glibc malloc with mimalloc; faster small-alloc paths and lower fragmentation across the desktop shell and pipeline crates |
| **Mel scratch reuse** | `lune-asr/src/mel/whisper.rs` | `padded`, `fft_in`, `power`, and `mel_f32` workspaces live behind a `Mutex` and grow monotonically — STFT/mel passes allocate once, then reuse |
| **KV dequant buffer reuse** | `lune-llm/src/backend/kv_cache.rs` | Per-layer dequant scratch is hoisted out of the unload loop and refilled in place |
| **ORT input pre-allocation** | `lune-llm/src/backend/ort_llm.rs` | `Inputs<'static>` is built into a `Vec::with_capacity(input_capacity(...))` instead of being constructed and then extended |
| **Corrector single-buffer output** | `lune-dict/src/corrector.rs` | Builds the corrected text directly into a sized `String` instead of a `Vec<String>` + `join(" ")` |
| **Pre-allocated audio snapshot** | `lune-asr/src/backend/qwen3.rs` | `mem::take` swaps a pre-allocated `Vec` in/out of the session — zero heap allocation during steady-state inference |
| **Persistent partial buffers** | `session.rs` | `partial_buf` and `audio_batch` are allocated once and reused across all iterations of the main loop |
| **Pre-computed capacity** | `confidence_router.rs`, `prompt.rs` | String builders compute exact capacity before allocation (context joins, ChatML prompts) |
| **Segment tracker drain** | `segment.rs` | `mem::replace` with a pre-allocated `Vec::with_capacity(8)` avoids reallocation on the next burst |
| **Static caches** | `mel/whisper.rs` | `OnceLock<Array2<f32>>` for the mel filterbank and Hann window — computed once, shared immutably |
| **Batch vec reuse** | `llm_worker.rs` | The `batch: Vec<LlmRequest>` is allocated once and cleared each iteration |
| **Pending buffer** | `confidence_router.rs` | 128-byte pre-allocated `String` with incremental word counting |

### Compute

| Technique | Location | Impact |
|-----------|----------|--------|
| **Shared softmax for HWCE** | `lune-core/src/hwce/features.rs` | Computes softmax once per token and feeds the probability slice to entropy/Tsallis/margin instead of paying for three independent passes |
| **Vectorized ASR argmax softmax** | `lune-asr/src/backend/qwen3.rs` | A branch-free range-reduced `fast_exp_nonpos` replaces the per-element libm `exp` across the 151,936-logit vocab; the LSE pass auto-vectorizes under `target-cpu=native`. ~2.2× faster argmax (1.37 → 0.62 ms/token), ~8% lower end-to-end latency, token output bit-identical |
| **Sequential-write 4-bit quant** | `lune-llm/src/backend/kv_cache.rs` | Row-major `(t, d)` min/max scan with stride-1 reads lets the auto-vectorizer emit SIMD min/max instead of stride-`dim` scalar loads |
| **O(1) risk lookup in corrector** | `lune-dict/src/corrector.rs` | Risk positions indexed in a `HashMap`, replacing a per-word linear scan that was O(words²) overall |
| **norm_sqr() vs norm()*norm()** | `mel/whisper.rs` | Eliminates a `sqrt()` call per bin per frame in the STFT power spectrum |
| **Sequential-write transpose** | `mel/whisper.rs` | Outer loop over time gives stride-1 writes (cache-friendly for L1) |
| **PCEN AGC frontend** | `lune-audio/src/vad.rs` | Default-on adaptive gain (`vad_pcen = true`) flattens loudness drift before VAD scoring; warmup holds unity gain while envelopes settle |
| **Incremental word counting** | `confidence_router.rs` | Counts only new words per accumulation, not the whole buffer |
| **Bit-shift length check** | `confidence_router.rs` | `(corr_len << 1) >= orig_len` for the default 0.5/2.0 thresholds |

### Concurrency

| Technique | Location | Impact |
|-----------|----------|--------|
| **Split-phase locking** | `lune-asr/src/backend/qwen3.rs:poll_results` | Session lock held only during snapshot copy; released before mel extraction and inference. Allows `push_audio` to make progress concurrently |
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

### Flamegraph (samply)

`samply` needs no root, but it does need `perf_event_paranoid <= 1`:

```bash
cargo install samply
echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid   # reverts on reboot

# Build with symbols + frame pointers (keeps the release optimization level)
RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
  cargo build --release --features cuda --example pipeline_stages -p lune-app

# The benchmark's PROFILE mode loops the full pipeline for a sampler to capture
PROFILE=1 PROFILE_SECS=30 samply record -- \
  ./target/release/examples/pipeline_stages
```

When `perf_event_paranoid` can't be lowered, the benchmark's built-in
decode-loop op breakdown (deterministic per-op timers) attributes the
decode stage without a sampler.

### Tracing

Set `RUST_LOG` to see per-component timing:

```bash
RUST_LOG=lune_asr=debug,lune_llm=debug,lune_app=debug ./target/release/lune-voice daemon
```

Key trace points:
- `lune_asr::backend::qwen3` — `inference` with `elapsed_ms`
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

# Quick regression pass
cargo test --workspace --release
```
