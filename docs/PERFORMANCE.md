# Performance

> Latency budgets, benchmark results, and optimization techniques used throughout the pipeline. For a quick overview, see the [README](../README.md).

---

## Table of Contents

- [Latency Budget](#latency-budget)
- [Benchmarks](#benchmarks)
  - [Pipeline Latency](#pipeline-latency)
  - [Mel-Spectrogram Throughput](#mel-spectrogram-throughput)
- [Optimization Techniques](#optimization-techniques)
  - [Memory & Allocation](#memory--allocation)
  - [Compute](#compute)
  - [Concurrency](#concurrency)
- [Tuning Parameters](#tuning-parameters)
- [Profiling Guide](#profiling-guide)

---

## Latency Budget

The user-perceived latency is the **after-speech wait** — time from when they stop talking to when corrected text appears. This is the sum of three stages:

```
After-speech wait = silence timeout + final inference + LLM correction + delivery

Current breakdown (typical, CPU-only):
┌─────────────────────────────────────────────────────────┐
│ Component              │ Latency    │ % of Total        │
├────────────────────────┼────────────┼───────────────────┤
│ Silence timeout        │ ~800 ms    │ dominant          │
│ Final ASR inference    │ ~100 ms    │ fixed cost        │
│ Mel extraction         │ ~0.06 ms   │ negligible        │
│ Dictionary apply       │ <1 ms      │ negligible        │
│ LLM post-processing    │ 500–2000ms │ varies by length  │
│ Text delivery (paste)  │ ~5 ms      │ negligible        │
├────────────────────────┼────────────┼───────────────────┤
│ TOTAL (short phrase)   │ ~1.4 s     │                   │
│ TOTAL (paragraph)      │ ~3.0 s     │                   │
└─────────────────────────────────────────────────────────┘
```

### What Does NOT Add to After-Speech Wait

- **Inference trigger threshold** (4800 samples / 300ms): Only affects how soon partials appear _during_ speech. The final delivery fires after silence, not after a threshold.
- **Audio accumulation**: Runs concurrently with speech. Frame coalescing adds ~100ns per frame.
- **Mel-spectrogram extraction**: At 0.06ms per 300ms chunk, it consumes 0.02% of the inference budget.

### Bottleneck Analysis

1. **Silence timeout (800ms):** The biggest knob. Lowering to 200ms saves 600ms but risks splitting mid-sentence pauses into separate utterances. The right value depends on speaking style.

2. **LLM post-processing (500–2000ms):** Proportional to input length. Short phrases (3–5 words) take ~500ms. Paragraphs (50+ words) can take 2s+. This is the cost of the hold-delivery model — you trade raw-speed for clean output.

3. **ASR inference (~100ms):** Fixed cost per poll. The TDT model is already INT8 quantized; further reduction requires GPU or a smaller model.

---

## Benchmarks

### Pipeline Latency

Measures each stage of the dictation pipeline by simulating realistic audio frame arrival and ASR behavior.

```bash
cargo run --example pipeline_latency -p lune-app --release

# Optional: force a specific backend during the benchmark run
ASR=qwen cargo run --example pipeline_latency -p lune-app --release
ASR=parakeet cargo run --example pipeline_latency -p lune-app --release
```

#### What It Measures

| Stage | Description |
|-------|-------------|
| Audio frame accumulation | `extend_from_slice` throughput for 100 frames (3s audio) |
| Inference trigger | Frames needed to reach threshold at various settings |
| Segment tracker | `push_partial`, `check_silence_finalization`, `drain` throughput (100K ops) |
| Silence timeout | Finalization latency at 500/300/200/150/100ms thresholds |
| Full pipeline simulation | End-to-end simulation: audio → ASR → segment → silence → finalize → delivery |

#### Sample Output

```
── Stage 1: Audio Frame Accumulation ──────────────────────────────
  Frame size:          480 samples (30 ms @ 16000 Hz)
  Frames pushed:       100 (3000 ms of audio)
  Buffer accumulation: 0.005 ms
  Per-frame cost:      51 ns

── Stage 3: Segment Tracker ──────────────────────────────────────
  push_partial × 100000:                 7.80 ms (78 ns/op)
  check_silence × 100000:               0.22 ms (2 ns/op)
  drain (1 segments):                    0.000 ms

── Stage 5: Full Pipeline Simulation ─────────────────────────────
  [Current (applied)]
    Trigger threshold: 4800 samples (300 ms)
    Silence timeout:   200 ms
    Inference latency: 100 ms (simulated)
    First inference:   300 ms after speech start
    Speech duration:   1500 ms
    PERCEIVED LATENCY: 1705 ms  (speech_end + silence + delivery)
    After-speech wait: 205 ms   (silence_timeout + delivery)
```

### Mel-Spectrogram Throughput

Profiles `extract_features()` across realistic audio lengths.

```bash
cargo run --example mel_throughput -p lune-asr --release
```

#### What It Measures

| Stage | Description |
|-------|-------------|
| OnceLock cache init | Cold vs warm call (filterbank + window construction) |
| Latency scaling | Extraction time for 30ms – 30s audio |
| Sustained streaming | 100 back-to-back 300ms chunks (simulates continuous inference) |
| Percentile distribution | min/p50/p90/p99/max over 100 samples |
| Shape & stability | Output dimensions and NaN/Inf checks |
| Budget analysis | Where mel fits in the 300ms inference cycle |

#### Sample Output (Intel i7, CPU-only)

```
── Stage 1: Static Cache Initialization (OnceLock) ──────────────
  Cold call (filterbank + window init):    0.361 ms
  Warm call (OnceLock cache hit):          0.018 ms
  One-time init overhead:                  0.343 ms

── Stage 2: Extraction Latency by Audio Length ──────────────────
  Duration    Samples    Frames      Latency     Throughput       RTF
  ────────────────────────────────────────────────────────────────────
  30 ms          480        5      0.003 ms   160,000,000 S/s  0.0001x
  300 ms        4800       31      0.058 ms    82,800,000 S/s  0.0002x
  1000 ms      16000      101      0.190 ms    84,200,000 S/s  0.0002x
  5000 ms      80000      501      0.942 ms    84,900,000 S/s  0.0002x
  30000 ms    480000     3001      5.620 ms    85,400,000 S/s  0.0002x

── Stage 3: Sustained Streaming Throughput ──────────────────────
  Chunks processed:    100
  Audio simulated:     30000 ms (30.0 s)
  Total wall time:     5.80 ms
  Sustained RTF:       0.0002x  → 0.02% of 300ms budget consumed

── Stage 4: Latency Percentiles (300ms chunk, 100 samples) ─────
  min=0.040  p50=0.058  p90=0.072  p99=0.120  max=0.180  (ms)

  Mel extraction consumes a fraction of the 300ms inference budget.
  The bottleneck is TDT ONNX inference, not preprocessing.

  300ms inference cycle budget:
    Mel extraction (p50):     0.058 ms  (0.0%)
    TDT inference (est):      100.0 ms  (33.3%)
    Segment tracker ops:        0.002 ms  (<0.1%)
    ────────────────────────────────────────────
    Remaining headroom:       199.9 ms  (66.6%)

  Throughput upper bound (mel only, no inference):
    17241 chunks/s  →  5172x real-time
```

#### Key Takeaways

- Mel extraction is **not a bottleneck**. At 0.058ms per 300ms chunk, it's 5000× faster than real-time.
- The OnceLock initialization cost (0.3ms) is paid exactly once per process lifetime.
- Sustained streaming shows consistent performance with no degradation over 100 consecutive chunks.
- p99 latency (0.12ms) is still negligible — no tail latency concern.

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
| **ASCII fast-path char_diff** | `lune-output` | Byte comparison for ASCII text (~3× faster than char iteration) |

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
# Both benchmarks
cargo run --example pipeline_latency -p lune-app --release
cargo run --example mel_throughput -p lune-asr --release

# Quick smoke test (unit tests include perf-relevant assertions)
cargo test --workspace --release
```
