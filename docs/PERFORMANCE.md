# Performance

> Where the time goes in the whisper.cpp pipeline, and how it is kept low. Styling runs on a remote OpenAI-compatible endpoint, so it is covered only as a network call. For an overview, see the [README](../README.md).

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

The delay you notice is the **after-speech wait**: the time from releasing the hotkey (or pressing it again to stop) until the text shows up in your app.

```
After-speech wait = poll_final (Silero endpoint trim + whisper.cpp decode) + optional LLM styling + delivery
```

| Stage | Cost profile | Notes |
|-------|-------------|-------|
| Audio flush | Negligible | 30 ms tail flush after hotkey release |
| whisper.cpp decode (`poll_final`) | **Primary fixed cost** | Batch decode of the full utterance; scales with audio length and model size |
| Silero VAD | Small, separate stage | Trims silence from the start and end before decode, and keeps the pauses inside speech because whisper reads them as punctuation. This is a separate pass, not whisper's in-decode `enable_vad` |
| LLM styling | Varies; skipped when no style is selected | One completion call to your OpenAI-compatible endpoint. It is plain async HTTP, so it doesn't block anything else |
| Delivery | Flat for a paste; ~8 ms per character when typed | Text over 80 characters, non-ASCII text and multi-line text are pasted (flat, ~100–360 ms). Shorter ASCII text is typed key by key, which is faster at that length and leaves your clipboard alone. On Windows, add 30 ms when focus has to be moved back to the original window first |

### What Does Not Add to the Wait

- **Recording** happens while you speak.
- **Model load** happens once at startup, and the loaded model is shared by every session. The exception is `model_idle_unload_minutes` (off by default): it frees the model after a quiet period, so the next hotkey press pays for the reload. On `Medium` that is ~160 ms if the file is still in the OS cache, ~750 ms if not.
- **Dictionary correction** is an in-memory string pass and costs next to nothing.

### Bottleneck Analysis

| Rank | Bottleneck | Why it matters |
|------|------------|----------------|
| 1 | whisper.cpp decode time | Scales with utterance length and model size; dominates on CPU |
| 2 | LLM styling call | A few hundred milliseconds per utterance; skipped when no style is selected |
| 3 | Memory use | whisper.cpp keeps the model weights in VRAM or RAM. Remote styling uses no local memory |

**GPU:** with a CUDA device, transcription runs on the GPU and decode time drops sharply. On Linux the CUDA backend comes from the downloadable GPU plugin ([docs/GPU_PLUGIN.md](GPU_PLUGIN.md)); a `whisper-cuda` build links it in directly instead. Without either, whisper.cpp runs on the CPU. Release builds ship two CPU backends and load the one your CPU supports: `haswell` (AVX2 + FMA + F16C) for Intel since 2013 and AMD since 2015, and `sse42` for older or low-end CPUs without AVX2. AVX-512 is not used. The native `cgs build` compiles a single backend tuned for the machine that builds it.

**Model size:** `Base` (~78 MB) is the fastest and the one to use on weak hardware. `Small` (~190 MB) comes next, then `Medium` (~539 MB). `LargeV3Turbo` (~874 MB) is the most accurate, and its turbo decoder keeps it closer to `Medium` in speed than to the full `Large` model.

---

## Benchmark Status

`cgs bench --env TYPE=<type>` runs the suite under `benchmarks/`:

| Type | Measures |
|---|---|
| `agc` | PCEN AGC frontend |
| `transcribe` | Whole capture → AGC → VAD → whisper (+ styling) pipeline on a WAV/f32/dir (`--env WAV=`) |
| `pipeline` / `stages` / `latency` | Real pipeline, per-stage timings, session latency |
| `domain` / `punctuation` / `dict` / `snippets` | Output-quality A/Bs |
| `leadin` | Opening-word errors on short utterances per lead-in policy (`--env LUNE_BENCH_DIR=` for your own `.wav` + `.txt` clips) |
| `styling` / `rulecmp` | Remote styling capability and rule comparison (endpoint via `LUNE_BENCH_LLM_*`, see `benchmarks/bench_utils.rs`) |
| `vram` | GPU memory use per model (`--env DEVICE=`, `CYCLES=`) |
| `modelload` | Cold and warm model load + unload timings (`--env REPEATS=`, `WHISPER_MODEL=`). The idle-unload figures above come from this |
| `live` / `livewer` | Live microphone transcription, and its word error rate |
| `confusion` / `noisify` | Recognition confusion pairs, and synthetic noise robustness |

`LUNE_BENCH_BASELINE` and `LUNE_BENCH_TOLERANCE` gate a run against a recorded
baseline. Targets that use a model compile whisper.cpp and need the model on disk. No numbers are checked in, because decode time varies tenfold across CPU, GPU, model size and utterance length. Run the targets on the hardware you care about, or use the profiling guide below.

---

## Optimization Techniques

### Memory & Allocation

| Technique | Location | Impact |
|-----------|----------|--------|
| **Global mimalloc allocator** | `desktop-ui/src-tauri/src/main.rs` | Replaces glibc malloc. Small allocations are faster and memory fragments less, across the desktop shell and the pipeline crates |
| **whisper.cpp audio buffer reuse** | `crates/lune-asr/src/backend/whisper_cpp.rs` | The per-session `Vec<f32>` audio buffer is pre-allocated for 30 s of audio (`Vec::with_capacity(16_000 * 30)`) and cleared, not freed, after each `poll_final`. Recording up to that length allocates nothing |
| **Corrector single-buffer output** | `crates/lune-dict/src/corrector.rs` | Builds the corrected text directly into a sized `String` instead of a `Vec<String>` + `join(" ")` |
| **Output buffer preallocation** | `crates/lune-llm/src/prompt.rs` | `fix_all_number_spaces` sizes its output buffer to the input length (`String::with_capacity(input.len())`) before filling |

### Compute

| Technique | Location | Impact |
|-----------|----------|--------|
| **O(1) risk lookup in corrector** | `crates/lune-dict/src/corrector.rs` | Risk positions are indexed in a `HashMap`, so each word costs an O(1) lookup instead of a linear scan over the risk list |
| **PCEN AGC frontend** | `crates/lune-audio/src/pcen.rs`, `crates/lune-voice/src/capture.rs` | Automatic gain on every captured frame through `MicAgc`, controlled by the `vad_pcen` setting (default on). It brings a quiet mic up toward a target level before whisper hears it |

### Concurrency

| Technique | Location | Impact |
|-----------|----------|--------|
| **Blocking-pool inference** | `crates/lune-asr/src/backend/whisper_cpp.rs` | Whisper decode runs in `block_in_place` (the session is `!Send`, so `spawn_blocking` can't take it). Decoding never starves the async runtime or the audio capture loop |
| **Shared model, per-session state (ASR)** | `crates/lune-asr/src/backend/whisper_cpp.rs` | `WhisperContext` (the model weights) is shared through an `Arc`. Each session gets its own `WhisperState`, which allocates its decode and KV-cache buffers when it is created |
| **Plain async HTTP (LLM)** | `crates/lune-llm/src/backend/openai.rs` | The styling call is one `async-openai` request on the async runtime. No local model and no blocking pool |

---

## Pipeline Tuning

These live in the `[pipeline]` section of `settings.toml`. For the full list of fields, see [Configuration → Pipeline Tuning](CONFIGURATION.md#pipeline-tuning).

| Parameter | Default | Effect | Tradeoff |
|-----------|---------|--------|----------|
| `max_speech_secs` | 300 (local Whisper), 120 (remote) | Ends the session after this many seconds of audio | Higher allows longer dictations, and decode time grows with the length |
| `llm_timeout_secs` | 30 | Uses the unstyled text (dictionary fixes still applied) if the styling call takes longer than this | Lower caps the worst-case wait; too low skips styling on a slow endpoint |

### Recommended Profiles

**Quick commands** (a stuck-open mic gives up after 30 s):
```toml
[pipeline]
max_speech_secs = 30
```

**Long dictation:** keep the default. With local Whisper it already allows 300 s (five minutes) per session.

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

For a statically linked GPU build, use `--features "whisper-cuda"` instead. To profile the plugin path, install the GPU plugin and run the CPU build; the backend is chosen at runtime.

### Tracing

Set `RUST_LOG` to see per-component timing:

```bash
RUST_LOG=lune_asr=debug,lune_llm=debug,lune_voice=debug ./target/release/lune-voice
```

Key trace points:
- `lune_asr::backend::whisper_cpp`: `poll_final` with `poll_final_ms`
- `lune_llm::backend::openai`: completion call timing
- `lune_voice::capture`: `asr: poll_final complete` with `poll_final_ms`

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
