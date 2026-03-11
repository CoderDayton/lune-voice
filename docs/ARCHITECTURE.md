# Architecture

> Deep dive into Lune Voice internals. For a quick overview, see the [README](../README.md).

---

## Table of Contents

- [System Overview](#system-overview)
- [Crate Dependency Graph](#crate-dependency-graph)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Audio Capture](#1-audio-capture)
  - [ASR](#2-asr)
  - [Confidence Router](#3-confidence-router)
  - [LLM Worker](#4-llm-worker)
  - [Text Delivery](#5-text-delivery)
- [Session Lifecycle](#session-lifecycle)
- [Hold-Delivery Model](#hold-delivery-model)
- [D-Bus Interface](#d-bus-interface)
- [Threading Model](#threading-model)
- [Data Flow Diagram](#data-flow-diagram)

---

## System Overview

Lune Voice is a privacy-first dictation daemon that converts speech to corrected text and can be driven either from the CLI/D-Bus surface or the Tauri desktop UI.

The runtime pipeline is split into interchangeable backends:

| Component | Primary Backends | Format | Runtime |
|-----------|------------------|--------|---------|
| ASR | Qwen3-ASR 0.6B, Parakeet TDT 0.6B | ONNX INT4 / INT8 | ONNX Runtime |
| LLM | Local correction model | GGUF | llama.cpp |

The daemon listens for D-Bus commands (`toggle`, `start`, `stop`, `cancel`) and manages a single active dictation session at a time. Backend selection is centralized through registries so the benchmark examples, app factory, and config parsing all share the same backend definitions.

---

## Crate Dependency Graph

```
lune-app (binary)
├── lune-audio       Microphone capture (cpal)
├── lune-asr         Speech recognition (ONNX Runtime)
│   └── lune-core    Shared traits + types
├── lune-llm         Text post-processing (llama.cpp)
│   └── lune-core
├── lune-commands    Voice command detection
├── lune-dict        User dictionary
├── lune-output      Text delivery controller
│   └── lune-core
└── lune-persist     Settings, DB (SQLite), model downloads
    └── lune-core
```

Each crate has a single responsibility. `lune-core` defines shared types and traits but contains no runtime logic. All inter-crate communication flows through trait objects and channels.

---

## Pipeline Walkthrough

### 1. Audio Capture

**Crate:** `lune-audio` | **File:** `crates/lune-audio/src/lib.rs`

- Uses `cpal` for cross-platform microphone access (ALSA/PipeWire on Linux, CoreAudio on macOS, WASAPI on Windows)
- Captures 16 kHz mono PCM in frames of 480 samples (30ms each)
- Frames are sent over an `mpsc` channel to the session loop
- The session loop coalesces back-to-back frames into a single batch before pushing to ASR, reducing await-point overhead during microphone bursts

### 2. ASR

**Crate:** `lune-asr`

Lune currently ships two local ASR paths:

- `ParakeetTdt` for the transducer-style Parakeet runtime
- `OrtQwenAsr` for the exported Qwen3-ASR encoder/embed/decoder split

Both backends share the same mel front-end and registry-driven selection in `registry.rs`. The benchmark examples can force either backend with `ASR=qwen|parakeet` or `-- --asr qwen|parakeet`.

#### 2a. Parakeet TDT

**Files:** `parakeet_backend.rs`, `mel.rs`, `tdt_decoder.rs`, `segment.rs`

The Parakeet path has four stages:

##### Audio Accumulation

`ParakeetSession` maintains a sliding window buffer (default 30 seconds). New audio is appended; when the buffer exceeds `max_context_samples`, the oldest samples are discarded via a shift-then-append pattern that saves one copy compared to extend-then-shift.

A pre-allocated `audio_snapshot` buffer is swapped in/out via `mem::take` for zero-allocation steady-state inference. The session lock is held only briefly — released before mel extraction so `push_audio` can make progress concurrently.

##### Mel-Spectrogram Extraction

Converts raw PCM → 128-mel spectrogram features. Pipeline:

```
preemphasis (y[i] = x[i] - 0.97*x[i-1])
  → STFT (512-pt Hann window, 160-sample hop, RustFFT)
    → mel filterbank (128 bands, Slaney scale, librosa-compatible)
      → log(x + 2^-24) guard
        → per-feature normalization (Welford's online algorithm)
```

Key implementation details:

- **Static caches:** `MEL_FILTERBANK` and `HANN_WINDOW` are computed once per process via `OnceLock`. Cold init costs ~0.3ms; subsequent calls are free.
- **Branchless triangle filters:** `max(0, min(lower, upper))` avoids branches in the filterbank inner loop.
- **Sequential-write transpose:** Outer loop over time gives stride-1 writes (cache-friendly).
- **Welford normalization:** Single-pass mean+variance per feature, eliminating one full strided traversal.

##### TDT Greedy Decode

The TDT (Token-and-Duration Transducer) decoder runs two ONNX models:

1. **Encoder:** Takes mel features `(1, 128, T)` → hidden states `(1, T', D)` with 8× stride
2. **Decoder-Joint:** Greedy autoregressive decode with LSTM states `(2, 1, 640)`

Each decode step:
- Feeds one encoder frame + last token + LSTM states to the joint network
- Extracts vocab logits and duration logits (5 possible frame advances)
- Computes softmax confidence from the argmax token's probability
- Advances the frame pointer by the predicted duration

The decoder tracks per-token confidence which is averaged into a per-utterance confidence score used by the confidence router.

##### Segment Tracking

`SegmentTracker` manages partial → final segment transitions:

- **Partial updates:** New ASR text replaces the current partial segment (no allocation on update)
- **Silence finalization:** If no new text arrives for `finalize_after_ms` (default 500ms), the partial is promoted to final
- **Drain pattern:** `mem::replace` swaps the segment vec with a pre-allocated empty one, avoiding reallocation on the next burst

#### 2b. Qwen3-ASR ONNX

**Files:** `ort_qwen_backend.rs`, `ort_qwen_stream.rs`, `mel.rs`

The Qwen path uses a three-part ONNX export produced by `scripts/export_onnx.py`:

1. `qwen3-asr-encoder.onnx` for the audio tower / projector
2. `qwen3-asr-embed-tokens.onnx` for token embeddings
3. `qwen3-asr-decoder.onnx` for the decoder with KV cache

The runtime feeds whole-clip or chunked mel features through the encoder, injects the exported prompt contract via the token embedder, and steps the decoder autoregressively while maintaining cache tensors between decode steps.

### 3. Confidence Router

**Crate:** `lune-app` | **File:** `confidence_router.rs`

Routes finalized ASR segments based on per-utterance confidence:

```
confidence ≥ 0.82  →  HighConfidence  →  queue to LLM immediately
confidence 0.50–0.82  →  Mid  →  accumulate until 12 words, then batch
confidence < 0.50  →  Low  →  accumulate until 8 words, then batch
```

The router maintains a `pending_buf` (pre-allocated 128 bytes) with incrementally-tracked word count. Words are counted via a branchless state machine on bytes — ASCII whitespace detection is correct for multi-byte UTF-8 because non-ASCII bytes are never ASCII whitespace.

When `None` confidence is received (e.g., OpenAI backend), the router treats it as 0.0 (low) for conservative batching.

**Context window:** The last 3 confirmed utterances are kept as context for LLM prompts, built with exact-capacity pre-allocation.

**Length sanity check:** LLM corrections are rejected if the output/input length ratio falls outside `[0.5, 2.0]`. Uses integer bit-shifts for the default thresholds, falling back to f32 division only for non-default ratios.

### 4. LLM Post-Processing

**Crate:** `lune-llm` | **File:** `crates/lune-llm/src/llamacpp_backend.rs`

The batch pipeline calls `TextPostProcessor::transform()` directly from `session.rs` — there is no separate LLM worker task.

#### llama.cpp Backend (default)

- Any GGUF model (e.g. Qwen3, Phi, Gemma) for contextual token resolution
- Loaded once via `LlamaModel`; fresh `LlamaContext` per call (stateless)
- Greedy decoding (deterministic)
- Input: raw text with `[?word?]` markers for low-confidence spans
- Output: corrected text
- `strip_markers()` in `prompt.rs` removes any residual `[?word?]`/`[word]` from output

### 5. Text Delivery

**Crate:** `lune-output` | **File:** `crates/lune-output/src/lib.rs`

`TextOutputController` emits two event types:

- `PartialText(String)` — in-progress text for live preview
- `FinalizedText(String)` — corrected text for delivery

The controller computes minimal edit actions via `char_diff()`:

- **ASCII fast path:** Byte comparison (~3× faster than char iteration)
- **Unicode path:** Single combined pass to find the common-prefix byte offset in both strings
- Returns `(delete_count, new_suffix)` for incremental editing

Delivery modes (handled in `main.rs`):

| Mode | Implementation |
|------|----------------|
| TypeInPlace | Clipboard paste via uinput: `wl-copy`/`xclip` → terminal detection → Ctrl+V or Ctrl+Shift+V. Adaptive post-paste delay (EMA of clipboard RTT). Per-key fallback for paste-hostile apps. |
| Clipboard | `wl-copy` (Wayland) / `xclip` (X11) — clipboard only, no paste chord |
| Both | Clipboard paste (same as TypeInPlace; clipboard is set as side effect) |

---

## Session Lifecycle

```
                          ┌─────────┐
                          │  Idle   │
                          └────┬────┘
                               │ start()
                          ┌────▼────┐
                          │ Active  │ ← audio flowing, ASR running
                          └────┬────┘
                               │ stop() / cancel() / audio closed
                          ┌────▼────┐
                          │Teardown │ ← flush pending, drain LLM, finalize DB
                          └────┬────┘
                               │
                          ┌────▼────┐
                          │ Stopped │ → SessionEvent::Stopped emitted
                          └─────────┘
```

Only one session may be active at a time (enforced by `Arc<Mutex<Option<ActiveHandle>>>`). The session task runs on `spawn_local` because it holds a `!Send` cpal audio stream.

**Teardown sequence:**

1. Promote any pending non-final text to the utterance buffer
2. Flush the confidence router's pending buffer → queue to LLM
3. Finalize the remaining utterance buffer → queue to LLM
4. Non-blocking `try_recv` loop to catch any LLM results that finished during teardown
5. Deliver any remaining pending entries as raw fallback
6. End the ASR session
7. Persist final session state to SQLite

---

## Hold-Delivery Model

This is the core architectural decision: **no raw ASR text is ever shown to the user.**

```
ASR finalizes utterance
  → text queued in pending_delivery: VecDeque<(u64, String)>
    → LlmRequest sent to worker channel
      → LLM processes (batching queued requests)
        → LlmResponse received
          → pending_delivery.retain(|(id, _)| *id > resp.utterance_id)
            → corrected text delivered via emit_final()
```

**Batching interaction:** The LLM worker returns the `max_utterance_id` from the batch. A single `retain` call clears all pending entries with `id <= max_id`, correctly handling both single and batched responses.

**Fallback path:** If `llm_tx.send()` fails (worker channel closed), the raw ASR text is delivered immediately and the pending entry is removed. This prevents text from being silently dropped.

**Stop-flow:** During teardown, `try_recv` catches any LLM results that finished. Anything still in `pending_delivery` after draining is delivered as raw fallback — the user always gets their text.

---

## D-Bus Interface

**Service:** `com.lunevoice.Daemon1` on the session bus  
**Object path:** `/com/lunevoice/Daemon1`

### Commands

| Method | Effect |
|--------|--------|
| `Toggle` | Start if idle, stop if active |
| `Start` | Begin dictation |
| `Stop` | End dictation gracefully |
| `Cancel` | Discard current session |
| `Quit` | Shut down daemon |
| `Status` | Returns `"idle"` / `"listening"` / `"error"` |

### Settings (all runtime-mutable via D-Bus)

| Getter / Setter | Type | Scope |
|----------------|------|-------|
| `Get/SetDictationMode` | string (`TypeInPlace`\|`Clipboard`\|`Both`) | Immediate (watch channel) |
| `Get/SetActivationMode` | string (`Toggle`\|`Hold`) | Immediate (watch channel) |
| `Get/SetDefaultStyle` | string (preset ID, validated against registry) | Next session start |
| `Get/SetLanguage` | string (BCP-47, well-formedness validated) | Next session start |
| `Get/SetPrivacyMode` | string (`LocalOnly`\|`LocalPreferred`\|`AllowRemote`) | Persist only |
| `Get/SetSessionHistoryDays` | i32 (-1 = disabled, 0+ = days) | Persist only |
| `Get/SetHotkey` | JSON string of `HotkeyConfig` | Persist only (external DE bindings) |
| `Get/SetAsrBackend` | JSON string of `AsrBackendConfig` | Daemon restart required |
| `Get/SetLlmBackend` | JSON string of `LlmBackendConfig` | Daemon restart required |
| `GetAllSettings` | TOML string (read-only dump) | — |

All setters persist changes to disk immediately via fire-and-forget `spawn_local`. Settings are stored in `Arc<RwLock<UserSettings>>` — the runtime source of truth.

### Signals

| Signal | Payload | Emitted when |
|--------|---------|-------------|
| `StateChanged` | `state: &str` | Pipeline state changes (idle/listening/error) |
| `DictationModeChanged` | `mode: &str` | Dictation mode setter called |
| `ActivationModeChanged` | `mode: &str` | Activation mode setter called |
| `SettingsChanged` | `key: &str` | Any setter changes a setting (key = field name) |

CLI subcommands (`lune-voice toggle`, `lune-voice settings style email`, etc.) are thin D-Bus proxy calls.

---

## Threading Model

```
Main Thread (tokio current_thread + LocalSet)
├── D-Bus connection (zbus) — initialized outside LocalSet
├── Command loop — receives AppCommand from D-Bus
├── Session task (spawn_local) — !Send due to cpal stream
│   ├── Audio coalescing (VAD-gated)
│   ├── ASR push + poll (streaming encoder-decoder)
│   ├── Per-word confidence marking
│   ├── LLM transform (llama.cpp, inline in session loop)
│   └── Text delivery
└── Event forwarder (spawn_local) — routes SessionEvents to delivery

spawn_blocking pool (tokio default)
└── LLM/ASR inference — runs under Mutex, CUDA or CPU
```

Why `current_thread` instead of `multi_thread`:
- Avoids work-stealing overhead for this single-pipeline daemon
- All hot paths stay on one core (cache-friendly)
- `spawn_local` support for `!Send` futures (cpal stream)

ORT sessions are wrapped in `Arc<Mutex<Session>>` for `Send` safety across `spawn_blocking`. The LLM backend uses `llama-cpp-2` with per-call contexts (no shared state). Since inference is single-threaded per model, contention is impossible in practice.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           lune-app (daemon)                                 │
│                                                                             │
│  ┌──────────┐    mpsc     ┌───────────────────────────────────────────┐     │
│  │ lune-    │ ─────────►  │            Session Task                   │     │
│  │ audio    │  AudioFrame │                                           │     │
│  │ (cpal)   │             │  ┌─────────┐  ┌──────────┐  ┌─────────┐ │     │
│  └──────────┘             │  │ ASR     │→ │Confidence│→ │ LLM     │ │     │
│                           │  │push+poll│  │ Router   │  │ Worker  │ │     │
│                           │  └─────────┘  └──────────┘  └────┬────┘ │     │
│                           │       │                          │      │     │
│                           │  ┌────▼────┐              ┌─────▼────┐ │     │
│                           │  │Segment  │              │lune-llm  │ │     │
│                           │  │Tracker  │              │(ORT      │ │     │
│                           │  └─────────┘              │spawn_blk)│ │     │
│                           │                           └─────┬────┘ │     │
│                           │                                 │      │     │
│                           │  ┌──────────────────────────────▼────┐ │     │
│                           │  │     pending_delivery: VecDeque    │ │     │
│                           │  │     (utterance_id, raw_text)      │ │     │
│                           │  └──────────────┬───────────────────┘ │     │
│                           │                 │ LLM response        │     │
│                           │           ┌─────▼─────┐              │     │
│                           │           │  Output    │              │     │
│                           │           │ Controller │              │     │
│                           │           └─────┬──────┘              │     │
│                           └─────────────────┼─────────────────────┘     │
│                                             │                           │
│  ┌──────────┐    mpsc     ┌────────────────▼───────────────────┐       │
│  │  D-Bus   │◄──────────  │       Event Forwarder              │       │
│  │Interface │  SessionEvt │  FinalizedText → DeliveryService   │       │
│  └──────────┘             └────────────────────────────────────┘       │
│                                                                         │
│  ┌──────────┐                                                           │
│  │lune-     │  SQLite — session history, settings                       │
│  │persist   │  HuggingFace — model downloads                            │
│  └──────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```
