# Architecture

Lune Voice has one dictation pipeline and one control surface: the desktop app, which drives the runtime through typed Tauri commands and a session event stream.

---

## Table of Contents

- [System Overview](#system-overview)
- [Crate Dependency Graph](#crate-dependency-graph)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Audio Capture](#1-audio-capture)
  - [ASR](#2-asr)
    - [Compute backend selection](#2a-i-compute-backend-selection)
  - [Dictionary Application](#3-dictionary-application)
  - [LLM Styling](#4-llm-styling)
  - [Text Delivery](#5-text-delivery)
- [Session Lifecycle](#session-lifecycle)
- [Desktop Bridge](#desktop-bridge)
- [Threading Model](#threading-model)
- [Data Flow Diagram](#data-flow-diagram)

---

## System Overview

Lune Voice turns microphone audio into styled text, then delivers that text to the focused app. The runtime keeps one active session at a time. Everything else exists to feed that session, configure it, or show its state.

Dictation is **hotkey-bounded**: the session collects audio for as long as the hotkey is held (Hold mode) or until the hotkey is pressed again (Toggle mode). There is no online VAD gate during recording; Silero trims silence off the ends of the buffered audio in a separate pass just before decode.

The runtime pipeline uses interchangeable backends:

| Component | Backends | Model format | Runtime |
|-----------|----------|--------------|---------|
| ASR | WhisperCpp (local), OpenAI-compatible (remote) | GGML `.bin` / remote API | whisper-rs (whisper.cpp) / HTTP |
| LLM | OpenAI-compatible (remote), disabled | remote API | HTTP |

The native ASR backend compiles its C/C++ library from source at build time via a `*-sys` crate. GPU (CUDA) support on Linux is a **runtime** capability: ggml's compute backends are built as `dlopen`-able plugins (`GGML_BACKEND_DL`), so the CUDA backend is downloaded on request rather than linked in — see [docs/GPU_PLUGIN.md](GPU_PLUGIN.md). The styling LLM is remote-only — point it at Ollama, llama-server, LM Studio, or a hosted API. See [docs/RELEASE.md](RELEASE.md) for the full build matrix.

Backend selection is centralized through registries and DTO mappers. The desktop UI never builds model paths or backend structs itself; it sends typed settings and receives snapshots.

---

## Crate Dependency Graph

```
lune-core (types, config, errors, hwce confidence)
├─ lune-audio (mic, resample, ringbuf, vad, pcen)
├─ lune-asr (whisper.cpp, openai remote, registry)
├─ lune-llm (openai remote, null passthrough, style pipeline, prompt, few-shot)
├─ lune-dict (corrector, vocab, voice commands)
└─ lune-persist (settings.toml, sqlite, model hub)
     └─ lune-voice (session, capture + AGC, delivery, sound cues)
          └─ desktop-ui/src-tauri (tauri IPC, hotkey, backend registry wiring, pill, persistence bridge)
```

Each crate has a single responsibility. `lune-core` defines shared types and traits but contains no runtime logic. All inter-crate communication flows through trait objects and channels.

---

## Pipeline Walkthrough

### 1. Audio Capture

**Crate:** `lune-audio` | **File:** `crates/lune-audio/src/lib.rs`  
**Orchestration:** `crates/lune-voice/src/capture.rs`

- Uses `cpal` for microphone access (ALSA/PipeWire)
- Delivers 16 kHz mono PCM in 80 ms chunks (1280 samples). Most hardware captures at 44.1 or 48 kHz, so a `rubato` async resampler converts to 16 kHz in the capture path whenever the device's native rate differs
- `MicAgc` (PCEN) normalizes each frame toward a target RMS before it leaves the capture loop, gated by the `vad_pcen` setting (default on). The volume meter reads the **post-AGC** signal, so it reflects what the recognizer hears
- Chunks are pushed to the ASR backend via `push_audio` as they arrive; there is no online VAD gate at this stage
- The session loop handles Stop/Cancel/deadline/max-speech control signals and terminates the capture loop accordingly
- After Stop, the producer is asked to flush and given a 30 ms grace period to hand over the chunk it still holds (up to ~80 ms — the last phonemes of the final word), followed by the AGC's held sub-hop remainder; then `poll_final` decodes the buffer

### 2. ASR

**Crate:** `lune-asr`

Both backends implement the same `AsrBackend` trait, so the session code does not branch on backend type.

#### 2a. WhisperCpp (local, default)

**File:** `crates/lune-asr/src/backend/whisper_cpp.rs`

The `WhisperCppBackend` wraps `whisper-rs`, which vendors and compiles `whisper.cpp`:

- The `WhisperContext` (model weights) is loaded **once** at backend construction and shared across all sessions. Per-session `WhisperState` is lightweight (a decode context).
- `push_audio` appends PCM to a per-session buffer pre-allocated for 30 s of audio (grows once for longer utterances; reset without freeing after each decode).
- `poll_final` decodes the entire buffer in one shot using **BeamSearch** (beam size 5) — whisper.cpp's accuracy-oriented default. Decoding is one-shot per utterance, so the extra compute is worthwhile.
- **Silero endpoint trim:** when the VAD model is present (`models/whisper-vad-silero/ggml-silero-v5.1.2.bin`), `poll_final` first runs `WhisperVadContext::segments_from_samples` over the buffer and decodes only the span between the first and last speech segment. Internal pauses are kept, because whisper reads them as punctuation cues. whisper.cpp's in-decode `enable_vad` is deliberately **not** used — it concatenates speech spans and strips those pauses, producing run-ons. The VAD context always runs on CPU (silero is tiny, and a GPU context crashed at init). If the model file is missing, the backend logs a warning and decodes the whole buffer.
- **Per-token confidence:** whisper exposes per-token probabilities. `poll_final` folds them into `AsrSegment::confidence` (mean) and `min_token_confidence` (minimum), excluding special tokens. Segments whose mean confidence falls below `GARBAGE_DROP_THRESHOLD` (0.25) are dropped by the capture layer before reaching the session.
- Language detection: the `language` field from `AsrSessionConfig` is passed to `set_language`; `None` uses `"auto"` (whisper detects from audio).

Three GGML model sizes are available, all from `ggerganov/whisper.cpp`:

| Variant | File | Size | Directory |
|---------|------|------|-----------|
| `Small` | `ggml-small-q5_1.bin` | ~190 MB | `whisper-small-q5_1` |
| `Medium` *(default)* | `ggml-medium-q5_0.bin` | ~539 MB | `whisper-medium-q5_0` |
| `LargeV3Turbo` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB | `whisper-large-v3-turbo-q8_0` |

#### 2a-i. Compute backend selection

**File:** `crates/lune-asr/src/gpu_backend.rs`

ggml's compute backends are built as `dlopen`-able plugins (`GGML_BACKEND_DL`), so the executable carries no `DT_NEEDED` on the CUDA runtime and starts on a machine with no NVIDIA driver at all. Two directories are scanned at startup: `<exe dir>/backends/` for the bundled CPU backends, and `<data dir>/backends/` for the CUDA plugin the user downloads.

`load_plugins` **must run before the first `WhisperContext` is created** — ggml selects a device from whatever is in the backend registry at context-creation time, which is why `desktop-ui/src-tauri/src/lib.rs` calls it before the Tauri builder. It is idempotent via a `OnceLock`, because the registry is process-global and has no unload.

Each plugin carries `RUNPATH=$ORIGIN` so it resolves its own CUDA libraries from its own directory. Setting `LD_LIBRARY_PATH` in-process would not work: glibc caches the search path at startup.

See [docs/GPU_PLUGIN.md](GPU_PLUGIN.md) for the states, the archive contents, and the publishing flow.

#### 2b. OpenAI-Compatible (remote)

Remote ASR is useful when the user explicitly allows remote processing. It returns transcript text through the same `AsrSegment` type but does not expose per-token confidence. Missing confidence disables LLM routing confidence gates; the styling call is still driven by the style preset.

##### Transport: when plaintext is allowed

Both remote backends (`crates/lune-asr/src/backend/openai.rs`, `crates/lune-llm/src/backend/openai.rs`) refuse to construct against a plaintext `http://` endpoint unless it is *local* by `lune_core::config::is_local_endpoint` — loopback, the RFC 1918 / RFC 4193 private ranges, and `.local` / `.localhost` names. A public host must use HTTPS.

That carve-out is wider than loopback on purpose, and it is a deliberate tradeoff worth stating plainly: **against a LAN endpoint, the API key and every dictated transcript cross the local network in cleartext.** Self-hosted inference servers (Ollama, llama-server, LM Studio) ship without TLS, so requiring HTTPS would break the common setup of running the model on another box on the same network. The exposure is bounded by the LAN, and the same predicate governs whether an endpoint may be used keyless, so the two decisions cannot drift apart.

Users who do not accept that tradeoff should put the endpoint behind HTTPS (a reverse proxy with a local certificate) rather than relying on the private-range carve-out. Tightening the predicate to loopback-only would refuse remote ASR at construction and silently degrade styling to passthrough for every LAN user, so it is a posture change rather than a fix.

### 3. Dictionary Application

**Crates:** `lune-voice`, `lune-dict`  
**Files:** `crates/lune-voice/src/session.rs`, `crates/lune-dict/src/corrector.rs`

After ASR segments are collected, the session runs a deterministic text pass:

1. Voice commands are stripped from the ASR text (whole-utterance matching).
2. The deterministic corrector (`LexicalGate::correct_with_risk_threshold` in `corrector.rs`, via `correct_and_format` in `session.rs`) rewrites the transcript: user-dictionary matches (exact, plus single-word fuzzy gated by `FUZZY_DICT_RISK_GATE`), bundled-vocab recognition (exact case-insensitive re-casing only — no fuzzy guessing), and the `ALWAYS_REPLACE` built-in mishear→canonical table. `formatting` then applies capitalization, terminal punctuation, filler stripping, and wrapping-quote removal.
3. The corrected text and the dictionary entries are packaged into `PostProcessInput` for the optional LLM styling stage.

Styling is not gated on word confidence. It runs only when a style preset is selected: when `default_style` is `None`, the LLM is skipped and the deterministic output is delivered as-is. (`apply_dictionary_seed`, below, is a secondary pass inside the styling stage — it only affects the text sent to the LLM, not this deterministic output.)

### 4. LLM Styling

**Crate:** `lune-llm` | **File:** `crates/lune-llm/src/pipeline.rs`

The session calls `StylePipeline::transform()` for LLM post-processing.

`PostProcessInput` carries:

| Field | Meaning |
|-------|---------| 
| `raw_text` | ASR transcript after dictionary substitution |
| `dictionary` | Entries for the pre-LLM seed substitution; never sent to the model |
| `style_preset` | Output style contract |
| `app_category` | Focused-application hint |
| `language` | BCP-47 code from ASR detection, `"en"` when unset |

`StylePipeline` runs two steps:

1. **Dictionary seed** — `apply_dictionary_seed` replaces dictionary phrases in the raw text (Unicode case-insensitive, longest-match). This happens before the LLM ever sees the text.
2. **Styling pass** — runs only when a style preset is selected. When `default_style` is `None` the LLM is skipped and the seeded text is returned; for any selected preset the LLM is called via `llm.complete()`. Output is run through `fix_all_number_spaces` and a `contains_forbidden_output` filter; a failing or empty output keeps the seeded text.

**Reasoning suppression:** a reasoning model puts its private thinking somewhere in the response, and none of it may reach the user's document. The backend reads `message.content` alone, so reasoning returned in a separate `reasoning_content` field is discarded by construction; reasoning inlined into the content is stripped by `strip_reasoning` — a wrapped `<think>` / `<thinking>` / `<reasoning>` / `<thought>` block in any case and repeated any number of times, an unwrapped block ending at one of those closing tags (what a template that pre-fills the opening tag produces), or harmony channels, of which only `<|channel|>final<|message|>` is kept. Reasoning that never closes collapses to an empty string, which the pipeline treats as "keep the text we already had".

When no styling engine is configured (`llm_backend` is `Disabled`), the runtime uses `NullPostProcessor`, which returns the seeded text unchanged. When the configured remote engine's request fails, `StylePipeline` keeps the seeded (dictionary-substituted) text instead of failing the session.

The LLM backend (`OpenAiLlmBackend`) wraps an `async-openai` `Client` and makes one chat-completion HTTP request per call, directly on the async runtime — no local model, no blocking thread pool.

### 5. Text Delivery

**Crate:** `lune-voice` | **Files:** `crates/lune-voice/src/delivery.rs`, `uinput_typer.rs`, `ibus.rs`

The session emits two `SessionEvent` variants directly on the event channel:

- `PartialText` — in-progress signal for live preview
- `FinalizedText { session_id, text }` — styled text for delivery

Delivery runs on a dedicated thread and picks a backend once at startup:

- **IBus** — used when IBus is the active input method (e.g. GNOME/Wayland; detected via `GTK_IM_MODULE` / `QT_IM_MODULE` / `XMODIFIERS`). Commits Unicode straight into the focused input context — no terminal detection, paste chord, or clipboard.
- **uinput** — the universal fallback. Drives a virtual keyboard through `/dev/uinput`.

`dictation_mode` selects the delivery method; `copy_to_clipboard` is an orthogonal toggle for whether a copy is *kept*:

| Mode | uinput path | IBus path |
|------|-------------|-----------|
| `TypeInPlace` | Types each key (US-QWERTY). Non-ASCII text routes through the clipboard paste path so it is never dropped. | `commit_text` |
| `Clipboard` | Sets the clipboard (`wl-copy`/`xclip`), detects terminals, sends Ctrl+V / Ctrl+Shift+V. Adaptive post-paste delay (EMA of clipboard RTT); per-key fallback for paste-hostile apps. | `commit_text` |

**Undeliverable text.** Delivery reports back whether the text reached the focused app. Multi-line text is never typed key-by-key — a newline would press Enter into whatever has focus — so if the clipboard is also unavailable the text is dropped rather than sent, and the session raises a warning instead of reporting success.

**Clipboard preservation.** The `Clipboard` path borrows the system clipboard as a transport. When `copy_to_clipboard` is off, the prior clipboard is snapshotted before delivery (its preferred MIME type and raw bytes, so images survive) and restored afterward. When `copy_to_clipboard` is on, the dictated text is kept and the otherwise-redundant re-copy is skipped. The IBus path never touches the clipboard.

---

## Session Lifecycle

```
Idle
  |
  | start()
  v
Active      audio flowing, ASR buffering
  |
  | stop() / cancel() / audio closed
  v
Teardown    poll final ASR, apply dictionary, style, finalize DB
  |
  v
Stopped     SessionEvent::Stopped emitted
```

Only one session may be active at a time (enforced by `Arc<Mutex<Option<ActiveHandle>>>`). The session task runs on the Tauri async runtime via `tokio::task::spawn`.

**Teardown sequence:**

1. Stop or cancel drains recording control messages.
2. Final ASR segments are polled (batch decode via `poll_final`) unless the session was cancelled.
3. Empty sessions finish without transcript output.
4. Voice commands are routed before dictionary application.
5. Dictionary seed and `StylePipeline` run on the transcript.
6. Final text is emitted, delivered, and persisted.

Cancellation is checked before LLM styling and again before final output. A cancellation during styling marks the session cancelled and skips `FinalizedText`.

Terminal events are sent with a timeout so a stalled listener cannot wedge teardown. A `Stopped` that follows a dropped `Error` carries `error_dropped`, so the listener still reports the session as failed.

---

## Desktop Bridge

The desktop app does not scrape logs or poll for session state. It uses typed Tauri commands for control and a `Channel<SessionStreamEvent>` for live runtime events.

### Commands

| Command | Window access | Result |
|---------|---------------|--------|
| `load_workspace_snapshot` | `main` | Settings, history, dictionary, style presets, service status |
| `save_workspace_settings` | `main` | Updated workspace snapshot |
| `create_dictionary_entry` / `delete_dictionary_entry` | `main` | Updated workspace snapshot |
| `subscribe_session_stream` / `detach_session_stream` | `main` | Live session event subscription |
| `start_dictation` / `stop_dictation` / `cancel_dictation` / `toggle_dictation` | `main`, `pill` | Session control |
| `load_pill_snapshot` | `pill` | Minimal state for the overlay |
| `nudge_pill` / `save_pill_position` | `pill` | Drag the overlay and persist where it lands (webview pill only) |
| `reset_pill_position` | `main` | Clear the saved position back to the anchor default |
| `list_monitors` | `main` | Connected displays for the pill's monitor picker |
| `gpu_backend_status` | `main` | Runtime GPU state: `unsupported`, `driver-missing`, `not-installed`, `ready`, `failed` |
| `install_gpu_backend` | `main` | Downloads and verifies the CUDA plugin, streaming progress over a channel |
| `list_history_page` / `delete_session` | `main` | History paging and deletion |
| `update_dictionary_entry` | `main` | Edit a dictionary entry in place |
| `save_snippet` / `delete_snippet` | `main` | Snippet management |
| `create_custom_style` / `update_custom_style` / `delete_custom_style` / `generate_style_examples` | `main` | Custom style presets |
| `list_audio_input_devices` | `main` | Input device picker |
| `start_trigger_capture` | `main` | Capture-only session for the snippet voice-trigger recorder (transcript streamed back, nothing typed, no history row) |
| `check_model_status` / `download_model_cmd` / `delete_model_cmd` | `main` | Model management |
| `export_transcript` | `main` | Write a session transcript to a file |
| `app_version` / `start_update_check` | `main` | Version string and on-demand update check |
| `initialize_pipeline` / `quit_app` | `main` | Runtime lifecycle |

### Session Stream

The stream uses a tagged enum serialized in camel case:

```json
{ "type": "sessionStarted", "sessionId": "..." }
{ "type": "audioLevel", "level": 0.42 }
{ "type": "finalizedText", "text": "..." }
```

The frontend store treats this stream as the source of truth for live runtime state. Workspace snapshots remain the source of truth for persisted data. When a completed session writes text, the backend emits `lune://history-changed`; the bridge reloads the snapshot once and updates history.

### The Pill Overlay

Two implementations, chosen at compile time, behind one interface in `pill.rs` — which owns the config, the saved position and the placement maths for both.

- **Linux — native.** A bare `gtk::Window` with a 32-bit ARGB visual, drawn with cairo (`pill_native.rs` + `pill_paint.rs`). WebKitGTK's accelerated compositor loses the alpha channel on the Nvidia + GNOME stack, so a webview pill can only ever be a flat opaque rectangle; cairo draws through XRender and never enters that path. Not a Tauri window: every Tauri window carries a webview, and the pill needs none. tao *is* GTK3 on Linux and drives the one GTK main loop, so the window just joins it.
- **Windows/macOS — webview.** A Tauri window rendering `pill.html` (`src/features/dictation/`), styled by the `.pill-*` rules in `components.css`.

Both draw the same thing: a bar with a circular control at each end while listening — stop-and-transcribe on the left (top when vertical), discard on the right — and the level meter between them, each chip centred in the space between the bar's end and the meter. Outside a live session both caps are absent, and the bar itself starts a session. The two implementations are kept visually in step by hand; what is checked is the shared numbers. `pill_paint.rs` mirrors the theme's oklch tokens and the meter's geometry, and unit tests read `tokens.css`/`daisyui.css` and `dot-style.ts` back, failing if either side moves.

State reaches the native pill by direct call (`pill::publish_status` / `publish_audio_level`), and the webview pill by targeted event, so it can react without owning the main session channel:

| Event | Payload |
|-------|---------| 
| `lune://pill-state` | `"ready"`, `"listening"`, `"processing"`, or `"error"` |
| `lune://audio-level` | Float in `[0, 1]` |
| `lune://pill-settings` | `pillLocked` |

Rust shows or hides the pill directly on status changes (`should_show_pill`) — the frontend never decides visibility, so the mapped window and its contents cannot desync.

### Sound Cues

Short audio cues play from the Rust backend on status transitions — Start on `listening`, Stop when leaving an active state, Error on `error` — deduplicated on the previous status so each fires once. Embedded 16-bit mono WAVs are decoded and played through a short-lived `cpal` output stream on one serialized worker thread, behind a small bounded queue so a burst of transitions cannot pile up streams. Playback is gated by `sound_cues_enabled` / `sound_cue_volume` and every failure is swallowed, so audio never disrupts dictation.

---

## Threading Model

```
Tauri async runtime (tokio multi_thread)
|-- Tauri IPC commands (commands.rs) - typed command handlers
|-- Session task (tokio::task::spawn) - orchestrates one session
|   |-- Audio push loop (raw PCM to ASR buffer)
|   |-- poll_final (batch decode; handed off via block_in_place)
|   |-- Dictionary + voice command pass
|   |-- StylePipeline transform (LLM call is a plain async HTTP request)
|   `-- Text delivery (handed to the delivery thread)
|-- Event forwarder (spawn) - routes SessionEvents to the UI stream
`-- Hotkey listener (spawn) - reacts to the global shortcut

tokio blocking pool (block_in_place / spawn_blocking)
|-- whisper.cpp inference (WhisperContext shared; per-session WhisperState via Mutex)
|-- spellcheck / dictionary work
`-- persistence I/O (run_persist, pill settings)

dedicated OS threads
|-- delivery thread (uinput / IBus / clipboard)
`-- hotkey-evdev thread (global shortcut capture)
```

The runtime is **tokio multi-thread** (Tauri's default). The `WhisperContext` is
wrapped in an `Arc` and shared across all sessions. Each session creates a
lightweight `WhisperState` via `ctx.create_state()`. The state is held behind a
`Mutex<WhisperCppSession>` so the async and blocking layers can both access it
safely.

Blocking work is dispatched as follows:

- **whisper decode** (`poll_final`) uses `block_in_place`, not `spawn_blocking`:
  the whisper session is `!Send` (it wraps raw C pointers) and cannot move to
  the blocking pool. `block_in_place` hands the current worker thread off to
  the blocking pool so other tasks keep running during the multi-second decode.
  On a multi-thread runtime (the app) this is a true handoff; on a current-thread
  runtime (unit tests) it runs inline and warns once.
- **LLM styling** is a plain async HTTP request (via `async-openai`), so it runs
  directly on the async runtime.
- **Persistence** never stalls the runtime. The Tauri command layer and the
  GPU-plugin unpack use `spawn_blocking`. The session lifecycle cannot: its
  `Database` lives behind a shared `tokio::sync::Mutex` whose guard cannot cross
  into a spawned task, so it uses `block_in_place` instead — same handoff, no
  `Send` bound, and no extra cancellation point in `start_inner`.

The audio capture callback runs on cpal's own high-priority thread and pushes
into a lock-free ring buffer; the session loop polls it, so the real-time path
does no allocation and takes no locks.

---

## Data Flow Diagram

```
                 lune-voice runtime

lune-audio (cpal)
  |
  | AudioFrame over mpsc (PCM, 16 kHz mono f32, post-AGC)
  v
Session task (capture.rs)
  |
  |-- push_audio → ASR buffer accumulates
  |-- [stop signal] → poll_final → Silero endpoint trim → whisper.cpp decode
  |-- voice command routing
  |-- apply_dictionary_seed
  |-- StylePipeline::transform (LLM styling, optional)
  `-- FinalizedText
        |
        v
Event forwarder
  |
  |-- delivery service
  |-- Tauri session stream
  `-- history-changed event

Tauri UI
  |
  `-- start, stop, cancel, settings

lune-persist
  |
  `-- SQLite history, settings, model downloads
```
