# Architecture

Lune Voice has one dictation pipeline and two control surfaces. The CLI talks to the daemon through D-Bus. The desktop app talks to the same runtime through Tauri commands and a typed session stream.

---

## Table of Contents

- [System Overview](#system-overview)
- [Crate Dependency Graph](#crate-dependency-graph)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Audio Capture](#1-audio-capture)
  - [ASR](#2-asr)
  - [Dictionary Application](#3-dictionary-application)
  - [LLM Styling](#4-llm-styling)
  - [Text Delivery](#5-text-delivery)
- [Session Lifecycle](#session-lifecycle)
- [Desktop Bridge](#desktop-bridge)
- [D-Bus Interface](#d-bus-interface)
- [Threading Model](#threading-model)
- [Data Flow Diagram](#data-flow-diagram)

---

## System Overview

Lune Voice turns microphone audio into styled text, then delivers that text to the focused app. The runtime keeps one active session at a time. Everything else exists to feed that session, configure it, or show its state.

Dictation is **hotkey-bounded**: the session collects audio for as long as the hotkey is held (Hold mode) or until the hotkey is pressed again (Toggle mode). There is no online VAD gate during recording. The Whisper backend applies its built-in Silero VAD to the buffered audio at decode time.

The runtime pipeline uses interchangeable backends:

| Component | Backends | Model format | Runtime |
|-----------|----------|--------------|---------|
| ASR | WhisperCpp (local), OpenAI-compatible (remote) | GGML `.bin` / remote API | whisper-rs (whisper.cpp) / HTTP |
| LLM | OpenAI-compatible (remote), disabled | remote API | HTTP |

The native ASR backend compiles its C/C++ library from source at build time via a `*-sys` crate. GPU (CUDA) support is selected at **build time** via the `whisper-cuda` Cargo feature; there is no runtime backend autodetection. The styling LLM is remote-only — point it at Ollama, llama-server, LM Studio, or a hosted API. See [docs/RELEASE.md](RELEASE.md) for the full build matrix.

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
     └─ lune-app (session, factory, hotkey, dbus, delivery, sound cues)
          └─ desktop-ui/src-tauri (tauri IPC, pill, persistence bridge)
```

Each crate has a single responsibility. `lune-core` defines shared types and traits but contains no runtime logic. All inter-crate communication flows through trait objects and channels.

---

## Pipeline Walkthrough

### 1. Audio Capture

**Crate:** `lune-audio` | **File:** `crates/lune-audio/src/lib.rs`  
**Orchestration:** `crates/lune-app/src/capture.rs`

- Uses `cpal` for cross-platform microphone access (ALSA/PipeWire on Linux, CoreAudio on macOS, WASAPI on Windows)
- Captures 16 kHz mono PCM in frames of 480 samples (30 ms each) — already in whisper.cpp's native input format; no resample or conversion needed
- Raw audio chunks are pushed straight to the ASR backend via `push_audio` as they arrive; there is no online VAD gate at this stage
- The session loop handles Stop/Cancel/deadline/max-speech control signals and terminates the capture loop accordingly
- After Stop, a 30 ms flush recovers the final callback buffer (last phonemes of the final word), then `poll_final` decodes the full buffer

### 2. ASR

**Crate:** `lune-asr`

Both backends implement the same `AsrBackend` trait, so the session code does not branch on backend type.

#### 2a. WhisperCpp (local, default)

**File:** `crates/lune-asr/src/backend/whisper_cpp.rs`

The `WhisperCppBackend` wraps `whisper-rs`, which vendors and compiles `whisper.cpp`:

- The `WhisperContext` (model weights) is loaded **once** at backend construction and shared across all sessions. Per-session `WhisperState` is lightweight (a decode context).
- `push_audio` appends PCM to a per-session buffer pre-allocated for 30 s of audio (grows once for longer utterances; reset without freeing after each decode).
- `poll_final` decodes the entire buffer in one shot using **BeamSearch** (beam size 5) — whisper.cpp's accuracy-oriented default. Decoding is one-shot per utterance, so the extra compute is worthwhile.
- **Built-in Silero VAD:** when the VAD model path is set (`models/whisper-vad-silero/ggml-silero-v5.1.2.bin`), `poll_final` calls `FullParams::set_vad_model_path` + `enable_vad` + `set_vad_params` before decode. Whisper gates the buffer to speech spans internally before running the transformer. If the VAD model file is missing at construction, the backend logs a warning and decodes without VAD.
- **Per-token confidence:** whisper exposes per-token probabilities. `poll_final` folds them into `AsrSegment::confidence` (mean) and `min_token_confidence` (minimum), excluding special tokens. Segments whose mean confidence falls below `GARBAGE_DROP_THRESHOLD` (0.25) are dropped by the capture layer before reaching the session.
- Language detection: the `language` field from `AsrSessionConfig` is passed to `set_language`; `None` uses `"auto"` (whisper detects from audio).

Three GGML model sizes are available, all from `ggerganov/whisper.cpp` as q8_0 quantizations:

| Variant | File | Size | Directory |
|---------|------|------|-----------|
| `Small` | `ggml-small-q8_0.bin` | ~264 MB | `whisper-small-q8_0` |
| `Medium` *(default)* | `ggml-medium-q8_0.bin` | ~823 MB | `whisper-medium-q8_0` |
| `LargeV3Turbo` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB | `whisper-large-v3-turbo-q8_0` |

#### 2b. OpenAI-Compatible (remote)

Remote ASR is useful when the user explicitly allows remote processing. It returns transcript text through the same `AsrSegment` type but does not expose per-token confidence. Missing confidence disables LLM routing confidence gates; the styling call is still driven by the style preset.

### 3. Dictionary Application

**Crates:** `lune-app`, `lune-dict`  
**Files:** `crates/lune-app/src/session.rs`, `crates/lune-dict/src/corrector.rs`

After ASR segments are collected, the session runs a deterministic text pass:

1. Voice commands are stripped from the ASR text (whole-utterance matching).
2. The deterministic corrector (`LexicalGate::correct_with_risk_threshold` in `corrector.rs`, via `correct_and_format` in `session.rs`) rewrites the transcript: user-dictionary matches (exact, plus single-word fuzzy gated by `FUZZY_DICT_RISK_GATE`), bundled-vocab recognition (exact case-insensitive re-casing only — no fuzzy guessing), and the `ALWAYS_REPLACE` built-in mishear→canonical table. `formatting` then applies capitalization, terminal punctuation, filler stripping, and wrapping-quote removal.
3. The corrected text and the top dictionary entries are packaged into `PostProcessInput` for the optional LLM styling stage.

Styling is not gated on word confidence. It runs only when a style preset is selected: when `default_style` is `None`, the LLM is skipped and the deterministic output is delivered as-is. (`apply_dictionary_seed`, below, is a secondary pass inside the styling stage — it only affects the text sent to the LLM, not this deterministic output.)

### 4. LLM Styling

**Crate:** `lune-llm` | **File:** `crates/lune-llm/src/pipeline.rs`

The session calls `StylePipeline::transform()` for LLM post-processing.

`PostProcessInput` carries:

| Field | Meaning |
|-------|---------| 
| `raw_text` | ASR transcript after dictionary substitution |
| `dictionary` | Top dictionary entries (injected into the prompt) |
| `style_preset` | Output style contract |

`StylePipeline` runs two steps:

1. **Dictionary seed** — `apply_dictionary_seed` replaces dictionary phrases in the raw text (Unicode case-insensitive, longest-match). This happens before the LLM ever sees the text.
2. **Styling pass** — runs only when a style preset is selected. When `default_style` is `None` the LLM is skipped and the seeded text is returned; for any selected preset the LLM is called via `llm.complete()`. Output is run through `fix_all_number_spaces` and a `contains_forbidden_output` filter; a failing or empty output keeps the seeded text.

**Thinking suppression:** a Qwen-style remote endpoint may emit an inline `<think>…</think>` reasoning block; the backend strips it from the response before it reaches the pipeline.

When no styling engine is configured (`llm_backend` is `Disabled`), the runtime uses `NullPostProcessor`, which returns the seeded text unchanged. When the configured remote engine's request fails, `StylePipeline` keeps the seeded (dictionary-substituted) text instead of failing the session.

The LLM backend (`OpenAiLlmBackend`) wraps an `async-openai` `Client` and makes one chat-completion HTTP request per call, directly on the async runtime — no local model, no blocking thread pool.

### 5. Text Delivery

**Crate:** `lune-app` | **Files:** `crates/lune-app/src/delivery.rs`, `uinput_typer.rs`, `ibus.rs`

The session emits two `SessionEvent` variants directly on the event channel:

- `PartialText` — in-progress signal for live preview
- `FinalizedText(String)` — styled text for delivery

Delivery runs on a dedicated thread and picks a backend once at startup:

- **IBus** — used when IBus is the active input method (e.g. GNOME/Wayland; detected via `GTK_IM_MODULE` / `QT_IM_MODULE` / `XMODIFIERS`). Commits Unicode straight into the focused input context — no terminal detection, paste chord, or clipboard.
- **uinput** — the universal fallback. Drives a virtual keyboard through `/dev/uinput`.

`dictation_mode` selects the delivery method; `copy_to_clipboard` is an orthogonal toggle for whether a copy is *kept*:

| Mode | uinput path | IBus path |
|------|-------------|-----------|
| `TypeInPlace` | Types each key (US-QWERTY). Non-ASCII text routes through the clipboard paste path so it is never dropped. | `commit_text` |
| `Clipboard` | Sets the clipboard (`wl-copy`/`xclip`), detects terminals, sends Ctrl+V / Ctrl+Shift+V. Adaptive post-paste delay (EMA of clipboard RTT); per-key fallback for paste-hostile apps. | `commit_text` |

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

Only one session may be active at a time (enforced by `Arc<Mutex<Option<ActiveHandle>>>`). The session task runs on `spawn_local` because it holds a `!Send` cpal audio stream.

**Teardown sequence:**

1. Stop or cancel drains recording control messages.
2. Final ASR segments are polled (batch decode via `poll_final`) unless the session was cancelled.
3. Empty sessions finish without transcript output.
4. Voice commands are routed before dictionary application.
5. Dictionary seed and `StylePipeline` run on the transcript.
6. Final text is emitted, delivered, and persisted.

Cancellation is checked before LLM styling and again before final output. A cancellation during styling marks the session cancelled and skips `FinalizedText`.

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
| `show_pill_window` / `hide_pill_window` / `reposition_pill_center` | `pill` | Overlay window control |
| `begin_pill_drag` / `drag_pill_tick` / `end_pill_drag` / `nudge_pill` | `pill` | Drag and fine-position the overlay |
| `save_pill_position` / `reset_pill_position` | `pill` | Persist a dragged position or clear it back to the anchor default |
| `list_monitors` | `main` | Connected displays for the pill's monitor picker |
| `gpu_support` | `main` | Returns `{ compiled_in: bool }` — true when built with `whisper-cuda` |

### Session Stream

The stream uses a tagged enum serialized in camel case:

```json
{ "type": "sessionStarted", "sessionId": "..." }
{ "type": "audioLevel", "level": 0.42 }
{ "type": "finalizedText", "text": "..." }
```

The frontend store treats this stream as the source of truth for live runtime state. Workspace snapshots remain the source of truth for persisted data. When a completed session writes text, the backend emits `lune://history-changed`; the bridge reloads the snapshot once and updates history.

### Pill Events

The pill receives targeted events so it can react without owning the main session channel:

| Event | Payload |
|-------|---------| 
| `lune://pill-state` | `"ready"`, `"listening"`, `"processing"`, or `"error"` |
| `lune://audio-level` | Float in `[0, 1]` |
| `lune://pill-settings` | Activation mode and pill visibility |

Rust also shows or hides the pill directly on status changes to avoid a frontend roundtrip for the common start/stop path.

### Sound Cues

Short audio cues play from the Rust backend on status transitions — Start on `listening`, Stop when leaving an active state, Error on `error` — deduplicated on the previous status so each fires once. Embedded 16-bit mono WAVs are decoded and played through a short-lived `cpal` output stream on a detached thread. Playback is gated by `sound_cues_enabled` / `sound_cue_volume` and every failure is swallowed, so audio never disrupts dictation.

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
| `Get/SetDictationMode` | string (`TypeInPlace`\|`Clipboard`; legacy `Both` maps to `TypeInPlace`) | Immediate (watch channel) |
| `Get/SetActivationMode` | string (`Toggle`\|`Hold`) | Immediate (watch channel) |
| `Get/SetDefaultStyle` | string (preset ID, validated against registry) | Next session start |
| `Get/SetLanguage` | string (BCP-47, well-formedness validated) | Next session start |
| `Get/SetPrivacyMode` | string (`LocalOnly`\|`LocalPreferred`\|`AllowRemote`) | Persist only |
| `Get/SetSessionHistoryDays` | i32 (-1 = disabled, 0+ = days) | Persist only |
| `Get/SetHotkey` | JSON string of `HotkeyConfig` | Persist only (external DE bindings) |
| `Get/SetAsrBackend` | JSON string of `AsrBackendConfig` | Daemon restart required |
| `Get/SetLlmBackend` | JSON string of `LlmBackendConfig` | Daemon restart required |
| `GetAllSettings` | TOML string (read-only dump) | - |

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
|-- D-Bus connection (zbus) - initialized outside LocalSet
|-- Command loop - receives AppCommand from D-Bus
|-- Session task (spawn_local) - !Send due to cpal stream
|   |-- Audio push loop (raw PCM to ASR buffer)
|   |-- poll_final (batch decode; blocking call dispatched to spawn_blocking)
|   |-- Dictionary + voice command pass
|   |-- StylePipeline transform (LLM call dispatched to spawn_blocking)
|   `-- Text delivery
`-- Event forwarder (spawn_local) - routes SessionEvents to delivery

spawn_blocking pool (tokio default)
`-- whisper.cpp inference (WhisperContext shared; per-call WhisperState via Mutex)
```

The remote LLM call is a plain async HTTP request (via `async-openai`), so it runs directly on the async runtime rather than `spawn_blocking`.

Why `current_thread` instead of `multi_thread`:
- Avoids work-stealing overhead for this single-pipeline daemon
- All hot paths stay on one core (cache-friendly)
- `spawn_local` support for `!Send` futures (cpal stream)

The `WhisperContext` is wrapped in an `Arc` and shared across all sessions. Each session creates a lightweight `WhisperState` via `ctx.create_state()`. The state is held behind a `Mutex<WhisperCppSession>` so the async and blocking layers can both access it safely.

---

## Data Flow Diagram

```
                 lune-app runtime

lune-audio (cpal)
  |
  | AudioFrame over mpsc (raw PCM, 16 kHz mono f32)
  v
Session task (capture.rs)
  |
  |-- push_audio → ASR buffer accumulates
  |-- [stop signal] → poll_final → whisper.cpp decode
  |   (built-in Silero VAD gates speech spans at decode time)
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

D-Bus / Tauri UI
  |
  `-- start, stop, cancel, settings

lune-persist
  |
  `-- SQLite history, settings, model downloads
```
