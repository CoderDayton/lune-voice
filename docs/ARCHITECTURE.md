# Architecture

Lune Voice has one dictation pipeline and two control surfaces. The CLI talks to the daemon through D-Bus. The desktop app talks to the same runtime through Tauri commands and a typed session stream.

---

## Table of Contents

- [System Overview](#system-overview)
- [Crate Dependency Graph](#crate-dependency-graph)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Audio Capture](#1-audio-capture)
  - [ASR](#2-asr)
  - [Correction and LLM Gate](#3-correction-and-llm-gate)
  - [LLM Post-Processing](#4-llm-post-processing)
  - [Text Delivery](#5-text-delivery)
- [Session Lifecycle](#session-lifecycle)
- [Desktop Bridge](#desktop-bridge)
- [D-Bus Interface](#d-bus-interface)
- [Threading Model](#threading-model)
- [Data Flow Diagram](#data-flow-diagram)

---

## System Overview

Lune Voice turns microphone audio into corrected text, then delivers that text to the focused app. The runtime keeps one active session at a time. Everything else exists to feed that session, configure it, or show its state.

The runtime pipeline is split into interchangeable backends:

| Component | Primary Backends | Format | Runtime |
|-----------|------------------|--------|---------|
| ASR | Qwen3-ASR 0.6B, OpenAI-compatible remote ASR | ONNX / remote API | ONNX Runtime / HTTP |
| LLM | Local style model, OpenAI-compatible remote LLM, disabled passthrough | ONNX / remote API | ONNX Runtime / HTTP |

Backend selection is centralized through registries and DTO mappers. The desktop UI never builds model paths or backend structs itself; it sends typed settings and receives snapshots.

---

## Crate Dependency Graph

```
lune-app (binary)
|-- lune-audio       Microphone capture (cpal)
|-- lune-asr         Speech recognition (ONNX Runtime)
|   `-- lune-core    Shared traits + types
|-- lune-llm         Text post-processing (local ONNX or OpenAI-compatible)
|   `-- lune-core
|-- lune-commands    Voice command detection
|-- lune-dict        User dictionary
`-- lune-persist     Settings, DB (SQLite), model downloads
    `-- lune-core
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

The active ASR choices are local Qwen3-ASR and OpenAI-compatible remote ASR. Both feed the same `AsrBackend` trait, so the session code does not branch on backend type.

#### 2a. Qwen3-ASR ONNX

**Files:** `backend/qwen3.rs`, `decode/`, `mel.rs`

The Qwen path uses a three-part ONNX export produced by `scripts/export_onnx.py`:

1. `qwen3-asr-encoder.onnx` for the audio tower / projector
2. `qwen3-asr-embed-tokens.onnx` for token embeddings
3. `qwen3-asr-decoder.onnx` for the decoder with KV cache

The runtime feeds whole-clip or chunked mel features through the encoder, injects the exported prompt contract via the token embedder, and steps the decoder autoregressively while maintaining cache tensors between decode steps.

The decoder also emits word confidence data when available. Session correction uses those word risks to decide which spans should be sent to the LLM.

#### 2b. OpenAI-Compatible ASR

Remote ASR is useful when the user explicitly allows remote processing. It returns transcript text through the same segment type, but it may not include per-word confidence. Missing confidence is treated conservatively by the LLM gate.

### 3. Correction and LLM Gate

**Crates:** `lune-app`, `lune-dict`
**Files:** `crates/lune-app/src/session.rs`, `crates/lune-dict/src/corrector.rs`

The session first gives cheap deterministic code the chance to fix the transcript. The LLM only receives text when the remaining risk justifies it.

The sequence is:

1. Voice commands are stripped from the ASR text.
2. The dictionary manager scans the remaining text.
3. `HeuristicCorrector` applies dictionary replacements and spelling fixes.
4. The corrector returns `word_risk_positions`, mapped to the corrected word positions.
5. `build_correction_spans` expands high-risk words into short editable spans.
6. The bypass gate decides whether to call the LLM.

The risk mapping lives in the corrector because that is the only point where original ASR words and corrected words are still paired. This keeps scores attached when `teh` becomes `the`, and it keeps duplicate words ordered by position instead of by string match.

The LLM gate uses two signals:

| Signal | Effect |
|--------|--------|
| `unresolved_words` | Heuristic could not safely fix a flagged word |
| `correction_spans` | A word risk crossed the LLM span threshold |

Unsupported dictionaries are handled conservatively. If the language has no bundled dictionary, high ASR risk can still force LLM review.

### 4. LLM Post-Processing

**Crate:** `lune-llm`

The session calls `TextPostProcessor::transform()` inline. There is no LLM worker queue in the current desktop runtime.

`PostProcessInput` carries structured data:

| Field | Meaning |
|-------|---------|
| `raw_text` | Heuristic-corrected transcript |
| `dictionary` | Relevant user dictionary entries |
| `correction_spans` | Editable ranges around high-risk words |
| `uncertain_words` | Words the heuristic left unresolved |
| `style_preset` | Output style contract |

Local LLM mode uses the ONNX style model when it is downloaded. If it is missing or fails to load, the runtime uses `NullPostProcessor` and returns the corrected text unchanged. Remote LLM mode uses an OpenAI-compatible chat API.

### 5. Text Delivery

**Crate:** `lune-app` | **File:** `crates/lune-app/src/delivery.rs`

The session emits two `SessionEvent` variants directly on the event channel:

- `PartialText` - in-progress signal for live preview
- `FinalizedText(String)` - corrected text for delivery

Delivery modes:

| Mode | Implementation |
|------|----------------|
| TypeInPlace | Clipboard paste via uinput: `wl-copy`/`xclip` -> terminal detection -> Ctrl+V or Ctrl+Shift+V. Adaptive post-paste delay (EMA of clipboard RTT). Per-key fallback for paste-hostile apps. |
| Clipboard | `wl-copy` (Wayland) / `xclip` (X11) - clipboard only, no paste chord |
| Both | Clipboard paste (same as TypeInPlace; clipboard is set as side effect) |

---

## Session Lifecycle

```
Idle
  |
  | start()
  v
Active      audio flowing, ASR running
  |
  | stop() / cancel() / audio closed
  v
Teardown    poll final ASR, correct, finalize DB
  |
  v
Stopped     SessionEvent::Stopped emitted
```

Only one session may be active at a time (enforced by `Arc<Mutex<Option<ActiveHandle>>>`). The session task runs on `spawn_local` because it holds a `!Send` cpal audio stream.

**Teardown sequence:**

1. Stop or cancel drains recording control messages.
2. Final ASR segments are polled unless the session was cancelled.
3. Empty sessions finish without transcript output.
4. Voice commands are routed before correction.
5. Dictionary and heuristic correction run.
6. LLM post-processing runs only when the gate asks for it.
7. Final text is emitted, delivered, and persisted.

Cancellation is checked before LLM work and again before final output. A cancellation during LLM work marks the session cancelled and skips `FinalizedText`.

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

The main window can start and stop dictation because its sidebar has a first-class dictation control. Pill-only commands stay pill-only.

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

Rust also shows or hides the pill directly on status changes. That avoids a frontend roundtrip for the common start/stop path.

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
| `GetAllSettings` | TOML string (read-only dump) | - |

All setters persist changes to disk immediately via fire-and-forget `spawn_local`. Settings are stored in `Arc<RwLock<UserSettings>>` - the runtime source of truth.

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
|   |-- Audio coalescing (VAD-gated)
|   |-- ASR push + poll (streaming encoder-decoder)
|   |-- Per-word confidence marking
|   |-- LLM transform (inline in session loop)
|   `-- Text delivery
`-- Event forwarder (spawn_local) - routes SessionEvents to delivery

spawn_blocking pool (tokio default)
`-- ASR / local LLM inference - runs under Mutex, CUDA or CPU
```

Why `current_thread` instead of `multi_thread`:
- Avoids work-stealing overhead for this single-pipeline daemon
- All hot paths stay on one core (cache-friendly)
- `spawn_local` support for `!Send` futures (cpal stream)

ORT sessions are wrapped in `Arc<Mutex<Session>>` for `Send` safety across `spawn_blocking`. Session control stays on the async side; model execution moves to the blocking pool.

---

## Data Flow Diagram

```
                 lune-app runtime

lune-audio (cpal)
  |
  | AudioFrame over mpsc
  v
Session task
  |
  |-- ASR push + poll
  |-- HeuristicCorrector
  |-- correction spans + word risks
  |-- LLM gate
  |-- TextPostProcessor or passthrough
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
