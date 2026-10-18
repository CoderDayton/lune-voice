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


| Component | Backends                                       | Model format             | Runtime                         |
| --------- | ---------------------------------------------- | ------------------------ | ------------------------------- |
| ASR       | WhisperCpp (local), OpenAI-compatible (remote) | GGML `.bin` / remote API | whisper-rs (whisper.cpp) / HTTP |
| LLM       | OpenAI-compatible (remote), disabled           | remote API               | HTTP                            |


The native ASR backend compiles its C/C++ library from source at build time via a `*-sys` crate. GPU (CUDA) support on Linux is a **runtime** capability: ggml's compute backends are built as `dlopen`-able plugins (`GGML_BACKEND_DL`), so the CUDA backend is downloaded on request rather than linked in — see [docs/GPU_PLUGIN.md](GPU_PLUGIN.md). The styling LLM is remote-only — point it at Ollama, llama-server, LM Studio, or a hosted API. See [docs/RELEASE.md](RELEASE.md) for the full build matrix.

Backend selection is centralized through registries and DTO mappers. The desktop UI never builds model paths or backend structs itself; it sends typed settings and receives snapshots.

---

## Crate Dependency Graph

```
lune-core (types, config, errors)
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

- Uses `cpal` for microphone access. Host choice lives in `crates/lune-audio/src/devices.rs` and nowhere else:
  - **Capture** (`capture_host`) uses cpal's PulseAudio host whenever a Pulse server answers — PulseAudio itself, or PipeWire through `pipewire-pulse` — and ALSA otherwise. Through the server, a mic another program holds (a noise filter, a browser) is shared; opening the raw ALSA `hw:` device instead fails with "device busy" while anything else has the card. The Pulse client is the pure-Rust `pulseaudio` crate: no system library, no headers, no extra runtime dependency
  - **Playback** (`playback_host`, sound cues) stays on ALSA's `default`, which PipeWire and PulseAudio serve through their ALSA plugins
- The picker and the device opened both go through `enumerate_input_devices`, so the list the user picks from and the device that opens cannot disagree. One entry per display name:
  - **Pulse:** sources read from the server (`crates/lune-audio/src/pulse.rs`), minus sink monitors (`monitor_of_sink_index`). Recording streams are not sources, so a filter's capture side never appears; its output source does. cpal keeps its Pulse device type private, so these facts come from a second, short-lived client of the same crate
  - **ALSA:** one card-backed capture route per name. ALSA enumerates one PCM per *route*, so otherwise 4 microphones list as 29 indistinguishable entries alongside `null` and `pipewire`
- A saved `input_device` matches a display name first, then a source's ALSA name (`"{alsa.card_name}, {alsa.name}"`, the name the ALSA host shows). Startup rewrites a setting matched the second way to the current name (`persistence::migrate_input_device`), only for a settings file that parsed. The rewrite is one-way: with no Pulse server the ALSA host has no record of a source name, so capture uses the system default and reports the fallback. Identity is the display name, so two mics that share one are a single picker entry
- Pulse capture requests a fixed 20 ms period (`capture_buffer_size`). Left to the server, a record stream's fragment defaults to about 2 s, and the first audio arrives 2 s after the stream opens
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

- The `WhisperContext` (model weights) is loaded **once** at backend construction and shared across all sessions. Per-session `WhisperState` is a decode context that allocates its own KV-cache and compute buffers when created.
- `push_audio` appends PCM to a per-session buffer pre-allocated for 30 s of audio (grows once for longer utterances; reset without freeing after each decode).
- `poll_final` decodes the entire buffer in one shot using **BeamSearch** (beam size 5) — whisper.cpp's accuracy-oriented default. Decoding is one-shot per utterance, so the extra compute is worthwhile.
- **Silero endpoint trim:** when the VAD model is present (`models/whisper-vad-silero/ggml-silero-v5.1.2.bin`), `poll_final` first runs `WhisperVadContext::segments_from_samples` over the buffer and decodes only the span between the first and last speech segment. Internal pauses are kept, because whisper reads them as punctuation cues. whisper.cpp's in-decode `enable_vad` is deliberately **not** used — it concatenates speech spans and strips those pauses, producing run-ons. The VAD context always runs on CPU (silero is tiny, and a GPU context crashed at init). If the model file is missing, the backend logs a warning and decodes the whole buffer.
- **Chunking at pauses:** a trimmed span longer than 28 s is cut in the middle of a pause between two VAD speech segments and each piece is decoded on its own (`plan_chunks`). Left to its own 30 s window, whisper decodes whatever is left over separately, and a leftover of a second or less comes back as invented text ("Thank you."). A cut leaves at least 3 s on both sides and prefers the longest pause in the back half of the window, so each piece keeps its own pauses whole. A stretch with no pause is decoded in one piece.
- **Per-token confidence:** whisper exposes per-token probabilities. `poll_final` folds them into `AsrSegment::confidence` (mean) and `min_token_confidence` (minimum), excluding special tokens. Segments whose mean confidence falls below `GARBAGE_DROP_THRESHOLD` (0.25) are dropped by the capture layer before reaching the session.
- Language detection: the `language` field from `AsrSessionConfig` is passed to `set_language`; `None` uses `"auto"` (whisper detects from audio). With several chunks, the first one detects and the rest reuse its language.

Four GGML model sizes are available, all from `ggerganov/whisper.cpp`:


| Variant              | File                           | Size    | Directory                     |
| -------------------- | ------------------------------ | ------- | ----------------------------- |
| `Base`               | `ggml-base-q8_0.bin`           | ~78 MB  | `whisper-base-q8_0`           |
| `Small`              | `ggml-small-q5_1.bin`          | ~190 MB | `whisper-small-q5_1`          |
| `Medium` *(default)* | `ggml-medium-q5_0.bin`         | ~539 MB | `whisper-medium-q5_0`         |
| `LargeV3Turbo`       | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB | `whisper-large-v3-turbo-q8_0` |


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
2. The deterministic corrector (`LexicalGate::correct_with_risk_threshold` in `corrector.rs`, via `correct_and_format` in `session.rs`) rewrites the transcript: user-dictionary matches (exact — a phrase of several words, or a word carrying punctuation such as `node.js` or `c++`, matches whole transcript words as written — plus single-word fuzzy — one edit, words of four characters or more — gated by `FUZZY_DICT_RISK_GATE` and never applied to a word the lexicon knows or one carrying a digit), bundled-vocab recognition (exact case-insensitive re-casing only — no fuzzy guessing), and the `ALWAYS_REPLACE` built-in mishear→canonical table. `formatting` then applies capitalization, terminal punctuation, filler stripping, and wrapping-quote removal.
3. The corrected text and the dictionary entries are packaged into `PostProcessInput` for the optional LLM styling stage.

Styling is not gated on word confidence. It runs only when a style preset is selected: when `default_style` is `None`, the LLM is skipped and the deterministic output is delivered as-is. (`apply_dictionary_seed`, below, is a secondary pass inside the styling stage — it only affects the text sent to the LLM, not this deterministic output.)

### 4. LLM Styling

**Crate:** `lune-llm` | **File:** `crates/lune-llm/src/pipeline.rs`

The session calls `StylePipeline::transform()` for LLM post-processing.

`PostProcessInput` carries:


| Field          | Meaning                                                            |
| -------------- | ------------------------------------------------------------------ |
| `raw_text`     | ASR transcript after dictionary substitution                       |
| `dictionary`   | Entries for the pre-LLM seed substitution; never sent to the model |
| `style_preset` | Output style contract                                              |
| `language`     | BCP-47 code from ASR detection, `"en"` when unset                  |


`StylePipeline` runs two steps:

1. **Dictionary seed** — `apply_dictionary_seed` replaces dictionary phrases in the raw text (Unicode case-insensitive, longest-match). This happens before the LLM ever sees the text.
2. **Styling pass** — runs only when a style preset is selected. When `default_style` is `None` the LLM is skipped and the seeded text is returned; for any selected preset the LLM is called via `llm.complete()`. Output is run through `fix_all_number_spaces` and a `contains_forbidden_output` filter; a failing or empty output keeps the seeded text.

**Reasoning suppression:** a reasoning model puts its private thinking somewhere in the response, and none of it may reach the user's document. The backend reads `message.content` alone, so reasoning returned in a separate `reasoning_content` field is discarded by construction; reasoning inlined into the content is stripped by `strip_reasoning` — a wrapped `<think>` / `<thinking>` / `<reasoning>` / `<thought>` block in any case and repeated any number of times, an unwrapped block ending at one of those closing tags (what a template that pre-fills the opening tag produces), or harmony channels, of which only `<|channel|>final<|message|>` is kept. Reasoning that never closes collapses to an empty string, which the pipeline treats as "keep the text we already had".

When no styling engine is configured (`llm_backend` is `Disabled`), the runtime uses `NullPostProcessor`, which returns the seeded text unchanged. When the configured remote engine's request fails, `StylePipeline` keeps the seeded (dictionary-substituted) text instead of failing the session.

The LLM backend (`OpenAiLlmBackend`) wraps an `async-openai` `Client` and makes one chat-completion HTTP request per call, directly on the async runtime — no local model, no blocking thread pool.

### 5. Text Delivery

**Crate:** `lune-voice` | **Files:** `crates/lune-voice/src/delivery.rs`, `uinput_typer.rs`, `ibus.rs`, `focus.rs` + `focus/unix.rs` + `focus/windows.rs`, `probe.rs`

`delivery.rs` defines one `PlatformDriver` trait with a half per platform. `focus.rs` owns focused-window detection, terminal classification and focus restore, with a half per platform: the compositor probes in `focus/unix.rs`, the Win32 calls in `focus/windows.rs`. Every external probe (compositor sockets, `wl-paste`, `swaymsg`, `xdotool`) runs through `probe.rs`, which time-bounds it and drains both child pipes for the whole wait so a chatty child is not killed mid-answer.

**The target window is captured once, when the dictation starts** (`capture_focus_target`, run off the start path because the Unix probes can cost ~500 ms) and consumed seconds later at delivery. Two things can move focus in between: the user, deliberately, and one of our own windows — the pill above all. So capture drops the restore token when the foreground window belongs to this process, and delivery restores the captured window before typing, but only while one of our own windows holds focus. Someone who switched apps on purpose keeps the app they switched to. The restore therefore covers focus that moves *after* capture only — a dictation that starts while one of our windows is already in front has no token to restore to, which is why the pill refuses activation in the first place.

The session emits the delivery payload directly on the event channel as
`FinalizedText { session_id, text }` — styled text for delivery.

Delivery runs on a dedicated thread and picks a backend once at startup. On Linux there are two:

- **IBus** — used when IBus is the active input method (e.g. GNOME/Wayland; detected via `GTK_IM_MODULE` / `QT_IM_MODULE` / `XMODIFIERS`). Commits Unicode straight into the focused input context — no terminal detection, paste chord, or clipboard.
- **uinput** — the universal fallback. Drives a virtual keyboard through `/dev/uinput`.

On Windows (and macOS) there is one: **arboard + enigo** — `arboard` for the clipboard, `enigo` for the paste chord and for per-key Unicode typing. `arboard` reads and writes the clipboard in-process, so there is no child process to bound; it reads text only, so a non-text clipboard is snapshotted as "nothing to restore" rather than as bytes. On Windows this driver first restores the captured window (`focus::restore_focus` — `SetForegroundWindow`, guarded by `IsWindow` because Windows reuses handles) and waits 30 ms for the activation to land, because the paste chord goes wherever focus is.

`dictation_mode` selects the delivery method; `copy_to_clipboard` is an orthogonal toggle for whether a copy is *kept*:


| Mode          | uinput (Linux)                                                                                                                                                                     | IBus (Linux)  | arboard + enigo (Windows / macOS)                                                |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------- |
| `TypeInPlace` | Types each key (US-QWERTY). Non-ASCII text routes through the clipboard paste path so it is never dropped.                                                                         | `commit_text` | `enigo.text()` — there is no uinput here                                         |
| `Clipboard`   | Sets the clipboard (`wl-copy`/`xclip`), detects terminals, sends Ctrl+V / Ctrl+Shift+V. Adaptive post-paste delay (EMA of clipboard RTT); per-key fallback for paste-hostile apps. | `commit_text` | In-process clipboard, then Ctrl+V — Cmd+V on macOS — **always**, terminal or not |


Terminal classification means something different on each side. On Linux it picks the chord. On Windows every console takes Ctrl+V, so it only decides whether the text must be flattened first.

**Undeliverable text.** Delivery reports back whether the text reached the focused app. Multi-line text is never typed key-by-key — a newline would press Enter into whatever has focus — so if the clipboard is also unavailable the text is dropped rather than sent, and the session raises a warning instead of reporting success.

**Text that could not be typed, but is recoverable.** Some failures leave the text on the clipboard instead of dropping it, and the session warning says to paste it: an IBus commit that did not land (the focused app has no IBus input context, so IBus never enables the engine — or the daemon is unreachable), a Windows target running at a higher integrity level (UIPI discards synthetic input without an error, so capture checks the target's elevation before sending), and a paste chord the system refused.

**One window, every mode.** `TypeInPlace` aims at the captured window the same way `Clipboard` does: focus is restored to it first, and terminal classification comes from it rather than from a live probe.

**Slow targets.** The text stays on the clipboard for at least `focus::paste_read_floor` after the chord for apps that read a paste late (Office, remote-desktop and VM clients, Electron shells). Windows also keeps a transient copy out of clipboard history and cloud sync.

**Delivery thread.** A driver that fails to start (a cancelled polkit prompt, say) is retried when the app next polls `health()`, at most every 30 s. A driver call that panics drops that one delivery and the thread carries on. IBus D-Bus calls time out after 2 s.

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


| Command                                                                                             | Window access  | Result                                                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_workspace_snapshot`                                                                           | `main`         | Settings, history, dictionary, style presets, service status                                                                                                                                                                                                                                            |
| `save_workspace_settings`                                                                           | `main`         | Updated workspace snapshot                                                                                                                                                                                                                                                                              |
| `create_dictionary_entry` / `delete_dictionary_entry`                                               | `main`         | Updated workspace snapshot                                                                                                                                                                                                                                                                              |
| `subscribe_session_stream` / `detach_session_stream`                                                | `main`         | Live session event subscription                                                                                                                                                                                                                                                                         |
| `start_dictation` / `stop_dictation` / `cancel_dictation` / `toggle_dictation`                      | `main`, `pill` | Session control                                                                                                                                                                                                                                                                                         |
| `load_pill_snapshot`                                                                                | `pill`         | Minimal state for the overlay                                                                                                                                                                                                                                                                           |
| `nudge_pill` / `save_pill_position`                                                                 | `pill`         | Drag the overlay and persist where it lands (webview pill only). Deltas arrive as webview CSS px and are scaled into the monitor's space, carrying the sub-pixel remainder between frames                                                                                                               |
| `reset_pill_position`                                                                               | `main`         | Clear the saved position back to the anchor default                                                                                                                                                                                                                                                     |
| `list_monitors`                                                                                     | `main`         | Named displays for the pill's monitor picker, sized in the pill's placement space — device px on Windows/macOS, GDK logical px on Linux                                                                                                                                                                 |
| `gpu_backend_status`                                                                                | `main`         | Runtime GPU state: `unsupported`, `driver-missing`, `not-installed`, `ready`, `failed`                                                                                                                                                                                                                  |
| `install_gpu_backend`                                                                               | `main`         | Downloads and verifies the CUDA plugin, streaming progress over a channel                                                                                                                                                                                                                               |
| `list_history_page` / `delete_session` / `delete_sessions`                                          | `main`         | History paging, single and bulk deletion                                                                                                                                                                                                                                                                |
| `update_dictionary_entry`                                                                           | `main`         | Edit a dictionary entry in place                                                                                                                                                                                                                                                                        |
| `save_snippet` / `delete_snippet`                                                                   | `main`         | Snippet management                                                                                                                                                                                                                                                                                      |
| `create_custom_style` / `update_custom_style` / `delete_custom_style` / `generate_style_examples`   | `main`         | Custom style presets                                                                                                                                                                                                                                                                                    |
| `list_audio_input_devices`                                                                          | `main`         | Input device picker                                                                                                                                                                                                                                                                                     |
| `start_trigger_capture`                                                                             | `main`         | Capture-only session for the snippet voice-trigger recorder (transcript streamed back, nothing typed, no history row)                                                                                                                                                                                   |
| `check_model_status` / `download_model_cmd` / `delete_model_cmd`                                    | `main`         | Model management                                                                                                                                                                                                                                                                                        |
| `required_models_ready`                                                                             | `main`         | Whether the configured ASR backend has what it needs on disk — the same check the pipeline builds against, so setup and runtime cannot drift                                                                                                                                                            |
| `active_model_download` / `cancel_model_download`                                                   | `main`         | The download slot is global, so a transfer survives the window closing and a reopened window re-attaches to it instead of restarting setup. A second `download_model_cmd` for the same model joins the running transfer and reports its outcome                                                         |
| `uinput_access_ready` / `grant_uinput_access` / `uinput_rule_installed` / `remove_uinput_rule`      | `main`         | The `/dev/uinput` udev rule, asked for during setup rather than mid-dictation. `grant_uinput_access` returns `false` when the rule is installed but needs a reboot to take effect; `Err` means the install itself failed. Removal is offered only for a rule this app wrote (a .deb/.rpm ships its own) |
| `export_transcript`                                                                                 | `main`         | Write a session transcript to a file                                                                                                                                                                                                                                                                    |
| `app_version` / `start_update_check` / `update_state` / `install_pending_update` / `dismiss_update` | `main`         | Version string and the in-app update flow. State is held in the backend, so an offer survives the window closing                                                                                                                                                                                        |
| `quit_app`                                                                                          | `main`         | Runtime lifecycle                                                                                                                                                                                                                                                                                       |


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
- **Windows/macOS — webview.** A Tauri window rendering `pill.html` (`src/features/dictation/`), styled by the `.pill-*` rules in `components.css`. It carries `WS_EX_NOACTIVATE` on Windows, because the window holding focus when a dictation ends is the window the transcript is typed into — an overlay that activates on click eats the user's text. The builder's `focused(false)` covers creation only. The Linux pill makes the same promise with `set_accept_focus(false)`.
- **`pill_all_workspaces` is Linux and macOS only.** Windows virtual desktops are not addressable per-window, so tao leaves the call unimplemented and the setting is ignored there. `WorkspaceSnapshot.capabilities` (`PlatformCapabilitiesDto`) reports this to the frontend, which drops the Settings row rather than showing a toggle that does nothing.

Both draw the same thing: a bar with a circular control at each end while listening — stop-and-transcribe on the left (top when vertical), discard on the right — and the level meter between them, each chip centred in the space between the bar's end and the meter. Outside a live session both caps are absent, and the bar itself starts a session. The two implementations are kept visually in step by hand; what is checked is the shared numbers. `pill_paint.rs` mirrors the theme's oklch tokens and the meter's geometry, and unit tests read `tokens.css`/`daisyui.css` and `dot-style.ts` back, failing if either side moves.

State reaches the native pill by direct call (`pill::publish_status` / `publish_audio_level`), and the webview pill by targeted event, so it can react without owning the main session channel:


| Event                  | Payload                                                |
| ---------------------- | ------------------------------------------------------ |
| `lune://pill-state`    | `"ready"`, `"listening"`, `"processing"`, or `"error"` |
| `lune://audio-level`   | Float in `[0, 1]`                                      |
| `lune://pill-settings` | `pillLocked`                                           |


Rust shows or hides the pill directly on status changes (`should_show_pill`) — the frontend never decides visibility, so the mapped window and its contents cannot desync.

**Placement is in the desktop's own coordinate space, and there are two of them.** `MonitorRect` holds GDK logical coordinates on Linux, where the whole layout is logical and `scale` is 1.0, and device pixels straight from the platform on Windows and macOS. A desktop spanning monitors at different DPI has exactly one coherent space and it is the physical one: dividing each monitor by its own scale makes neighbouring screens claim overlapping coordinates, so a drag lands on the wrong screen.

The pill's own geometry stays logical, because that is what the webview lays itself out in. `pill_extent` converts the footprint and the edge margin into the target monitor's space, so a 168×50 pill stays 168×50 to the eye on a 150% panel. `monitor_for_pill` resolves in two passes — the corner picks a candidate, that candidate's scale gives the footprint, the centre picks the answer — because the pill's size in device pixels depends on the monitor being searched for. The saved position is in the same space, so `pill_position` in `settings.toml` means different numbers on different platforms.

**A geometry change rebuilds the window.** The webview pill is never resized live (compositor ghosting during resize), so any change to size, orientation, corner or anchor destroys it and builds a new one. `destroy` unregisters on the event loop, so the label frees asynchronously and building under one still taken is a silent no-op; the rebuild waits up to 5 s for it. If the label never frees, the overlay stays hidden until the next pill settings change or a restart, and says so at `error` level.

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
`WhisperState` via `ctx.create_state()`, which allocates that session's KV-cache
and compute buffers. The state is held behind a
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
`Send` bound, and no extra cancellation point in `start`.

The audio capture callback runs on cpal's own high-priority thread and pushes
into a lock-free ring buffer; the session loop polls it, so the real-time path
does no allocation and takes no locks.

---

## Data Flow Diagram

```
                 lune-voice runtime

lune-audio (cpal)
  |
  | AudioChunk polled via AudioPull::try_next_chunk (PCM, 16 kHz mono f32, post-AGC)
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

