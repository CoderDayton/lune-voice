# Architecture

Lune Voice has one dictation pipeline and one way to control it: the desktop app, which talks to the runtime through typed Tauri commands and a session event stream.

---

## Table of Contents

- [System Overview](#system-overview)
- [Crate Dependency Graph](#crate-dependency-graph)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Audio Capture](#1-audio-capture)
  - [ASR](#2-asr)
    - [ParakeetCpp (local)](#2a-i-parakeetcpp-local)
    - [Compute backend selection](#2a-ii-compute-backend-selection)
  - [Dictionary Application](#3-dictionary-application)
  - [LLM Styling](#4-llm-styling)
  - [Text Delivery](#5-text-delivery)
- [Session Lifecycle](#session-lifecycle)
- [Desktop Bridge](#desktop-bridge)
- [Threading Model](#threading-model)
- [Data Flow Diagram](#data-flow-diagram)

---

## System Overview

Lune Voice turns microphone audio into text, optionally styles it, and sends it to the focused app. Only one session runs at a time. Everything else feeds that session, configures it, or shows its state.

The **hotkey decides when dictation starts and stops**: the session records while the hotkey is held (Hold mode) or until it is pressed again (Toggle mode). No VAD runs during recording. Silero trims silence off the ends of the recorded audio in a separate pass just before decode.

The runtime pipeline uses interchangeable backends:


| Component | Backends                                       | Model format             | Runtime                         |
| --------- | ---------------------------------------------- | ------------------------ | ------------------------------- |
| ASR       | WhisperCpp (local), ParakeetCpp (local), OpenAI-compatible (remote) | GGML `.bin` / remote API | whisper-rs (whisper.cpp, incl. its `parakeet` library) / HTTP |
| LLM       | OpenAI-compatible (remote), disabled           | remote API               | HTTP                            |


The native ASR backend compiles its C/C++ library from source at build time through a `*-sys` crate. GPU support is decided at **runtime**: ggml's compute backends are built as `dlopen`-able plugins (`GGML_BACKEND_DL`), so the Vulkan backend is downloaded on request instead of linked in. See [docs/GPU_PLUGIN.md](GPU_PLUGIN.md). The styling LLM is remote only; point it at Ollama, llama-server, LM Studio, or a hosted API. See [docs/RELEASE.md](RELEASE.md) for the full build matrix.

Backends are chosen in one place, through registries and DTO mappers. The desktop UI never builds model paths or backend structs itself. It sends typed settings and gets snapshots back.

---

## Crate Dependency Graph

```
lune-core (types, config, errors)
├─ lune-audio (mic, resample, ringbuf, vad, pcen)
├─ lune-asr (whisper.cpp, parakeet, openai remote, registry)
├─ lune-llm (openai remote, null passthrough, style pipeline, prompt, few-shot)
├─ lune-dict (corrector, vocab, voice commands)
└─ lune-persist (settings.toml, sqlite, model hub)
     └─ lune-voice (session, capture + AGC, delivery, sound cues)
          └─ desktop-ui/src-tauri (tauri IPC, hotkey, backend registry wiring, pill, persistence bridge)
```

Each crate has one job. `lune-core` defines shared types and traits but has no runtime logic. Crates talk to each other only through trait objects and channels.

---

## Pipeline Walkthrough

### 1. Audio Capture

**Crate:** `lune-audio` | **File:** `crates/lune-audio/src/lib.rs`  
**Orchestration:** `crates/lune-voice/src/capture.rs`

- Uses `cpal` for microphone access. Host choice lives in `crates/lune-audio/src/devices.rs` and nowhere else:
  - **Capture** (`capture_host`) uses cpal's PulseAudio host whenever a Pulse server answers (PulseAudio itself, or PipeWire through `pipewire-pulse`), and ALSA otherwise. Through the server, a mic that another program is using (a noise filter, a browser) can be shared. Opening the raw ALSA `hw:` device instead fails with "device busy" while anything else has the card. The Pulse client is the pure-Rust `pulseaudio` crate, so there is no system library, no headers, and no extra runtime dependency
  - **Playback** (`playback_host`, sound cues) stays on ALSA's `default`, which PipeWire and PulseAudio serve through their ALSA plugins
- The picker and the device that opens both go through `enumerate_input_devices`, so they can't disagree. There is one entry per display name:
  - **Pulse:** sources read from the server (`crates/lune-audio/src/pulse.rs`), minus sink monitors (`monitor_of_sink_index`). Recording streams aren't sources, so a filter's input side never shows up, but its output source does. cpal keeps its Pulse device type private, so this information comes from a second, short-lived client of the same crate. That client opens its own socket and shuts it down if the server doesn't answer within 2 s, so a stuck server leaves no thread behind
  - **ALSA:** one card-backed capture route per name. ALSA lists one PCM per *route*, so without this, 4 microphones would show up as 29 identical-looking entries next to `null` and `pipewire`
- A saved `input_device` matches a display name first, then a source's ALSA name (`"{alsa.card_name}, {alsa.name}"`, the name the ALSA host shows). At startup, a setting that matched the second way is rewritten to the current name (`persistence::migrate_input_device`), but only if the settings file parsed, and only if the saved name looks like an ALSA name (it contains `", "`), so most launches skip the device listing. The lookup gives up after 3 s and keeps the saved name. The rewrite only goes one way: with no Pulse server, the ALSA host doesn't know source names, so capture uses the system default and says so. A device is identified by its display name, so two mics with the same name are one picker entry
- Pulse capture asks for a fixed 20 ms period (`capture_buffer_size`). Left to the server, a recording stream's fragment defaults to about 2 s, and the first audio arrives 2 s after the stream opens
- Delivers 16 kHz mono PCM in 80 ms chunks (1280 samples). Most hardware records at 44.1 or 48 kHz, so a `rubato` async resampler converts to 16 kHz whenever the device's rate differs
- `MicAgc` (PCEN) brings each frame toward a target level before it leaves the capture loop, controlled by the `vad_pcen` setting (default on). The level meter shows the signal **after** this gain, so it matches what whisper hears
- Chunks go to the ASR backend through `push_audio` as they arrive. No VAD runs at this stage
- The session loop handles Stop, Cancel, deadline and max-speech signals and ends the capture loop when one arrives
- After Stop, the producer is asked to flush and gets 30 ms to hand over the chunk it still holds (up to ~80 ms, which is the end of the last word), then the AGC's leftover partial frame. Then `poll_final` decodes the buffer

### 2. ASR

**Crate:** `lune-asr`

Both backends implement the same `AsrBackend` trait, so the session code never checks which one it has.

#### 2a. WhisperCpp (local, default)

**File:** `crates/lune-asr/src/backend/whisper_cpp.rs`

The `WhisperCppBackend` wraps `whisper-rs`, which vendors and compiles `whisper.cpp`:

- The `WhisperContext` (model weights) is loaded **once** when the backend is built and shared by every session. Each session gets a `WhisperState`, a decode context that allocates its own KV-cache and compute buffers.
- `push_audio` appends PCM to a per-session buffer sized for 30 s of audio. It grows once for longer utterances, and is cleared without freeing after each decode.
- `poll_final` decodes the whole buffer at once with **beam search** (beam size 5), whisper.cpp's accuracy-first setting. There is only one decode per utterance, so the extra compute is worth it.
- **Silero trim:** when the VAD model is there (`models/whisper-vad-silero/ggml-silero-v5.1.2.bin`), `poll_final` first runs `WhisperVadContext::segments_from_samples` over the buffer and decodes only the span from the first speech segment to the last. Pauses inside that span are kept, because whisper uses them to place punctuation. whisper.cpp's built-in `enable_vad` is **not** used on purpose: it glues speech together and removes those pauses, which gives run-on sentences. The VAD always runs on the CPU, because Silero is tiny and a GPU VAD context crashes at init. If the model file is missing, a warning is logged and the whole buffer is decoded.
- **Splitting at pauses:** a trimmed span longer than 28 s is cut in the middle of a pause between two speech segments, and each piece is decoded on its own (`plan_chunks`). Left alone, whisper works in 30 s windows and decodes the leftover separately, and a leftover of a second or less comes back as made-up text ("Thank you."). Each cut leaves at least 3 s on both sides and prefers the longest pause in the back half of the window, so every piece keeps its pauses whole. A stretch with no pause is decoded in one piece.
- **Per-token confidence:** whisper reports a probability for each token. `poll_final` turns them into `AsrSegment::confidence` (mean) and `min_token_confidence` (minimum), leaving out special tokens. The capture layer drops segments whose mean is below `GARBAGE_DROP_THRESHOLD` (0.25) before the session sees them.
- Language: the `language` field from `AsrSessionConfig` goes to `set_language`. `None` means `"auto"`, so whisper detects it from the audio. With several pieces, the first one detects the language and the rest reuse it.

Four GGML model sizes are available, all from `ggerganov/whisper.cpp`:


| Variant              | File                           | Size    | Directory                     |
| -------------------- | ------------------------------ | ------- | ----------------------------- |
| `Base`               | `ggml-base-q8_0.bin`           | ~78 MB  | `whisper-base-q8_0`           |
| `Small`              | `ggml-small-q5_1.bin`          | ~190 MB | `whisper-small-q5_1`          |
| `Medium` *(default)* | `ggml-medium-q5_0.bin`         | ~539 MB | `whisper-medium-q5_0`         |
| `LargeV3Turbo`       | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB | `whisper-large-v3-turbo-q8_0` |


#### 2a-i. ParakeetCpp (local)

**File:** `crates/lune-asr/src/backend/parakeet_cpp.rs`

NVIDIA Parakeet TDT 0.6B v3, run by the `parakeet` library that whisper.cpp 1.9 ships beside `whisper` in the same source tree. It links the same ggml, so it needs no second native stack: the vendored `whisper-rs-sys` binds `parakeet.h` and links `libparakeet` next to `libwhisper` (marked `LUNE FORK` in its `build.rs` and `wrapper.h`), and the CPU backend plugins and the downloadable CUDA plugin serve both engines. `whisper-rs` has no Parakeet wrapper, so the backend calls the raw bindings and keeps each native handle in a small owning type that frees it.

- **One decoder state, reused.** `parakeet_init_state` allocates the decoder state, so the backend builds it once at load instead of per session (the library still re-plans the encoder graph when the clip length changes). Every decode runs with `no_context`, which clears the prediction network's LSTM state and the decoded tokens first, so the same audio decodes the same way whatever came before (an integration test pins this).
- **Chunked at pauses, inside the model's window.** v3 declares a 5000-mel-frame (50 s) encoder window. Past it whisper.cpp's Parakeet switches to a dynamic encoder graph that silently drops speech (a 55 s test clip lost its second half), and attention stays full and quadratic up to ~655 s, so a 5-minute single pass would also need 1–2 GB. A trimmed span is therefore cut in the middle of pauses by the planner whisper uses (`local::plan_chunks_within`, whisper with a 28 s window) into pieces that fit the window read from the model (`piece_limit`, ~49.7 s for v3); continuous speech with no pause is split evenly as a last resort. Each piece costs tens of MB. The real-model test pins this with an 86 s dictation. Clips under 0.5 s are padded with trailing silence so the 8× convolutional subsampling never sees an empty input.
- **Language.** v3 detects which of its 25 European languages is spoken but takes no hint and reports nothing back, so the dictation language has to come from the user: `ParakeetModel::accepts_dictation_language` refuses `"auto"` and anything outside the 25, the settings save enforces it, and the UI offers only those languages (switching engines picks the system language when it fits, else English, in the same save). `start_session` still refuses an uncovered language with an error that names the fix, and tolerates a hand-edited `"auto"` (formatted as English) rather than block dictation. `detected_language` on each segment is the dictation language. `lune_core::lang::display_name` names all 25, so none falls back to English formatting rules.
- **Logging.** Loading Parakeet re-points ggml's process-global log callback at Parakeet's own (`parakeet_backend_init_gpu` calls `ggml_log_set`). `install_logging_hooks` therefore registers a `tracing` sink for Parakeet too; left at its stderr default, it would silently undo the ggml hook for whisper as well.

**Shared with whisper** (`crates/lune-asr/src/backend/local.rs`): the Silero endpoint trim, the ggml file sanity check, the CPU-backend guard (both native loaders *throw* out of their state setup when ggml has no CPU device, which aborts the process — a broken plugin install now gets an error instead), decode-thread count, and the per-word confidence rule. A tuning change there reaches both engines.

| Variant | File | Size | Directory |
| ------- | ---- | ---- | --------- |
| `TdtV3Q4K` | `ggml-parakeet-tdt-0.6b-v3-q4_k.bin` (`ggml-org/parakeet-GGUF`) | ~416 MB | `parakeet-tdt-0.6b-v3-q4_k` |
| `TdtV3Q8` *(default)* | `ggml-parakeet-tdt-0.6b-v3-q8_0.bin` (`ggml-org/parakeet-GGUF`) | ~669 MB | `parakeet-tdt-0.6b-v3-q8_0` |

The two files were uploaded to the repository in separate commits, so each `PinnedFile` row in `lune-persist`'s hub carries its own commit, and a model whose files are all pinned is downloaded from those pins alone, without asking the hub for a file listing.

#### 2a-ii. Compute backend selection

**File:** `crates/lune-asr/src/gpu_backend.rs`

ggml's compute backends are built as `dlopen`-able plugins (`GGML_BACKEND_DL`), so the executable has no GPU dependency and starts on any machine. Two directories are scanned at startup: `<exe dir>/backends/` for the bundled CPU backends, and `<data dir>/backends/vulkan/` for the Vulkan plugin the user downloads.

`load_plugins` **must run before the first `WhisperContext` or Parakeet context is created**. ggml picks a device from whatever is in the backend registry when the context is created, which is why `desktop-ui/src-tauri/src/lib.rs` calls it before the Tauri builder. A `OnceLock` makes it run only once, because the registry is global to the process and has no unload.

On Linux each plugin carries `RUNPATH=$ORIGIN`. Setting `LD_LIBRARY_PATH` from inside the app wouldn't work, because glibc caches the search path at startup. The Vulkan plugin's only external dependency is the Vulkan loader the graphics driver installs.

See [docs/GPU_PLUGIN.md](GPU_PLUGIN.md) for the states, the archive contents, and the publishing flow.

#### 2b. OpenAI-Compatible (remote)

Remote ASR is only used when the user allows remote services. It returns text through the same `AsrSegment` type, but without per-token confidence. Styling doesn't depend on confidence, so it still follows the style preset.

##### Transport: when plaintext is allowed

Both remote backends (`crates/lune-asr/src/backend/openai.rs`, `crates/lune-llm/src/backend/openai.rs`) refuse a plain `http://` endpoint unless `lune_core::config::is_local_endpoint` calls it *local*: loopback, the RFC 1918 / RFC 4193 private ranges, and `.local` / `.localhost` names. A public host must use HTTPS.

This allows more than loopback on purpose, and the cost is real: **with a LAN endpoint, the API key and every transcript cross your local network unencrypted.** Self-hosted servers (Ollama, llama-server, LM Studio) ship without TLS, so requiring HTTPS would break the common setup of running the model on another machine on the same network. The exposure stays inside the LAN. The same check decides whether an endpoint may be used without a key, so the two rules can't drift apart.

If you don't accept that, put the endpoint behind HTTPS (a reverse proxy with a local certificate). Narrowing the check to loopback only would block remote ASR and quietly turn off styling for everyone on a LAN, so it would be a policy change, not a fix.

### 3. Dictionary Application

**Crates:** `lune-voice`, `lune-dict`  
**Files:** `crates/lune-voice/src/session.rs`, `crates/lune-dict/src/corrector.rs`

Once the ASR segments are in, the session runs a fixed, rule-based text pass:

1. Voice commands are removed from the text (the whole utterance must match).
2. The corrector (`LexicalGate::correct_with_risk_threshold` in `corrector.rs`, called from `correct_and_format` in `session.rs`) rewrites the transcript in three ways:
   - **Your dictionary.** Exact matches always apply. A phrase of several words, or a word with punctuation such as `node.js` or `c++`, matches whole words exactly as written. A single word of four or more characters can also match with one edit, but only when `FUZZY_DICT_RISK_GATE` allows it, and never on a word the lexicon knows or a word with a digit in it.
   - **Bundled vocabulary.** Exact, case-insensitive matches get their capitals fixed. There is no fuzzy guessing here.
   - **`ALWAYS_REPLACE`.** A built-in table of common mishearings and their fixes.

   `formatting` then handles capitalization, end punctuation, filler words, and quotes wrapped around the whole text.
3. The corrected text and the dictionary entries go into `PostProcessInput` for the optional styling stage.

Styling doesn't depend on word confidence. It only runs when a style is selected: when `default_style` is `None`, the LLM is skipped and this output is delivered as is. (`apply_dictionary_seed`, below, is a second pass inside the styling stage. It only changes the text sent to the LLM, not this output.)

### 4. LLM Styling

**Crate:** `lune-llm` | **File:** `crates/lune-llm/src/pipeline.rs`

The session calls `StylePipeline::transform()` for styling.

`PostProcessInput` carries:


| Field          | Meaning                                                            |
| -------------- | ------------------------------------------------------------------ |
| `raw_text`     | ASR transcript after dictionary substitution                       |
| `dictionary`   | Entries for the pre-LLM seed substitution; never sent to the model |
| `style_preset` | Output style contract                                              |
| `language`     | BCP-47 code from ASR detection, `"en"` when unset                  |


`StylePipeline` runs two steps:

1. **Dictionary seed:** `apply_dictionary_seed` replaces dictionary phrases in the text (case-insensitive, Unicode-aware, longest match wins). This happens before the LLM sees anything.
2. **Styling:** only when a style is selected. When `default_style` is `None`, the LLM is skipped and the seeded text is returned. Otherwise the LLM is called through `llm.complete()`. Its output goes through `fix_all_number_spaces` and the `contains_forbidden_output` filter. If the output fails the filter or is empty, the seeded text is kept.

**Removing reasoning:** a reasoning model puts its thinking somewhere in the reply, and none of it may reach the user's text. The backend reads only `message.content`, so reasoning in a separate `reasoning_content` field is never seen. Reasoning inside the content is removed by `strip_reasoning`:

- `<think>`, `<thinking>`, `<reasoning>` and `<thought>` blocks, in any case and any number of times
- a block with only a closing tag, which is what you get from a template that fills in the opening tag itself
- harmony channels, keeping only `<|channel|>final<|message|>`

Reasoning that never closes becomes an empty string, which the pipeline treats as "keep the text we already had".

When styling is off (`llm_backend` is `Disabled`), the runtime uses `NullPostProcessor`, which returns the seeded text unchanged. When the remote request fails, `StylePipeline` keeps the seeded (dictionary-corrected) text instead of failing the session.

The LLM backend (`OpenAiLlmBackend`) wraps an `async-openai` `Client` and makes one chat-completion request per call, right on the async runtime. There is no local model and no blocking thread pool.

### 5. Text Delivery

**Crate:** `lune-voice` | **Files:** `crates/lune-voice/src/delivery.rs`, `uinput_typer.rs`, `ibus.rs`, `focus.rs` + `focus/unix.rs` + `focus/windows.rs`, `probe.rs`

`delivery.rs` defines one `PlatformDriver` trait with a Linux half and a Windows half. `focus.rs` handles finding the focused window, deciding whether it is a terminal, and restoring focus, also split by platform: compositor queries in `focus/unix.rs`, Win32 calls in `focus/windows.rs`. Every outside command (compositor sockets, `wl-paste`, `swaymsg`, `xdotool`) runs through `probe.rs`, which gives it a time limit and keeps reading both of its output pipes the whole time, so a command with a lot to say isn't killed halfway through its answer.

**The target window is recorded once, when dictation starts** (`capture_focus_target`, run outside the start path because the Unix queries can take ~500 ms), and used seconds later at delivery. Two things can move focus in between: the user, on purpose, and one of our own windows, the pill most of all. So if the focused window belongs to this app at start, no restore target is kept. At delivery, the recorded window is focused again before typing, but only if one of our own windows has focus right then. If you switched apps on purpose, you stay where you went. This means the restore only covers focus that moves *after* dictation starts. If one of our windows was already in front at the start, there is nothing to go back to, which is why the pill refuses focus in the first place.

The session sends the final text on the event channel as
`FinalizedText { session_id, text }`.

Delivery runs on its own thread and picks a driver once at startup. Linux has two:

- **IBus:** used when IBus is the input method (GNOME and Wayland, for example; detected from `GTK_IM_MODULE`, `QT_IM_MODULE` or `XMODIFIERS`). It commits Unicode text straight into the focused input field, with no terminal check, paste keystroke, or clipboard.
- **uinput:** the fallback that works everywhere. It drives a virtual keyboard through `/dev/uinput`.

Windows has one: **arboard + enigo**. `arboard` handles the clipboard, and `enigo` sends the paste keystroke and types Unicode key by key. `arboard` works inside the app, so there is no helper process to time out. It only reads text, so a clipboard holding anything else counts as "nothing to restore". Before pasting, this driver focuses the recorded window again (`focus::restore_focus`: `SetForegroundWindow`, checked with `IsWindow` because Windows reuses handles) and waits 30 ms for it to take effect, because the paste goes wherever focus is.

`dictation_mode` picks how the text is delivered. `copy_to_clipboard` separately decides whether a copy is *kept*:


| Mode | uinput (Linux) | IBus (Linux) | arboard + enigo (Windows) |
| --- | --- | --- | --- |
| `TypeInPlace` | Types each key (US-QWERTY). Text over 80 characters, multi-line text, and non-ASCII text is pasted instead, so it is fast and never dropped. | `commit_text` | `enigo.text()` |
| `Clipboard` | Sets the clipboard (`wl-copy`/`xclip`), checks for a terminal, sends Ctrl+V or Ctrl+Shift+V. The wait after the paste adapts to the clipboard's measured speed (an EMA of its round trip). Falls back to typing if the clipboard fails. | `commit_text` | Clipboard inside the app, then Ctrl+V **always**, terminal or not |


Spotting a terminal does different things on each platform. On Linux it picks the paste keystroke. On Windows every console accepts Ctrl+V, so it only decides whether line breaks must be flattened first.

**Text that can't be delivered.** Delivery reports whether the text reached the app. On Linux, multi-line text is never typed key by key, because a newline would press Enter in whatever has focus. So if the clipboard is also unavailable, the text is not sent, and the session shows a warning instead of reporting success.

**Text that can't be typed but can be recovered.** Some failures leave the text on the clipboard instead, and the warning tells the user to paste it:

- an IBus commit that didn't land (the app has no IBus input field, so IBus never turns the engine on, or the IBus daemon can't be reached)
- a Windows app running with higher privileges (Windows silently ignores simulated input sent to it, so the app checks the target's privilege level before sending)
- a paste keystroke the system refused

**One window for every mode.** `TypeInPlace` aims at the recorded window the same way `Clipboard` does: focus goes back to it first, and the terminal check uses it, not whatever is focused right then.

**Slow apps.** After the paste, the text stays on the clipboard for at least `focus::paste_read_floor`, for apps that read a paste late (Office, remote desktop and VM clients, Electron apps). On Windows, that temporary copy is also kept out of clipboard history and cloud sync.

**Delivery thread.** If a driver fails to start (say the user cancelled the polkit prompt), it is tried again the next time the app checks `health()`, at most every 30 s. If a driver call panics, that one delivery is lost and the thread keeps going. IBus D-Bus calls time out after 2 s.

**Keeping your clipboard.** A paste borrows the clipboard to carry the text. When `copy_to_clipboard` is off, the old clipboard is saved before delivery and put back afterwards. On Linux it is saved as its preferred MIME type and raw bytes, so images survive. When `copy_to_clipboard` is on, the dictated text is kept and the extra copy step is skipped. The IBus path never touches the clipboard.

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

Only one session can be active at a time (`Arc<Mutex<Option<ActiveHandle>>>` enforces this). The session task runs on the Tauri async runtime through `tokio::task::spawn`.

**Teardown sequence:**

1. Stop or cancel clears out the pending recording control messages.
2. Unless the session was cancelled, the final ASR segments are collected (one decode, through `poll_final`).
3. An empty session ends with no transcript.
4. Voice commands are handled before the dictionary runs.
5. Dictionary seed and `StylePipeline` run on the transcript.
6. Final text is emitted, delivered, and persisted.

Cancellation is checked before styling and again before the final output. Cancelling during styling marks the session cancelled and skips `FinalizedText`.

The final events are sent with a timeout, so a stuck listener can't hang the teardown. If an `Error` event was dropped, the `Stopped` that follows carries `error_dropped`, so the listener still reports the session as failed.

---

## Desktop Bridge

The desktop app doesn't read logs or poll for session state. It uses typed Tauri commands for control and a `Channel<SessionStreamEvent>` for live events.

### Commands


| Command                                                                                             | Window access  | Result                                                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_workspace_snapshot`                                                                           | `main`         | Settings, history, dictionary, style presets, service status                                                                                                                                                                                                                                            |
| `save_workspace_settings`                                                                           | `main`         | Updated workspace snapshot                                                                                                                                                                                                                                                                              |
| `create_dictionary_entry` / `delete_dictionary_entry`                                               | `main`         | Updated workspace snapshot                                                                                                                                                                                                                                                                              |
| `subscribe_session_stream` / `detach_session_stream`                                                | `main`         | Live session event subscription                                                                                                                                                                                                                                                                         |
| `start_dictation` / `stop_dictation` / `cancel_dictation` / `toggle_dictation`                      | `main`, `pill` | Session control                                                                                                                                                                                                                                                                                         |
| `load_pill_snapshot`                                                                                | `pill`         | Minimal state for the overlay                                                                                                                                                                                                                                                                           |
| `nudge_pill` / `save_pill_position`                                                                 | `pill`         | Drag the overlay and save where it lands (webview pill only). Movements arrive in webview CSS px and are scaled into the monitor's space, carrying the sub-pixel remainder between frames                                                                                                               |
| `reset_pill_position`                                                                               | `main`         | Clear the saved position back to the anchor default                                                                                                                                                                                                                                                     |
| `list_monitors`                                                                                     | `main`         | Named displays for the pill's monitor picker, sized in the pill's placement space: device px on Windows, GDK logical px on Linux                                                                                                                                                                        |
| `gpu_backend_status`                                                                                | `main`         | Runtime GPU state: `unsupported`, `driver-missing`, `not-installed`, `ready`, `failed`                                                                                                                                                                                                                  |
| `install_gpu_backend`                                                                               | `main`         | Downloads and verifies the Vulkan plugin, streaming progress over a channel                                                                                                                                                                                                                               |
| `list_history_page` / `delete_session` / `delete_sessions`                                          | `main`         | History paging, single and bulk deletion                                                                                                                                                                                                                                                                |
| `update_dictionary_entry`                                                                           | `main`         | Edit a dictionary entry in place                                                                                                                                                                                                                                                                        |
| `save_snippet` / `delete_snippet`                                                                   | `main`         | Snippet management                                                                                                                                                                                                                                                                                      |
| `create_custom_style` / `update_custom_style` / `delete_custom_style` / `generate_style_examples`   | `main`         | Custom style presets                                                                                                                                                                                                                                                                                    |
| `list_audio_input_devices`                                                                          | `main`         | Input device picker                                                                                                                                                                                                                                                                                     |
| `start_trigger_capture`                                                                             | `main`         | Capture-only session for the snippet voice-trigger recorder (transcript streamed back, nothing typed, no history row)                                                                                                                                                                                   |
| `check_model_status` / `download_model_cmd` / `delete_model_cmd`                                    | `main`         | Model management                                                                                                                                                                                                                                                                                        |
| `required_models_ready`                                                                             | `main`         | Whether the configured ASR backend has what it needs on disk. It is the same check the pipeline uses, so setup and runtime can't disagree                                                                                                                                                               |
| `active_model_download` / `cancel_model_download`                                                   | `main`         | There is one download slot for the whole app, so a download keeps going when the window closes, and a reopened window picks it up instead of restarting setup. A second `download_model_cmd` for the same model joins the running download and reports its result                                                         |
| `uinput_access_ready` / `grant_uinput_access` / `uinput_rule_installed` / `remove_uinput_rule`      | `main`         | The udev rule (`assets/udev/70-lune-voice-input.rules`). Its number must sort before systemd's `73-seat-late.rules`, which only grants `uaccess` to devices already tagged. It gives write access to `/dev/uinput` for typing and read access to keyboards for the hotkey. The app asks for it during setup, not mid-dictation, and a .deb/.rpm installs it, so packaged users are never asked. Ready means `/dev/uinput` is writable and at least one keyboard is readable; the typing driver only needs `/dev/uinput`. `grant_uinput_access` returns `false` when the rule is installed but needs a reboot, and `Err` when the install itself failed. Removal is only offered for a rule this app wrote (a .deb/.rpm ships its own) |
| `export_transcript`                                                                                 | `main`         | Write a session transcript to a file                                                                                                                                                                                                                                                                    |
| `app_version` / `start_update_check` / `update_state` / `install_pending_update` / `dismiss_update` | `main`         | Version string and the in-app update flow. The state lives in the backend, so an offer survives the window closing                                                                                                                                                                                        |
| `quit_app`                                                                                          | `main`         | Quit the app                                                                                                                                                                                                                                                                                            |


### Session Stream

The stream uses a tagged enum serialized in camel case:

```json
{ "type": "sessionStarted", "sessionId": "..." }
{ "type": "audioLevel", "level": 0.42 }
{ "type": "finalizedText", "text": "..." }
```

The frontend trusts this stream for live state, and workspace snapshots for saved data. When a finished session saves its text, the backend emits `lune://history-changed`, and the frontend reloads the snapshot once to update history.

### The Pill Overlay

There are two implementations, picked at compile time, behind one interface in `pill.rs`. `pill.rs` owns the config, the saved position and the placement maths for both.

- **Linux: native.** A bare `gtk::Window` with a 32-bit ARGB visual, drawn with cairo (`pill_native.rs` + `pill_paint.rs`). On Nvidia + GNOME, WebKitGTK's accelerated compositor loses the alpha channel, so a webview pill could only ever be a flat, opaque rectangle. cairo draws through XRender and avoids that path. It is not a Tauri window, because every Tauri window has a webview and the pill doesn't need one. On Linux tao *is* GTK3 and runs the one GTK main loop, so the window just joins it.
- **Windows: webview.** A Tauri window that renders `pill.html` (`src/features/dictation/`), styled by the `.pill-*` rules in `components.css`. It has `WS_EX_NOACTIVATE`, because the transcript is typed into whichever window has focus when dictation ends, and an overlay that takes focus on click would swallow the user's text. The builder's `focused(false)` only covers creation. The Linux pill does the same with `set_accept_focus(false)`.
- **`pill_all_workspaces` is Linux only.** Windows doesn't let an app pin one window to every virtual desktop, so tao doesn't implement the call and the setting is ignored. `WorkspaceSnapshot.capabilities` (`PlatformCapabilitiesDto`) tells the frontend, which hides the Settings row instead of showing a switch that does nothing.

Both draw the same thing. While listening, the bar has a round button at each end (stop and transcribe on the left, or top when vertical; discard on the right) with the level meter between them, and each button is centred between the bar's end and the meter. Outside a session there are no buttons, and clicking the bar starts a session. The two are kept looking the same by hand, but the shared numbers are tested: `pill_paint.rs` copies the theme's oklch tokens and the meter's geometry, and unit tests read `tokens.css`/`daisyui.css` and `dot-style.ts` back and fail if either side changes.

The native pill gets state through direct calls (`pill::publish_status` / `publish_audio_level`). The webview pill gets targeted events, so it can react without owning the main session channel:


| Event                  | Payload                                                |
| ---------------------- | ------------------------------------------------------ |
| `lune://pill-state`    | `"ready"`, `"listening"`, `"processing"`, or `"error"` |
| `lune://audio-level`   | Float in `[0, 1]`                                      |
| `lune://pill-settings` | `pillLocked`                                           |


Rust shows or hides the pill itself when the status changes (`should_show_pill`). The frontend never decides visibility, so the window and its contents can't get out of sync.

**Placement uses the desktop's own coordinates, and there are two kinds.** `MonitorRect` holds GDK logical coordinates on Linux, where the whole layout is logical and `scale` is 1.0, and device pixels straight from the system on Windows. A desktop with monitors at different DPI has only one consistent space, the physical one. Dividing each monitor by its own scale makes neighbouring screens claim the same coordinates, so a drag lands on the wrong screen.

The pill's own size stays logical, because that is what the webview lays itself out in. `pill_extent` converts the size and the edge margin into the target monitor's space, so a 168×50 pill still looks 168×50 on a 150% screen. `monitor_for_pill` works in two passes (the corner picks a candidate monitor, that monitor's scale gives the size, and the centre picks the final answer), because the pill's size in device pixels depends on which monitor it is on. The saved position uses the same space, so `pill_position` in `settings.toml` holds different numbers on different platforms.

**Changing the geometry rebuilds the window.** The webview pill is never resized live, because the compositor leaves ghost frames during a resize. Any change to size, orientation, corner or anchor destroys it and builds a new one. `destroy` unregisters on the event loop, so the window's label is freed later, and building with a label still taken silently does nothing. The rebuild waits up to 5 s for the label. If it never frees, the pill stays hidden until the next pill setting change or a restart, and an `error` is logged.

### Sound Cues

The Rust backend plays short sounds when the status changes: Start on `listening`, Stop when leaving an active state, Error on `error`. Each plays once per change. Built-in 16-bit mono WAVs are decoded and played through a short-lived `cpal` output stream on one worker thread, behind a small queue, so a burst of changes can't pile up streams. `sound_cues_enabled` and `sound_cue_volume` control playback, and every failure is ignored, so sound never gets in the way of dictation.

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
|-- Parakeet inference (one context + one reused decoder state behind a Mutex)
|-- spellcheck / dictionary work
`-- persistence I/O (run_persist, pill settings)

dedicated OS threads
|-- delivery thread (uinput / IBus / clipboard)
`-- hotkey-evdev thread (global shortcut capture)
```

The runtime is **tokio multi-thread** (Tauri's default). The `WhisperContext` is
wrapped in an `Arc` and shared by all sessions. Each session creates a
`WhisperState` with `ctx.create_state()`, which allocates that session's KV-cache
and compute buffers. The state sits behind a `Mutex<WhisperCppSession>` so the
async and blocking code can both use it safely.

Blocking work is dispatched as follows:

- **whisper decode** (`poll_final`) uses `block_in_place`, not `spawn_blocking`.
The whisper session is `!Send` (it wraps raw C pointers), so it can't move to
the blocking pool. `block_in_place` hands the current worker thread over to
blocking work, so other tasks keep running during a decode that takes seconds.
On the app's multi-thread runtime that is a real handoff. On a current-thread
runtime (unit tests) it runs inline and warns once.
- **LLM styling** is a plain async HTTP request (through `async-openai`), so it
runs on the async runtime.
- **Persistence** never blocks the runtime. The Tauri commands and the GPU
plugin unpack use `spawn_blocking`. The session code can't, because its
`Database` sits behind a shared `tokio::sync::Mutex` whose guard can't move
into a spawned task. It uses `block_in_place` instead: the same handoff, no
`Send` requirement, and no extra cancellation point in `start`.

The audio callback runs on cpal's own high-priority thread and writes into a
lock-free ring buffer that the session loop reads. The real-time path never
allocates or takes a lock.

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

