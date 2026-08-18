# Configuration

Lune Voice stores user settings as TOML, exposes the same settings through the desktop UI, and applies most changes without restarting the app.

---

## Table of Contents

- [Runtime Settings](#runtime-settings)
- [Settings File](#settings-file)
- [ASR Backends](#asr-backends)
  - [WhisperCpp (Local, default)](#whispercpp-local-default)
  - [OpenAI-Compatible (Remote)](#openai-compatible-remote-asr)
- [LLM Backends](#llm-backends)
  - [Disabled](#disabled-llm)
  - [OpenAI-Compatible (Remote)](#openai-compatible-remote-llm)
- [GPU / Device Settings](#gpu--device-settings)
- [Privacy Mode](#privacy-mode)
- [Dictation Modes](#dictation-modes)
- [Activation Modes](#activation-modes)
- [Pill Overlay](#pill-overlay)
- [Hotkey Configuration](#hotkey-configuration)
- [Style Presets](#style-presets)
  - [Built-in Presets](#built-in-presets)
  - [Prompt Structure](#prompt-structure)
  - [Few-Shot Examples](#few-shot-examples)
- [User Dictionary](#user-dictionary)
- [Session History](#session-history)
- [Pipeline Tuning](#pipeline-tuning)
- [Model Management](#model-management)
- [Environment Variables](#environment-variables)

---

## Runtime Settings

Settings are edited in the desktop UI or by hand in `settings.toml`. There is no
command-line interface. Most changes apply through the runtime settings channel
or on the next dictation session; backend changes are persisted immediately and
the desktop runtime rebuilds the pipeline before the next start when needed.

Settings mutability tiers:

| Tier | When Applied | Fields |
|------|-------------|--------|
| **Hot** | On save | `hotkey`, `activation_mode`, `mic_prewarm`, `input_device`, `sound_cues_enabled`, `sound_cue_volume`, every `pill_*` field |
| **Warm** | Next session start, when the pipeline rebuilds | `asr_backend`, `llm_backend`, `device`, `privacy_mode`, `default_style`, `language`, `dictation_language`, `vad_pcen`, `dictation_mode`, `copy_to_clipboard`, `session_history_days` (it decides whether a dictation is recorded at all) |
| **Cold** | Next app start | `terminal_mode`, `start_minimized`, `check_updates_on_launch` |

All changes are persisted to disk immediately. The desktop app reloads a workspace snapshot after save, so the UI reflects the stored settings rather than a local guess.

---

## Settings File

Lune Voice is Linux-only. Settings are persisted as TOML at
`~/.config/lune-voice/settings.toml`.

If the file doesn't exist or fails to parse, defaults are used and a warning is logged. The daemon never crashes on invalid settings — it falls back gracefully.

### Full Default Configuration

```toml
device = "auto"
# default_style is optional — omit for no styling (raw transcript, the default).
# Set it to a preset id (e.g. "casual") to enable the LLM style pass.
language = "auto"
dictation_language = "auto"
privacy_mode = "LocalOnly"
dictation_mode = "TypeInPlace"
copy_to_clipboard = false
activation_mode = "Toggle"
pill_visibility = "Always"
pill_anchor = "BottomCenter"
pill_size = "Standard"
pill_orientation = "Horizontal"
pill_corner = "Rounded"
pill_edge_margin = 72
pill_all_workspaces = true
pill_locked = false
# pill_monitor is optional — omit to follow the active monitor, or set a
# connector name (e.g. "DP-1") to pin the pill to one display.
# pill_position is set automatically when you drag the pill; omit for the anchor default.
terminal_mode = "Auto"
sound_cues_enabled = true
sound_cue_volume = 1.0
session_history_days = 30
mic_prewarm = true
# input_device is optional — omit to follow the system default input.
vad_pcen = true
start_minimized = false
check_updates_on_launch = false

[asr_backend]
type = "WhisperCpp"
model = "Medium"

[llm_backend]
type = "Disabled"

[hotkey.toggle_dictation]
modifiers = ["Ctrl"]
key = "Space"

[hotkey.cancel_dictation]
modifiers = []
key = "Escape"
```

Model files live under the app model directory. The settings file stores backend choice and policy, not resolved model paths.

---

## ASR Backends

### WhisperCpp (Local, default)

The default backend. Runs whisper.cpp locally via `whisper-rs` using GGML models downloaded from `ggerganov/whisper.cpp`.

```toml
[asr_backend]
type = "WhisperCpp"
model = "Medium"
```

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `model` | `Small`, `Medium`, `LargeV3Turbo` | `Medium` | Which GGML model to load |

**Available models** (multilingual):

| `model` value | File | Size | Speed / Accuracy |
|---------------|------|------|-----------------|
| `Small` | `ggml-small-q5_1.bin` | ~190 MB | Fastest; lowest accuracy |
| `Medium` | `ggml-medium-q5_0.bin` | ~539 MB | Balanced (recommended) |
| `LargeV3Turbo` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB | Best accuracy; turbo decoder keeps it fast |

Device choice (CPU vs CUDA) comes from the top-level `device` field. GPU inference requires a registered CUDA device, which on Linux means the downloadable GPU plugin (or a build compiled with `whisper-cuda`) — see [GPU / Device Settings](#gpu--device-settings).

**VAD:** Silero runs as a standalone endpoint-trim pass *before* decoding — it trims leading and trailing silence and keeps internal pauses, which whisper reads as a punctuation cue. whisper.cpp's in-decode `enable_vad` is deliberately not used: it concatenates speech spans and strips those pauses, producing run-on sentences. The VAD model (`ggml-silero-v5.1.2.bin`, downloaded to `models/whisper-vad-silero/`) is loaded automatically when present and always runs on CPU. If it is missing, the backend decodes without it and logs a warning.

**Inference details:**
- Input: 16 kHz mono PCM (already in whisper's native format — no conversion)
- Decoding: BeamSearch (beam size 5) for accuracy; one-shot per utterance (no streaming autoregressive loop)
- Confidence: per-token probabilities folded into segment `confidence` (mean) and `min_token_confidence` (minimum); segments below the garbage threshold (0.25) are dropped before the session sees them

### OpenAI-Compatible (Remote ASR)

Use any OpenAI-compatible speech-to-text API (OpenAI Whisper, Groq, local Whisper server, etc.).

```toml
[asr_backend]
type = "OpenAiCompatible"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
model = "whisper-1"
streaming = "HttpPolling"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | URL | — | API base URL |
| `api_key` | string | — | API key (kept in the OS keyring when one is available, inline in `settings.toml` otherwise; redacted in logs, zeroized on drop) |
| `model` | string | — | Model identifier |
| `streaming` | enum | — | `"WebSocket"` or `"HttpPolling"` |

> **Privacy note:** Remote backends send audio data to external servers, so they require `privacy_mode = "LocalPreferred"` or `"AllowRemote"`. See [Privacy Mode](#privacy-mode).

> **Confidence:** Remote backends do not expose per-token confidence. Styling is still applied based on the style preset.

---

## LLM Backends

### Disabled LLM

Disables LLM post-processing. Dictionary substitutions still run (applied via the dictionary seed before any LLM call). This is the default.

```toml
[llm_backend]
type = "Disabled"
```

### OpenAI-Compatible (Remote LLM)

Use any OpenAI-compatible chat completion API.

```toml
[llm_backend]
type = "OpenAiCompatible"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
model = "gpt-4o-mini"
max_tokens = 512
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | URL | — | API base URL |
| `api_key` | string | — | API key (kept in the OS keyring when one is available, inline in `settings.toml` otherwise; redacted in logs, zeroized on drop) |
| `model` | string | — | Model identifier |
| `max_tokens` | u32 | — | Maximum output tokens |

> Sends transcript text to the configured endpoint, so it requires `privacy_mode = "LocalPreferred"` or `"AllowRemote"`. See [Privacy Mode](#privacy-mode).

---

## GPU / Device Settings

```toml
device = "auto"
```

| Value | Behavior |
|-------|----------|
| `"auto"` | Use CUDA when a CUDA device is registered; fall back to CPU otherwise. **(default)** |
| `"cuda"` | Require CUDA. Runtime falls back to CPU if CUDA is unavailable even when explicitly requested. |
| `"cpu"` | Force CPU-only inference. |

**GPU support is a runtime capability.** The CUDA backend is a plugin the app downloads on request, so `device = "cuda"` can start working on a binary that was never built with CUDA linked in. What it needs is an NVIDIA driver, which is not redistributable and so has to be installed by you. Query the current state with the `gpu_backend_status` command, or read it in Settings under Models, where the plugin is also downloaded. The onboarding acceleration card appears only when accepting it can change something: the state is `ready`, `needs-restart`, or `not-installed` with a published plugin archive for this build. Setup downloads the plugin itself when the card is accepted and the state is `not-installed`.

The plugin installs into `~/.local/share/lune-voice/backends/`, so removing that directory reverts to CPU. Because ggml builds its device registry once per process, a freshly installed plugin only takes effect after a restart — the UI says so rather than silently doing nothing.

See [docs/GPU_PLUGIN.md](GPU_PLUGIN.md) for how detection and installation work, and [docs/RELEASE.md](RELEASE.md) for the build matrix.

---

## Privacy Mode

Controls whether remote backends are allowed.

```toml
privacy_mode = "LocalOnly"
```

| Mode | Behavior |
|------|----------|
| `LocalOnly` | Rejects every remote backend. **(default)** |
| `LocalPreferred` | Allows remote backends, which only run when explicitly configured |
| `AllowRemote` | Use whatever is configured — no restrictions |

Under `LocalOnly`, a remote transcription backend fails the pipeline build with
an actionable message — there is no local fallback to silently substitute.
Remote styling is optional, so it degrades to passthrough (the deterministic
transcript) and logs the reason.

---

## Dictation Modes

`dictation_mode` selects how text is delivered to the focused app after styling. The orthogonal `copy_to_clipboard` toggle decides whether a copy is also left on the clipboard.

```toml
dictation_mode = "TypeInPlace"
copy_to_clipboard = false
```

| Mode | Behavior |
|------|----------|
| `TypeInPlace` | Types the text into the focused field, character by character, through a `uinput` virtual keyboard. Non-ASCII text (accents, CJK, emoji) is delivered via a clipboard paste instead, so nothing is dropped. **(default)** |
| `Clipboard` | Delivers by setting the clipboard and sending a paste chord — `Ctrl+V`, or `Ctrl+Shift+V` when a terminal is focused. Uses an adaptive post-paste delay (EMA of clipboard round-trip) and falls back to per-key typing if `wl-copy` / `xclip` are missing. Text that cannot be typed safely per key — anything multi-line — is dropped with a warning rather than delivered, since a newline would press Enter into the focused app. |

> When IBus is the active input method (common on GNOME/Wayland), both modes commit text directly through IBus and never touch the clipboard. See [Architecture → Text Delivery](ARCHITECTURE.md#5-text-delivery).

### Copy to Clipboard

Independent of the delivery method — works with either `TypeInPlace` or `Clipboard`.

```toml
copy_to_clipboard = false
```

When `true`, the dictated text is left on the system clipboard after delivery. When `false` (default), the clipboard is preserved: if the `Clipboard` paste path borrowed it to send `Ctrl+V`, the previous contents — text or images — are snapshotted first and restored byte-for-byte afterward.

---

## Activation Modes

Controls how dictation sessions are started and stopped.

```toml
activation_mode = "Toggle"
```

| Mode | Behavior |
|------|----------|
| `Toggle` | Press hotkey once to start, press again to stop **(default)** |
| `Hold` | Hold hotkey to dictate, release to stop |

---

## Pill Overlay

The floating dictation pill shows live audio levels and session state and can be
placed anywhere. All fields are edited from the **Pill** section in Settings and
take effect on save.

```toml
pill_visibility = "Always"       # Always | WhenDictating | Never
pill_anchor = "BottomCenter"     # default position when not dragged
pill_size = "Standard"           # Compact | Standard | Large
pill_orientation = "Horizontal"  # Horizontal | Vertical
pill_corner = "Rounded"          # Pill | Rounded | Square
pill_edge_margin = 72            # gap (px) from the screen edge for the anchor
pill_all_workspaces = true       # follow you across virtual desktops
pill_locked = false              # true disables dragging
# pill_monitor = "DP-1"          # pin to a connector; omit to follow the active monitor
# pill_position = { x = 40, y = 900 }  # set automatically when you drag the pill
```

| Field | Values | Notes |
|-------|--------|-------|
| `pill_visibility` | `Always`, `WhenDictating`, `Never` | Applies instantly |
| `pill_anchor` | `TopLeft`, `TopCenter`, `TopRight`, `BottomLeft`, `BottomCenter`, `BottomRight` | Default placement; only used while `pill_position` is unset |
| `pill_size` | `Compact`, `Standard`, `Large` | Overall footprint |
| `pill_orientation` | `Horizontal`, `Vertical` | `Vertical` swaps width and height |
| `pill_corner` | `Pill`, `Rounded`, `Square` | Corner rounding of the bar |
| `pill_edge_margin` | `0`–`512` | Distance from the anchored screen edge |
| `pill_all_workspaces` | `true` / `false` | Show on every workspace |
| `pill_locked` | `true` / `false` | Lock the pill against dragging |
| `pill_monitor` | connector name or unset | Pin to a specific display |
| `pill_position` | `{ x, y }` or unset | Written when you drag the pill; clears when you pick a new anchor |

On GNOME/Mutter and other compositors without `wlr-layer-shell`, the app runs the
overlay under XWayland so it can position itself and stay on top.

---

## Hotkey Configuration

The app reads the keyboard devices directly (evdev), so its hotkeys are global
and need no desktop-environment binding. Record a combination from Settings
rather than editing this by hand.

```toml
[hotkey]
toggle_dictation = { modifiers = ["Ctrl"], key = "Space" }
cancel_dictation = { modifiers = [], key = "Escape" }
```

| Field | Type | Description |
|-------|------|-------------|
| `modifiers` | list | Combination of `"Ctrl"`, `"Shift"`, `"Alt"`, `"Meta"` |
| `key` | string | Key name (e.g., `"Space"`, `"Escape"`, `"F5"`) |

---

## Style Presets

### Built-in Presets

| ID | Name | Description | Behavior |
|----|------|-------------|----------|
| `casual` | Casual | Texting, messaging, chat | Lowercase, contractions, comma-joined clauses |
| `professional` | Professional | Clear workplace communication | Standard capitalization, broken run-ons, no contractions |
| `formal` | Formal | Official correspondence and documents | Formal register, expanded contractions, structured sentences |
| `technical` | Technical | Code and developer communication | Wraps identifiers/keywords in backticks; preserves jargon |
| `writing` | Writing | Long-form content — articles, docs, essays | Polished prose, varied sentence length, filler words removed |

Set the default style (omit the key entirely for no styling):

```toml
default_style = "casual"
```

### Prompt Structure

Each style preset builds a `(system, user)` message pair via `StylePreset::build_styling_prompt`:

| Role | Content |
|------|---------|
| `system` | Style name, rules, and few-shot examples (when the preset has any) |
| `user` | The transcript to rewrite |

These are sent as standard `system` + `user` chat-completion messages to the configured OpenAI-compatible endpoint — no custom prompt templating or chat-template rendering happens client-side. A reasoning model's private thinking never reaches your document: reasoning returned in a separate `reasoning_content` field is not read at all, and reasoning inlined into the response is stripped — `<think>` / `<thinking>` / `<reasoning>` / `<thought>` blocks in any case, an unwrapped block ending at one of those closing tags, and harmony channels, of which only the final one is kept. A response that is nothing but unclosed reasoning is discarded and the unstyled text is delivered instead.

### Few-Shot Examples

Each preset includes 2 input/output pairs that teach the LLM the expected transformation:

**Casual:**
```
Input:  She said: "I will be there"; however, I do not think she meant it.
Output: she said, i'll be there. however, i don't think she meant it.
```

**Professional:**
```
Input:  i talked to the client and they're fine with the delay but they want weekly updates going forward.
Output: I spoke with the client, and they are fine with the delay. They want weekly updates going forward.
```

**Technical:**
```
Input:  The user profile class needs a get by ID method that returns an optional user.
Output: The `UserProfile` class needs a `get_by_id` method that returns an `Optional<User>`.
```

**Writing:**
```
Input:  Basically the problem is that it just scales poorly and once you hit about ten thousand users things start to break down.
Output: The problem is that it scales poorly. Once you hit ten thousand users, things start to break down.
```

---

## User Dictionary

The dictionary provides substitution rules applied before the LLM ever sees the transcript. Useful for proper nouns, brand names, and domain-specific terms.

`apply_dictionary_seed` performs Unicode case-insensitive matching on the alphanumeric core of each whitespace-delimited token, with longest-match wins at each position.

Each entry has:

| Field | Type | Description |
|-------|------|-------------|
| `phrase` | string | The spoken/misheard form |
| `replacement` | string | The correct written form |
| `priority` | u8 | Higher priority entries are matched first |

Dictionary entries are not sent to the styling LLM — the substitution happens before the call, so the model only ever sees the corrected text.

---

## Session History

Completed dictation sessions are stored in a SQLite database.

```toml
session_history_days = 30    # days to retain, or null to record nothing
```

| Field | Type | Description |
|-------|------|-------------|
| `session_history_days` | u32 (1–3650) or null | Days to retain session history. Older sessions are pruned at startup and whenever settings are saved. `null` disables history: no session row, transcript or correction pair is written, and anything already stored is cleared when the setting changes |

Each session record contains:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Unique session identifier |
| `started_at` | datetime | Session start timestamp |
| `ended_at` | datetime | Session end timestamp (null if crashed) |
| `style` | string | Style preset ID used |
| `backend` | enum | ASR backend type (`WhisperCpp` / `OpenAiCompatible`) |
| `status` | enum | `Active`, `Completed`, `Failed(reason)`, or `Cancelled` |
| `transcript` | string | Final styled transcript (null if empty) |

Database location: `~/.local/share/lune-voice/lune.db`.

---

## Pipeline Tuning

These live in the `[pipeline]` section of `settings.toml` (`PipelineOverrides`); each overrides the ASR backend's resolved default. All fields are optional — omitting a field inherits the backend's recommended default.

| Parameter | Default | Effect | Tradeoff |
|-----------|---------|--------|----------|
| `max_speech_secs` | 60 | Auto-end the session after this many seconds of audio | Higher allows longer single dictations |
| `llm_timeout_secs` | 30 | Fall back to unseeded text if LLM styling takes too long | Lower bounds worst-case latency |

**Note on VAD parameters:** `sentence_close_ms`, `hard_close_ms`, `speech_threshold`, `speech_threshold_end`, `min_speech_ms`, and `adaptive_vad` are preserved in `PipelineOverrides` for future use and potential use by alternative backends, but they have no effect on the whisper.cpp backend — whisper handles VAD internally during `poll_final`.

### Recommended Profiles

**Quick commands:**
```toml
[pipeline]
max_speech_secs = 30
```

**Long dictation:**
```toml
[pipeline]
max_speech_secs = 120
```

---

## Model Management

Models are downloaded automatically on first run from HuggingFace Hub.

| Model | HuggingFace repo | Directory | Files | Size |
|-------|-----------------|-----------|-------|------|
| Whisper Small | `ggerganov/whisper.cpp` | `whisper-small-q5_1` | `ggml-small-q5_1.bin` | ~190 MB |
| Whisper Medium *(default)* | `ggerganov/whisper.cpp` | `whisper-medium-q5_0` | `ggml-medium-q5_0.bin` | ~539 MB |
| Whisper Large v3 Turbo | `ggerganov/whisper.cpp` | `whisper-large-v3-turbo-q8_0` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB |
| Whisper VAD | `ggml-org/whisper-vad` | `whisper-vad-silero` | `ggml-silero-v5.1.2.bin` | ~9 MB |

Models are stored in `~/.local/share/lune-voice/models/`.

Styling runs against a remote OpenAI-compatible endpoint; custom model paths are not part of the public settings surface. To configure one:

```toml
[llm_backend]
type = "OpenAiCompatible"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
model = "gpt-4o-mini"
max_tokens = 512
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------| 
| `RUST_LOG` | Tracing filter (e.g., `lune_asr=debug,lune_llm=info`) |
| `LUNE_LOG_TRANSCRIPTS` | Set to `1` to include transcript text in logs. Redacted by default — leave it unset outside recognition debugging |
| `LUNE_DISABLE_IBUS` | Set to any value to skip the IBus delivery path and always type through `uinput` |
| `HF_TOKEN` | HuggingFace token for model downloads; only needed for gated or private repos |
| `LUNE_UPDATER_ENDPOINT` | Override the release manifest URL used by the update check |
| `LIBCLANG_PATH` | Required at build time for bindgen (e.g., `/lib/x86_64-linux-gnu`); not a runtime variable |
