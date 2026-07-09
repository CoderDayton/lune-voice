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
  - [Local Style LLM](#local-style-llm)
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

Most settings apply through the runtime settings channel or on the next dictation session. Backend changes are persisted immediately and the desktop runtime rebuilds the pipeline before the next start when needed.

```bash
lune-voice settings                     # dump all settings as TOML
lune-voice settings style technical     # switch style preset
lune-voice settings language en-GB      # change language (BCP-47 validated)
lune-voice settings privacy AllowRemote # change privacy mode
lune-voice settings history 90          # set retention to 90 days
lune-voice settings history -1          # disable session history
```

Settings mutability tiers:

| Tier | When Applied | Fields |
|------|-------------|--------|
| **Hot** | Immediately (watch channel) | `dictation_mode`, `copy_to_clipboard`, `activation_mode`, `pill_visibility`, `terminal_mode`, `sound_cues_enabled`, `sound_cue_volume` |
| **Warm** | Next session start | `default_style`, `language`, `dictation_language`, `mic_prewarm` |
| **Cold** | Persist only | `privacy_mode`, `session_history_days`, `start_minimized`, `hotkey` |
| **Reload** | Before next session | `asr_backend`, `llm_backend`, `device` |

All changes are persisted to disk immediately. The desktop app reloads a workspace snapshot after save, so the UI reflects the stored settings rather than a local guess.

---

## Settings File

Settings are persisted as TOML. Location depends on platform:

| Platform | Path |
|----------|------|
| Linux | `~/.config/lune-voice/settings.toml` |
| macOS | `~/Library/Application Support/lune-voice/settings.toml` |
| Windows | `%APPDATA%\lune-voice\settings.toml` |

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
start_minimized = false

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

**Available models** (all q8_0 quantizations, multilingual):

| `model` value | File | Size | Speed / Accuracy |
|---------------|------|------|-----------------|
| `Small` | `ggml-small-q8_0.bin` | ~264 MB | Fastest; lowest accuracy |
| `Medium` | `ggml-medium-q8_0.bin` | ~823 MB | Balanced (recommended) |
| `LargeV3Turbo` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB | Best accuracy; turbo decoder keeps it fast |

Device choice (CPU vs CUDA) comes from the top-level `device` field. GPU inference requires a build compiled with `whisper-cuda` — see [GPU / Device Settings](#gpu--device-settings).

**VAD:** the backend uses whisper.cpp's built-in Silero VAD at decode time. The VAD model (`ggml-silero-v5.1.2.bin`, downloaded to `models/whisper-vad-silero/`) is loaded automatically when present. If it is missing, the backend decodes without VAD and logs a warning.

**Inference details:**
- Input: 16 kHz mono PCM (already in whisper's native format — no conversion)
- Decoding: BeamSearch (beam size 5) for accuracy; one-shot per utterance (no streaming autoregressive loop)
- Confidence: per-token probabilities folded into segment `confidence` (mean) and `min_token_confidence` (minimum); segments below the garbage threshold (0.25) are dropped before the session sees them

**Migration from old configs:** settings files that still contain `type = "Qwen3Asr"` are automatically migrated to `WhisperCpp` with the `Medium` model on first load — the daemon does not fail on the unknown type.

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
| `api_key` | string | — | API key (stored in settings file; redacted in logs, zeroized on drop) |
| `model` | string | — | Model identifier |
| `streaming` | enum | — | `"WebSocket"` or `"HttpPolling"` |

> **Privacy note:** Remote backends send audio data to external servers. Requires `privacy_mode = "AllowRemote"`.

> **Confidence:** Remote backends do not expose per-token confidence. Styling is still applied based on the style preset.

---

## LLM Backends

### Local Style LLM

Runs the bundled local style model (Qwen3.5-0.8B GGUF via llama.cpp) when it has been downloaded. If the model files are missing or fail to load, Lune uses passthrough mode for that session.

```toml
[llm_backend]
type = "Local"
```

The local backend applies the dictionary seed and, if a style preset is selected, rewrites the text in the chosen style; dictionary substitution runs regardless of styling. It does not perform grammar correction — Whisper already emits punctuated text.

Model: `unsloth/Qwen3.5-0.8B-GGUF`, file `Qwen3.5-0.8B-UD-Q5_K_XL.gguf`, stored under `models/qwen3-styler-0.8b-gguf/`.

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
| `api_key` | string | — | API key (redacted in logs, zeroized on drop) |
| `model` | string | — | Model identifier |
| `max_tokens` | u32 | — | Maximum output tokens |

> Requires `privacy_mode = "AllowRemote"` or `"LocalPreferred"`.

---

## GPU / Device Settings

```toml
device = "auto"
```

| Value | Behavior |
|-------|----------|
| `"auto"` | Use CUDA when compiled with `whisper-cuda` / `llama-cuda`; fall back to CPU otherwise. **(default)** |
| `"cuda"` | Require CUDA. Runtime falls back to CPU if CUDA is unavailable even when explicitly requested. |
| `"cpu"` | Force CPU-only inference. |

**GPU support is a build-time choice.** The `device` setting only takes effect when the binary was compiled with the matching CUDA features (`whisper-cuda`, `llama-cuda`). A CPU build (`whisper-cpp`, `llama-cpp`) ignores `device = "cuda"` and runs on CPU regardless. The onboarding GPU acceptance card is only shown when `gpu_support.compiled_in` is true.

See [docs/RELEASE.md](RELEASE.md) for the full build matrix and how to build CPU vs CUDA artifacts.

---

## Privacy Mode

Controls whether remote backends are allowed.

```toml
privacy_mode = "LocalOnly"
```

| Mode | Behavior |
|------|----------|
| `LocalOnly` | Reject any remote backend usage. Only local models allowed. **(default)** |
| `LocalPreferred` | Use local backends when available; allow remote if explicitly configured and consented |
| `AllowRemote` | Use whatever is configured — no restrictions |

The daemon enforces privacy mode at startup. If a remote backend is configured but privacy mode is `LocalOnly`, the daemon will refuse to start and log an error.

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
| `Clipboard` | Delivers by setting the clipboard and sending a paste chord — `Ctrl+V`, or `Ctrl+Shift+V` when a terminal is focused. Uses an adaptive post-paste delay (EMA of clipboard round-trip) and falls back to per-key typing if `wl-copy` / `xclip` are missing. |

> When IBus is the active input method (common on GNOME/Wayland), both modes commit text directly through IBus and never touch the clipboard. See [Architecture → Text Delivery](ARCHITECTURE.md#5-text-delivery).

The combined `Both` mode was removed; configs that still set `dictation_mode = "Both"` migrate to `TypeInPlace` with `copy_to_clipboard = true`.

Switch at runtime:

```bash
lune-voice mode dictation Clipboard
```

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

Switch at runtime:

```bash
lune-voice mode activation Hold
```

**Recommended hotkey bindings:**

| Mode | Press | Release |
|------|-------|---------| 
| Toggle | `lune-voice toggle` | — |
| Hold | `lune-voice start` | `lune-voice stop` |

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

Hotkeys are configured in settings but must be bound externally in your desktop environment's keyboard settings.

```toml
[hotkey]
toggle_dictation = { modifiers = ["Ctrl"], key = "Space" }
cancel_dictation = { modifiers = [], key = "Escape" }
```

| Field | Type | Description |
|-------|------|-------------|
| `modifiers` | list | Combination of `"Ctrl"`, `"Shift"`, `"Alt"`, `"Meta"` |
| `key` | string | Key name (e.g., `"Space"`, `"Escape"`, `"F5"`) |

The daemon itself does not capture global hotkeys. Instead, bind your OS keyboard shortcuts to invoke the CLI subcommands:

**COSMIC / GNOME:**
```
Settings -> Keyboard -> Custom Shortcuts
  Name: Lune Toggle
  Command: lune-voice toggle
  Shortcut: Ctrl+Space
```

**KDE:**
```
System Settings -> Shortcuts -> Custom Shortcuts
  Add -> Command/URL
  Trigger: Ctrl+Space
  Action: lune-voice toggle
```

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

Switch at runtime:

```bash
lune-voice settings style technical
```

### Prompt Structure

Each style preset defines a full prompt template:

| Component | Purpose |
|-----------|---------| 
| `system_prompt` | Instructions for the LLM's role and behavior |
| `few_shot_pairs` | Example input/output pairs for in-context learning |

The final prompt is assembled as ChatML. The assistant turn is prefilled with an empty `<think>\n\n</think>` block to suppress Qwen3.5's chain-of-thought output:

```
<|im_start|>system
{system_prompt}

Dictionary terms (always use these exact spellings):
- lunevoice -> Lune Voice
- ...

Preceding context: {previous utterances}<|im_end|>
<|im_start|>user
{few_shot_input_1}<|im_end|>
<|im_start|>assistant
<think>
</think>
{few_shot_output_1}<|im_end|>
...
<|im_start|>user
{raw_text}<|im_end|>
<|im_start|>assistant
<think>
</think>
```

### Few-Shot Examples

Each preset includes 2 input/output pairs that teach the LLM the expected transformation:

**Casual:**
```
Input:  She said: "I will be there"; however, I do not think she meant it.
Output: she said she'd be there, but i don't think she meant it.
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

`apply_dictionary_seed` performs Unicode case-insensitive matching on the alphanumeric core of each whitespace-delimited token, with longest-match wins at each position. Dictionary entries are also injected into the LLM prompt as "Dictionary terms (always use these exact spellings)" to reinforce correct spelling during styling.

Each entry has:

| Field | Type | Description |
|-------|------|-------------|
| `phrase` | string | The spoken/misheard form |
| `replacement` | string | The correct written form |
| `priority` | int | Higher priority entries are matched first |

The top 20 most relevant dictionary entries (by fuzzy match against the current text) are included in each LLM prompt.

---

## Session History

Completed dictation sessions are stored in a SQLite database.

```toml
session_history_days = 30    # number of days to retain, or null to disable
```

| Field | Type | Description |
|-------|------|-------------|
| `session_history_days` | u32 or null | Days to retain session history. `null` disables persistence entirely |

Each session record contains:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Unique session identifier |
| `started_at` | datetime | Session start timestamp |
| `ended_at` | datetime | Session end timestamp (null if crashed) |
| `style` | string | Style preset ID used |
| `backend` | enum | ASR backend type (`WhisperCpp` / `OpenAiCompatible`) |
| `status` | enum | Completed / Failed(reason) |
| `transcript` | string | Final styled transcript (null if empty) |

Database location:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/lune-voice/lune.db` |
| macOS | `~/Library/Application Support/lune-voice/lune.db` |
| Windows | `%APPDATA%\lune-voice\lune.db` |

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
| Whisper Small | `ggerganov/whisper.cpp` | `whisper-small-q8_0` | `ggml-small-q8_0.bin` | ~264 MB |
| Whisper Medium *(default)* | `ggerganov/whisper.cpp` | `whisper-medium-q8_0` | `ggml-medium-q8_0.bin` | ~823 MB |
| Whisper Large v3 Turbo | `ggerganov/whisper.cpp` | `whisper-large-v3-turbo-q8_0` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB |
| Whisper VAD | `ggml-org/whisper-vad` | `whisper-vad-silero` | `ggml-silero-v5.1.2.bin` | ~9 MB |
| Style LLM | `unsloth/Qwen3.5-0.8B-GGUF` | `qwen3-styler-0.8b-gguf` | `Qwen3.5-0.8B-UD-Q5_K_XL.gguf` | ~600 MB |

Models are stored in:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/lune-voice/models/` |
| macOS | `~/Library/Application Support/lune-voice/models/` |
| Windows | `%APPDATA%\lune-voice\models\` |

Custom model paths are not part of the current public settings surface. To use a remote LLM instead of the local model:

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
| `LUNE_VOICE_CONFIG` | Override settings file path |
| `HF_HOME` | Override HuggingFace cache directory (affects model downloads) |
| `CUDA_VISIBLE_DEVICES` | Control which GPU is used for inference (CUDA builds only) |
| `LIBCLANG_PATH` | Required at build time for bindgen (e.g., `/lib/x86_64-linux-gnu`); not a runtime variable |
