# Configuration

Lune Voice keeps its settings in a TOML file. The same settings are in the app's Settings screens, and most changes apply without a restart.

---

## Table of Contents

- [Runtime Settings](#runtime-settings)
- [Settings File](#settings-file)
- [ASR Backends](#asr-backends)
  - [WhisperCpp (Local, default)](#whispercpp-local-default)
  - [ParakeetCpp (Local)](#parakeetcpp-local)
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

You change settings in the app, or by hand in `settings.toml`. There is no
command-line interface. Most changes apply right away or at the next dictation.
Backend changes are saved immediately, and the pipeline is rebuilt before the
next session when needed.

When each setting takes effect:

| Tier | When Applied | Fields |
|------|-------------|--------|
| **Hot** | On save | `hotkey`, `activation_mode`, `model_idle_unload_minutes`, `input_device`, `sound_cues_enabled`, `sound_cue_volume`, every `pill_*` field |
| **Warm** | Next session start, when the pipeline rebuilds | `asr_backend`, `llm_backend`, `device`, `privacy_mode`, `default_style`, `language`, `dictation_language`, `vad_pcen`, `brand_casing`, `dictation_mode`, `copy_to_clipboard`, `session_history_days` (it decides whether a dictation is recorded at all) |
| **Cold** | Next app start | `terminal_mode`, `start_minimized`, `check_updates_on_launch` |

Every change is written to disk right away. After saving, the app reloads the settings from disk, so what you see is what was stored.

---

## Settings File

Settings are persisted as TOML at `~/.config/lune-voice/settings.toml` on Linux
and `%APPDATA%\lune-voice\settings.toml` on Windows.

If the file is missing or can't be parsed, the app uses the defaults and logs a warning. Bad settings never crash it.

### Full Default Configuration

```toml
device = "auto"
# default_style is optional. Leave it out for no styling (the default).
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
# pill_all_workspaces is ignored on Windows (see Pill Overlay).
# pill_monitor is optional. Leave it out to follow the active monitor, or set a
# display name (e.g. "DP-1") to pin the pill to one.
# pill_position is set when you drag the pill. Leave it out to use the anchor.
# Its units differ per platform (see Pill Overlay).
terminal_mode = "Auto"
sound_cues_enabled = true
sound_cue_volume = 1.0
session_history_days = 30
# model_idle_unload_minutes is optional. Leave it out to keep the model loaded.
# input_device is optional. Leave it out to use the system default input. The
# value is a name from the mic picker.
vad_pcen = true
start_minimized = false
check_updates_on_launch = false
brand_casing = true

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

Model files live in the app's model directory. The settings file stores which model to use, not where the file is.

---

## ASR Backends

### WhisperCpp (Local, default)

The default. Runs whisper.cpp locally through `whisper-rs`, using GGML models downloaded from `ggerganov/whisper.cpp`.

```toml
[asr_backend]
type = "WhisperCpp"
model = "Medium"
```

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `model` | `Base`, `Small`, `Medium`, `LargeV3Turbo` | `Medium` | Which GGML model to load |

**Available models** (multilingual):

| `model` value | File | Size | Speed / Accuracy |
|---------------|------|------|-----------------|
| `Base` | `ggml-base-q8_0.bin` | ~78 MB | Fastest; lowest accuracy. For low-end hardware |
| `Small` | `ggml-small-q5_1.bin` | ~190 MB | Fast; less accurate than Medium |
| `Medium` | `ggml-medium-q5_0.bin` | ~539 MB | Balanced (recommended) |
| `LargeV3Turbo` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB | Best accuracy; turbo decoder keeps it fast |

CPU or GPU is set by the top-level `device` field. Running on the GPU needs the downloadable Vulkan plugin (or a dev build compiled with `whisper-cuda`). See [GPU / Device Settings](#gpu--device-settings).

**VAD:** Silero runs as its own pass *before* decoding. It trims silence from the start and end and keeps the pauses inside your speech, because whisper uses them to place punctuation. whisper.cpp's built-in `enable_vad` is not used on purpose: it glues the speech together and removes those pauses, which gives run-on sentences. The VAD model (`ggml-silero-v5.1.2.bin`, in `models/whisper-vad-silero/`) loads automatically when it is there and always runs on the CPU. If it is missing, decoding runs without it and a warning is logged.

**Inference details:**
- Input: 16 kHz mono PCM, which is whisper's native format, so nothing is converted
- Decoding: beam search (beam size 5), once per utterance, with no streaming
- Confidence: per-token probabilities become each segment's `confidence` (mean) and `min_token_confidence` (minimum). Segments below 0.25 are treated as garbage and dropped

### ParakeetCpp (Local)

NVIDIA [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), run on-device by whisper.cpp's own `parakeet` library — the same whisper.cpp tree and the same ggml as the Whisper backend, so the CPU backends and the downloadable GPU plugin apply to it unchanged. The model is whisper.cpp's GGML conversion from `ggml-org/parakeet-GGUF`.

```toml
[asr_backend]
type = "ParakeetCpp"
model = "TdtV3Q8"
```

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `model` | `TdtV3Q8`, `TdtV3Q4K` | `TdtV3Q8` | Which GGML quantization to load |

**Available models** (the same NVIDIA weights, two quantizations):

| `model` value | File | Size | Trade-off |
|---------------|------|------|-----------|
| `TdtV3Q4K` | `ggml-parakeet-tdt-0.6b-v3-q4_k.bin` | ~416 MB | Smaller download and memory; slightly less accurate |
| `TdtV3Q8` | `ggml-parakeet-tdt-0.6b-v3-q8_0.bin` | ~669 MB | Practically the original model's accuracy (recommended) |

**When to pick it over Whisper:** dictation on a CPU without a GPU, in one of its languages. Its FastConformer encoder and TDT decoder skip the token-by-token beam search Whisper runs, so it finishes a dictation many times faster on the same CPU, and it writes punctuation and capitalization itself.

**Languages:** Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish and Ukrainian.

**`dictation_language` must be set explicitly — no `"auto"`.** Parakeet recognizes which of its languages is spoken, but never reports it back, and the correction and formatting passes need to know (with `"auto"` they would treat every dictation as English: English filler removal, English casing rules). So with Parakeet the language selector offers only these 25 languages, without Auto-detect. Switching the engine to Parakeet while the language is `"auto"` or one it does not cover (Japanese, Korean, Chinese) moves it, in the same save, to your system language if Parakeet covers it, otherwise English, and the Parakeet panel says which. Saving Parakeet with `"auto"` or an unsupported language is refused. A hand-edited `settings.toml` that still pairs them keeps dictating, formatted as English, with a warning in Settings; a language outside the list makes a dictation refuse to start with an error naming the fix.

**Inference details:**
- Input: 16 kHz mono PCM, endpoint-trimmed by the same Silero pass as Whisper (it uses the same `models/whisper-vad-silero/` model, downloaded alongside)
- Decoding: greedy TDT. A dictation longer than the model's ~50 s encoder window is cut in the middle of a pause and decoded piece by piece. Past that window whisper.cpp's Parakeet drops speech, and a 5-minute single pass would also need 1–2 GB; each piece needs tens of MB
- Confidence: per-token probabilities folded into segment and per-word confidences exactly as for Whisper, so correction sensitivity works the same way. The default `word_confidence_threshold` is `0.50`
- `detected_language` on each segment is the dictation language when one is set; Parakeet does not report what it heard

### OpenAI-Compatible (Remote ASR)

Use any OpenAI-compatible speech-to-text API: OpenAI, Groq, a local Whisper server, and so on.

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
| `base_url` | URL | none | API base URL |
| `api_key` | string | none | API key (kept in the OS keyring when one is available, inline in `settings.toml` otherwise; redacted in logs, zeroized on drop) |
| `model` | string | none | Model identifier |
| `streaming` | enum | none | `"WebSocket"` or `"HttpPolling"` |

> **Privacy:** a remote backend sends your audio to another server, so it needs `privacy_mode = "AllowRemote"` (**Allow remote services** in Settings → General). See [Privacy Mode](#privacy-mode).

> **Confidence:** remote backends don't report per-token confidence. Styling still works as usual.

---

## LLM Backends

### Disabled LLM

Turns styling off. Dictionary fixes still apply. This is the default.

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
| `base_url` | URL | none | API base URL |
| `api_key` | string | none | API key (kept in the OS keyring when one is available, inline in `settings.toml` otherwise; redacted in logs, zeroized on drop) |
| `model` | string | none | Model identifier |
| `max_tokens` | u32 | none | Maximum output tokens |

> This sends your transcript to the endpoint, so it needs `privacy_mode = "AllowRemote"` (**Allow remote services** in Settings → General). See [Privacy Mode](#privacy-mode).

---

## GPU / Device Settings

```toml
device = "auto"
```

| Value | Behavior |
|-------|----------|
| `"auto"` | Use the GPU when a GPU device is registered; fall back to CPU otherwise. **(default)** |
| `"cuda"` | Ask for the GPU (any vendor, despite the name). If none is available, it still falls back to CPU. |
| `"cpu"` | Always use the CPU. |

Only a discrete GPU counts. On a machine with just integrated graphics, transcription stays on the CPU, which is usually faster there.

**GPU support is decided at runtime.** The GPU backend is a Vulkan plugin the app downloads when you ask, so a CPU-only binary can still use the GPU. You need a graphics driver with Vulkan 1.2, which you install yourself. You can see the current state in Settings → Transcription, under Use hardware acceleration, which is also where you download the plugin (or with the `gpu_backend_status` command). The GPU card during setup only appears when accepting it would change something: the state is `ready`, `needs-restart`, or `not-installed` with a published plugin for this build. If you accept it while the state is `not-installed`, setup downloads the plugin.

The plugin installs into `~/.local/share/lune-voice/backends/`. Delete that folder to go back to CPU. ggml builds its device list once per run, so a newly installed plugin only works after a restart, and the app tells you so.

See [docs/GPU_PLUGIN.md](GPU_PLUGIN.md) for how detection and installation work, and [docs/RELEASE.md](RELEASE.md) for the build matrix.

---

## Privacy Mode

Controls whether remote backends are allowed. In the app this is the
**Allow remote services** switch under Settings → General.

```toml
privacy_mode = "LocalOnly"
```

| Mode | Behavior |
|------|----------|
| `LocalOnly` | Rejects every remote backend. **(default)** |
| `AllowRemote` | Allows remote backends. Each one still runs only once it is configured |

`LocalPreferred` is a former name for `AllowRemote` and still loads as it.

Under `LocalOnly`, a remote transcription backend stops the pipeline from
starting, with a message that says what to change. It doesn't quietly switch to
local Whisper. Remote styling is optional, so it is skipped instead: you get the
unstyled transcript, and the reason is logged.

---

## Dictation Modes

`dictation_mode` picks how the text gets into your app. The separate `copy_to_clipboard` switch decides whether a copy is also left on the clipboard.

```toml
dictation_mode = "TypeInPlace"
copy_to_clipboard = false
```

| Mode | Behavior |
|------|----------|
| `TypeInPlace` | Types the text. On Linux, short ASCII text is typed key by key through a `uinput` virtual keyboard, and text over 80 characters, multi-line text, or non-ASCII text (accents, CJK, emoji) is pasted instead, so it is fast and nothing is lost. On Windows, `enigo` types all of it. **(default)** |
| `Clipboard` | Puts the text on the clipboard and sends the paste keystroke. |

If the clipboard fails, the app types the text instead. On Linux, multi-line text is the exception: typing it would press Enter in your app, so it is not sent, and a warning is logged. It is still in history.

**The paste keystroke depends on the platform.** On Linux it is `Ctrl+V`, or `Ctrl+Shift+V` in a terminal. The wait after a paste adapts to how fast your clipboard is, and typing takes over if `wl-copy` or `xclip` is missing. On Windows it is always `Ctrl+V`, which every console accepts. There, spotting a terminal decides whether newlines and tabs are flattened first. Windows handles the clipboard inside the app, so there is no helper program to install.

**The text goes to the window that was focused when you started speaking**, not when it arrives. On Windows, if one of the app's own windows took focus in between (say you clicked the pill), focus is moved back before the paste. See [Architecture → Text Delivery](ARCHITECTURE.md#5-text-delivery).

> When IBus is your input method (common on GNOME and Wayland), both modes send the text through IBus and never touch the clipboard.

### Copy to Clipboard

Works with either `TypeInPlace` or `Clipboard`.

```toml
copy_to_clipboard = false
```

When `true`, the dictated text stays on the clipboard afterwards. When `false` (the default), your clipboard is left as it was: if a paste had to borrow it, whatever was there before is saved first and put back. On Linux that includes images. On Windows only text can be saved, so if the clipboard held something else, the dictated text is left there instead of wiping it.

---

## Activation Modes

Controls how dictation sessions are started and stopped.

```toml
activation_mode = "Toggle"
```

| Mode | Behavior |
|------|----------|
| `Toggle` | Press the hotkey to start, press it again to stop **(default)** |
| `Hold` | Hold the hotkey while you talk, release to stop |

---

## Pill Overlay

The floating pill shows your mic level and what the app is doing, and you can
put it anywhere. Every field below is in the **Pill** section of Settings and
applies on save.

```toml
pill_visibility = "Always"       # Always | WhenDictating | Never
pill_anchor = "BottomCenter"     # default position when not dragged
pill_size = "Standard"           # Compact | Standard | Large
pill_orientation = "Horizontal"  # Horizontal | Vertical
pill_corner = "Rounded"          # Pill | Rounded | Square
pill_edge_margin = 72            # gap from the screen edge for the anchor, in logical px
pill_all_workspaces = true       # follow you across workspaces (Linux only)
pill_locked = false              # true disables dragging
# pill_monitor = "DP-1"          # pin to a display by name; omit to follow the active one
# pill_position = { x = 40, y = 900 }  # set automatically when you drag the pill
```

| Field | Values | Notes |
|-------|--------|-------|
| `pill_visibility` | `Always`, `WhenDictating`, `Never` | Applies instantly |
| `pill_anchor` | `TopLeft`, `TopCenter`, `TopRight`, `BottomLeft`, `BottomCenter`, `BottomRight` | Default placement; only used while `pill_position` is unset |
| `pill_size` | `Compact`, `Standard`, `Large` | Overall footprint |
| `pill_orientation` | `Horizontal`, `Vertical` | `Vertical` swaps width and height |
| `pill_corner` | `Pill`, `Rounded`, `Square` | Corner rounding of the bar |
| `pill_edge_margin` | `0`–`512` | Distance from the anchored screen edge, in **logical** px. On a 150% display, 72 becomes 108 device px |
| `pill_all_workspaces` | `true` / `false` | Show on every workspace. **Linux only**: Windows doesn't let an app pin one window to every virtual desktop, so the value is ignored there and Settings hides the switch |
| `pill_locked` | `true` / `false` | Lock the pill against dragging |
| `pill_monitor` | display name or unset | Pin to one display. The name is what the system reports: a connector like `DP-1` on Linux, a device name on Windows. Displays without a name aren't listed |
| `pill_position` | `{ x, y }` or unset | Written when you drag the pill; clears when you pick a new anchor |

**`pill_position` is not relative to a monitor, and its units differ by
platform:** logical pixels on Linux, device pixels on Windows, matching how each
system reports monitor positions. With Linux layer-shell it is a margin on that
output instead. On a scaled display, editing it by hand is guesswork, so drag
the pill and let it write the value.

On Windows, changing the size, orientation, corner or anchor rebuilds the pill
window instead of resizing it. You normally won't notice. If the rebuild fails,
the pill stays hidden until you change a pill setting again or restart, and the
log says so. See
[Architecture → The Pill Overlay](ARCHITECTURE.md#the-pill-overlay).

On GNOME (Mutter) and other compositors without `wlr-layer-shell`, the app runs
under XWayland so the pill can position itself and stay on top.

---

## Hotkey Configuration

Hotkeys are global and don't need a shortcut set up in your desktop. On Linux
the app reads the keyboards directly through evdev; on Windows it uses a
low-level keyboard hook. Record a combination in Settings instead of editing
this by hand.

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
| `writing` | Writing | Long-form writing: articles, docs, essays | Polished prose, varied sentence length, filler words removed |

Set the default style (leave the key out for no styling):

```toml
default_style = "casual"
```

### Brand Casing

`brand_casing` (default `true`) fixes the capitals of known names: `github` becomes `GitHub`, `home depot` becomes `Home Depot`. Turn it off to keep Whisper's own capitals. Your dictionary entries always apply, and so does the short built-in list of fixed replacements.

A style can override the global setting. Each style's `formatting.brand_casing` is `true`, `false`, or unset, and unset follows `brand_casing`. Built-in presets leave it unset. In the app, the global switch is in Settings → Transcription → Correction, and the per-style choice is in the style editor.

### Prompt Structure

Each style builds a `(system, user)` message pair with `StylePreset::build_styling_prompt`:

| Role | Content |
|------|---------|
| `system` | Style name, rules, and few-shot examples (when the preset has any) |
| `user` | The transcript to rewrite |

They go to your endpoint as normal `system` + `user` chat messages. The app does no chat-template rendering of its own.

A reasoning model's thinking never ends up in your text:

- Reasoning in a separate `reasoning_content` field is ignored.
- Reasoning inside the reply is removed. That covers `<think>`, `<thinking>`, `<reasoning>` and `<thought>` blocks in any case, a block that only has the closing tag, and harmony channels (only the final channel is kept).
- A reply that is nothing but unfinished reasoning is thrown away, and you get the unstyled text.

### Few-Shot Examples

Each preset has two input/output examples that show the LLM what to do:

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

The dictionary holds replacements that run before styling. Use it for names, brands, and terms from your field.

`apply_dictionary_seed` matches each word without regard to case (Unicode-aware), ignoring punctuation around it. When several entries match at the same spot, the longest one wins.

Each entry has:

| Field | Type | Description |
|-------|------|-------------|
| `phrase` | string | What you say, or what Whisper mishears |
| `replacement` | string | How it should be written |
| `priority` | u8 | Higher priority entries are matched first |

Your dictionary is not sent to the styling LLM. The replacements happen first, so the model only sees the corrected text.

---

## Session History

Completed dictation sessions are stored in a SQLite database.

```toml
session_history_days = 30    # days to retain, 0 to keep forever, or null to record nothing
```

| Field | Type | Description |
|-------|------|-------------|
| `session_history_days` | u32 (0, or 1–3650) or null | How many days of history to keep. Older sessions are deleted at startup and whenever settings are saved. `0` means **keep forever**. `null` turns history off: nothing is recorded, and anything already stored is deleted when you change the setting |

In code, read this through `UserSettings::history_retention()`
(`Off` / `Forever` / `Days(n)`). Treating the raw field as a number of days
would make `0` delete everything.

You can also delete sessions by hand on the Sessions page, one at a time, or by
ticking several and using **Delete selected** (up to 1000 at once,
`HISTORY_BULK_DELETE_MAX`).

Each session record contains:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Unique session identifier |
| `started_at` | datetime | Session start timestamp |
| `ended_at` | datetime | Session end timestamp (null if the app crashed) |
| `style` | string | Style preset ID used |
| `backend` | enum | ASR backend type (`WhisperCpp` / `ParakeetCpp` / `OpenAiCompatible`) |
| `status` | enum | `Active`, `Completed`, `Failed(reason)`, or `Cancelled` |
| `transcript` | string | Final styled transcript (null if empty) |

Database location: `~/.local/share/lune-voice/lune.db` on Linux, `%APPDATA%\lune-voice\lune.db` on Windows.

---

## Pipeline Tuning

These go in the `[pipeline]` section of `settings.toml` (`PipelineOverrides`). Each one overrides the transcription backend's default, and all of them are optional.

| Parameter | Default | Effect | Tradeoff |
|-----------|---------|--------|----------|
| `word_confidence_threshold` | 0.50 (whisper.cpp, Parakeet) | Words below this count as uncertain. Only uncertain single words can get a near-miss dictionary fix | Higher allows more near-miss fixes |
| `max_speech_secs` | 300 (local Whisper and Parakeet), 120 (remote) | Ends the session after this many seconds of audio | Higher allows longer dictations |
| `llm_timeout_secs` | 30 | Uses the unstyled text if styling takes longer than this | Lower caps the worst-case wait |

There are no VAD timing settings. The hotkey decides when you start and stop, and the VAD only trims silence at the ends.

### Recommended Profiles

**Quick commands** (a stuck-open mic gives up after 30 s):
```toml
[pipeline]
max_speech_secs = 30
```

**Long dictation:** keep the default. With local Whisper it already allows 300 s (five minutes) per session.

---

## Model Management

Models are downloaded automatically on first run from HuggingFace Hub.

| Model | HuggingFace repo | Directory | Files | Size |
|-------|-----------------|-----------|-------|------|
| Whisper Base | `ggerganov/whisper.cpp` | `whisper-base-q8_0` | `ggml-base-q8_0.bin` | ~78 MB |
| Whisper Small | `ggerganov/whisper.cpp` | `whisper-small-q5_1` | `ggml-small-q5_1.bin` | ~190 MB |
| Whisper Medium *(default)* | `ggerganov/whisper.cpp` | `whisper-medium-q5_0` | `ggml-medium-q5_0.bin` | ~539 MB |
| Whisper Large v3 Turbo | `ggerganov/whisper.cpp` | `whisper-large-v3-turbo-q8_0` | `ggml-large-v3-turbo-q8_0.bin` | ~874 MB |
| Parakeet TDT 0.6B v3 (q4_k) | `ggml-org/parakeet-GGUF` | `parakeet-tdt-0.6b-v3-q4_k` | `ggml-parakeet-tdt-0.6b-v3-q4_k.bin` | ~416 MB |
| Parakeet TDT 0.6B v3 (q8_0) | `ggml-org/parakeet-GGUF` | `parakeet-tdt-0.6b-v3-q8_0` | `ggml-parakeet-tdt-0.6b-v3-q8_0.bin` | ~669 MB |
| Whisper VAD | `ggml-org/whisper-vad` | `whisper-vad-silero` | `ggml-silero-v5.1.2.bin` | ~9 MB |

The VAD downloads with whichever speech model you get first, and both local engines share it. Models are stored in `~/.local/share/lune-voice/models/` on Linux and `%APPDATA%\lune-voice\models\` on Windows.

Styling has no local model to manage. It runs against an OpenAI-compatible endpoint, set up like this:

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
| `LUNE_LOG_TRANSCRIPTS` | Set to `1` to include transcript text in logs. It is redacted by default; only set this while debugging recognition |
| `LUNE_DISABLE_IBUS` | Set to any value to skip the IBus delivery path and always type through `uinput` |
| `HF_TOKEN` | HuggingFace token for model downloads; only needed for gated or private repos |
| `LUNE_UPDATER_ENDPOINT` | Override the release manifest URL used by the update check |
| `LIBCLANG_PATH` | Required at build time for bindgen (e.g., `/lib/x86_64-linux-gnu`); not a runtime variable |
