# Configuration

> Complete reference for all settings, backends, style presets, and runtime modes. For a quick overview, see the [README](../README.md).

---

## Table of Contents

- [Runtime Settings](#runtime-settings)
- [Settings File](#settings-file)
- [ASR Backends](#asr-backends)
  - [Qwen3-ASR (Local, default)](#qwen3-asr-local-default)
  - [Parakeet TDT (Local)](#parakeet-tdt-local)
  - [OpenAI-Compatible (Remote)](#openai-compatible-remote-asr)
- [LLM Backends](#llm-backends)
  - [LlamaCpp Local (default)](#llamacpp-local-default)
  - [OpenAI-Compatible (Remote)](#openai-compatible-remote-llm)
- [Privacy Mode](#privacy-mode)
- [Dictation Modes](#dictation-modes)
- [Activation Modes](#activation-modes)
- [Hotkey Configuration](#hotkey-configuration)
- [Style Presets](#style-presets)
  - [Built-in Presets](#built-in-presets)
  - [Prompt Structure](#prompt-structure)
  - [Few-Shot Examples](#few-shot-examples)
- [User Dictionary](#user-dictionary)
- [Session History](#session-history)
- [Confidence Router](#confidence-router)
- [Model Management](#model-management)
- [Environment Variables](#environment-variables)

---

## Runtime Settings

All settings are mutable at runtime via D-Bus — no daemon restart required for most changes. The CLI provides a convenient interface:

```bash
lune-voice settings                     # dump all settings as TOML
lune-voice settings style email         # switch style preset
lune-voice settings language en-GB      # change language (BCP-47 validated)
lune-voice settings privacy AllowRemote # change privacy mode
lune-voice settings history 90          # set retention to 90 days
lune-voice settings history -1          # disable session history
```

Settings mutability tiers:

| Tier | When Applied | Fields |
|------|-------------|--------|
| **Hot** | Immediately (watch channel) | `dictation_mode`, `activation_mode` |
| **Warm** | Next session start | `default_style`, `language` |
| **Cold** | Persist only | `privacy_mode`, `session_history_days`, `hotkey` |
| **Restart** | Daemon restart | `asr_backend`, `llm_backend` |

All changes are persisted to disk immediately. D-Bus signals (`SettingsChanged`, `DictationModeChanged`, `ActivationModeChanged`) are emitted on every change for UI integration.

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
language = "en-US"
privacy_mode = "LocalOnly"
dictation_mode = "TypeInPlace"
activation_mode = "Toggle"
default_style = "general"
session_history_days = 30

[asr_backend]
type = "OrtQwenAsr"
model_dir = ""          # auto-populated from model downloads
use_gpu = true

[llm_backend]
type = "LlamaCppLocal"
model_path = ""         # auto-populated from model downloads
n_gpu_layers = 1000
n_ctx = 2048
max_output_tokens = 256

[hotkey]
toggle_dictation = { modifiers = ["Ctrl"], key = "Space" }
cancel_dictation = { modifiers = [], key = "Escape" }
```

> Model paths are auto-populated on first run after models are downloaded from HuggingFace. You generally don't need to set them manually.

---

## ASR Backends

### Qwen3-ASR (Local, default)

The default backend. Runs Qwen3-ASR 0.6B locally via ONNX Runtime with streaming transcription.

```toml
[asr_backend]
type = "OrtQwenAsr"
model_dir = "/path/to/qwen3-asr-0.6b-onnx"
use_gpu = true
stream_chunk_sec = 0.5
stream_rollback = 5
stream_unfixed_chunks = 2
stream_max_new_tokens = 32
past_text_conditioning = true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_dir` | path | auto | Directory containing encoder.onnx, embed_tokens.onnx, decoder_merged.onnx, and tokenizer.json |
| `use_gpu` | bool | `true` | Enable CUDA GPU acceleration |
| `stream_chunk_sec` | f32 | `0.5` | Streaming chunk size in seconds |
| `stream_rollback` | i32 | `5` | Number of rollback tokens kept for boundary stability |
| `stream_unfixed_chunks` | i32 | `2` | Number of initial unfixed chunks before stable commit |
| `stream_max_new_tokens` | i32 | `32` | Decoder cap per stream update |
| `past_text_conditioning` | bool | `true` | Whether stream chunks are conditioned on prior committed text |

**Model files expected in `model_dir`:**

| File | Purpose |
|------|---------|
| `encoder.onnx` | AuT audio encoder + multi-modal projector |
| `embed_tokens.onnx` | Token embedding lookup |
| `decoder_merged.onnx` | Qwen3 decoder with KV cache |
| `tokenizer.json` | HuggingFace tokenizer |

**Inference details:**

- Input: 16 kHz mono PCM → 128-mel spectrogram (Slaney scale, reflection padding)
- Streaming: rollback-buffered emission with degeneracy detection and periodic re-anchoring
- Output: text with automatic speech-end detection

### Parakeet TDT (Local)

Alternative ASR backend using NVIDIA's Parakeet TDT 0.6B model locally via ONNX Runtime.

```toml
[asr_backend]
type = "ParakeetTdt"
model_dir = "/path/to/parakeet-tdt-0.6b-v3-onnx"
use_gpu = true
context_secs = 30.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_dir` | path | auto | Directory containing ONNX model files and `vocab.txt` |
| `use_gpu` | bool | `true` | Enable CUDA GPU acceleration (requires `--features lune-asr/cuda` at build time) |
| `context_secs` | f32 | `30.0` | Sliding context window in seconds. Longer = better context for extended dictation, but more memory |

**Model files expected in `model_dir`:**

| File | Purpose |
|------|---------|
| `encoder-model.onnx` or `encoder-model.int8.onnx` | Encoder network |
| `decoder_joint-model.onnx` or `decoder_joint-model.int8.onnx` | Decoder-joint network |
| `vocab.txt` | SentencePiece vocabulary |
| `*.onnx.data` | External weight data (if model uses external format) |

The backend searches for files in priority order and falls back to glob matching (`encoder*.onnx`).

**Inference details:**

- Input: 16 kHz mono PCM → 128-mel spectrogram (Slaney scale)
- Encoder stride: 8× (1280 samples per output frame)
- Inference trigger: fires after 300ms (4800 samples) of accumulated audio
- Output: text with per-token confidence scores (softmax probability)

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

> **Confidence:** Remote backends don't expose per-token confidence. The confidence router treats `None` confidence as 0.0 (low), causing all text to accumulate before LLM dispatch.

---

## LLM Backends

### LlamaCpp Local (default)

The default backend. Runs any GGUF model locally via llama.cpp for `[?word?]` span correction.

```toml
[llm_backend]
type = "LlamaCppLocal"
model_path = "/path/to/model.gguf"
n_gpu_layers = 1000
n_ctx = 2048
max_output_tokens = 256
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | path | auto | Path to the `.gguf` model file |
| `n_gpu_layers` | u32 | `1000` | GPU layers to offload (0 = CPU only, 1000 = all GPU) |
| `n_ctx` | u32 | `2048` | Context window size in tokens |
| `max_output_tokens` | u32 | `256` | Maximum tokens to generate per request |

**Inference:** Greedy argmax decoding via llama.cpp. A fresh context is created per call (stateless). Input is raw text with `[?word?]` markers; output is corrected text.

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

Controls what happens with corrected text after LLM processing.

```toml
dictation_mode = "TypeInPlace"
```

| Mode | Behavior |
|------|----------|
| `TypeInPlace` | Types corrected text into the currently focused input field using keyboard synthesis (`enigo` / `wtype`) **(default)** |
| `Clipboard` | Copies corrected text to the system clipboard only (via `arboard`) |
| `Both` | Types into focused field AND copies to clipboard |

Switch at runtime:

```bash
lune-voice mode dictation Clipboard
```

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
Settings → Keyboard → Custom Shortcuts
  Name: Lune Toggle
  Command: lune-voice toggle
  Shortcut: Ctrl+Space
```

**KDE:**
```
System Settings → Shortcuts → Custom Shortcuts
  Add → Command/URL
  Trigger: Ctrl+Space
  Action: lune-voice toggle
```

**macOS:**
Use Automator or a tool like Hammerspoon to bind `lune-voice toggle` to a hotkey.

**Windows:**
Use AutoHotkey or PowerToys to bind to `lune-voice toggle`.

---

## Style Presets

### Built-in Presets

| ID | Name | Description | Use Case |
|----|------|-------------|----------|
| `general` | General | Clean grammar and remove disfluencies | Everyday dictation |
| `email` | Email | Formal register with paragraph breaks | Professional emails |
| `slack` | Slack | Casual, concise messaging tone | Chat messages |
| `meeting_notes` | Meeting Notes | Bullet points with `Action:` items | Meeting summaries |
| `code_comments` | Code Comments | Terse, imperative, technical doc comments | Documentation |

Set the default style:

```toml
default_style = "general"
```

Switch at runtime:

```bash
lune-voice settings style email
```

### Prompt Structure

Each style preset defines a full prompt template:

| Component | Purpose |
|-----------|---------|
| `system_prompt` | Instructions for the LLM's role and behavior |
| `user_prompt_template` | Template with `{raw_text}` placeholder for the input |
| `few_shot_pairs` | Example input/output pairs for in-context learning |

The final prompt is assembled as ChatML:

```
<|im_start|>system
/no_think
{system_prompt}

Dictionary terms (always use these exact spellings):
- lunevoice → Lune Voice
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

The `/no_think` directive disables Qwen3's chain-of-thought mode. The pre-filled `<think>\n</think>\n` at the start of each assistant turn further suppresses reasoning output.

### Few-Shot Examples

Each preset includes 2 input/output pairs that teach the LLM the expected transformation:

**General:**
```
Input:  "so um I was thinking that we should like go to the store"
Output: "I was thinking that we should go to the store."

Input:  "the the meeting is uh scheduled for tomorrow at 3"
Output: "The meeting is scheduled for tomorrow at 3."
```

**Email:**
```
Input:  "hey uh just wanted to follow up on the thing we discussed last week about the project timeline"
Output: "I wanted to follow up on our discussion last week regarding the project timeline."
```

**Meeting Notes:**
```
Input:  "so we talked about the Q3 roadmap and uh John is going to handle the API redesign and Sarah needs to review the mockups by next Tuesday"
Output: "- Discussed Q3 roadmap
         - Action: John — API redesign
         - Action: Sarah — review mockups by next Tuesday"
```

---

## User Dictionary

The dictionary provides phonetic substitution rules that are applied before LLM processing. Useful for proper nouns, brand names, and domain-specific terms.

Dictionary entries are also injected into the LLM prompt as "Dictionary terms (always use these exact spellings)" to reinforce correct spelling.

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
| `backend` | enum | ASR backend type (ParakeetTdt / OpenAiCompatible) |
| `status` | enum | Completed / Failed(reason) |
| `transcript` | string | Final corrected transcript (null if empty) |

Database location:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/lune-voice/lune-voice.db` |
| macOS | `~/Library/Application Support/lune-voice/lune-voice.db` |
| Windows | `%APPDATA%\lune-voice\lune-voice.db` |

---

## Confidence Router

The confidence router controls how ASR output is batched before LLM processing. These settings are internal (not exposed in the TOML file) but can be modified in code.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `high_threshold` | 0.82 | Confidence above which text is queued to LLM immediately (single utterance) |
| `mid_threshold` | 0.50 | Boundary between low and mid word-count thresholds |
| `low_word_threshold` | 8 | Words to accumulate before dispatching low-confidence text |
| `mid_word_threshold` | 12 | Words to accumulate before dispatching mid-confidence text |
| `context_window_size` | 3 | Number of recent confirmed utterances kept as LLM context |
| `max_length_ratio` | 2.0 | Reject LLM correction if output/input length ratio exceeds this |
| `min_length_ratio` | 0.5 | Reject LLM correction if output/input length ratio is below this |

**Routing logic:**

```
confidence ≥ 0.82  →  Queue to LLM immediately (single utterance)
0.50 ≤ conf < 0.82 →  Accumulate until 12 words, then batch to LLM
conf < 0.50        →  Accumulate until 8 words, then batch to LLM
conf = None        →  Treated as 0.0 (conservative batching)
```

**Length sanity check:** After the LLM returns a correction, the ratio `len(corrected) / len(original)` is checked. If outside `[0.5, 2.0]`, the correction is rejected and the original text is delivered instead.

---

## Model Management

Models are downloaded automatically on first run from HuggingFace Hub.

| Model | Repository | Files | Size |
|-------|-----------|-------|------|
| Qwen3-ASR 0.6B | `lune-voice/qwen3-asr-0.6b-onnx` | encoder.onnx, embed_tokens.onnx, decoder_merged.onnx, tokenizer.json | ~1.2 GB |
| LLM Corrector | `lune-voice/llm-corrector-gguf` | model.gguf | varies |

Models are stored in:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/lune-voice/models/` |
| macOS | `~/Library/Application Support/lune-voice/models/` |
| Windows | `%APPDATA%\lune-voice\models\` |

To use a custom model, set the path directly in the config:

```toml
[asr_backend]
type = "OrtQwenAsr"
model_dir = "/path/to/custom/asr-model"

[llm_backend]
type = "LlamaCppLocal"
model_path = "/path/to/custom/model.gguf"
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG` | Tracing filter (e.g., `lune_asr=debug,lune_llm=info`) |
| `LUNE_VOICE_CONFIG` | Override settings file path |
| `HF_HOME` | Override HuggingFace cache directory (affects model downloads) |
| `ORT_DYLIB_PATH` | Override ONNX Runtime shared library path |
| `CUDA_VISIBLE_DEVICES` | Control which GPU is used for inference |
