# 03 – Detailed Component Design

← [Architecture](./02-architecture.md) | → [Data Design](./04-data-design.md)

---

## 5.1 Audio Capture Service (`lune-audio`)

### Responsibilities
- Configure and open microphone stream (mono, fixed sample rate).
- Chunk audio into fixed frames (frame size + hop aligned to Voxtral encoder expectations).
- Normalize amplitude; apply lightweight VAD to reduce unnecessary ASR load.
- Push frames into an async `mpsc` channel consumed by the ASR Engine.
- Manage per-session state (start/stop with hotkey).

### Key Design Points

```rust
pub struct AudioCaptureService {
    device:     cpal::Device,
    config:     cpal::SupportedStreamConfig,
    frame_size: usize,      // samples per chunk, aligned to ASR encoder
    hop_size:   usize,
    vad:        Option<Box<dyn VadFilter>>,
}

impl AudioCaptureService {
    pub async fn start_session(
        &self,
        tx: mpsc::Sender<AudioFrame>,
    ) -> Result<SessionHandle>;

    pub async fn stop_session(&self, handle: SessionHandle) -> Result<()>;
}
```

### Checklist
- [ ] `cpal` input stream opens without error on macOS CoreAudio
- [ ] `cpal` input stream opens without error on Windows WASAPI
- [ ] Frame size matches Voxtral encoder (document exact value)
- [ ] Mono downmix from stereo sources implemented
- [ ] Sample rate resampling if device default ≠ model expected rate
- [ ] VAD: configurable on/off; default on to reduce load
- [ ] Silent frames filtered before pushing to ASR channel
- [ ] `AudioFrame` type carries timestamp + sample data
- [ ] Session handle allows clean teardown (no cpal stream leak)
- [ ] Unit test: stream produces frames at expected rate

---

## 5.2 ASR Engine Abstraction (`lune-asr`)

### Trait Definition

```rust
#[async_trait::async_trait]
pub trait AsrBackend: Send + Sync {
    type SessionHandle: Send;

    async fn start_session(
        &self,
        config: AsrSessionConfig,
    ) -> Result<Self::SessionHandle>;

    async fn push_audio(
        &self,
        session: &Self::SessionHandle,
        pcm: &[f32],
    ) -> Result<()>;

    async fn poll_results(
        &self,
        session: &Self::SessionHandle,
    ) -> Result<Vec<AsrSegment>>;

    async fn end_session(
        &self,
        session: Self::SessionHandle,
    ) -> Result<()>;
}
```

### Segment Model

```rust
pub struct AsrSegment {
    pub id:       u64,
    pub text:     String,
    pub is_final: bool,       // immutable once true
    pub start_ms: Option<u64>,
    pub end_ms:   Option<u64>,
}

pub struct AsrSessionConfig {
    pub language:       Option<String>,   // BCP-47 tag or None for auto-detect
    pub sample_rate:    u32,
    pub frame_size:     usize,
    pub model_hint:     Option<String>,   // model name override for remote backends
}
```

### 5.2.1 VoxtralLocalBackend

- Wraps a Rust Voxtral inference implementation (GGUF/safetensors).
- Manages internal sliding-window context for unbounded streaming.
- Emits partial and final segments with timestamps when available.
- GPU/CPU backend selected via build feature flags.

#### Checklist
- [ ] GGUF weights loading without panic on missing file (clear error)
- [ ] Encoder frame ingestion aligned to model's expected input shape
- [ ] Sliding context window size documented and configurable
- [ ] Partial segment emission: fires at each internal decode step
- [ ] Final segment marking: triggered on silence or end-of-session
- [ ] GPU acceleration path tested (CUDA / Metal)
- [ ] CPU fallback confirmed functional with quantized weights
- [ ] RTF ≤ 0.5 measured on reference hardware (document device spec)
- [ ] Thread safety: `Send + Sync` on backend struct
- [ ] Integration test: push 5 s of audio → receive transcription segments

### 5.2.2 OpenAiCompatibleAsrBackend

- Sends audio to `POST /v1/audio/transcriptions`.
- Streaming preferred via WebSocket when backend supports it.
- Fallback: periodic short segments buffered client-side, sent as individual requests.

#### Checklist
- [ ] Configurable base URL and `model` field
- [ ] multipart/form-data request construction (audio + model + optional prompt/language)
- [ ] WebSocket streaming mode implemented
- [ ] Fallback HTTP polling mode implemented with configurable segment duration
- [ ] Auth header (Bearer token) injected from settings — never hardcoded
- [ ] TLS required; plaintext connections rejected unless `127.0.0.1`
- [ ] Retry on transient errors (5xx, network timeout) with exponential backoff
- [ ] Integration test against local mock server

---

## 5.3 Text Post-Processor (`lune-llm`)

### Trait Definition

```rust
#[async_trait::async_trait]
pub trait TextPostProcessor: Send + Sync {
    async fn transform(
        &self,
        input: &PostProcessInput,
    ) -> Result<PostProcessOutput>;
}

pub struct PostProcessInput {
    pub raw_text:      String,
    pub prev_context:  Option<String>,   // preceding text in target app
    pub style_preset:  StylePreset,
    pub app_category:  Option<AppCategory>, // email, ide, chat, …
    pub dictionary:    Vec<DictionaryEntry>,
}

pub struct PostProcessOutput {
    pub cleaned_text: String,
    pub metadata:     Option<OutputMetadata>, // bullets, headings, etc.
}
```

### 5.3.1 LocalLlmBackend (llama-cpp-rs)

- Small chat-style model (1–3B params, Q4 quantized).
- Prompt template per style preset (system + few-shot user/assistant).
- Batched on multi-sentence windows, not per-token.

#### Checklist
- [ ] llama-cpp-rs version pinned in Cargo.toml
- [ ] Model path loaded from `UserSettings`; clear error if missing
- [ ] Style preset → system prompt mapping implemented
- [ ] Few-shot examples cover: grammar fix, disfluency removal, style enforcing
- [ ] Dictionary entries injected as few-shot correction examples
- [ ] Inference runs off the hot path (separate Tokio task or blocking thread pool)
- [ ] Max latency budget documented per preset (target: < 500 ms for short segments)
- [ ] Unit test: known disfluent input → expected cleaned output

### 5.3.2 OpenAiCompatibleLlmBackend

- Calls `POST /v1/chat/completions` or `/v1/completions`.
- Uses style-specific prompt templates.
- Extracts `choices[0].message.content` as transformed text.

#### Checklist
- [ ] Prompt templates per style preset stored in settings/config
- [ ] System + user message construction validated
- [ ] `max_tokens` bounded to prevent runaway cost
- [ ] Auth header from settings; never hardcoded
- [ ] Streaming response (`stream: true`) supported for perceived latency reduction
- [ ] Error path: LLM failure → return raw ASR text with warning logged

### Prompt Strategy

Each style preset defines:

| Field | Contents |
|-------|----------|
| `system_prompt` | Tone, constraints (e.g., "no emojis", "formal register") |
| `user_prompt_template` | Parameterized with `{raw_text}`, `{context}` |
| `few_shot_pairs` | 3–5 input→output pairs covering disfluency + grammar |

#### Checklist
- [ ] Prompts peer-reviewed for instruction following quality
- [ ] Prompts tested against target model(s) — not just assumed to work
- [ ] `StylePreset` serialization/deserialization round-trips correctly
- [ ] Default presets ship with app: General, Email, Slack, Meeting Notes, Code Comments

---

## 5.4 Session Manager (`lune-core`)

### Responsibilities

- Dictation session lifecycle (start/stop via hotkey or voice command).
- Wire: `AudioCapture → AsrEngine → TextPostProcessor → TextOutput`.
- Rolling buffer for partial ASR segments.
- Stabilization logic: mark segments final after configurable silence window.
- Minimal diff of output text to reduce target-app flicker.

### Key Design Points

```rust
pub struct SessionManager {
    audio:     Arc<AudioCaptureService>,
    asr:       Arc<dyn AsrBackend<SessionHandle = Box<dyn Any + Send>>>,
    llm:       Arc<dyn TextPostProcessor>,
    output:    Arc<TextOutputController>,
    commands:  Arc<VoiceCommandRouter>,
    settings:  Arc<RwLock<UserSettings>>,
}

pub enum SessionEvent {
    Started,
    PartialText(String),
    FinalizedText(String),
    CommandDetected(VoiceCommand),
    Error(anyhow::Error),
    Stopped,
}
```

### Checklist
- [ ] `start_session` → wires all components and returns `SessionHandle`
- [ ] `stop_session` → graceful teardown (flush buffer, end ASR session)
- [ ] Stabilization window configurable (default: 500 ms silence)
- [ ] Rolling buffer: partial segments accumulate, replaced on update, finalized on stable
- [ ] Output diff: character-level diff against last emitted text; backspace+retype minimal changes
- [ ] `SessionEvent` stream exposed to UI layer via Tauri event bridge
- [ ] Concurrent sessions rejected (one at a time)
- [ ] Integration test: full hotkey → audio → text → output round-trip

---

## 5.5 Text Output Controller (`lune-output`)

### Responsibilities

- Route cleaned text to:
  1. Active OS text field (keyboard injection).
  2. Internal Notes view.
- Support modes: "Live typing" (stream partials) vs "Type on stop" (emit on session end).
- Auto-insert line breaks, bullets, headings per style preset.

### OS-Specific Implementation

| Platform | Mechanism | Status |
|----------|-----------|--------|
| macOS | Accessibility API (`AXUIElement`) or `CGEventPost` | Planned v0.1 (Notes only), v0.2 (injection) |
| Windows | `SendInput` / `keybd_event` | Planned v0.2 |
| Linux | `xdotool` / `ydotool` (Wayland) | Stretch goal |

### Checklist
- [ ] Notes view sink implemented (Tauri event → frontend)
- [ ] macOS Accessibility permission check + user prompt on first run
- [ ] macOS keyboard injection via `CGEventPost` implemented
- [ ] Windows `SendInput` implemented
- [ ] "Live typing" mode: streams partial text with incremental backspace correction
- [ ] "Type on stop" mode: emits final post-processed text on session end
- [ ] Auto-formatting: newlines, bullets controlled by `OutputMetadata`
- [ ] Injection blocked in password fields (check `AXSecureTextField` / Windows equivalent)
- [ ] Unit test: injection string encodes special characters correctly

---

## 5.6 Voice Command Router (`lune-commands`)

### Responsibilities

- Detect command phrases from finalized ASR segments **before** post-processing.
- Dispatch app-level actions without passing trigger phrases to LLM.

### Built-in Command Grammar

| Spoken Phrase | Action |
|---------------|--------|
| "start dictation" | `SessionManager::start_session` |
| "stop dictation" / "stop" | `SessionManager::stop_session` |
| "new line" | Insert `\n` |
| "new paragraph" | Insert `\n\n` |
| "list item" | Insert `\n- ` |
| "switch to formal" | Change active `StylePreset` |
| "make this a Slack message" | Change active `StylePreset` |
| "insert [snippet name]" | `DictionaryAndSnippetManager::expand_snippet` |
| "scratch that" | Delete last emitted segment |

### Checklist
- [ ] Command detection runs on raw ASR text before LLM transform
- [ ] Exact phrase matching implemented (case-insensitive)
- [ ] Optional fuzzy/LLM classification for ambiguous commands (configurable)
- [ ] Commands filtered out of text before passing to post-processor
- [ ] User-customizable command phrases in settings
- [ ] Unit test: each built-in command phrase triggers correct action
- [ ] "Scratch that" correctly identifies and removes last emitted segment

---

## 5.7 Dictionary & Snippet Manager (`lune-dict`)

### Personal Dictionary

- Stores user-specific terms with preferred casing/spelling.
- Applied as:
  1. Post-ASR string replacement (near-match substitution).
  2. Few-shot examples injected into LLM prompt.

```rust
pub struct DictionaryEntry {
    pub phrase:      String,   // spoken form
    pub replacement: String,   // correct casing/spelling
    pub priority:    u8,       // higher = applied first
}
```

### Snippet Library

- Maps trigger phrases to text templates.
- Supports `{placeholder}` slots filled by context or interactive prompt.

```rust
pub struct Snippet {
    pub id:             String,
    pub trigger_phrase: String,
    pub body:           String,    // may contain {date}, {name}, etc.
}
```

### Checklist
- [ ] `DictionaryEntry` CRUD operations (add, edit, delete, list)
- [ ] Near-match replacement: Levenshtein distance ≤ 1 for single words
- [ ] Dictionary entries included in LLM few-shot examples (≤ 10 most relevant)
- [ ] `Snippet` CRUD operations
- [ ] Snippet trigger detection in `VoiceCommandRouter`
- [ ] Placeholder `{date}` auto-filled with current date
- [ ] Other placeholders prompt user via tray notification or Notes view
- [ ] Persisted to `lune-persist` (survives app restart)
- [ ] Import/export dictionary as plain text (one entry per line)

---

## 5.8 Persistence Layer (`lune-persist`)

### Data Categories

| Category | Format | Location |
|----------|--------|----------|
| Settings | TOML | `$CONFIG_DIR/lune-voice/settings.toml` |
| Dictionary | TOML / SQLite | `$DATA_DIR/lune-voice/dictionary.db` |
| Snippets | TOML | `$DATA_DIR/lune-voice/snippets.toml` |
| Session history | SQLite (optional) | `$DATA_DIR/lune-voice/sessions.db` |
| Style presets | TOML | `$DATA_DIR/lune-voice/presets.toml` |

### Checklist
- [ ] `serde` + `toml` for human-editable files (settings, snippets, presets)
- [ ] SQLite via `rusqlite` or `sqlx` for dictionary and session history
- [ ] Migration strategy for schema changes (versioned migrations)
- [ ] `$CONFIG_DIR` / `$DATA_DIR` resolved via `dirs` crate (XDG on Linux, correct paths on macOS/Windows)
- [ ] Settings file created with defaults on first run
- [ ] All writes atomic (temp file + rename) to prevent corruption
- [ ] Session history retention configurable (default: 30 days or disabled)
- [ ] No raw audio written to disk unless explicitly enabled for debugging
- [ ] Unit test: round-trip serialize/deserialize for each data type
