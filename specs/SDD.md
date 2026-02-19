# Project SDD – Local Flow (Working Name)

## 1. Introduction

### 1.1 Purpose

This document defines the software design for **Lune Voice**, a cross‑platform Rust application that provides WisprFlow‑style voice dictation using **local, low‑latency speech recognition** plus a **text‑editing LLM** for grammar cleanup, filler removal, and style transformation.

### 1.2 Scope

Local Flow focuses on:

- Real‑time, streaming dictation for desktop (initially macOS and Windows, optional Linux).
- Primary ASR backend: **Mistral Voxtral Mini Realtime** family via local inference (GGUF) or compatible servers.
- Optional ASR via **OpenAI‑compatible `/v1/audio/transcriptions`** APIs (local or remote).
- A small LLM backend (local or OpenAI‑compatible) for:
  - Grammar and punctuation fixes.
  - Removal of disfluencies (“um”, “uh”, restarts).
  - Text style transforms (e.g., “concise email”, “meeting notes”, “friendly Slack”).
- User‑level features: personal dictionary, snippets/shortcuts, voice commands.

Mobile clients and full note‑taking UX (like WisprFlow’s notes view) are explicitly out of scope for v1.

### 1.3 Definitions and Acronyms

- **ASR** – Automatic Speech Recognition.
- **LLM** – Large Language Model used here as a text post‑processor.
- **Voxtral Realtime** – Mistral’s streaming ASR model family optimized for low‑latency, on‑device transcription.
- **GGUF** – Quantized model format used by llama.cpp‑style runtimes and TrevorJS’s Voxtral Q4 build.
- **RTF** – Real‑Time Factor, ratio of inference time to audio duration.[web:15]

---

## 2. System Overview

Lune Flow is a background desktop app that:

1. Captures the user’s microphone audio continuously while a **push‑to‑talk** or **always‑listening with wake‑word** mode is active.
2. Streams audio into an **ASR Engine** (local Voxtral or OpenAI‑compatible API) that emits incremental text segments.
3. Passes finalized text segments into a **Text Post‑Processor** (small LLM) that:
   - Fixes grammar and punctuation.
   - Removes filler words and hesitations.
   - Applies a configurable **style preset** (e.g., “formal email”, “bullet‑point summary”).
4. Outputs the transformed text into:
   - The active text field via OS‑level keyboard injection.
   - Or a **Notes view** inside the app for long‑form capture.

In addition, Local Flow maintains:

- A **Personal Dictionary** of user‑specific terms (names, jargon) similar to WisprFlow.
- A **Snippet Library** of reusable text blocks invoked via spoken cues (“insert my calendar link”).
- A **Voice Command Router** for app‑level actions (start/stop, style switching, snippet insertion, simple formatting).

---

## 3. Goals, Non‑Goals, and Constraints

### 3.1 Primary Goals

- **Low latency streaming dictation** competitive with cloud APIs while running locally.
- **High‑quality text output** via post‑processing, approaching WisprFlow’s “AI auto edits”.
- **Privacy‑preserving** operation: all audio and text can stay on device when using local backends.
- **Configurable backends**: local Voxtral vs. OpenAI‑compatible ASR/LLM.
- **Simple UX**: single global hotkey, minimal visible chrome during dictation, clear feedback.

### 3.2 Non‑Goals (v1)

- Full voice navigation of the OS or apps (beyond simple commands).
- Multi‑user diarization or speaker identification (Voxtral Realtime itself does not provide diarization; that lives in a separate batch model).[web:12]
- Complex project management or multi‑document editing flows.
- Native mobile applications (can be future phases).

### 3.3 Constraints and Assumptions

- The primary local ASR backend is **Voxtral Mini 4B Realtime**, via either:
  - A Rust implementation with GGUF/safetensors weights (TrevorJS’s implementation or equivalent).
  - A local vLLM/vLLM‑like server exposing an OpenAI‑style realtime or websockets API.
- Llama.cpp currently does not yet have upstream Voxtral support, but the design should keep a **pluggable inference layer** so llama‑cpp‑rs can be integrated once support is available.
- Hardware target: modern consumer laptops and desktops; GPU acceleration is preferred but CPU‑only operation should remain usable with quantized models.

---

## 4. High‑Level Architecture

### 4.1 Component Diagram (Logical)

- **UI Layer**
  - Global Tray / Menu Bar UI
  - Notes View
  - Settings Panel
- **Application Core**
  - Session Manager
  - Audio Capture Service
  - ASR Engine Abstraction
  - Text Post‑Processor (LLM) Abstraction
  - Text Output Controller
  - Voice Command Router
  - Dictionary & Snippet Manager
  - Persistence Layer
- **Backends**
  - Local Voxtral Backend (Rust / GGUF)
  - OpenAI‑compatible ASR Backend
  - Local LLM Backend (llama‑cpp‑rs) for text edits
  - OpenAI‑compatible LLM Backend

All components are implemented in Rust with async orchestration (Tokio).

### 4.2 Technology Choices (Initial)

- **Language**: Rust (safety, performance, good FFI story).
- **Async runtime**: `tokio`.
- **Audio capture**: `cpal` for cross‑platform input streams.
- **Desktop UI**: Tauri (Rust backend + WebView UI).
- **Model backends**:
  - Voxtral local: Custom Inference Backend in rust.
  - LLM local: `llama-cpp-rs` for small instruction‑tuned models.
  - Remote: generic OpenAI‑style HTTP client configurable via settings.

---

## 5. Detailed Component Design

### 5.1 Audio Capture Service

**Responsibilities**

- Configure and open microphone stream (mono, fixed sample rate).
- Chunk audio into small frames (fixed frame size + hop).
- Normalize and optionally apply simple VAD (voice activity detection) to reduce unnecessary ASR load.
- Push frames into an async mpsc channel consumed by the ASR Engine.

**Key Design Points**

- Maintain consistent frame size that aligns with the Voxtral encoder’s expectations。
- Stateful management per **Dictation Session** (start/stop with hotkey, per‑session configuration).

### 5.2 ASR Engine Abstraction

Define a trait:

```rust
#[async_trait::async_trait]
pub trait AsrBackend {
    type SessionHandle;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle>;
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()>;
    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>>;
    async fn end_session(&self, session: Self::SessionHandle) -> Result<()>;
}
```

**Implementations**

1. **VoxtralLocalBackend**
    - Wraps a Rust Voxtral implementation (e.g., `voxtral-mini-realtime-rs`) in a streaming API.[web:15]
    - Responsible for:
        - Feeding audio frames into the encoder.
        - Managing internal sliding‑window context for “infinite” streaming.[web:3][web:12]
        - Emitting partial and final text segments, plus timestamps if available.
2. **OpenAiCompatibleAsrBackend**
    - Sends chunked or full audio to `/v1/audio/transcriptions` style endpoints (OpenAI Whisper, custom servers, vLLM realtime gateway).
    - For true streaming:
        - Preferred mode is websockets when backend supports it (vLLM realtime example).
        - Fallback mode: periodic short segments buffered on the client and sent as small requests.

**Segment Model**

```rust
struct AsrSegment {
    id: u64,
    text: String,
    is_final: bool,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
}
```

Segments are immutable once marked `is_final`.

### 5.3 Text Post‑Processor (LLM)

Define a trait:

```rust
#[async_trait::async_trait]
pub trait TextPostProcessor {
    async fn transform(
        &self,
        input: &PostProcessInput
    ) -> Result<PostProcessOutput>;
}
```

`PostProcessInput` includes:

- Raw ASR text (possibly multiple joined segments).
- Optional context: previous text in the target app, user’s style preset, app category (email, IDE, chat).

`PostProcessOutput` includes:

- Cleaned text.
- Optional structured metadata (e.g., bullet lists, headings) for formatting.

**Backends**

- **LocalLlmBackend**: uses `llama-cpp-rs` with a small chat‑style model for fast inference.
- **OpenAiCompatibleLlmBackend**: calls `/v1/chat/completions` or `/v1/completions` with prompt templates specialized for:
    - Grammar fixes.
    - Disfluency removal.
    - Style enforcement.

**Prompt Strategy**

For each style preset, define:

- A system prompt specifying:
    - Target tone (e.g., “friendly but professional email”).
    - Constraints (no emojis, no hedging phrases if desired).
- A few‑shot set to robustly remove disfluencies and partial restarts.


### 5.4 Session Manager

Orchestrates:

- Dictation session lifecycle (start/stop).
- Wiring between Audio Capture, ASR Engine, Text Post‑Processor, and Text Output.
- Debouncing and buffering partial ASR segments:
    - Uses a rolling buffer of text.
    - Compares new partial transcripts to last stable text and only applies minimal diffs to the target application (reduces flicker).


### 5.5 Text Output Controller

Responsibilities:

- Decide when to emit text to:
    - Active OS text field (using keyboard injection APIs or OS accessibility APIs).
    - Internal Notes view (for long‑form dictation).
- Provide options:
    - “Live typing” vs. “Type on stop”.
    - Automatic insertion of line breaks, bullets, headings based on style.

Implementation details depend on OS:

- macOS: Accessibility API or AppleScript for text injection.
- Windows: low‑level keyboard events; later Windows Text Services Framework.


### 5.6 Voice Command Router

Provides:

- **Command grammar**: simple phrases for:
    - Start/stop recording (“start dictation”, “stop dictation”).
    - Formatting (“new line”, “new paragraph”, “list item”).
    - Style changes (“switch to formal”, “make this a Slack message”).
    - Snippets (“insert calendar link”).
- Command detection options:
    - Lightweight keyword matching on ASR text.
    - Optional LLM classification for ambiguous commands.

Commands are resolved before post‑processing to avoid style transforms interfering with trigger phrases.

### 5.7 Dictionary \& Snippet Manager

**Personal Dictionary**

- Stores user terms and preferred casing/spelling (e.g., product names, internal acronyms), similar to WisprFlow’s personal dictionary.
- Used by:
    - ASR post‑processor (replace near‑matches).
    - LLM prompt (few‑shot examples to preserve spelling).

**Snippet Library**

- Maps short keys (spoken or text) to larger templates (e.g., “insert PTO policy boilerplate”).
- Trigger detection via Voice Command Router.
- Supports parameter slots (e.g., `{date}`, `{name}`) filled by current context or quick prompts.


### 5.8 Persistence Layer

- Local file‑based storage using `serde` + `ron`/`toml`/`sqlite`.
- Data categories:
    - Settings (model paths, backend URLs, hotkeys).
    - Dictionaries and snippets.
    - Recent sessions and notes (optional; user‑configurable retention).

---

## 6. Data Design

### 6.1 Core Data Structures

Key structs (conceptual):

```rust
struct UserSettings {
    asr_backend: AsrBackendConfig,
    llm_backend: LlmBackendConfig,
    hotkey: HotkeyConfig,
    default_style: StyleId,
    language: String,
    privacy_mode: PrivacyMode,
}

struct StylePreset {
    id: StyleId,
    name: String,
    description: String,
    system_prompt: String,
    user_prompt_template: String,
}

struct DictionaryEntry {
    phrase: String,
    replacement: String,
    priority: u8,
}

struct Snippet {
    id: String,
    trigger_phrase: String,
    body: String,
}

struct DictationSession {
    id: Uuid,
    started_at: DateTime<Utc>,
    ended_at: Option<DateTime<Utc>>,
    style: StyleId,
    backend: AsrBackendKind,
    status: SessionStatus,
}
```


### 6.2 Text Processing Pipeline

1. **Raw ASR segments** are appended to a session buffer.
2. **Stabilization** logic marks segments as final when they stop changing for a configurable window.
3. Finalized text is sent to the **Text Post‑Processor** with:
    - Aggregated session text.
    - Style preset metadata.
    - Dictionary hints.
4. Post‑processed text is diffed against current output and applied to the target sink.

---

## 7. External Interfaces

### 7.1 OpenAI‑Compatible ASR API

- Endpoint: `POST /v1/audio/transcriptions`
- Request: multipart/form‑data with:
    - `file` (audio).
    - `model` (string).
    - Optional `prompt`, `language`.
- Response: JSON with `text` and optional `segments`.

Local Flow must support:

- Custom base URL (self‑hosted inferencer).
- Custom `model` name (e.g., `voxtral-mini-realtime-2602`).


### 7.2 OpenAI‑Compatible LLM API

- Endpoint: `POST /v1/chat/completions` or `/v1/completions`.
- Request: includes style‑specific system + user prompts.
- Response: single `choices[^0].message.content` used as transformed text.


### 7.3 Local Voxtral Interface

Two possible approaches:

1. **In‑process Rust Library (Preferred)**
    - Use `voxtral-mini-realtime-rs` as a dependency or similar Rust implementation.
    - Provides direct functions:
        - `VoxtralSession::push_audio(&[f32])`.
        - `VoxtralSession::poll_text() -> Vec<Segment>`.
2. **Sidecar Server**
    - Spawn a local HTTP/websocket server (e.g., vLLM Voxtral deployment) listening on localhost.
    - Communicate over WebSocket for low‑latency streaming.
3. **Custom Inference Backend**
    - Create a novel custom inference backend in rust

The design should keep this behind the `AsrBackend` trait so both are swappable.

---

## 8. Performance and Resource Targets

### 8.1 Latency and Throughput

Based on published Voxtral Realtime materials and Rust GGUF benchmarks:

- **Target end‑to‑end latency** (speech to displayed text) is **well under one second** for typical desktop hardware, with typical streaming configurations using a sub‑half‑second internal model delay.
- Reference Rust implementation reports an **RTF ≈ 0.42**, meaning transcription completes in noticeably less than real time for a benchmark device; converting informal testing into an interval yields approximately $0.42 \pm 0.1$ @ 90% confidence for similar hardware classes.

Design implications:

- Default configuration:
    - Conservative chunk sizes that avoid overrunning CPU/GPU.
    - Batching of LLM post‑processing on multi‑sentence windows rather than every token.
- Optional advanced config:
    - Adjustable trade‑off between “instant” partials vs. more stable, slightly delayed text (mirroring Voxtral’s configurable delay).


### 8.2 Memory and Model Size

- Q4 quantized Voxtral GGUF weights are on the order of a few GB, with working memory on a modern GPU in the low single‑digit GB range.
- A compact local LLM (1–3B parameters) can be quantized to a similar footprint.

Target hardware guideline (approximate):

- Recommended: discrete GPU with mid‑range VRAM and 16+ GB system RAM; interpreted as `VRAM ≈ 8–12 GB` with an uncertainty of ±4 GB at 80% confidence, acknowledging vendor and driver variation.

---

## 9. Security and Privacy

- **On‑device preference**: default configuration uses local ASR + LLM when models are available.
- **Explicit remote opt‑in**: using cloud APIs (OpenAI or others) is a clear choice in settings.
- **Data handling**:
    - No permanent storage of raw audio unless explicitly enabled for debugging.
    - Optional local transcript history; can be disabled or auto‑expired.
- **Network security**:
    - HTTPS required for remote APIs.
    - For local sidecars, restrict binding to `127.0.0.1` unless overridden.

---

## 10. Error Handling, Telemetry, and Observability

- User‑visible:
    - Non‑intrusive toast notifications for backend failures.
    - Clear status indicators (listening, processing, idle).
- Logging:
    - Structured logs (e.g., `tracing`) with categories:
        - Audio, ASR, LLM, UI, Backend.
    - Optional anonymized performance metrics (RTF, latency percentiles) for local debugging.
- Recovery:
    - Automatic reconnection to local/remote backends on transient failures.
    - Graceful degradation from streaming to chunked modes if websockets fail.

---

## 11. Open Issues and Risks

- **Model support in llama.cpp**: full Voxtral support is not yet merged, so reliance on an external Rust implementation or vLLM sidecar is necessary initially.
- **GPU variability**: performance will vary widely across devices; defaults must be conservative, with advanced tuning options.
- **Cross‑platform text injection**: OS‑level APIs differ; some environments may block simulated input.
- **LLM latency**: even small LLMs can introduce noticeable delays; careful batching and prompt design are required.

---

## 12. Roadmap (High‑Level)

**v0.1 (MVP)**

- Local Voxtral backend (Rust, GGUF) with streaming dictation into internal Notes view.
- Basic LLM post‑processor via local model unsloth/Qwen3-0.6B-GGUF.
- Single global hotkey, minimal window UI, manual style selection.
- Personal dictionary and simple snippet insertion.

**v0.2**

- Local LLM integration (llama‑cpp‑rs).
- OS text‑field injection for “type anywhere” behavior.
- Voice command grammar and formatting commands.

**v1.0**

- Configurable ASR/LLM backends (local vs remote).
- Per‑app style presets similar to WisprFlow.
- Improved stability, performance tuning, and basic telemetry controls.

