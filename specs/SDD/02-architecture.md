# 02 – High-Level Architecture

← [Introduction](./01-introduction.md) | → [Components](./03-components.md)

---

## 4.1 Component Diagram (Logical)

```
┌──────────────────────────────────────────────────────────────┐
│                        UI Layer                              │
│  ┌─────────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ Global Tray /   │  │ Notes View │  │ Settings Panel   │  │
│  │ Menu Bar UI     │  │            │  │                  │  │
│  └────────┬────────┘  └─────┬──────┘  └────────┬─────────┘  │
└───────────┼────────────────┼─────────────────┼─────────────┘
            │                │                 │
┌───────────▼────────────────▼─────────────────▼─────────────┐
│                     Application Core                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Session Manager                     │   │
│  └────┬──────────────────────────────────────────┬──────┘   │
│       │                                          │           │
│  ┌────▼──────────┐   ┌──────────────────┐  ┌────▼─────────┐ │
│  │ Audio Capture │──▶│  ASR Engine      │  │ Text Output  │ │
│  │ Service       │   │  Abstraction     │  │ Controller   │ │
│  └───────────────┘   └────────┬─────────┘  └──────────────┘ │
│                               │                              │
│                      ┌────────▼──────────┐                  │
│                      │ Text Post-        │                  │
│                      │ Processor (LLM)   │                  │
│                      └───────────────────┘                  │
│                                                             │
│  ┌───────────────────┐   ┌───────────────────────────────┐  │
│  │ Voice Command     │   │ Dictionary & Snippet Manager  │  │
│  │ Router            │   │                               │  │
│  └───────────────────┘   └───────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Persistence Layer                  │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
            │                                   │
┌───────────▼───────────┐     ┌─────────────────▼──────────────┐
│     ASR Backends       │     │       LLM Backends             │
│                        │     │                                │
│ ┌────────────────────┐ │     │ ┌────────────────────────────┐ │
│ │ VoxtralLocalBackend│ │     │ │ LocalLlmBackend            │ │
│ │ (Rust/GGUF)        │ │     │ │ (llama-cpp-rs)             │ │
│ └────────────────────┘ │     │ └────────────────────────────┘ │
│ ┌────────────────────┐ │     │ ┌────────────────────────────┐ │
│ │ OpenAiCompatible   │ │     │ │ OpenAiCompatibleLlmBackend │ │
│ │ AsrBackend         │ │     │ │ (/v1/chat/completions)     │ │
│ └────────────────────┘ │     │ └────────────────────────────┘ │
└────────────────────────┘     └────────────────────────────────┘
```

---

## 4.2 Data Flow (Dictation Session)

```
Microphone
    │
    ▼ (pcm f32 frames)
AudioCaptureService  ──(mpsc channel)──▶  AsrEngine
                                              │
                                    partial/final AsrSegments
                                              │
                                              ▼
                                       SessionManager
                                       (stabilization buffer)
                                              │
                                    finalized text batch
                                              │
                              ┌───────────────┴──────────────────┐
                              ▼                                   ▼
                     VoiceCommandRouter                  TextPostProcessor
                     (check for commands)               (LLM transform)
                              │                                   │
                        dispatch command                  cleaned text
                              │                                   │
                              └───────────────┬──────────────────┘
                                              ▼
                                    TextOutputController
                                    (OS keyboard inject
                                     or Notes view)
```

---

## 4.3 Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | **Rust** | Safety, performance, strong FFI for model runtimes |
| Async runtime | **Tokio** | Ecosystem standard; fits cpal/network/IPC patterns |
| Audio capture | **cpal** | Cross-platform microphone input (macOS CoreAudio, Windows WASAPI) |
| Desktop UI | **Tauri** | Rust backend + WebView frontend; small binary |
| Local ASR | Custom Rust GGUF backend | Pluggable behind `AsrBackend` trait |
| Local LLM | **llama-cpp-rs** | Proven Rust binding for GGUF inference |
| Remote APIs | Generic HTTP client (reqwest) | Configurable base URL + model name |
| Serialization | **serde** + toml/ron/sqlite | Settings, dictionary, snippets |
| Logging | **tracing** | Structured, async-safe, category-filtered |

---

## 4.4 Crate Layout (Proposed)

```
lune-voice/
├── Cargo.toml                  # workspace
├── crates/
│   ├── lune-core/              # Session orchestration, traits, shared types
│   ├── lune-audio/             # AudioCaptureService (cpal)
│   ├── lune-asr/               # AsrBackend trait + implementations
│   │   ├── src/local/          # VoxtralLocalBackend
│   │   └── src/openai/         # OpenAiCompatibleAsrBackend
│   ├── lune-llm/               # TextPostProcessor trait + implementations
│   │   ├── src/local/          # LocalLlmBackend (llama-cpp-rs)
│   │   └── src/openai/         # OpenAiCompatibleLlmBackend
│   ├── lune-output/            # TextOutputController (OS injection)
│   ├── lune-commands/          # VoiceCommandRouter
│   ├── lune-dict/              # DictionaryAndSnippetManager
│   ├── lune-persist/           # PersistenceLayer
│   └── lune-app/               # Tauri app shell + UI bridge
└── specs/
```

---

## Section Checklist

### Architecture Decisions
- [ ] Component boundaries documented and agreed
- [ ] `AsrBackend` trait interface finalized (see §5.2)
- [ ] `TextPostProcessor` trait interface finalized (see §5.3)
- [ ] Crate layout created in Cargo workspace
- [ ] Tokio selected and documented as async runtime

### Technology Choices
- [ ] `cpal` evaluated for macOS (CoreAudio) and Windows (WASAPI) — no blockers
- [ ] Tauri version selected and evaluated for target OSes
- [ ] `llama-cpp-rs` version pinned; GPU backend flags configured
- [ ] `reqwest` selected as HTTP client with TLS support verified
- [ ] `tracing` + `tracing-subscriber` wired with JSON formatter

### Pluggability
- [ ] Inference backends fully decoupled via traits — no concrete types in `lune-core`
- [ ] Backend selection driven by `UserSettings` at runtime
- [ ] llama.cpp Voxtral support tracked; `VoxtralLocalBackend` designed for easy swap
