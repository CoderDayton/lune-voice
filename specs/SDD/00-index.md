# Lune Voice – Software Design Document Index

> Cross-platform Rust voice dictation with local ASR + LLM post-processing.

## Document Map

| File | Section | Contents |
|------|---------|----------|
| [01-introduction.md](./01-introduction.md) | §1–3 | Purpose, scope, goals, non-goals, constraints |
| [02-architecture.md](./02-architecture.md) | §4 | Component diagram, technology choices |
| [03-components.md](./03-components.md) | §5 | Detailed component design (Audio, ASR, LLM, Session, Output, Commands, Dictionary, Persistence) |
| [04-data-design.md](./04-data-design.md) | §6 | Core structs, text processing pipeline |
| [05-external-interfaces.md](./05-external-interfaces.md) | §7 | OpenAI-compatible ASR/LLM APIs, local Voxtral interface |
| [06-performance.md](./06-performance.md) | §8 | Latency targets, memory/model size budgets |
| [07-security.md](./07-security.md) | §9 | Privacy model, data handling, network security |
| [08-observability.md](./08-observability.md) | §10 | Logging, telemetry, error recovery |
| [09-risks-roadmap.md](./09-risks-roadmap.md) | §11–12 | Open issues, risks, versioned roadmap |

---

## Master Implementation Checklist

Track overall progress here. Details live in each section file.

### Infrastructure
- [ ] Cargo workspace scaffolded with crate layout
- [ ] Tokio async runtime configured
- [ ] `tracing` subscriber wired (structured JSON logs)
- [ ] CI pipeline (format → lint → typecheck → test)

### Core Components
- [ ] `AudioCaptureService` — cpal stream, VAD, mpsc feed
- [ ] `AsrBackend` trait + session lifecycle
- [ ] `VoxtralLocalBackend` implementation
- [ ] `OpenAiCompatibleAsrBackend` implementation
- [ ] `TextPostProcessor` trait
- [ ] `LocalLlmBackend` (llama-cpp-rs)
- [ ] `OpenAiCompatibleLlmBackend`
- [ ] `SessionManager` orchestration
- [ ] `TextOutputController` (Notes view first, then OS injection)
- [ ] `VoiceCommandRouter`
- [ ] `DictionaryAndSnippetManager`
- [ ] `PersistenceLayer` (serde + toml/sqlite)

### UI
- [ ] Tauri app shell
- [ ] System tray / menu bar icon
- [ ] Global hotkey registration
- [ ] Notes view (internal long-form dictation)
- [ ] Settings panel

### Backends (external)
- [ ] OpenAI-compatible ASR HTTP client
- [ ] OpenAI-compatible LLM HTTP client
- [ ] Local Voxtral GGUF backend
- [ ] WebSocket streaming fallback

### Quality Gates
- [ ] End-to-end latency < 1 s on target hardware (RTF ≈ 0.42)
- [ ] Unit tests for all trait implementations
- [ ] Integration test: full dictation session round-trip
- [ ] Security review: no raw audio persisted by default
- [ ] Privacy review: remote API opt-in is explicit

---

## Versioned Milestones

| Milestone | Status | Key Deliverables |
|-----------|--------|-----------------|
| v0.1 MVP | `[ ] pending` | Local Voxtral + Qwen3-0.6B, Notes view, hotkey, dictionary |
| v0.2 | `[ ] pending` | llama-cpp-rs, OS text injection, voice commands |
| v1.0 | `[ ] pending` | Configurable backends, per-app style presets, telemetry |
