# 09 – Open Issues, Risks & Roadmap

← [Observability](./08-observability.md) | → [Index](./00-index.md)

---

## 11. Open Issues & Risks

### 11.1 Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Status |
|----|------|-----------|--------|------------|--------|
| R-01 | `llama.cpp` lacks upstream Voxtral support | High | High | Use external Rust impl or vLLM sidecar; `AsrBackend` trait makes swap straightforward | Open |
| R-02 | GPU variability across consumer hardware | High | Medium | Conservative defaults; advanced tuning options exposed in Settings | Open |
| R-03 | Cross-platform text injection blocked by OS security policy | Medium | High | Degrade to Notes view; document workarounds per OS | Open |
| R-04 | LLM latency too high for real-time feel | Medium | High | Batch on sentences not tokens; offer "TypeOnStop" mode; smaller models | Open |
| R-05 | Voxtral GGUF weights not publicly available or licensed | Medium | High | Track Mistral licensing; fallback to OpenAI-compatible ASR backend | Open |
| R-06 | cpal stream instability on some Windows audio drivers | Low | Medium | Test with WASAPI exclusive + shared modes; expose fallback to ASIO | Open |
| R-07 | Tauri version incompatibility with target OS versions | Low | Medium | Pin Tauri version; test on minimum supported OS versions | Open |
| R-08 | Accessibility permission prompts confuse users | Medium | Low | Clear first-run onboarding explaining why permission is needed | Open |

---

### 11.2 Open Technical Questions

| ID | Question | Owner | Resolution Target |
|----|---------|-------|-----------------|
| Q-01 | Which specific Rust Voxtral implementation to use (or build)? | — | Before v0.1 start |
| Q-02 | Exact frame size expected by Voxtral encoder | — | During VoxtralLocalBackend implementation |
| Q-03 | Sliding context window size for Voxtral streaming | — | During VoxtralLocalBackend implementation |
| Q-04 | Tauri version: v1 vs v2? | — | Before app scaffold |
| Q-05 | SQLite vs pure TOML for dictionary storage? | — | Before persistence implementation |
| Q-06 | Hotkey library: `global-hotkey` crate vs Tauri built-in? | — | Before v0.1 hotkey feature |
| Q-07 | VAD algorithm: webrtc-vad FFI vs pure Rust impl? | — | Before audio capture implementation |

---

## 12. Versioned Roadmap

### v0.1 — MVP

**Goal:** Prove the local Voxtral + LLM pipeline end-to-end. Internal use only.

#### Features
- [ ] Cargo workspace created with all crates stubbed
- [ ] `AudioCaptureService` with cpal, frame chunking, basic VAD
- [ ] `VoxtralLocalBackend` streaming transcription
- [ ] `LocalLlmBackend` using `unsloth/Qwen3-0.6B-GGUF`
- [ ] `SessionManager` orchestrating the full pipeline
- [ ] Notes view (Tauri WebView) as sole text output sink
- [ ] Single global hotkey: toggle dictation on/off
- [ ] Minimal system tray: listening / idle indicator
- [ ] `StylePreset`: General only (grammar fix + disfluency removal)
- [ ] `DictionaryAndSnippetManager`: basic CRUD, applied to ASR output
- [ ] `PersistenceLayer`: settings.toml + snippets.toml
- [ ] Structured logging via `tracing`
- [ ] Manual style selection in Settings panel

#### Quality Gates
- [ ] End-to-end round-trip test passes (audio file → notes view text)
- [ ] RTF measured and documented
- [ ] No API keys in logs (CI grep check)
- [ ] Runs on macOS (primary dev platform)

---

### v0.2 — Type Anywhere

**Goal:** OS text field injection; voice commands; richer LLM presets.

#### Features
- [ ] `TextOutputController`: macOS keyboard injection via Accessibility API
- [ ] `TextOutputController`: Windows `SendInput` keyboard injection
- [ ] `VoiceCommandRouter`: built-in command grammar implemented
- [ ] Formatting commands: "new line", "new paragraph", "list item"
- [ ] Style switch commands: "switch to formal", "make this a Slack message"
- [ ] `LocalLlmBackend`: integrated via `llama-cpp-rs` (replaces direct binary call)
- [ ] LLM streaming responses (`stream: true`) for perceived latency reduction
- [ ] Additional `StylePreset` library: Email, Slack, Meeting Notes, Code Comments
- [ ] "Scratch that" voice command (delete last segment)
- [ ] Per-app style preset auto-selection (rule-based, by active app name)
- [ ] Improved stabilization: adaptive window based on speech rate

#### Quality Gates
- [ ] Injection works in Safari, Chrome, VS Code, Terminal (macOS)
- [ ] Injection works in Notepad, VS Code, browsers (Windows)
- [ ] Voice commands tested for all built-in phrases
- [ ] End-to-end latency ≤ 1 s on recommended hardware (documented)

---

### v1.0 — General Release

**Goal:** Polished, configurable, privacy-respecting release.

#### Features
- [ ] `OpenAiCompatibleAsrBackend`: full HTTP + WebSocket implementation
- [ ] `OpenAiCompatibleLlmBackend`: full implementation
- [ ] Backend switching in Settings (no restart required)
- [ ] Per-app style presets (user-configurable rules)
- [ ] Snippet library with placeholder support
- [ ] Session history (opt-in, configurable retention)
- [ ] Performance diagnostics view in Settings
- [ ] Model checksum verification (SHA-256) for local GGUF weights
- [ ] Onboarding flow: first-run wizard for mic + accessibility permissions
- [ ] Auto-update mechanism (Tauri updater)
- [ ] Linux support (Wayland `ydotool` + X11 `xdotool`)
- [ ] Privacy mode: `LocalOnly` enforcement (reject remote backend config)
- [ ] Telemetry opt-in with local-only metrics storage

#### Quality Gates
- [ ] All v0.1 and v0.2 quality gates still passing
- [ ] Accessibility audit: tray icon + settings panel keyboard-navigable
- [ ] Security review: threat model (§07) checklist fully addressed
- [ ] Performance regression suite passing (CI)
- [ ] 30-day soak test on macOS and Windows without crash or data loss
- [ ] CHANGELOG.md current; user-facing docs complete

---

## Section Checklist

### Risk Management
- [ ] Risk register reviewed at each milestone kickoff
- [ ] R-01 (Voxtral GGUF availability): Mistral license confirmed before v0.1 ships
- [ ] R-03 (text injection blocking): tested on macOS 14+ with SIP enabled
- [ ] R-04 (LLM latency): latency budget measured before v0.2; TypeOnStop added if needed
- [ ] R-05 (model licensing): legal review of GGUF weight redistribution policy

### Open Questions
- [ ] Q-01 resolved before v0.1 implementation starts
- [ ] Q-02, Q-03 resolved during VoxtralLocalBackend spike
- [ ] Q-04 resolved before Tauri scaffold committed
- [ ] All open questions tracked in issue tracker with assigned owner

### Roadmap
- [ ] v0.1 scope locked (no scope creep without explicit decision)
- [ ] v0.2 scope reviewed after v0.1 quality gates pass
- [ ] v1.0 scope reviewed after v0.2 quality gates pass
- [ ] Each milestone has a designated QA checklist sign-off before release
