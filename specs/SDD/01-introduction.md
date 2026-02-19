# 01 – Introduction, Scope & Goals

← [Index](./00-index.md) | → [Architecture](./02-architecture.md)

---

## 1. Purpose

**Lune Voice** is a cross-platform Rust application providing WisprFlow-style voice dictation using:

- **Local, low-latency ASR** — Mistral Voxtral Mini Realtime (GGUF) or OpenAI-compatible endpoints.
- **LLM text post-processing** — grammar cleanup, filler removal, style transformation.

---

## 2. Scope

### In Scope (v1)

- [ ] Real-time streaming dictation on **macOS** and **Windows** (Linux optional).
- [ ] Primary ASR: **Voxtral Mini 4B Realtime** via local GGUF inference or compatible server.
- [ ] Optional ASR via **OpenAI-compatible `/v1/audio/transcriptions`** (local or remote).
- [ ] Small LLM backend (local or OpenAI-compatible) for:
  - [ ] Grammar and punctuation correction.
  - [ ] Disfluency removal ("um", "uh", false starts).
  - [ ] Style transforms ("concise email", "meeting notes", "friendly Slack").
- [ ] Personal dictionary (user-specific terms, names, jargon).
- [ ] Snippet library (reusable text blocks via spoken cues).
- [ ] Voice command routing (start/stop, formatting, style switching).

### Out of Scope (v1)

- Full OS/app voice navigation.
- Multi-user diarization or speaker identification.
- Complex project management or multi-document editing.
- Native mobile applications.

---

## 3. Goals, Non-Goals & Constraints

### 3.1 Primary Goals

| Goal | Acceptance Criteria |
|------|-------------------|
| Low-latency streaming dictation | End-to-end < 1 s; RTF ≈ 0.42 on target hardware |
| High-quality output | LLM post-processing removes disfluencies and fixes grammar reliably |
| Privacy-preserving | Full on-device path possible; no audio leaves device unless user opts in |
| Configurable backends | Hot-swap ASR/LLM between local GGUF and OpenAI-compatible remotes |
| Simple UX | Single global hotkey, minimal chrome, clear status feedback |

### 3.2 Constraints & Assumptions

- [ ] Primary local ASR: **Voxtral Mini 4B Realtime** via:
  - Custom Rust inference with GGUF/safetensors weights, **or**
  - Local vLLM server exposing OpenAI-style WebSocket/HTTP API.
- [ ] `llama.cpp` upstream Voxtral support not yet merged → design must keep inference layer **pluggable** (trait-based).
- [ ] GPU acceleration preferred; CPU-only must remain usable with quantized models.
- [ ] Target hardware: modern consumer laptop/desktop.

### 3.3 Definitions

| Term | Definition |
|------|-----------|
| ASR | Automatic Speech Recognition |
| LLM | Large Language Model used as text post-processor |
| Voxtral Realtime | Mistral's streaming ASR model family for low-latency on-device transcription |
| GGUF | Quantized model format used by llama.cpp-style runtimes |
| RTF | Real-Time Factor: ratio of inference time to audio duration |
| VAD | Voice Activity Detection |
| PTT | Push-To-Talk |

---

## Section Checklist

- [ ] Scope boundaries documented and agreed with stakeholders
- [ ] All acronyms defined in §1.3
- [ ] Target platforms confirmed (macOS, Windows; Linux stretch)
- [ ] v1 out-of-scope items acknowledged in issue tracker
