# 05 – External Interfaces

← [Data Design](./04-data-design.md) | → [Performance](./06-performance.md)

---

## 7.1 OpenAI-Compatible ASR API

### Endpoint

```
POST {base_url}/v1/audio/transcriptions
Authorization: Bearer {api_key}
Content-Type: multipart/form-data
```

### Request Fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | binary | Yes | Audio data (wav/mp3/flac/ogg) |
| `model` | string | Yes | e.g. `voxtral-mini-realtime-2602`, `whisper-1` |
| `prompt` | string | No | Hint for transcription (dictionary terms) |
| `language` | string | No | BCP-47 tag; omit for auto-detect |
| `response_format` | string | No | `json` (default) or `verbose_json` |

### Response (JSON)

```json
{
  "text": "Transcribed text goes here.",
  "segments": [
    {
      "id": 0,
      "text": "Transcribed text",
      "start": 0.0,
      "end": 1.23
    }
  ]
}
```

### Configuration Requirements

- [ ] `base_url` fully configurable (default: `https://api.openai.com`)
- [ ] `model` field configurable per-backend
- [ ] `api_key` stored via `secrecy::Secret`; never logged
- [ ] Dictionary terms injected into `prompt` field (≤ 224 tokens)
- [ ] Audio chunked into ≤ 25 MB segments per OpenAI limit
- [ ] `response_format=verbose_json` used when timestamps are needed

### Checklist
- [ ] HTTP client sends correct `Authorization: Bearer` header
- [ ] multipart/form-data body construction tested
- [ ] `verbose_json` response parsed into `Vec<AsrSegment>`
- [ ] Plain `json` response (text only) parsed as single `AsrSegment { is_final: true }`
- [ ] Network errors → retry with exponential backoff (max 3 attempts, 2 s initial)
- [ ] Timeout: configurable (default 30 s per request)
- [ ] TLS certificate validation enabled; no `danger_accept_invalid_certs`

---

## 7.2 OpenAI-Compatible LLM API

### Endpoint

```
POST {base_url}/v1/chat/completions
Authorization: Bearer {api_key}
Content-Type: application/json
```

### Request Body

```json
{
  "model": "qwen3-0.6b",
  "messages": [
    { "role": "system", "content": "<style_system_prompt>" },
    { "role": "user",   "content": "<few_shot_example_1_input>" },
    { "role": "assistant", "content": "<few_shot_example_1_output>" },
    { "role": "user",   "content": "<raw_asr_text>" }
  ],
  "max_tokens": 512,
  "temperature": 0.2,
  "stream": false
}
```

### Response Extraction

```
choices[0].message.content  →  PostProcessOutput.cleaned_text
```

### Configuration Requirements

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_tokens` | 512 | Bounded to prevent runaway cost |
| `temperature` | 0.2 | Low for deterministic cleanup |
| `stream` | false (v0.1) | true in v0.2 for perceived latency |

### Checklist
- [ ] Request JSON serialization tested (no missing fields)
- [ ] `choices[0].message.content` extraction handles empty array gracefully
- [ ] `max_tokens` enforced on every call
- [ ] Temperature configurable per preset
- [ ] Streaming (`"stream": true`) response parsing implemented (v0.2)
- [ ] LLM failure → log warning + return raw ASR text (graceful degradation)
- [ ] Prompt injection guard: raw ASR text escaped before insertion into prompt
- [ ] Total prompt token count estimated before call; truncate if over limit

---

## 7.3 Local Voxtral Interface

Three implementation options, all hidden behind `AsrBackend` trait:

### Option A — In-Process Rust Library (Preferred)

```rust
// voxtral-mini-realtime-rs or equivalent
let session = VoxtralSession::new(model_path, config)?;
session.push_audio(&pcm_frames)?;
let segments: Vec<Segment> = session.poll_text()?;
session.close()?;
```

**Checklist:**
- [ ] Crate identified or implemented (document chosen crate + version)
- [ ] `VoxtralSession` is `Send`; audio pushing is non-blocking (queues internally)
- [ ] Sliding context window managed inside session (document window size)
- [ ] Segment output includes `is_final` flag
- [ ] Build feature flags: `feature = "cuda"`, `feature = "metal"`, `feature = "cpu"`
- [ ] GPU selection validated at startup with clear error if requested GPU unavailable

### Option B — Sidecar HTTP/WebSocket Server

```
Local vLLM / compatible server on 127.0.0.1:{port}
Protocol: WebSocket for streaming
```

**Checklist:**
- [ ] Server spawned as child process by `lune-core` on startup
- [ ] Port assigned dynamically; bound to `127.0.0.1` only
- [ ] Health-check endpoint polled before session start
- [ ] Server stdout/stderr forwarded to `tracing` log
- [ ] Server terminated on app exit (clean shutdown)
- [ ] Reconnect logic on unexpected server crash

### Option C — Custom Inference Backend

- Novel Rust inference backend built in-house.
- Must expose same `AsrBackend` trait interface.
- Architecture TBD; tracked as separate spec when scoped.

**Checklist:**
- [ ] Separate design document created before implementation starts
- [ ] Benchmarked against Option A for RTF parity

### Interface Selection Checklist
- [ ] Backend selection persisted in `AsrBackendConfig` variant
- [ ] Runtime switching of ASR backend supported (requires session restart)
- [ ] Error message clearly indicates which backend failed
- [ ] Fallback chain documented: Local → OpenAI-compatible (optional, requires user consent)
