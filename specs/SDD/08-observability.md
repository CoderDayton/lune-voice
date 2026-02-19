# 08 – Error Handling, Telemetry & Observability

← [Security](./07-security.md) | → [Risks & Roadmap](./09-risks-roadmap.md)

---

## 10.1 Error Handling Philosophy

- **Surface, don't swallow**: all errors must be logged and either recovered from or surfaced to the user.
- **Graceful degradation**: backend failure → fall back to simpler mode, not crash.
- **No silent failures**: every `Result::Err` must be handled (no `let _ = ...` without comment).
- **User-visible errors**: non-intrusive toast notifications for actionable failures.

---

## 10.2 Error Categories & Recovery

| Error Category | Examples | Recovery Strategy |
|---------------|---------|-----------------|
| Audio device | Mic unplugged, permission denied | Toast + status indicator; retry on device reconnect |
| ASR backend — transient | Network timeout, 5xx | Exponential backoff (max 3 retries, 2 s initial); then surface error |
| ASR backend — fatal | Model load failure, invalid weights | Prompt user to check settings; disable backend |
| LLM backend — transient | Timeout, 5xx | Retry once; then return raw ASR text with warning |
| LLM backend — fatal | Model file missing, llama-cpp panic | Log + return raw ASR text; notify user |
| WebSocket disconnect | Connection dropped mid-session | Attempt reconnect; fallback to HTTP polling mode |
| Settings parse error | Invalid TOML, missing required field | Log WARN for unknown fields; abort startup for invalid required fields with clear message |
| Persistence | SQLite locked, write failure | Log ERROR; notify user; continue with in-memory data for session |
| OS text injection | Accessibility denied | Notify user; degrade to Notes view |

---

## 10.3 Structured Logging

### Setup

```rust
use tracing_subscriber::{
    fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};

tracing_subscriber::registry()
    .with(EnvFilter::from_env("LUNE_LOG"))   // default: "info"
    .with(fmt::layer().json())
    .init();
```

### Log Levels

| Level | Usage |
|-------|-------|
| `ERROR` | Unrecoverable failures requiring user attention |
| `WARN` | Degraded operation; recovered, but worth noting |
| `INFO` | Session start/stop, backend selection, style changes |
| `DEBUG` | Per-segment ASR output, LLM request/response shapes (no content) |
| `TRACE` | Audio frame timing, raw segment text (only in dev builds) |

### Log Categories (spans)

```rust
tracing::info_span!("audio",   session_id = %id);
tracing::info_span!("asr",     session_id = %id, backend = %kind);
tracing::info_span!("llm",     session_id = %id, style   = %style);
tracing::info_span!("output",  session_id = %id, sink    = %sink);
tracing::info_span!("command", session_id = %id);
tracing::info_span!("ui",);
tracing::info_span!("backend", name = %name);
```

### Sensitive Data Rules

- **Never log**: API keys, raw audio content, full ASR/LLM text at INFO or above.
- `DEBUG` may log segment shapes and token counts, **not** content.
- `TRACE` may log content only in `#[cfg(debug_assertions)]` builds.

---

## 10.4 User-Facing Status Indicators

| State | Indicator | Location |
|-------|-----------|---------|
| Idle | Grey mic icon | System tray |
| Listening | Animated green mic icon | System tray |
| Processing (LLM) | Spinner / pulsing icon | System tray |
| Error | Red icon + toast | System tray + desktop notification |
| Backend offline | Yellow warning badge | System tray |

### Toast Notification Guidelines

- **Content**: action-oriented ("Microphone permission denied — open System Preferences").
- **Duration**: auto-dismiss after 5 s for warnings; persistent for errors until user dismisses.
- **Frequency**: deduplicate identical errors within a 30 s window.

---

## 10.5 Performance Telemetry (Optional, Opt-In)

Anonymized local metrics for user's own debugging:

| Metric | Collection Point | Purpose |
|--------|-----------------|---------|
| ASR RTF | Per-session | Identify slow hardware / backend |
| LLM latency (ms) | Per-transform | Identify prompt / model issues |
| End-to-end latency (ms) | Per-session | Overall health |
| Session duration | On stop | Usage pattern |
| Backend type | On start | Track which backend is used |

Metrics stored locally only (no network upload). Viewable in Settings → Diagnostics.

---

## 10.6 Panic Handling

- `std::panic::set_hook` installed at startup to log panic info via `tracing::error!`.
- Panic message must not contain user audio/text content.
- On panic: attempt graceful session teardown before process exit.
- In release builds: `panic = "abort"` in `Cargo.toml` (fast, avoids partial state).

---

## Section Checklist

### Error Handling
- [ ] All `Result` returns handled; `unwrap()` / `expect()` banned outside tests (enforced via `clippy::unwrap_used`)
- [ ] Recovery strategy documented for each error category above
- [ ] Exponential backoff implemented for transient network errors (ASR + LLM)
- [ ] LLM failure path returns raw ASR text and logs WARN — confirmed via test
- [ ] WebSocket reconnect logic implemented with max-retry cap

### Logging
- [ ] `tracing` + `tracing-subscriber` wired at app startup
- [ ] `LUNE_LOG` env var respected for log level override
- [ ] JSON log format enabled in production; pretty format in dev (`debug_assertions`)
- [ ] All spans carry `session_id` for per-session correlation
- [ ] No API keys appear in any log line (grep test in CI)
- [ ] No user audio/text content at INFO level or above

### User Feedback
- [ ] System tray icon state reflects: idle / listening / processing / error
- [ ] Toast notifications implemented for all error categories
- [ ] Toast deduplication (30 s window) implemented
- [ ] Persistent error toasts dismissible by user

### Telemetry
- [ ] Opt-in telemetry: disabled by default (`UserSettings.telemetry_enabled = false`)
- [ ] Metrics stored locally; no network upload path exists unless explicitly built
- [ ] Diagnostics view in Settings shows last N session metrics

### Panics
- [ ] `set_hook` installed at startup; logs to `tracing::error!`
- [ ] `panic = "abort"` set in `[profile.release]`
- [ ] Integration test: forced panic path produces log output (dev build only)
