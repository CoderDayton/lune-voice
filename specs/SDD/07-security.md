# 07 – Security & Privacy

← [Performance](./06-performance.md) | → [Observability](./08-observability.md)

---

## 9.1 Threat Model

| Threat | Attack Vector | Mitigation |
|--------|--------------|------------|
| API key exfiltration | Log files, crash dumps | `secrecy::Secret` wrapper; keys never logged or serialized plain |
| Raw audio persistence | App bug writes audio to disk | No audio written unless debug mode explicitly enabled |
| Remote data leakage | Cloud API configured without user awareness | Remote backends require explicit opt-in + settings acknowledgment |
| Prompt injection via ASR text | Adversarial audio → crafted LLM prompt | ASR text escaped before prompt interpolation; system prompt hardened |
| Local network eavesdropping | Sidecar server on non-loopback | Sidecar bound to `127.0.0.1` only; connection refused otherwise |
| Malicious model weights | GGUF loaded from untrusted path | Model path set only via settings UI; SHA-256 checksum verification (v0.2) |
| Settings file tampering | External process modifies `settings.toml` | Settings reloaded with validation; schema errors abort startup |

---

## 9.2 Privacy Model

### Data Handling Principles

1. **On-device default**: all audio and text processing stays local when local backends are configured.
2. **Explicit remote opt-in**: remote API usage requires user action in Settings; not a default.
3. **No permanent audio**: raw PCM never written to disk in production builds.
4. **Minimal transcript retention**: session history disabled by default; user enables if desired.
5. **No telemetry by default**: anonymized performance metrics opt-in only.

### Data Inventory

| Data Type | Stored? | Location | Retention |
|-----------|---------|----------|-----------|
| Raw audio (PCM) | No (default) | Memory only | Cleared on session end |
| Raw ASR text | No (default) | Memory only | Cleared on session end |
| Post-processed text | No (default) | Memory only | Cleared on session end |
| Session transcript | Optional | SQLite local | User-configurable (default: disabled) |
| API keys | Yes | Settings file | Until user deletes |
| Dictionary entries | Yes | SQLite local | Until user deletes |
| Snippets | Yes | TOML local | Until user deletes |
| App logs | Yes (debug) | Log file | Rotated; no audio content |

---

## 9.3 Secrets Management

- API keys stored in `settings.toml` as opaque strings; wrapped in `secrecy::Secret<String>` in memory.
- Keys are **never**:
  - Logged (even at `TRACE` level).
  - Included in crash reports or bug reports.
  - Transmitted to any endpoint other than the configured `base_url`.
- Key display in Settings UI: masked by default; "reveal" requires user action.

---

## 9.4 Network Security

| Rule | Details |
|------|---------|
| TLS required for remote APIs | `reqwest` with default TLS; `danger_accept_invalid_certs` never enabled |
| Local sidecar binding | `127.0.0.1:{port}` only; `0.0.0.0` disallowed |
| No CORS / cross-origin issues | Local-only; not applicable |
| Request headers | Only `Authorization`, `Content-Type`, `User-Agent`; no fingerprinting headers |

---

## 9.5 OS Permissions

| Permission | Platform | Purpose | User Prompt |
|-----------|----------|---------|-------------|
| Microphone access | macOS, Windows | Audio capture | On first dictation session |
| Accessibility access | macOS | Keyboard injection | On first "type anywhere" use |
| `SendInput` permission | Windows | Keyboard injection | No special prompt required |

---

## 9.6 Input Validation

- **Settings file**: parsed with `serde`; unknown fields logged at WARN, not silently ignored.
- **LLM prompt**: ASR text inserted as literal string, not interpolated as instructions.
- **Snippet body**: `{placeholder}` slots resolved from controlled values only (date, settings data).
- **Model paths**: validated as existing files before loading; symlink traversal noted.
- **API base URLs**: parsed via `url::Url`; scheme must be `https` (or `http` for `127.0.0.1`).

---

## Section Checklist

### Secrets
- [ ] `secrecy` crate added as dependency; API keys typed as `Secret<String>`
- [ ] `Debug` impl for settings struct redacts key fields
- [ ] Settings serialization: API keys written as-is to file (user-owned); redacted in logs
- [ ] No API key in any log line at any level (test with grep on log output)
- [ ] Settings UI masks key by default; "copy to clipboard" action clearly labeled

### Audio Privacy
- [ ] No audio write path exists in production build (`#[cfg(debug_assertions)]` gates debug dump)
- [ ] Session end clears audio buffer from memory (verified with sanitizer or test)
- [ ] Session history: disabled by default; opt-in requires explicit settings change
- [ ] "Delete all history" action implemented in Settings

### Network
- [ ] `reqwest` built with TLS; `danger_accept_invalid_certs` never called
- [ ] Sidecar server bind address validated to be `127.0.0.1`
- [ ] Remote backend only contacted after user configures it in Settings
- [ ] HTTP requests do not leak API key in URL query params (key only in `Authorization` header)

### Input Validation
- [ ] `url::Url::parse` used for base URLs; plain strings rejected
- [ ] Unknown TOML keys in settings file produce WARN log, not panic
- [ ] ASR text inserted into prompts with explicit role boundary (not injected into system prompt)
- [ ] Snippet placeholder resolution limited to allowlisted variables

### OS Permissions
- [ ] Microphone permission requested only when dictation starts (not on app launch)
- [ ] Accessibility permission check on macOS: prompt user if denied, degrade gracefully
- [ ] Permission denial handled gracefully (clear error message, not crash)
