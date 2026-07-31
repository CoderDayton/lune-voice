# Runbook

Common failure modes and how to get out of them. Lune Voice is a single-user
desktop app, so "recovery" almost always means restoring local state, not
failing over.

Everything below assumes the default paths:

| What | Where |
|---|---|
| Settings | `~/.config/lune-voice/settings.toml` |
| Database | `~/.local/share/lune-voice/lune.db` |
| Models | `~/.local/share/lune-voice/models/` |
| GPU plugin | `~/.local/share/lune-voice/backends/` |
| IME recovery record | `~/.local/share/lune-voice/displaced-input-engine` |

First diagnostic step for anything: run with logs on.

```sh
RUST_LOG=info lune-voice          # or: RUST_LOG=info cgs dev
```

Transcript text is **redacted by default**. To include it while debugging a
recognition problem — and only then — set `LUNE_LOG_TRANSCRIPTS=1`. Never attach
a log captured that way to a public issue without reading it first: it contains
verbatim everything you dictated.

---

## Typing is broken everywhere after a crash

**Symptom:** after Lune Voice crashed or was killed, you cannot type in any
application, or your input method is stuck.

**Cause:** IBus delivery briefly makes `lune-voice` the global input engine to
commit text. If the process dies inside that window, the engine is never
restored.

**Recovery:** relaunch Lune Voice. It records the displaced engine to
`displaced-input-engine` before switching and restores it on the next start.

If it does not recover — the record is missing or names an engine that no longer
exists — set your engine back manually:

```sh
ibus engine                       # what is active now
ibus list-engine | head           # what is available
ibus engine xkb:us::eng           # set it back
```

Then remove the stale record so it is not replayed:

```sh
rm -f ~/.local/share/lune-voice/displaced-input-engine
```

**Avoiding it entirely:** `LUNE_DISABLE_IBUS=1` pins delivery to the uinput
path, which never touches your input method.

---

## Dictation produces no text

Work down this list:

1. **No model.** Check `~/.local/share/lune-voice/models/` has the model named
   in settings. Re-download from Settings → Models.
2. **Mic not captured.** `RUST_LOG=info` and look for capture logs; check the
   input device in Settings. The level meter reads *post-AGC*, so a silent meter
   means no audio is arriving at all.
3. **Everything dropped as non-speech.** Look for `dropping non-speech segment` /
   `dropping garbage segment` at WARN. Usually a mic gain problem.
4. **Backend not compiled in.** An error containing `not compiled in` means the
   binary was built without `whisper-cpp`. Rebuild with `cgs build`.

---

## Styling makes dictation slow or hangs

Styling is a remote call, so it inherits your endpoint's latency.

- Timeout is `pipeline.llm_timeout_secs` (default 30, max 300). On timeout the
  raw transcript is used — dictation degrades, it does not fail.
- **`llm_timeout_secs = 0` disables the timeout entirely.** A hung endpoint then
  hangs the pipeline with no bound. Do not set it to 0 unless you are debugging.
- To rule styling out completely, set the style to "off" (no styling) in
  Settings. That skips the LLM call.

---

## Remote ASR / styling endpoint failing

- ASR retries transient failures (5xx, 429) three times with jittered
  exponential backoff, honouring `Retry-After` up to 30 s. A 4xx fails
  immediately — that is a configuration or credential problem, not a blip.
- `ASR API error 401` → the key is wrong or missing. Re-enter it in Settings.
- The base URL must be HTTPS unless it is loopback; plaintext to a remote host
  is refused at construction.

---

## API key problems

Keys live in the OS keyring, not in `settings.toml`.

- Headless, or a locked session? The keyring is unavailable, and the key stays
  inline in `settings.toml` as a documented fallback. That file is written
  `0600` and re-hardened on every boot.
- **A key in `settings.toml` is a real secret.** Treat that file as sensitive
  and never paste it into an issue.
- To clear a key: blank it in Settings, or
  `secret-tool clear service lune-voice account llm_api_key`.

---

## GPU plugin will not install

- **"no published GPU backend for this platform yet"** — the release is
  unpublished (its SHA-256 is still the `UNPUBLISHED` sentinel). Nothing to fix
  locally; the plugin has not shipped.
- **"failed verification"** — the download did not match the pinned digest.
  Treated as hostile and discarded. Retry; if it persists, do not work around it.
- **"exceeded the N bytes expected"** — the server sent more than the release
  size plus 10%. Same conclusion: do not work around it.
- Download stalls are bounded by a 30 s connect and 60 s read timeout.

**Reverting to CPU:** delete the downloaded plugin and restart.

```sh
rm -rf ~/.local/share/lune-voice/backends/
```

---

## Database is corrupt

A corrupt DB is reported at startup rather than panicking. The database holds
history and custom styles — **not** settings, models, or dictionary defaults —
so discarding it loses history, not configuration.

```sh
# Inspect first.
sqlite3 ~/.local/share/lune-voice/lune.db "PRAGMA integrity_check;"

# Salvage what is readable.
sqlite3 ~/.local/share/lune-voice/lune.db ".recover" | \
  sqlite3 ~/.local/share/lune-voice/lune-recovered.db

# Last resort: move it aside; a fresh one is created on next start.
mv ~/.local/share/lune-voice/lune.db ~/.local/share/lune-voice/lune.db.broken
```

---

## Disk full

Writes fail loudly rather than corrupting state: settings use write-to-temp +
rename, and the GPU plugin stages every file under `.lune-part` before
publishing. The usual culprit is `models/` (hundreds of MB each) or a partial
GPU plugin download (~500 MB).

```sh
du -sh ~/.local/share/lune-voice/*
rm -f ~/.local/share/lune-voice/backends/.gpu-plugin.tar.gz.part
```

---

## Escalating a bug

Include: the `RUST_LOG=info` log (redacted by default — check it anyway), your
OS and desktop session (`echo $XDG_SESSION_TYPE $XDG_CURRENT_DESKTOP`), whether
the build is CPU or CUDA, and the model in use. Do not include `settings.toml`
without first removing any `api_key` line.
