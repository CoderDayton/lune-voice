# Runbook

Common problems and how to fix them. Lune Voice is a single-user desktop app,
so recovery almost always means fixing local state.

Entries marked **(Linux)** or **(Windows)** apply only to that platform.
Everything else applies to both.

Everything below assumes the default paths:

| What | Linux | Windows |
|---|---|---|
| Settings | `~/.config/lune-voice/settings.toml` | `%APPDATA%\lune-voice\settings.toml` |
| Database | `~/.local/share/lune-voice/lune.db` | `%APPDATA%\lune-voice\lune.db` |
| Models | `~/.local/share/lune-voice/models/` | `%APPDATA%\lune-voice\models\` |
| GPU plugin | `~/.local/share/lune-voice/backends/` | not published for Windows |
| IME recovery record | `~/.local/share/lune-voice/displaced-input-engine` | n/a |

Whatever the problem, start by running with logs on:

```sh
RUST_LOG=info lune-voice          # or: RUST_LOG=info cgs dev
```

In PowerShell, run `$env:RUST_LOG="info"` on its own line first, then start
the app.

Transcript text is **redacted by default**. Set `LUNE_LOG_TRANSCRIPTS=1` only
when you are debugging recognition itself. A log captured that way contains
everything you dictated, word for word, so read it before you attach it to a
public issue.

---

## Typing is broken everywhere after a crash

**Symptom:** after Lune Voice crashed or was killed, you can't type in any app,
or your input method is stuck.

**Cause:** to insert text through IBus, the app briefly makes `lune-voice` the
active input engine. If it dies during that moment, your own engine is never
put back.

**Recovery:** start Lune Voice again. It saves your engine to
`displaced-input-engine` before switching, and restores it on the next start.

If that doesn't work (the file is missing, or names an engine that no longer
exists), set your engine back by hand:

```sh
ibus engine                       # what is active now
ibus list-engine | head           # what is available
ibus engine xkb:us::eng           # set it back
```

Then delete the old record so it isn't applied again:

```sh
rm -f ~/.local/share/lune-voice/displaced-input-engine
```

**To avoid it entirely:** set `LUNE_DISABLE_IBUS=1`. The app then always types
through uinput, which never touches your input method.

---

## Dictation produces no text

Work down this list:

1. **No model.** Check `~/.local/share/lune-voice/models/` has the model named
   in settings. Download it again from Settings → Models. A stuck download
   gives up on its own: 30 s to connect, 30 s to respond, one hour for the
   file.
2. **No audio from the mic.** Run with `RUST_LOG=info`, look for capture logs,
   and check the input device in Settings.
   - The level meter shows the signal *after* automatic gain, so a flat meter
     means no audio is arriving at all.
   - The picker shows one entry per name. A mic that never shows up either
     isn't a capture device or has the same name as one already listed.
   - With a Pulse server (PulseAudio, or PipeWire's `pipewire-pulse`), the
     names are the server's source names, like `Shure MV6 Mono`. Without one,
     they are ALSA card names.
   - `audio init: … temporarily busy` means no Pulse server answered, so
     capture fell back to raw ALSA while another program held the card.
     `opening capture stream host=Alsa` at INFO confirms it. Check
     `systemctl --user status pipewire-pulse` (or `pulseaudio`).
   - `host=PulseAudio` plus a WARN ending in `listing every input cpal reports
     instead` means the server answered cpal but not the picker's own query.
     The picker then lists every input cpal sees, including sink monitors.
3. **Everything thrown out as non-speech.** Look for `dropping non-speech
   segment` or `dropping garbage segment` at WARN. This is usually a mic gain
   problem.
4. **Backend not compiled in.** An error containing `not compiled in` means the
   binary was built without `whisper-cpp`. Rebuild with `cgs build`.
5. **Remote transcription with remote services off.** An error saying
   `remote services are turned off` means transcription is set to a remote
   service but `privacy_mode = "LocalOnly"`. Switch transcription back to
   Whisper, or turn on **Allow remote services** in Settings → General.

---

## Transcript is recognized but never appears (Linux)

**Symptom:** the session finishes and history shows the text, but nothing was
inserted, and the app says the dictation wasn't delivered.

**Cause:** the clipboard isn't available (`wl-copy` or `xclip` is missing or
failing) and the text has more than one line. Typing it key by key would press
Enter in whatever has focus, so the app doesn't send it at all.

**Recovery:** install `wl-clipboard` (Wayland) or `xclip` (X11). Nothing is
lost: copy the transcript from history in the meantime.

---

## The pill disappears after changing a pill setting (Windows)

**Symptom:** the overlay vanishes right after you change its size, orientation,
corner or anchor, and doesn't come back on its own.

**Cause:** on Windows the pill is rebuilt, not resized, when its geometry
changes. The old window still held its label 5 seconds after it was destroyed,
so the new window couldn't take it.

**Diagnosis:** `pill window label still registered 5s after destroy` at `error`
level.

**Recovery:** change any pill setting again, or restart the app. Dictation
isn't affected, and the hotkey still works while the overlay is gone.

---

## Transcript lands in the pill instead of the app (Windows)

**Symptom:** dictation started by clicking the pill delivers nothing to the
target app. History has the text.

**Cause:** the overlay took focus, so the paste keystroke went to it. The pill
is created with `WS_EX_NOACTIVATE` (`pill.rs::deny_activation`) so it can't be
activated, and that is the only guard for this case. The focus restore at
delivery (`focus/windows.rs::restore_focus`) only handles focus that moves
*after* dictation starts. If the pill already had focus when the session began,
there is no earlier window to go back to. Seeing this means WebView2 called
`SetFocus` on itself before the session started.

**Diagnosis:** `RUST_LOG=info` and look for `could not restore focus to the
window the dictation started in`, or `foreground window belongs to this
process` at capture time.

**Recovery:** start dictation with the hotkey instead of the pill. The hotkey
never moves focus. The transcript is in history either way.

---

## Styling makes dictation slow or hangs

Styling is a call to your LLM endpoint, so it is only as fast as that endpoint.

- The timeout is `pipeline.llm_timeout_secs` (default 30, max 300). If it runs
  out, the unstyled transcript is used. You get plain text, not an error.
- The call can't wait forever. `0`, or anything above 300, is set to 300 when
  the pipeline is built, and the change is logged.
- To rule styling out, set the style to "off" in Settings. That skips the LLM
  call entirely.

---

## Remote ASR / styling endpoint failing

- ASR and styling share one retry policy (`lune_core::retry`): transport
  errors, 5xx and 429 get three attempts with jittered exponential backoff,
  honouring `Retry-After` up to 30 s. A 4xx fails right away, since it means a
  config or credential problem, not a hiccup.
- Response bodies are capped (ASR 8 MiB, styling 1 MiB). A bigger response is
  rejected, not buffered.
- `ASR API error 401` → the key is wrong or missing. Re-enter it in Settings.
- The base URL must be HTTPS unless it is localhost. Plain HTTP to a remote
  host is rejected up front.

---

## API key problems

Keys are stored in the OS keyring when there is one. Otherwise they stay in
`settings.toml`.

- On a headless machine or a locked session, the keyring isn't available, so
  the key stays in `settings.toml`. That file is written with mode `0600`, and
  the app resets it to `0600` every time it starts.
- **A key in `settings.toml` is a real secret.** Treat that file as sensitive
  and never paste it into an issue.
- To clear a key: blank it in Settings, or
  `secret-tool clear service lune-voice account llm_api_key`.

---

## GPU plugin will not install

- **"no published GPU backend for this platform yet":** the plugin for your
  platform hasn't been released (its SHA-256 in
  `crates/lune-persist/src/gpu_plugin.rs` is still `UNPUBLISHED`). There is
  nothing to fix on your end. You can build one yourself with `cgs gpu-plugin`.
- **"failed verification":** the download didn't match the pinned digest, so it
  was treated as tampered with and deleted. Retry. If it keeps happening, don't
  work around it.
- **"exceeded the N bytes expected":** the server sent more than the release
  size plus 10%. Same advice: don't work around it.
- A stalled download gives up after 30 s to connect or 60 s without data.

**Reverting to CPU:** delete the downloaded plugin and restart.

```sh
rm -rf ~/.local/share/lune-voice/backends/
```

---

## Database is corrupt

A corrupt database is reported at startup instead of crashing the app. It holds
history and custom styles, **not** settings, models, or dictionary defaults, so
throwing it away loses history but not your configuration.

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

When the disk is full, writes fail with an error instead of leaving broken
files. Settings are written to a temp file and renamed into place, and the GPU
plugin stages every file as `.lune-part` first. The usual space hogs are
`models/` (hundreds of MB each) and a half-finished GPU plugin download
(~500 MB).

```sh
du -sh ~/.local/share/lune-voice/*
rm -f ~/.local/share/lune-voice/backends/.gpu-plugin.tar.gz.part
```

---

## Window has no icon (tray icon is fine)

This is expected when you run a binary straight from `target/`. The desktop
shows the icon of the *application* it matched the window to, and that match
goes through an installed desktop entry. The tray gets its icon as raw bytes
over DBus, so it isn't affected.

Tauri does set an icon: it applies `default_window_icon` to any window built
without one, and on X11 that becomes `_NET_WM_ICON`. On GNOME the app
runs under XWayland (`GDK_BACKEND=x11`, forced in `lib.rs` so the pill can
position itself), so the window does carry an icon; GNOME just prefers the
matched app's icon in the dash and overview. On a native-Wayland session
(COSMIC, wlroots) there is no `_NET_WM_ICON` at all and the desktop entry is
the only source.

An installed `.deb` is fine: it ships `LuneVoice.desktop` carrying
`StartupWMClass=lune-voice`, which is what gnome-shell matches X11 windows
against. For a locally built binary, install a user-level entry:

```sh
cgs desktop-entry                      # points at target/release
cgs desktop-entry --env PROFILE=debug  # for `cgs dev` builds
cgs desktop-entry --env REMOVE=1       # uninstall
```

`StartupWMClass` and the entry's basename both have to be `lune-voice`: GTK
derives WM_CLASS (and, on a Wayland session, `app_id`) from `g_get_prgname()`,
the binary name, because Tauri only sets a GTK application id when
`app.enableGtkAppId` is true and it is not. Turning that on would change the
identity to the bundle identifier and break the match the bundled desktop entry
relies on.

---

## Escalating a bug

Include the `RUST_LOG=info` log (redacted by default, but check it anyway),
your OS and desktop session (`echo $XDG_SESSION_TYPE $XDG_CURRENT_DESKTOP`),
whether the build is CPU or CUDA, and the model you use. If you include
`settings.toml`, remove every `api_key` line first.
