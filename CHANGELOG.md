# Changelog

All notable changes to Lune Voice are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

First release, for Linux and Windows.

### Dictation

- Hotkey-bounded dictation, toggle or push-to-talk hold, `Ctrl+Space` by default, with a separate cancel key. The hotkey is global and the text lands in whatever window has focus
- Two on-device speech engines:
  - **Whisper** (whisper.cpp) in four sizes: Base (82 MB), Small (190 MB), Medium (539 MB, the default) and Large v3 Turbo (874 MB)
  - **Parakeet** (NVIDIA Parakeet TDT 0.6B v3) in 669 MB (q8_0, recommended) or 416 MB (q4_k). Many times faster than Whisper on a CPU, and punctuates and capitalizes by itself. It cannot report which language it heard, so it has no Auto-detect: the selector lists only its 25 European languages and switching to it picks your system language (or English)
- Whisper offers Auto-detect plus 28 languages (the same 25 European ones, and Japanese, Korean and Chinese), each formatted by its own rules
- Remote transcription through any OpenAI-compatible endpoint, as an alternative to the local engines
- Silero endpoint trim before decode: silence is cut off both ends and internal pauses are kept, because whisper reads them as punctuation. Dictations longer than 28 s are decoded in pieces split at pauses
- PCEN automatic gain control on the capture path, so a quiet microphone still reaches the recognizer at a usable level. Toggle it in Settings → Dictation
- Correction pass over the transcript:
  - your dictionary, matched exactly, plus a one-letter near miss on longer words the recognizer was confident about
  - about 2,170 bundled brand, product and tech names re-cased exactly (**Capitalize known names**, on by default; a style can override it)
  - a **Correction sensitivity** slider for how unsure a word must be before it is flagged
- Formatting pass: capitalization, terminal punctuation, English filler-word removal, spoken key chords ("control shift p" → `Ctrl+Shift+P`), and the quotes whisper wraps around a read-aloud utterance
- Optional styling through any OpenAI-compatible endpoint, local or hosted: five built-in presets (casual, professional, formal, technical, writing), or your own with custom prompts and generated few-shot examples. Spaced digit groups that are not thousands separators, such as phone numbers, are left intact in styled text
- Voice commands at the start of an utterance: "switch to <preset>" changes the style, "insert <snippet>" types a snippet. There are no fixed phrases, so words like "new line" are always dictated as text
- Snippets: saved text delivered verbatim by voice, with a recorder for the trigger phrase
- The loaded model can be freed after an idle period while the microphone stays warm, so the next dictation still starts immediately. Off by default; **Free the model when idle** in Settings → Transcription
- Microphone picker, falling back to the system default when the chosen device is gone

### Text delivery

- Linux: IBus when it is the active input method, otherwise a `uinput` virtual keyboard, with a clipboard paste path for non-ASCII. Multi-line text is never typed key by key, so newlines cannot press Enter in the focused app
- Windows: per-key typing by default, clipboard paste as the alternative. Line breaks and tabs are flattened when delivering into a terminal
- Delivery targets the window that was focused when you started speaking
- **Copy to clipboard** (off by default) keeps the dictation on the clipboard. With it off, the previous clipboard is restored: text or images on Linux, text on Windows
- Paste compatibility: Automatic, Always `Ctrl+V` or Always `Ctrl+Shift+V`
- The temporary clipboard copy used for a paste carries the clipboard-history exclusion hint, so it stays out of clipboard managers

### Platforms

- **Linux** (X11 or Wayland): evdev hotkey grab and focused-window detection on Hyprland, Sway, `wlr-foreign-toplevel` compositors, COSMIC, KDE and GNOME, with xdotool on X11. On compositors without `wlr-layer-shell`, such as GNOME, the app runs under XWayland. Ships as `.deb`, `.rpm`, `.AppImage` and a portable `.tar.gz`. The `.deb` and `.rpm` install the udev rule for `/dev/uinput` and keyboard access, so packaged installs never ask for a password
- **Windows** 10/11: low-level keyboard hook and an NSIS installer. If one of the app's own windows took focus, focus goes back to the target window before delivery
- In-app updates for the `.deb`, `.rpm`, `.AppImage` and Windows installer. The portable `.tar.gz` announces a new version and links to the download instead of asking for root. The check on launch is off by default, so the app makes no network request until you turn it on or check by hand

### GPU

- GPU acceleration as an optional download rather than a separate build: ggml's Vulkan backend as a plugin (about 17 MB) that runs on NVIDIA, AMD and Intel GPUs, for both engines. One build serves every machine, GPU or not
- Offered during onboarding and in Settings → Transcription, only when a discrete GPU with Vulkan 1.2 is present. Machines with only integrated graphics stay on the CPU, which is usually faster there, and the GPU section is hidden
- The GPU card names the active device and explains every other state: restart needed, no driver, driver too old, or why the plugin failed
- The download is checked against a pinned SHA-256 and installed under the data directory (`~/.local/share/lune-voice/backends/vulkan/` on Linux)

### Interface

- Settings → Transcription holds everything about the voice model: tabs for Whisper, Parakeet and Remote, with download progress, disk usage, delete, and the GPU card
- Dictation pill overlay with live audio levels and session state: choose its visibility, monitor, anchor, size, orientation, corner style and edge margin, drag it anywhere, and lock it in place. It never takes keyboard focus, so clicking it cannot divert your text. On Linux it can stay visible across workspaces (on by default)
- The pill places itself correctly across monitors running at different scale factors, so a laptop panel next to a scaled external display puts it where you dropped it
- Onboarding that can be cancelled and resumed. A model download survives the window closing, and a reopened window re-attaches to it
- Searchable session history with per-session transcript, timing, and the style applied. Retention runs from off through 7, 30 (the default), 90 and 365 days to forever, and sessions can be deleted in bulk
- Dictionary management with inline editing and priority ranking
- Sound cues on start, stop, and error, with a volume slider and an off switch
- Tray menu: open the app or Settings, start dictation, and copy the last dictation. Start minimized to the tray
- Sidebar quick actions: theme, keyboard shortcuts, check for updates, help and feedback, quit
- Light, dark or system theme
- Licence, warranty and source-code notice in Settings → General
- Interface in 12 languages: Arabic, Chinese, English, French, German, Hindi, Italian, Japanese, Korean, Portuguese, Russian and Spanish. Keyboard-navigable with visible focus rings

### Performance

- mimalloc as the global allocator; release builds use thin LTO
- The recording buffer is sized up front, so recording does not allocate on the audio path, and the model is loaded once and shared across sessions
- History statistics read a maintained word count, so the overview does not slow down as history grows

### Security and privacy

- `privacy_mode` is enforced: under the default `LocalOnly`, a remote transcription backend is refused with an actionable message and remote styling degrades to passthrough. **Allow remote services** in Settings → General lifts it
- The transcription and styling endpoints require HTTPS for anything but a local address (loopback, private LAN, `.local`/`.localhost`)
- Transcript text is redacted from logs by default; `LUNE_LOG_TRANSCRIPTS=1` opts in while debugging recognition
- API keys are kept in the OS keyring where one is available. `settings.toml` and the session-history database, with its WAL/SHM sidecars, are created owner-only
- Transcript export writes through a validated Rust command; the webview has no file-write capability
- Every model file is pinned to a source commit, size and SHA-256, so a truncated or swapped file is caught
- Content Security Policy on the Tauri webview
- GPL-3.0-or-later, with `cargo deny` gating licenses and advisories across the dependency tree
- Every bundle ships `THIRD_PARTY_NOTICES.md` for the native libraries, models, fonts and data, and a generated `THIRD_PARTY_LICENSES.md` carrying the full licence text of every Rust crate and JavaScript package. The release build fails if the generated file is out of date
