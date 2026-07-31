<div align="center">

# Lune Voice

**Open source dictation for Linux that runs on your own hardware.**

[Docs](docs/) · [Architecture](docs/ARCHITECTURE.md) · [GPLv3](LICENSE)

</div>

---

Lune Voice converts speech to text without sending your audio anywhere. You hold a hotkey (or toggle one), speak, and the transcribed text is inserted into whatever window has focus: an editor, an email draft, a chat box.

Transcription and correction run locally, so the app keeps working after you disconnect from the internet. It is free software under the GPLv3: read the code, change it, redistribute it, and anyone you pass it to gets those same freedoms.

## How it works

```
Mic → Whisper (speech-to-text + voice-activity trim) → Dictionary + formatting → [optional styling] → Paste
```

Whisper handles transcription on your machine. Your dictionary corrects the names, acronyms, and terminology it would otherwise get wrong, and the formatter applies capitalization, punctuation, and filler removal.

Styling is optional and off by default. When you enable it, an LLM rewrites the transcript in a tone you select: casual, professional, technical, or one you write yourself. Point it at a local Ollama or llama-server instance to keep transcripts on your machine, or at a hosted API if you would rather not run a model.

First launch downloads roughly 1 GB of Whisper models. After that, dictation needs no network access.

## Features

**Dictation pill.** A small floating overlay showing audio levels and session state. You can stop or cancel a session without switching away from the application you are working in.

**Style presets.** "Professional" tightens up loose phrasing. "Technical" preserves jargon and code formatting. You can define your own with custom prompts and few-shot examples, or leave styling off and keep only the dictionary and punctuation passes.

**Custom dictionary.** Add the names, acronyms, and terms specific to your work. Exact matches always apply, near misses in pronunciation are caught as well, and priority ranking decides between competing entries.

**Voice commands.** "New line." "Scratch that." "Stop dictation." These work mid-sentence, and you can map additional phrases to your own actions.

**Hold or toggle.** Hold the hotkey for short corrections, or toggle it for longer passages. Hotkeys are global and work in any application.

**Searchable history.** Every session is recorded with its transcript, timing, and the style applied.

**Audio cues.** Short sounds mark the start, end, and failure of a session, so you can track state without watching the screen. Volume is adjustable, and the cues can be turned off.

## Privacy

The app makes exactly one kind of network request on its own: downloading the Whisper models on first launch. Transcription, dictionary correction, and formatting never reach the network.

Styling is the one feature that can, and only after you turn it on. Pointing it at a local endpoint keeps transcripts on your machine.

The app collects no telemetry and no analytics. Since the source is public, you do not have to take that on faith.

## Building from source

You need Rust, [Bun](https://bun.sh), CMake, and LLVM. On Linux, install the Tauri WebView dependencies as well. A CUDA toolkit is optional and enables GPU transcription.

```bash
git clone https://github.com/CoderDayton/lune-voice
cd lune-voice
cgs dev      # run the app with hot reload
cgs build    # build a desktop bundle
```

Both commands detect CUDA automatically and fall back to CPU. You can force either path with `--env BACKEND=cpu` or `--env BACKEND=cuda`. See [docs/RELEASE.md](docs/RELEASE.md) for the full build matrix, and run `cgs show` to list every available script.

| | |
|---|---|
| **OS** | Linux (X11 or Wayland) |
| **GPU** | Optional. CUDA gives the lowest latency; CPU-only works but is slower. |
| **Disk** | Roughly 2 GB for models |
| **Internet** | Once, for the model download |

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: pipeline internals, session lifecycle, threading model
- **[Configuration](docs/CONFIGURATION.md)**: settings, backends, hotkeys, style presets
- **[Performance](docs/PERFORMANCE.md)**: benchmarks, latency breakdowns, tuning
- **[Release](docs/RELEASE.md)**: build matrix, packaging, third-party notices

## License

GNU General Public License, version 3 or later. See [LICENSE](LICENSE) for the full text, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream projects bundled into the binary.

If you distribute Lune Voice, modified or not, you have to make the source available to whoever you give it to, under these same terms.
