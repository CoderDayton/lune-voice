<div align="center">

# Lune Voice

**Your voice, your machine, your text.**

Offline dictation that actually understands what you meant to say.

[Get Lune Voice](https://lunevoice.com) · [Docs](docs/)

</div>

---

Most dictation tools send your audio to someone else's server. Lune Voice doesn't. It runs entirely on your hardware — no cloud, no subscriptions, no data leaving your machine. Buy it once, own it forever.

You press a hotkey. You talk. Clean, formatted text lands wherever your cursor is — your editor, your email, your chat window. That's it. No accounts, no internet required after setup.

<br>

## How It Works

Speech flows through a pipeline that gets smarter at each stage:

```
Mic → Whisper (speech-to-text + built-in VAD) → Dictionary → LLM Style Pass → Paste
```

Raw transcription is rarely ready to use. Lune runs it through your custom dictionary to fix names and jargon, then — if you've turned styling on — hands it to an LLM that reformats the result to match a style you pick: casual, professional, technical, or one you define yourself. Point styling at a local Ollama or llama-server instance to keep it fully on-machine, or at a hosted API if you'd rather not run one.

First launch downloads ~1 GB of Whisper speech-to-text models (plus its voice-activity model) to your machine. After that, the ASR + dictionary pipeline is air-gapped; styling only reaches the network if you enable it and point it at a remote endpoint.

<br>

## What Makes It Worth It

**A dictation pill that stays out of your way** — A floating overlay shows live audio levels and session state. Stop or cancel without ever leaving the app you're working in.

**Style presets that match how you write** — "Professional" tightens prose for workplace messages. "Technical" preserves jargon and code formatting. Prefer your words untouched? Leave styling off — your dictionary and punctuation still apply, no LLM call is made. Roll your own with custom prompts and few-shot examples that teach the LLM your voice.

**A dictionary that learns your world** — Names, acronyms, slang — anything the ASR has never heard. Entries are fuzzy-matched and priority-ranked, so corrections land even when pronunciation is rough.

**Voice commands that feel natural** — "New line." "Scratch that." "Stop dictation." Say them mid-sentence. Add your own phrase mappings for anything you say often.

**Hold or toggle — your call** — Hold-to-talk for quick bursts, toggle for long-form dictation. Global hotkeys work everywhere.

**Session history you can search** — Every dictation is logged with its transcript, timing, and style. Go back and find that thing you said yesterday.

**Audio cues, if you want them** — A short sound when dictation starts, finishes, or hits an error, so you know the app's state without looking. Toggle it off or set the volume to taste.

<br>

## Pricing

Buy once. Use forever. No subscriptions.

| | Devices | Price |
|---|---|---|
| **Personal** | 1 | $10 |
| **Power User** | 3 | $25 |
| **Small Business** | 10 | $80 |

Each license key activates on the number of devices in your tier. Need more seats later? Just grab another key at the current price.

A one-time activation ping confirms your key — after that, everything runs offline.

<br>

## Privacy, for Real

This isn't a marketing claim with asterisks. Speech-to-text and your dictionary run with zero network calls — your audio never leaves your sound card. Styling is off by default; if you turn it on, point it at a local Ollama/llama-server instance and transcripts still never leave your machine, or at a hosted API if you'd rather not run one yourself. Model download and license activation are the only other times the app talks to the internet — both happen once.

There is no telemetry. There are no analytics. There is nothing to opt out of because there is nothing to opt into.

<br>

## Getting Started

After purchasing, download the app and enter your license key. Lune Voice will activate, download models, and you're ready to go.

System requirements:
- **OS:** Linux (X11 or Wayland)
- **GPU:** Recommended for best latency (CUDA supported). CPU-only works but is slower.
- **Disk:** ~2 GB for models
- **Internet:** Required once for activation and model download

<br>

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — pipeline internals, session lifecycle, threading model
- **[Configuration](docs/CONFIGURATION.md)** — settings, backends, hotkeys, style presets
- **[Performance](docs/PERFORMANCE.md)** — benchmarks, latency breakdowns, tuning

<br>

## License

Lune Voice is proprietary software. See [LICENSE](LICENSE) for terms.
