# Lune Voice — ASR Pipeline Audit

**Scope:** Full dictation pipeline, audio capture → output delivery. Offline, local, Linux.
**Method:** Source read of every stage with `file:line` grounding; key claims independently verified by grep.
**Pipeline (actual):** `Mic (cpal) → downmix/resample → ring buffer → VAD (Earshot + PCEN) → preroll/pause-injection gate → ASR (Qwen3-ASR ONNX, batch greedy) → voice-command routing → dictionary + heuristic correction → punctuation → bypass gate → LLM correct/style (Qwen3-0.6B ONNX, greedy) → sentence-cap backstop → delivery (uinput type / clipboard paste) → persist`.

All paths relative to repo root. Citations re-verified against `main` @ `fc5eb1e` (the audit was first written at `98e782d`). Line numbers in the changed stages are exact; deeper internal citations in unchanged stages may be ±4 from intervening edits. Changes since the first cut are folded in: streaming trait surface removed, orphaned dict scorer/candidate files deleted, cross-utterance LLM context removed, non-ASCII delivery fixed, and a homophone resolver + conformal calibration subsystem added (built, not yet live-wired).

---

## 0. Top-level orchestration (`crates/lune-app/src/session.rs`)

`SessionManager<A: AsrBackend>` is the conductor. Only **one session active at a time**; enforced by `active: Arc<Mutex<Option<ActiveHandle>>>` (`session.rs:81`, guard at `:176-179`).

**Lifecycle.** `start()` / `start_with_audio()` / `start_with_handle()` all funnel into `start_inner` (`:137`):
1. Resolve style preset **before** taking the active lock — a missing style is a config error, not a session-state error (`:147-166`). Fallback chain: requested id → custom styles → `"minimal"` → hard error.
2. Take active lock; reject if already `Some` (`:176-179`).
3. Insert session row into sqlite (`:184-197`); persist failure is surfaced as a non-fatal `Warning`, not fatal.
4. Spawn `run_session` task + a watchdog task that forces cleanup + emits `Error("internal session panic")` if the session task panics (`:213-242`).
5. Assign `*lock = Some(...)` last — **cancellation-safe**: if `start_inner` is dropped at any await before this, the slot stays `None` and a later start succeeds (`:168-175`, regression test `:1764-1799`).

**Control plane.** `stop()` (graceful), `stop_and_wait()` (registers the `Notify` *before* sending Stop to avoid a TOCTOU race, `:271-280`), `cancel()` (discard). Signals travel over an unbounded mpsc to the session task.

**Constants in the spine:**
- `MAX_SEGMENTS = 500` (`:301`) — ring of accepted ASR segments; oldest dropped past the cap.
- `GARBAGE_DROP_THRESHOLD = 0.25` (`:303`) — segments with `confidence < 0.25` are dropped in `append_segments` (`:352`).
- `PREROLL_CHUNKS = 4` (`:310`) — ~320 ms of pre-speech audio kept to recover hard word onsets ("hello" → "ello" without it).
- `PAUSE_TOTAL_MIN_SAMPLES = SAMPLE_RATE` = 1000 ms (`:318`) — gaps shorter than this are breaths, not deliberate pauses.
- `PAUSE_TOTAL_CAP_SAMPLES = 1.5·SAMPLE_RATE` = 1500 ms (`:321`) — ceiling on injected pause.
- `ASR_START_TIMEOUT = 15s` (`:441`) — hard ceiling on backend init.

---

## 1. Audio Capture (`crates/lune-audio/src/{capture,ring}.rs`, `lune-core/src/audio.rs`)

**Canonical rate:** `SAMPLE_RATE = 16_000` (`lune-core/src/audio.rs:3`). All downstream math assumes 16 kHz mono f32 in [-1, 1].

### 1.1 Backend & device
- **cpal**, `default_host().default_input_device()` → `AudioError::NoDevice` if absent (`capture.rs:223-224`). **No device enumeration/selection** — always the system default input.
- Stream built at the **device native** sample rate and channel count, `BufferSize::Default` (`:233-253`). The app does *not* request 16 kHz from the device; it resamples in software.
- Sample formats: `F32` (passthrough), `I16` (`s/32768`), `U16` (`(s-32768)/32768`) via `build_stream!` macro; any other → `StreamBuild("Unsupported sample format")` (`:351-360`).

### 1.2 Per-callback processing chain (`:540-614`)
1. **Downmix to mono** (`downmix_mono_into`, `:123-217`). Each sample clamped to [-1,1] and zeroed if non-finite (`clean_sample`). Stereo/3–32ch path computes per-channel power; if imbalance `power_ratio > POWER_RATIO_LIMIT (16.0)` or antiphase (`stereo_corr < STEREO_CORR_FLOOR 0.1`), it **drops the weak channel** instead of averaging (avoids 12 dB-imbalance and antiphase cancellation). >32 ch → plain average.
2. **Resample** (rubato `Async` poly, `Septic` degree, `FixedAsync::Input`) — built **only if native ≠ 16 kHz** (`:267-283`). Backlog capped at `MAX_RESAMPLE_BACKLOG = 48_000` samples (~1 s @48 kHz), oldest drained on overflow (`:548-556`). Resampler `process` error → **clears the input buffer** and breaks (`:576-582`).
3. **Chunk + push.** While `output_buf.len() >= chunk_samples`, drain exactly `chunk_samples` into an `AudioChunk { samples, timestamp_ms }` and `ring.push` (`:589-601`). `timestamp_ms = start_time.elapsed()` **at push time** — processing wall-clock, not sample-accurate capture time (drifts under backlog).

### 1.3 AudioConfig
`target_sample_rate = 16_000`, `chunk_duration_ms = 80`, `buffer_capacity = 128` (~10 s) (`:13-36`). **Chunk = 16000·80/1000 = 1280 samples / 80 ms.** Zero/overflow configs rejected by `validate_config` (`:49-70`).

### 1.4 Tail flush
`request_flush()` sets an atomic; the callback then `mem::take`s the sub-chunk remainder (< 1280 samples) and pushes it as a short final chunk (`:607-614`). Recovers the last phonemes on session stop. The session calls this just after the capture loop, then waits 30 ms for one callback cycle (`session.rs`).

### 1.5 Ring buffer (`ring.rs`)
- `crossbeam_channel::bounded(capacity)` + `tokio::sync::Notify` for async wakeups (`:6-23`).
- **Overwrite-on-full:** `push` evicts the oldest chunk when full so the **newest always lands** (`:33-49`). The lib doc "no audio silently dropped" refers only to the new item — older chunks *are* evicted under consumer lag.
- `pop()` awaits `Notify` on empty; `try_pop()` non-blocking. SPSC-oriented (one audio producer, one session consumer).

### 1.6 AudioPull abstraction & offline path
`AudioPull` trait (`lib.rs:44-78`): `try_next_chunk`, `is_offline`, `request_flush`. Three session sources (`session.rs:57-65`): `Open` (fresh live stream), `Owned(AudioStream)` (file replay/tests), `Shared(AudioStreamHandle)` (pre-warmed mic; `drain_keep_last(6)` discards idle audio but keeps ~500 ms of hotkey-press window, `session.rs:463-467`).
- `from_samples` builds an **offline** stream (no audio thread, `_stream: None`); `is_offline()` → true (`capture.rs:382-421`).
- ⚠️ **`AudioStreamHandle::is_offline()` is hardcoded `false`** (`lib.rs:75-77`) — a handle wrapping an offline stream cannot report drain-complete.

### 1.7 Capture-stage risks
1. Three independent audio-drop points under load: ring eviction, resample backlog cap, resampler-error buffer clear.
2. `timestamp_ms` is processing-time, not capture-time.
3. A parallel/legacy config type `lune-core::AudioCaptureConfig` (`audio.rs:43-79`, defaults `frame_size 480`, `vad_threshold 0.01`) is **unused** by the live path and disagrees with it — a confusion hazard.

---

## 2. Voice Activity Detection (`crates/lune-audio/src/{vad,pcen,calibrate}.rs`)

**VAD is Earshot (pure Rust, no ONNX model)** — `use earshot::Detector; Detector::default()` (`vad.rs:3,99`). This is intentional: VAD runs entirely on CPU with no neural model. The CLAUDE.md "Qwen3-ASR-ONNX" line refers to ASR, not VAD; nothing in the repo (or its git history) references Silero or any learned VAD model. No model file is loaded for VAD. *(Verified by grep: `vad.rs:3,53,55,99`; a repo-wide search for "silero" returns nothing.)*

### 2.1 Framing
- `HOP_SIZE = 256` samples = **16 ms @ 16 kHz** (`vad.rs:7-8`). Earshot is fed fixed 256-sample frames.
- Leftover samples between chunks are carried (`carry: Vec<f32>`, cap 256) and completed with the head of the next chunk (`:135-167`), so framing is continuous across the 1280-sample audio chunks.
- `vad_score` returned per chunk = **max frame score** across that chunk (`:153-163,188`).

### 2.2 VadConfig defaults (`vad.rs:22-51`)
| Field | Default | Role |
|---|---|---|
| `threshold` | **0.5** | speech-START probability |
| `threshold_end` | **0.3** | speech-END probability (hysteresis) |
| `hangover_ms` | **240** | trailing speech hold (was 30 ms — too eager, spliced unrelated speech) |
| `pcen` | `Some(default)` | PCEN AGC **on by default** |

Construction validates thresholds ∈ [0,1] and `threshold_end ≤ threshold` (`:67-85`). `hangover_samples = hangover_ms·16`, floored at `HOP_SIZE` (`:86-88`); at 240 ms → 3840 samples.

### 2.3 Hysteresis & hangover (`:169-184`)
- State-dependent threshold: in speech → compare against `threshold_end (0.3)`; idle → against `threshold (0.5)`. Enter at 0.5, stay until below 0.3.
- On a crossing → `is_speech = true`, `hangover_remaining = hangover_total` (refill). Below threshold but hangover left → still speech, decrement by **`samples.len()`** (chunk length). So hangover real-time duration couples to caller chunk size (correct at 1280, but a coupling).

### 2.4 PCEN AGC (`pcen.rs`)
Per-hop adaptive gain normalizing each Earshot frame **in place before scoring**; diagnostic `energy` is computed on the **raw** input (`vad.rs:130`), independent of PCEN. Tracks signal + noise power envelopes (noise: asymmetric IIR, falls fast / rises slow), applies one scalar gain bounded by a target output level and a noise ceiling so amplified background noise can't exceed `noise_ceiling_rms`.

Defaults (`pcen.rs:57-73`): `target_rms 0.10`, `noise_ceiling_rms 0.02`, `max_gain 12.0`, `min_gain 0.25`, `alpha 0.85`, `tau_signal 250 ms`, `tau_noise_up 5000 ms`, `tau_noise_down 200 ms`, `tau_gain 80 ms`, `warmup_frames 16` (hold gain=1 during warmup so hotkey-onset speech isn't mis-gained), `eps 1e-8`. Toggled by `UserSettings.vad_pcen` (default **true**); `init_vad` sets `vad_config.pcen = None` when disabled (`session.rs:490-499`).

### 2.5 Calibration (`calibrate.rs`)
Pure function `recommend(CalibrationInput) → CalibrationResult` over two pre-recorded VAD-score vectors (quiet room, read prompt). Does **not** read live audio — the `calibration.rs` session in lune-app records and supplies the scores. Derives `speech_threshold` / `speech_threshold_end` (mapped onto `VadConfig`) plus `noise_p95`, `speech_p10`, `pause_p20`. Gates: noise ≥ 2 s, speech ≥ 8 s, separation `speech_p10 − noise_p95 ≥ 0.20` else `InsufficientSignal`. End threshold is forced below start by ≥ `HYSTERESIS_GAP (0.05)`.

---

## 3. The VAD Gate / Pause Injection (`session.rs` capture loop, `:598` onward)

This is the most subtle stage: the capture loop streams VAD-passed audio to ASR while reconstructing the pauses VAD strips, so Qwen3 still hears prosody for punctuation.

**Loop structure** — `tokio::select!` with `biased` priority (`'capture: loop` at `:598`):
1. `control_rx` (highest) — Stop/Cancel break promptly even mid-stream (regression test `stop_breaks_capture_loop_promptly` `:1538`).
2. `session_deadline` wall-clock guard = `3 × max_speech_seconds` (`:580-581`).
3. `audio_poll_tick` every 10 ms (Skip on miss) — drains all available chunks.

**Per chunk:**
- `vad.process_chunk` → `VadResult { is_speech, vad_score, energy }`. `peak_rms` tracked for the level meter.
- **Not speech:** if speech already started, accumulate `silence_samples`; push chunk into the `preroll` ring (cap `PREROLL_CHUNKS=4`); `was_speech=false` (`:646-656`).
- **Speech:**
  - Hangover-tail accounting: if `vad_score < speech_threshold_end`, this chunk is silence fed live as "speech" → add to `gap_fed_samples` (`:666-671`).
  - **On speech onset** (`!was_speech`): compute `pause_injection_samples(silence, preroll_len, gap_fed)` and push that many **zero samples** to ASR so the model hears the gated pause, then flush the preroll ring to ASR (`:672-689`). Pause math (`pause_injection_samples` `:333`) makes the *heard* pause exactly `min(true_gap, 1500 ms)` sample-for-sample: injects `silence − preroll`, capped by `1500 ms − already_heard`, and **zero if total gap < 1000 ms** (breaths skipped). Verified exact by tests `:1257-1282`.
  - Push the speech chunk to ASR via `asr.push_audio` (`:694`). push errors are recorded (`first_asr_error`) but the loop continues.
  - `speech_samples += len`; break if `≥ max_speech_samples` (`:701-708`).
- **Level meter:** `AudioLevel(min(peak_rms·6, 1.0))` emitted at ~15 fps via `try_send` (`:711-719`).
- **Starvation guard:** if a live mic delivers no audio for 5 s → `Error("audio device stopped responding")`, `Failed("audio starvation")` (`:721-730`). Offline streams exit cleanly on exhaustion (`:732-735`).

**After the loop:** tail-flush the producer (`:744-755`), emit `Processing`, then `poll_final` the backend (`:766-781`). On cancel, `reset_context` discards the buffer instead.

**Empty-output handling** (`:802-822`): if no speech / no segments / empty text — surface a held ASR error as `Failed`, else `Cancelled` (speech but no output) or silent `Completed` (no speech).

---

## 4. ASR — Qwen3-ASR ONNX (`crates/lune-asr/`)

### 4.1 Trait & buffering model
`AsrBackend` (`lune-core/src/asr.rs`) is **batch-only by design**: `poll_final` (`:152`) is the single required transcription entry point — the earlier `poll_results` / `supports_streaming` streaming surface has been **removed**. Methods: `push_audio` (`:146`), `poll_final` (`:152`), `end_session` (`:154`), defaulted `reset_context` (`:159`), `segment_capacity_hint`. Qwen3 impl (`backend/qwen3.rs`):
- `SessionHandle = Mutex<Qwen3Session>` holding `pending: Vec<f32>`, `total_samples`, `sample_rate`, `next_id`, `forced_language`.
- `push_audio` **appends raw PCM** to `pending`; trims oldest beyond `MAX_WINDOW_SAMPLES = 300·16000 = 4_800_000` (5 min) (`:55`, push `:318-327`).
- `poll_final` (`:329`) `mem::take`s the whole buffer and decodes once via `transcribe_pcm`. `reset_context` clears `pending`; `end_session` is a no-op.

So the session streams audio in but ASR produces **one segment per utterance** at finalize. No partial/incremental text reaches the UI from Qwen3 — and there is no longer any dormant streaming machinery behind it (the trait was slimmed to match).

### 4.2 Models & session options
Three ONNX sessions (`qwen3.rs:161-198`): `qwen3-asr-encoder.onnx`, `qwen3-asr-embed-tokens.onnx`, `qwen3-asr-decoder.onnx` + HF `tokenizer.json`. Model dir `qwen3-asr-0.6b` (`lune-core/src/models.rs:1`). Decoder weights **FP16**. Constants: `VOCAB_SIZE 151_936`, `NUM_LAYERS 28`, `NUM_KV_HEADS 8`, `HEAD_DIM 128` (`:26-47`).

Session build (`ort_utils.rs:226-278`): `GraphOptimizationLevel::Level3`, 4 intra-threads, optimized-graph caching (`*.onnx.opt`), shared env CUDA allocators. **CUDA EP** only under `feature="cuda"`: heuristic conv search, tf32 on, `error_on_failure()` (no silent in-ORT CPU fallback). Explicit CPU retry one level up only when `use_gpu && !cuda_required`, with a loud error log. Arena-shrink run options used at prefill and a post-decode "janitor" pass to release ~1 GB KV high-water back to the driver (`:1009-1051`).

### 4.3 Mel front-end (`mel/whisper.rs`)
Whisper-style, **native f16, mel-major**. `N_FFT 400`, `HOP_LENGTH 160`, `WIN_LENGTH 400`, `N_MELS 128`, 16 kHz (`:12-15`). Center reflect-pad by 200; periodic Hann; power spectrum over 201 bins (rustfft); **Slaney mel filterbank** (`fmin 0`, `fmax 8000`, area-normalized triangles); `ln`-floor `1e-10`; Whisper log-norm `log10(v)/4 + 1` then clamp to `peak − 2.0`. Padded to `MEL_MIN_FRAMES = 3000` (30 s) using an analytic silence frame (no FFT on silence), reshaped `[1,128,3000]`, tensor name `"mel"` (`qwen3.rs:724-744`). A boost-only loudness pre-norm toward `NORM_TARGET_RMS 0.08` (≤ `NORM_MAX_GAIN 10×`) runs first (`:70-96,717`).

### 4.4 Decode loop
This is a **decoder-only multimodal LM**: the encoder produces audio embeddings that are **scattered into the text-embedding sequence at `<audio_pad>` positions** (not cross-attention).

Prompt layout (`build_prompt_ids` `:245-292`): ChatML `system` (fixed ASR-engine instruction + optional `"\nPreferred terms: …"` boost csv) + `user` block wrapping `AUDIO_START · n×AUDIO_PAD (151676) · AUDIO_END` + `assistant` header; if a language is forced, `"language {lang}"` tokens are appended so the model skips its own LID. Special tokens `:37-47`. Position ids = mRoPE `[3,1,seq]`, three identical `0..seq` rows for audio-only.

Generation (`:886-1006`): **greedy argmax only** — no sampling/temperature/top-p (`argmax_on_logits` `:377-479`). KV-cached autoregressive, one token/step, attention mask sliced from a preallocated ones row. **Stop:** `last ∈ EOS_IDS {151643,151645}` or `tokens ≥ MAX_NEW_TOKENS (512)` (`:889-892`). *(All verified by grep.)*

**Output parsing** (`decode_output` `:1103-1249`): split raw text on `<asr_text>` → `(detected_language, text)` via `parse_auto_asr_output`; strip meta tokens and a leaked system-prompt prefix; empty → no segment. `bcp47_to_qwen3_language` maps BCP-47 → Qwen labels for ~20 languages (`:1292-1331`).

### 4.5 Confidence
- **Per token:** `argmax_on_logits` fuses argmax + log-sum-exp softmax + top-16 logits in one f16→f32 pass over the last position (`MABLE_TOP_K = 16`). Confidence = softmax prob of the chosen token (raw, even on the boost/ban-constrained path).
- **Per segment:** `mean_conf` → `AsrSegment.confidence`; `min_conf` → `min_token_confidence` (`:1176-1178`). Raw softmax at the backend; the session layer is documented to later apply an isotonic calibrator turning these into `P(correct)` (`asr.rs:39-57`).
- **Per word (HWCE):** word spans built from token boundaries (space / `▁`), then `hwce::estimate` produces `word_error_risk`; `WordConfidence.confidence = 1 − risk` (`:1180-1246`). No CTC/attention aligner in this path. (HWCE detailed in §6.4.)

### 4.6 Repetition / garbage containment
1. **No-repeat-3-gram ban** (`REPEAT_GUARD_ORDER 3`): masks tokens that would re-complete a seen 3-block (constrained argmax path; empty in the common case so the fast path is preserved).
2. **Degenerate-cycle trim:** detects an n-token block (n∈1..8) repeated 4× in the last 4n tokens (≥16 tokens), truncates `len − 3n`, breaks the decode with a warn.
3. Word-level `has_repetition` feeds HWCE risk.
4. Segment-level `GARBAGE_DROP_THRESHOLD 0.25` drop in the session (`session.rs:372-381`).

### 4.7 Timestamps & boost
Qwen3 emits **no internal timestamps**: segment carries `start_ms = None`, `end_ms = total_samples·1000/sample_rate`. Boost (`set_boost_phrases`): dual-layer — phrases in the system prompt as "Preferred terms" **and** prefix-constrained additive logit bias (entry token at `score`, continuation at `2×score` on prefix match). Controlled by `word_boost_enabled` / `word_boost_score` (default off / 1.0 for Qwen3 per registry).

### 4.8 OpenAI-compatible ASR backend (`backend/openai.rs`)
Thin retrying HTTP client. Buffers f32; `poll_final` (`:143`) force-flushes the buffer (the incremental `poll_results` path was removed with the trait slimming). Encodes 16-bit mono WAV in memory, posts multipart `/audio/transcriptions` with `response_format=verbose_json`; 3 attempts, 2 s doubling backoff on 5xx/429; restores audio on error for retry. Per-segment `start/end` → ms; **no confidence/word signal** (no logprobs). Non-HTTPS rejected unless loopback.

### 4.9 Registry & pipeline defaults (`backend/registry.rs`)
2-variant enum (`Qwen3 | OpenAi`). `from_config` resolves `use_gpu`/`cuda_required` from `DevicePolicy`. `default_pipeline_config`:
- **Qwen3Asr:** `max_segment_ms / sentence_close_ms / hard_close_ms = 300_000` (so VAD never mid-utterance flushes — the dictation toggle drives flush), `speech_threshold 0.5`, `speech_threshold_end 0.3`, `min_speech_ms 200`, `word_confidence_threshold Some(0.55)`, `llm_skip_threshold Some(0.65)`, `llm_skip_noisy_threshold Some(0.30)`, `max_speech_samples 4_800_000` (300 s), `llm_timeout_secs 30`, `itn_enabled true`, `word_boost_enabled false`.
- **OpenAiCompatible:** `max_segment_ms 30_000`, `sentence_close_ms 800`, `hard_close_ms 2_000`, all confidence/skip thresholds `None` (no logprobs), `max_speech_samples 120·16000`.

---

## 5. Post-Capture Processing (`session.rs::process_and_deliver`, `:850`)

Runs once per session on `(raw_text, segments)`. Order: voice commands → dictionary scan + heuristic correction → punctuation → bypass gate → LLM → sentence-cap backstop → deliver + persist.

### 5.1 Voice command routing (`crates/lune-dict/src/commands.rs`)
`router.route_full(&raw_text)` → `RouteResult { commands, remaining }`. Grammar (`commands.rs:23-50`):
- **Exact builtins, sorted longest-first** so short matches don't shadow: `stop listening`/`stop dictation`/`stop` → StopDictation, `new paragraph` → NewParagraph, `scratch that` → ScratchThat, `list item` → ListItem, `new line` → NewLine.
- **Parametric prefixes** (run first, claim their span before exact matching): `switch preset to `/`switch to ` → `SwitchStyle(arg)`, `expand snippet `/`insert ` → `InsertSnippet(arg)`; arg terminated by `, . ;`.
- Matching is case-insensitive on a lowercased copy; `remaining` is rebuilt from the **original-case** text with command spans removed and whitespace collapsed.

`SwitchStyle` is applied **live**: the session swaps `style_preset` mid-pipeline (`session.rs:882-913`). Each command is emitted as `CommandDetected`. If `remaining` is empty after stripping, the session finalizes with no delivery (`:914-928`).

### 5.2 Dictionary scan + heuristic correction
**Dictionary** (`lune-dict/src/lib.rs`). `DictionaryManager` holds a priority-DESC-sorted cache of `{phrase, replacement, priority}`.
- `top_relevant(text, 20)` → up to 20 entries for the LLM prompt (CI substring or Levenshtein-≤1 single-word match). Forced-title-case replacements ("Dev", "Server") are **softened to lowercase before the prompt/seed** so the sentence-cap backstop can re-capitalize correctly (`session.rs:944-949`).
- `scan(text)` → `Vec<(matched, replacement)>` without mutating; multi-word = exact CI window match, single-word = Levenshtein-≤1 CI (`lev_le1_ci`, an O(n) early-exit primitive).

**LexicalGate** (`corrector.rs`), `correct_with_risk_threshold(text, word_risk_scores, dict_replacements, lang, flag_threshold)`:
- ⚠️ **No spelling-candidate generation in the live path.** It does dictionary replacement, canonical-casing, proper-noun/known-word preservation, and *flagging* only — it never rewrites a flagged word to a guessed candidate (it emits the original verbatim and marks it for the LLM). The dormant weighted-edit-distance scorer and candidate enumerator (`scorer.rs`, `candidate_source.rs`) flagged in the first audit have since been **deleted**. `lune-dict` now declares `caprate, corrector, freq, homophone, homophone_calib, lexicon, ngram, punctuation, vocab` (`lib.rs:1-9`); the `homophone*` modules are the new resolver subsystem (§5.5), separate from the live `LexicalGate`.
- Decision order per word (`process_single_word` `:60-180`): digit/empty passthrough → dictionary replacement (case-insensitive exact phrase, highest priority) → known&unflagged passthrough → canonical-casing override (github→GitHub, guarded against flattening sentence caps / force-capping common nouns) → known-word verbatim under `CANONICAL_OVERRIDE_RISK_CEIL 0.75` (Vocab words pass even above) → proper-noun/acronym/camelCase preservation under `PROPER_NOUN_RISK_CEIL 0.80` → else flag as unresolved if `risk ≥ flag_threshold`.
- Thresholds: `HWCE_FLAG_THRESHOLD 0.40`, canonical ceiling `0.75`, proper-noun ceiling `0.80`. `risk` is `word_error_risk` from HWCE (the `1 − threshold` form, `session.rs:964-968`).
- Result: `CorrectionResult { corrected_text, word_risk_positions, unresolved_words, all_resolved, dropped_risk_entries }`. Word↔risk alignment is a greedy forward cursor; entries that don't align count as `dropped`.
- The corrector no longer holds the n-gram model — the former `#[allow(dead_code)]` field was removed and `HeuristicCorrector` renamed `LexicalGate`. The trained n-gram still powers homophone calibration at startup (§5.5).

Lexicon (`lexicon.rs`): tiered `lookup` over a frequency dict (`CONFIDENT_FREQ 5000`), curated supplemental vocab (`Vocab`, freq floor 10000, dev terms), and English Wordnik membership. 11 bundled frequency dictionaries (en/de/es/fr/it/ja/ko/nl/pt/ru/zh).

### 5.3 Punctuation (`formatting.rs`)
`normalize(text) → { text, needs_llm }`:
- Non-Latin → verbatim. Title abbreviations (mister→Mr.) applied when followed by a capitalized word.
- **Tier-1 deterministic:** `capitalize_sentence_starts` + append a terminal `.` for **declaratives only**.
- **`needs_llm` = run-on OR (interrogative-lead AND no terminal mark).** Run-on = `≥ RUN_ON_WORD_THRESHOLD (12)` words with no internal separator (`, ; : — – . ! ?`). Interrogative = WH-lead, or AUX-lead + question-subject (subject-aux inversion; bare aux like "have a great day" stays declarative). The end mark for these is deferred to the LLM.
- `capitalize_sentence_starts` preserves existing brand casing (iPhone), handles dotted acronyms (`U.S.`), abbreviations (`mr/dr/st/etc/vs/prof`), and single-letter initials as non-terminal.

### 5.4 The Bypass Gate (`session.rs:996-1051`)
Decides whether to invoke the LLM at all. Three "needs correction" signals OR-ed:
1. `!all_resolved` — a word the heuristic could neither resolve nor confidently preserve.
2. `punctuation.needs_llm` — ambiguous punctuation (run-on / interrogative without end mark).
3. `conf_attention` — `min_conf < llm_skip_threshold (0.65)`, where `min_conf = 1 − max_word_risk`. Catches homophone mishears ("no"/"know") even when everything "resolved".

Plus a **skip-when-broken veto** (`too_broken`): segment-average `avg_conf = 1 − mean_risk < llm_skip_noisy_threshold (0.30)` → transcript too degraded across the board; correction would *degrade* quality (arXiv:2509.25048), so the LLM is skipped.

Final: `needs_correction = signal1 || signal2 || conf_attention`; `wants_styling = style_preset.needs_styling()`; **`needs_llm = (needs_correction || wants_styling) && !too_broken`** (`:1030-1036`). Styling is orthogonal — a clean transcript still gets a non-`minimal` style pass. A `GateDecision { used_llm, lang, unresolved }` event is emitted for the UI.

The text that goes forward is `text_after_heuristic` (heuristic-corrected + Tier-1 punctuation). If `!needs_llm`, this is the final output (fast path).

### 5.5 Homophone resolver + conformal calibration (`lune-dict/{homophone,homophone_calib}.rs`) — built, NOT yet live-wired

A self-evaluating homophone disambiguator exists but is **not called from the live correction path** — `process_and_deliver`/`LexicalGate` never invoke it. It is fully built, tested, and exercised at startup (calibration is computed and logged), pending integration.

- **Resolver** (`homophone.rs`): 15 curated, tokenizer-distinct confusion sets (`CONFUSION_SETS` `:52` — e.g. `there/their/they're`, `to/too/two`; apostrophe-colliding pairs like `it's/its` deliberately excluded). `ResolverPolicy` (`:102`): `Off`, `RerankArgmax` (the lune-ngram blend, ungated — over-corrects), and `MarginGate { tau }` (default **2.0**). A flip fires only when **all** gates pass: confusion-set membership, n-gram avg-log-prob margin > τ, sufficient trigram **context support**, word HWCE risk below `max_word_risk` (0.85), and an optional **contextual-gain** gate. Scoring shares `best_member_margin` with the calibrator so offline and live decisions match exactly. `resolve_words` carries the **observed (ASR)** left-context forward — byte-for-byte what the calibrator replays.
- **Calibration** (`homophone_calib.rs`): per-user, per-set self-evaluation over `(raw → final)` history pairs.
  - `calibrate` (`:44`) replays the resolver counterfactually and, per set, sets the margin threshold via **Conformal Risk Control** (`conformal_threshold` `:137`; Angelopoulos et al., arXiv:2208.02814): τ̂ = the smallest threshold whose expected over-correction rate is provably ≤ a target α (default 0.01). Over-correction is monotone in τ, so split-conformal calibration applies directly.
  - **Selective enablement** (reject option): a set is enabled only if its net benefit is positive under a Wilson lower bound (Chow 1970; Geifman & El-Yaniv 2017) — sets with no learnable per-user signal auto-disable. An **isotonic** margin→P(correct) reliability curve is fitted alongside.
  - Feeds on the `correction_pairs` table (§7.5). `NGramModel` gained `context_support` + `unigram_avg_logprob` for the gates.
- **Wiring today:** `runtime.rs:886` trains the n-gram on history, then calls `calibrate` and **logs** the per-set enable/τ/net decisions at startup. The resolver is not yet placed in the correction path.
- **Benchmarks:** `benchmarks/homophone_tune.rs` (synthetic, seed-averaged, empirically validates the conformal α-guarantee) and `benchmarks/homophone_pipeline.rs` (runs on the real sqlite DB; compares integration strategies — PreLLM static/conformal, PostLLM safety-net, gate-signal).

---

## 6. LLM Correction & Styling (`crates/lune-llm/`)

Invoked only when the gate sets `needs_llm`. Wrapped in `tokio::time::timeout(llm_timeout_secs)` (default 30 s); on timeout or `Err`, the pipeline **falls back to `text_after_heuristic`** — output is never worse than the heuristic fast path. The LLM input (`PostProcessInput`, `lune-core/src/llm.rs:12-26`) carries `raw_text` (heuristic output), the resolved style preset, up to 20 dictionary entries, detected `language`, and `correction_needed`. The former cross-utterance `context` field (a rolling 400-char tail of prior transcripts) has been **removed** end-to-end — the LLM now sees only the current utterance.

### 6.1 Two-pass transform (`pipeline.rs:115-194`)
`CorrectionPipeline` wraps any `LlmService` and runs:
1. **Dictionary pre-seed** (`apply_dictionary_seed`, before the LLM) — longest-phrase-first, Unicode-CI, punctuation/whitespace preserved. Replacements arrive at the validator already done, not as LLM edits.
2. **Pass 1 — Correction**, gated on `correction_needed`. Builds system+user prompt, `llm.complete(system, user)`. Output is run through `fix_all_number_spaces` → `contains_forbidden_output` guard → `validate_and_merge`. On `Err`, warns and falls back to seeded raw (styling still runs).
3. **Pass 2 — Styling**, gated on `style_preset.needs_styling()`. `llm.complete("", style_prompt)` — **empty system prompt** (bypasses the prefix-KV cache). Accepted only if non-empty and not forbidden output; **not** run through `validate_and_merge`. Errors keep the corrected text.

### 6.2 Prompts (`config/style_llm.rs`, `backend/ort_llm.rs`)
- `BASE_SYSTEM_PROMPT`: fixed, language-agnostic ASR-corrector instruction ("smallest edits", homophones only when unambiguous, preserve punctuation, no translate/paraphrase/reorder, no censoring profanity, no conversational replies) + 6 inline few-shot examples. Dictionary terms appended as `"- {phrase} → {replacement}"`.
- **Correction user prompt:** `"Correct the following {Language} transcript:"` + up to 6 dictionary carrier-sentence examples + transcript + `"\n\n/no_think"`. (The earlier "Earlier dictation (context only…)" block was removed with the context field.)
- **Styling user prompt:** self-contained, no system prompt — `"Rewrite … Output only the rewritten text."` + style name + rules + few-shot pairs + transcript + `"\n/no_think"`.

### 6.3 ONNX backend (`backend/ort_llm.rs`) — Qwen3-0.6B
- **Model discovery:** quant priority `q4f16 > int8 > fp16 > q4 > quantized > ""` — q4f16 deliberately outranks int8 (int8 leaks `<think>`/few-shot text and ignores `/no_think`). Split export `decoder_model_merged{suffix}.onnx` + `embed_tokens{suffix}.onnx`, else single `model{suffix}.onnx`.
- **Session:** Level3, 2–8 intra-threads. CUDA EP with `error_on_failure`; CPU fallback only when `use_gpu && !cuda_required`. Deliberately **not** `with_env_allocators` so a TTL unload returns VRAM.
- **ChatML template** pre-fills a **closed empty think block** `"<think>\n\n</think>\n\n"` (`:475-482`) — the quantized model ignores the soft `/no_think` and would otherwise burn the whole budget inside an unterminated `<think>`.
- **Decode budget:** `min((user_len + user_len/4 + 48), max_new_tokens 512)` — output bounded by input length +25% +48. IO-binding loop keeps KV on-device; logits to CPU for the sampler.
- **Sampler is deterministic greedy in production:** `temperature 1.0, top_k 1, top_p 1.0, repetition_penalty 1.0, all penalties identity` (`:308-318`). *(Verified.)* `top_k==1` + identity penalties → fused-argmax fast path; the full softmax/top-p/min-p machinery in `sampler.rs` is **never exercised in production**. Chat defaults (temp 0.7/presence 1.5) are explicitly noted as destructive here (presence penalty punishes re-emitting input → paraphrase drift).
- **KV cache:** GPU = lossless device-resident frozen system-prefix KV bound by reference. CPU = Hadamard-rotated, 4-bit quantized prefix (`sink_tokens 4` unquantized). Prefix invalidated by a hash of the system-prompt text; empty system (styling) skips caching so it never evicts the correction prefix.
- **Output:** `strip_think_tags` returns text after `</think>` (empty if it starts with an unterminated `<think>`).

### 6.4 Anti-hallucination guards
1. `correction_needed` gate (skip the LLM when predicted no-op).
2. `contains_forbidden_output` — rejects meta-text (`"here is"`, `"corrected transcript"`, `"input:"`, `"output:"`, `"possibly misheard"`, …) on both passes.
3. **`validate_and_merge`** (correction only) — the core validator:
   - Word-level LCS DP on normalized tokens.
   - **Reject-to-input gates:** < **55%** of input words survived → return input unchanged; word count moved > ~50% → return input.
   - **Case preservation:** ASR casing wins for interior-uppercase tokens (GitHub, NASA) and sentence-initial caps unless the LLM legitimately merged sentences.
   - **Substitution control:** accepted only for English, only if `is_allowed_sub` (curated homophone/contraction table) or `looks_like_spelling_repair` (Levenshtein ≤1 for ≥4 chars, ≤2 for ≥7). `BLOCKED_SUBS` hard-blocks grammar-damaging expansions (`your→youre`, `there→theyre`, `their→theyre`). Non-English → all subs rejected.
   - **Insertion control:** only `SAFE_INSERTIONS` function words, English only, only with following input (drops trailing function-word hallucinations); negations excluded from safe insertions.
   - Net: the correction pass **can never output less-trustworthy text than its seeded input** — every failure degrades to passthrough.

### 6.5 Style presets (`style_llm.rs`)
6 builtins: `minimal` (correction-only, `skip_styling: true` → `needs_styling()` false), `casual`, `professional`, `formal`, `technical`, `writing` (removes fillers 'so'/'just'/'basically'/'like'). Each carries 2–3 hardcoded few-shot pairs injected into the styling prompt. Default `default_style = "minimal"` (correction-only). ⚠️ There is **no separate Rust filler-removal/capitalization pass** in this crate — filler removal exists only as the `writing` preset's prompt instruction; capitalization is enforced inside `validate_and_merge` and the sentence-cap backstop.

### 6.6 Other backends
`NullPostProcessor` returns raw text verbatim (`complete` always errors) — the default when `llm_backend = Disabled` (which is the `UserSettings::default()`, though the checked-in TOML uses `Local`). OpenAI backend: chat completion, default `gpt-4o-mini`, `max_tokens 512`, `temperature 0.2`; graceful degradation handled at the pipeline level.

### 6.7 HWCE — word confidence estimation (`lune-core/src/hwce/`)
Produces the `word_error_risk ∈ [0,1]` consumed by the corrector and gate. Per-token features from the top-16 logits: one softmax, **normalized Shannon entropy** `H/ln(k)`, **Tsallis confidence** (α=1/3), **top1−top2 margin**. Per-word features: mean/min token entropy, mean margin, length, rare-casing, proper-noun candidate, nearby entropy spike (>0.35 within ±2 tokens), repetition (adjacent equal token ids). Scoring: weighted sum (IR weights summing to 1; proper-noun is *negative* −1/15) → `sigmoid(scale·(raw − 0.28))`, `scale = 4·ln(3) ≈ 4.39`. Length risk is a Gaussian penalty (μ=5, σ=3). Degenerate spans get a fixed risk 0.5.

---

## 7. Finalize & Delivery

### 7.1 Sentence-cap backstop & finalize (`session.rs:1098-1170`)
After the LLM, `capitalize_sentence_starts` runs as a **deterministic backstop** so any sentence the model left lowercase still gets its initial capital. Then:
- `FinalizedText(post_llm_text)` event emitted (`:1118`) — this is the authoritative delivery payload.
- The `(raw_text, final)` pair is written to the `correction_pairs` table (`:1133`, best-effort) as labelled calibration data for the homophone subsystem (§5.5, §7.5).
- `finalize_db` awaited **inline** so a runtime shutdown can't drop the transcript; the sqlite row gets status/`ended_at`/transcript.
- `Stopped` event + `clear_active`.

(The shared `recent_text` rolling-context buffer that previously lived here was removed alongside the LLM context field.)

Cancellation is re-checked via `drain_processing_controls` at two points (after the gate, after the LLM); `Stop` during processing means "wrap up and deliver", only `Cancel` discards (`:404-413`, regression tests `:1730-1757`).

### 7.2 DeliveryService (`crates/lune-app/src/delivery.rs`)
⚠️ **Push-based, not event-subscribing.** It owns a dedicated OS thread and a `mpsc` queue. The `SessionEvent::FinalizedText → deliver_to` wiring lives in `desktop-ui/src-tauri/src/runtime.rs:446-481`, which reads `dictation_mode` from settings and passes the session-start-captured `focus_target`. **`SessionEvent::CorrectedText` is UI-only — explicitly NOT delivered** (`runtime.rs:475-481`).

**Mode → method** (`delivery_thread` `:103-160`):
- `DictationMode::TypeInPlace` (the **default**) → `type_keys` = **per-key synthetic typing**, not clipboard.
- `DictationMode::Both` → clipboard copy + per-key type.
- `Clipboard` / catch-all → `type_text` = **clipboard-paste chord**.

**Feedback-loop guard:** on dequeue, `busy_until = now + 3600s`; reset to `now + DELIVERY_SETTLE (300ms)` after the driver returns. `is_busy()` suppresses new dictation starts (runtime `:300-303,415-418`) so synthetic keystrokes landing on the focused toggle button don't drive a feedback loop.

### 7.3 uinput typer (`uinput_typer.rs`, Linux)
- **Device:** `evdev` `VirtualDevice` named `"lune-voice"`, key codes 1–254; 200 ms post-create settle. `/dev/uinput` access: if not writable, installs a udev `uaccess` rule via `pkexec` (GUI prompt), reloads/triggers udev, 500 ms settle, re-checks — one-time, then permanent.
- **Keymap:** US-QWERTY ASCII tables (letters with shift for upper, digits/symbols with shift variants, space/enter/tab). Non-ASCII is handled by **falling back to the clipboard-paste path**: `is_keystroke_typeable` (`:767`) tests the string, and `type_str_keys` (`:966`) routes any text containing an unmappable char (accents, CJK, emoji, smart quotes) through `do_clipboard_paste` instead of dropping it. The raw per-key emitter is the last resort only when the clipboard itself is unavailable, where an unmappable char is logged and skipped (`type_char` `:986`).
- **Per-key path:** 30 ms modifier settle, then key press/release + **2 ms per char**.
- **Paste chord:** clipboard set (`wl-copy` / `xclip`, 2 s timeout) timed for an adaptive EMA delay (`clamp(rtt_ema·2, 40, 300) ms`), then Ctrl+V (or **Ctrl+Shift+V** in terminal mode). Clipboard failure falls back to per-key typing.

### 7.4 Focus & app identity
`capture_focus_target()` runs at **session start** (off the critical path via `spawn_blocking`; the COSMIC/Wayland probes block ~400-500 ms) and is stored, then reused at delivery. Probe order: Hyprland → Sway → generic Wayland (`wlr-foreign-toplevel`, `wayland_toplevel.rs`) → COSMIC (`zcosmic-toplevel-info`, `cosmic_toplevel.rs`) → X11. ⚠️ The generic-Wayland and COSMIC paths carry **no `focus_token`**, so `restore_focus` is a no-op there. Terminal detection normalizes the app-id/class against ~46 known terminals → drives Ctrl+Shift+V.

### 7.5 Persistence (`crates/lune-persist/`)
sqlite-backed `Database`. Session rows written on start (`insert_session` `:284`) and finalize (`update_session` `:402`). A separate **`correction_pairs`** table (migration `00000000000003`) stores `(raw ASR, final delivered)` per session: `insert_correction_pair` (`:423`) caps the table at the newest 5000 rows, and `list_correction_pairs` (`:452`) feeds the homophone calibrator. Both `delete_session` and the day-based `prune_sessions` cascade to `correction_pairs`, so deleting/pruning history leaves no transcript text behind. All persist failures during a session are non-fatal `Warning` events — the transcript still reaches the active app. `session_history_days` (default 30) bounds retention.

---

## 8. End-to-End Data Flow (condensed)

```
hotkey
  → SessionManager.start_inner: resolve style, insert DB row, spawn run_session
  → capture loop (10ms tick):
       AudioStream → 1280-sample chunks → VAD(Earshot+PCEN, 256-frame)
       speech?  yes → inject gated pause (zeros) + flush 4-chunk preroll → asr.push_audio
                no  → accumulate silence + buffer preroll
       emit AudioLevel @15fps; starvation/deadline/max-speech guards
  → Stop → tail-flush producer → Processing event → asr.poll_final
  → Qwen3 batch decode (mel[1,128,3000] → encoder → audio-pad inline → greedy KV decode)
       → AsrSegment{text, mean/min conf, word_error_risk via HWCE, lang}
  → append_segments (drop conf<0.25)
  → router.route_full → strip commands (apply SwitchStyle live)
  → dict.scan + LexicalGate (replace/case/flag; no candidate gen)
  → punctuation.normalize (Tier-1 caps+period; set needs_llm)
  → bypass gate: needs_llm = (unresolved | punct-ambiguous | min_conf<0.65 | wants_style) & avg_conf≥0.30
       no  → text_after_heuristic (fast path)
       yes → CorrectionPipeline: dict-seed → correct(validate_and_merge) → style → strip_think
  → capitalize_sentence_starts backstop
  → FinalizedText event → DeliveryService → uinput type / clipboard paste into focused app
  → finalize_db (status/ended_at/transcript) → Stopped → clear_active
```

---

## 9. Findings & Risks

**Correctness / behavior**
1. **VAD is Earshot (energy/GMM-style, pure Rust)** — intentional; there is no learned/ONNX VAD model and the repo never referenced Silero. (An earlier "contradicts the documented pipeline" read was wrong — CLAUDE.md never claims a VAD model.) (`vad.rs:3`)
2. **Qwen3 ASR is fully batch (by design)** — the `poll_results`/`supports_streaming` trait surface was removed, leaving `poll_final` (`qwen3.rs:329`) as the single decode entry point. No partial/incremental text reaches the UI. The capture loop is still chunk-by-chunk "streaming-shaped," but the backend only decodes at finalize.
3. **No live spelling correction** — the orphaned `scorer.rs`/`candidate_source.rs` were **deleted**; the heuristic only replaces dictionary terms, fixes casing, and *flags* low-confidence words for the LLM. Correction quality on non-dictionary mishears rests entirely on the LLM pass. The new homophone resolver (§5.5) targets this gap but is not yet wired into the live path. (`lune-dict/lib.rs:1-7`)
4. ~~**ASCII-only output on Linux**~~ **(fixed)** — the uinput typer now routes any non-ASCII text through the clipboard-paste path (`is_keystroke_typeable` / `type_str_keys`, `uinput_typer.rs:767,966`) instead of dropping characters, so multilingual output types correctly.
5. **`AudioStreamHandle::is_offline()` hardcoded false** — a handle over an offline stream can't signal drain-complete. (`lib.rs:75-77`)

**Robustness**
6. Three silent audio-drop points under load (ring eviction, resample backlog cap, resampler-error clear).
7. Generic-Wayland/COSMIC focus paths have no re-focus token → `restore_focus` is a no-op; delivery relies on focus not changing between session start and finalize.
8. `timestamp_ms` is processing-time, not capture-time.

**Dead / inconsistent code (low risk, hygiene)**
9. The corrector no longer carries the n-gram model — the dead `#[allow(dead_code)]` field was removed (and `HeuristicCorrector` is now `LexicalGate`); the trained model still drives homophone calibration at startup (§5.5).
10. `lune-core::AudioCaptureConfig` / `AudioFrame` parallel/legacy types disagree with the live capture path (480 vs 1280 samples, 0.01 vs 0.5 thresholds).
11. Stale test names: `default_has_skip_threshold_at_0_95` asserts `Some(0.65)`; a pipeline comment says "40% LCS gate" while code is 55%; `BASE_SYSTEM_PROMPT` `{lang}` token removed but `.replace("{lang}", …)` callers remain (no-ops).
12. `LlmBackendConfig::default() = Local` vs `UserSettings::default() = Disabled` — divergent defaults reconciled only by the checked-in TOML (`Local`).

**Strengths worth noting**
- The homophone resolver's per-user **conformal calibration** (§5.5) bounds over-correction with a finite-sample guarantee and auto-disables sets that don't help a given user — a principled answer to "when is it safe to correct." (Built, not yet live-wired.)
- The bypass gate + skip-when-broken design (arXiv:2509.25048) and the layered `validate_and_merge` guards make the LLM strictly safe — output never degrades below the heuristic fast path.
- Cancellation safety is carefully engineered (active-slot assignment last; `stop_and_wait` TOCTOU fix; Stop-vs-Cancel semantics during processing) and regression-tested.
- Pause re-injection reconstructs exact prosody gaps (proven sample-accurate by tests) so VAD gating doesn't cost the model its punctuation evidence.

---

## 10. Verification

- This revision re-grounds the changed stages against `fc5eb1e`: ASR trait slimmed to `poll_final` (`asr.rs:152`); `lune-dict` modules `corrector/freq/homophone/homophone_calib/lexicon/punctuation/vocab` with `scorer.rs`/`candidate_source.rs` deleted; `PostProcessInput` has no `context` field (`llm.rs:12-26`); uinput non-ASCII clipboard fallback (`:767,966`); `correction_pairs` table + `insert_correction_pair` (`:423`).
- Unchanged stages (audio capture, VAD/PCEN, mel front-end, decoder internals) retain their original citations; deep qwen3 internal anchors are within ~4 lines after the `poll_results` removal.
- Qwen3 still greedy argmax + `MAX_NEW_TOKENS 512` (`:35`) + `EOS_IDS` (`:37`); LLM sampler still greedy (`top_k 1`, `temperature 1.0`, `repetition_penalty 1.0`). The conformal over-correction bound is empirically verified by `homophone_tune` EXP 4.
```
VERIFICATION: grep cross-checks at `fc5eb1e` — asr.rs (poll_final only), lune-dict/lib.rs (homophone modules; scorer/candidate_source absent), qwen3.rs (poll_final:329, greedy), llm.rs (no context field), uinput_typer.rs (is_keystroke_typeable), persist/db/mod.rs (correction_pairs) → all confirmed.
```
