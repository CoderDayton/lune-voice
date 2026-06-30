> **HISTORICAL DOCUMENT — PRE-MIGRATION ARCHITECTURE**
>
> This audit describes the **Qwen3-ASR ONNX / Silero-streaming VAD / Qwen3-0.6B ONNX correction** pipeline
> that was in production before the `feat/whisper-cpp-backend` migration. All code it references
> (Qwen3-ASR ONNX, the `crates/lune-app/src/vad/` hybrid gate, `ort_llm.rs`, `kv_cache.rs`,
> `sampler.rs`, `mel/`) has been **deleted** from the repository.
>
> For the current architecture, see [docs/ARCHITECTURE.md](ARCHITECTURE.md).
> The pre-migration code is preserved on the `archive/onnx-qwen` branch.

---

# Lune Voice — ASR Pipeline Audit (Historical: Qwen3-ASR/ONNX)

**Scope:** Full dictation pipeline, audio capture → output delivery. Offline, local, Linux.
**Method:** Source read of every stage with `file:line` grounding; key claims independently verified by grep.
**Pipeline (actual):** `Mic (cpal) → downmix/resample → ring buffer → Hybrid VAD (Earshot acoustic gate → Silero neural verify → contiguous span gate; PCEN AGC) → preroll + intra-utterance pause preservation → ASR (Qwen3-ASR ONNX, batch greedy) → voice-command routing (whole-utterance) → dictionary + heuristic correction → formatting/punctuation → bypass gate → LLM correct/style (Qwen3-0.6B ONNX, penalized greedy) → sentence-cap backstop → delivery (IBus commit / uinput type / clipboard paste) → persist`.

All paths relative to repo root. Citations re-grounded against `main` @ `3e8d213` (earlier cuts: first written `98e782d`, re-verified `fc5eb1e`). Stages changed since the last cut are re-read line-by-line; deeper internal anchors in unchanged stages may be ±4 from intervening edits. **Material changes folded in since `fc5eb1e`:** a hybrid two-stage VAD with a bundled Silero ONNX model (the old "no learned VAD" claim is reversed); the capture loop extracted to `capture.rs`; the bypass gate extracted to `gate.rs` with a ≥2-word attention band; `punctuation.rs` renamed `formatting.rs` and `lune-commands` folded into `lune-dict`; the voice-command grammar reduced to whole-utterance matching; an in-process IBus delivery backend (preferred over uinput on Linux); the LLM sampler switched from plain greedy to **penalized greedy**; the ASR system prompt rewritten + a prompt-echo segment-drop; the session pipeline's args bundled into `SessionDeps`; the deterministic shaping chain extracted to `correct_and_format`; and `session_id` correlation threaded end-to-end through delivery.

---

## 0. Top-level orchestration (`crates/lune-app/src/session.rs`)

`SessionManager<A: AsrBackend>` is the conductor. Only **one session active at a time**; enforced by `active: Arc<Mutex<Option<ActiveHandle>>>` (`session.rs:79`, guard at `:169-172`).

**Lifecycle.** `start()` / `start_with_audio()` / `start_with_handle()` all funnel into `start_inner` (`:135`):
1. Resolve style preset **before** taking the active lock — a missing style is a config error, not a session-state error (`:143-159`). Fallback chain: requested id → custom styles → `"minimal"` → hard error.
2. Take active lock; reject if already `Some` (`:169-172`).
3. Insert session row into sqlite (`:177-190`); persist failure is surfaced as a non-fatal `Warning`, not fatal.
4. Build a `SessionDeps` bundle and spawn `run_session` + a watchdog task that forces cleanup + emits `Error("internal session panic")` if the session task panics (`:191-237`).
5. Assign `*lock = Some(...)` last — **cancellation-safe**: if `start_inner` is dropped at any await before this, the slot stays `None` and a later start succeeds (`:235-240`).

**`SessionDeps` (`:346-358`).** The per-session shared handles are bundled into one struct (`session_id`, `router`, `post_processor`, `dict`, `persist`, `heuristic`, `config`, `event_tx`, `active_arc`, `idle_notify`, `settings`) instead of being threaded positionally. `run_session(deps, asr, style_preset, control_rx, audio_override)` (`:360`) and `process_and_deliver(deps, raw_text, segments, status, style_preset, control_rx, cancelled)` (`:564`) now take it; `process_and_deliver` destructures it back to locals at `:573-585`. Both `#[allow(clippy::too_many_arguments)]` are gone (param counts dropped 15→5 and 17→7).

**Control plane.** `stop()` (graceful), `stop_and_wait()` (registers the `Notify` *before* sending Stop to avoid a TOCTOU race, `:249-276`), `cancel()` (discard). Signals travel over an unbounded mpsc to the session task. Mid-processing cancellation is re-checked via `drain_processing_controls` (def `:301-310`) at two points (after the gate `:721`, after the LLM `:775`); `Stop` during processing means "wrap up and deliver", only `Cancel` discards.

**Constants** moved out of `session.rs` when the capture loop was extracted:
- `MAX_SEGMENTS = 500` (`capture.rs:26`) — ring of accepted ASR segments; oldest dropped past the cap (`append_segments` `:30-66`).
- `GARBAGE_DROP_THRESHOLD = 0.25` (`capture.rs:28`) — segments with `confidence < 0.25` are dropped (`:42-51`).
- `ASR_START_TIMEOUT = 15s` (`capture.rs:97`) — hard ceiling on backend init.
- The old `PREROLL_CHUNKS` / `PAUSE_TOTAL_MIN_SAMPLES` / `PAUSE_TOTAL_CAP_SAMPLES` constants are **deleted** — preroll and pause handling moved into the hybrid VAD (`vad/config.rs`: `preroll_ms 500`, `long_gap_ms 2000`; §2–§3).

---

## 1. Audio Capture (`crates/lune-audio/src/{capture,ring}.rs`, `lune-core/src/audio.rs`)

**Canonical rate:** `SAMPLE_RATE = 16_000` (`lune-core/src/audio.rs:4`). All downstream math assumes 16 kHz mono f32 in [-1, 1].

### 1.1 Backend & device
- **cpal**, `default_host().default_input_device()` → `AudioError::NoDevice` if absent (`capture.rs:223-224`). **No device enumeration/selection** — always the system default input.
- Stream built at the **device native** sample rate and channel count, `BufferSize::Default` (`:233-253`). The app does *not* request 16 kHz from the device; it resamples in software.
- Sample formats: `F32` (passthrough), `I16` (`s/32768`), `U16` (`(s-32768)/32768`) via the `build_stream!` macro (`:296`); any other → `StreamBuild("Unsupported sample format")` (`:351-360`).

### 1.2 Per-callback processing chain (`process_audio_callback`, `:524-614`)
1. **Downmix to mono** (`downmix_mono_into`, `:123-217`). Each sample clamped to [-1,1] and zeroed if non-finite (`clean_sample`). Stereo/3–32ch path computes per-channel power; if imbalance `power_ratio > POWER_RATIO_LIMIT (16.0)` or antiphase (`stereo_corr < STEREO_CORR_FLOOR 0.1`), it **drops the weak channel** instead of averaging (avoids 12 dB-imbalance and antiphase cancellation; consts `:111-112`). >32 ch → plain average.
2. **Resample** (rubato `Async` poly, `Septic` degree, `FixedAsync::Input`) — built **only if native ≠ 16 kHz** (`:267-283`). Backlog capped at `MAX_RESAMPLE_BACKLOG = 48_000` samples (~1 s @48 kHz), oldest drained on overflow (`:548-556`). Resampler `process` error → **clears the input buffer** and breaks (`:576-582`).
3. **Chunk + push.** While `output_buf.len() >= chunk_samples`, drain exactly `chunk_samples` into an `AudioChunk { samples, timestamp_ms }` and `ring.push` (`:589-601`). `timestamp_ms = start_time.elapsed()` **at push time** (`:593`) — processing wall-clock, not sample-accurate capture time (drifts under backlog).

### 1.3 AudioConfig
`target_sample_rate = 16_000`, `chunk_duration_ms = 80`, `buffer_capacity = 128` (~10 s) (`:13-33`). **Chunk = 16000·80/1000 = 1280 samples / 80 ms.** Zero/overflow configs rejected by `validate_config` (`:49-70`).

### 1.4 Tail flush
`request_flush()` sets an atomic; the callback then `mem::take`s the sub-chunk remainder (< 1280 samples) and pushes it as a short final chunk (`:607-614`). Recovers the last phonemes on session stop. The capture loop calls this just after the loop, then waits ~30 ms for one callback cycle (`capture.rs`).

### 1.5 Ring buffer (`ring.rs`)
- `crossbeam_channel::bounded(capacity)` + `tokio::sync::Notify` for async wakeups (`:6-30`).
- **Overwrite-on-full:** `push` evicts the oldest chunk when full so the **newest always lands** (`:34-53`). The lib doc "no audio silently dropped" refers only to the new item — older chunks *are* evicted under consumer lag.
- `pop()` awaits `Notify` on empty (`:57-67`); `try_pop()` non-blocking (`:71-73`). SPSC-oriented (one audio producer, one session consumer).

### 1.6 AudioPull abstraction & offline path
`AudioPull` trait (`lib.rs:44-54`): `try_next_chunk`, `is_offline`, `request_flush`. Three session sources (`session.rs`): `Open` (fresh live stream), `Owned(AudioStream)` (file replay/tests), `Shared(AudioStreamHandle)` (pre-warmed mic; `drain_keep_last(6)` discards idle audio but keeps ~500 ms of hotkey-press window).
- `from_samples` builds an **offline** stream (no audio thread, `_stream: None`); `is_offline()` → true (`capture.rs:382-421`).
- ⚠️ **`AudioStreamHandle::is_offline()` is hardcoded `false`** (`lib.rs:75-77`) — a handle wrapping an offline stream cannot report drain-complete.

### 1.7 Capture-stage risks
1. Three independent audio-drop points under load: ring eviction, resample backlog cap, resampler-error buffer clear.
2. `timestamp_ms` is processing-time, not capture-time.
3. A parallel/legacy config type `lune-core::AudioCaptureConfig` (`audio.rs:48-71`, defaults `frame_size 480`, `vad_threshold 0.01`) is **unused** by the live path and disagrees with it — a confusion hazard.

---

## 2. Voice Activity Detection — hybrid two-stage gate (`crates/lune-app/src/vad/`, `crates/lune-audio/src/{vad,pcen,calibrate}.rs`)

⚠️ **Reversal of the previous audit:** the live VAD is **no longer Earshot-alone**. It is `HybridVad` (`vad/mod.rs`), a three-stage gate that adds a learned **Silero** ONNX model. The 2.3 MB model `crates/lune-app/data/silero_vad.onnx` is **baked into the binary** via `include_bytes!` (`vad/silero.rs:20`). `silero_enabled` defaults **true** (`vad/config.rs:64`). The earlier statement "nothing in the repo references Silero / no VAD model file" is **false** as of `3e8d213`. `lune-audio` itself remains ONNX-free (Earshot only) and is consumed as stage 1 — that narrower scoping is the only part of the old claim that still holds (`vad/mod.rs:1-5`).

The three stages (`vad/mod.rs:1-26`):
1. **Earshot** (stage 1, `lune-audio`) — cheap energy/GMM activity pre-gate (+ PCEN), high recall; also feeds the UI level meter.
2. **Silero** (stage 2, `vad/silero.rs`) — neural speech probability; rejects fans/keyboard/music/static Earshot lets through.
3. **Span gate** (stage 3, `vad/chunker.rs`) — a pure, model-free state machine that opens on onset and stays open through short pauses, closing only on a long trailing-silence run or a max-length cap.

### 2.1 Stage 1 — Earshot acoustic gate (`lune-audio/src/vad.rs`)
- Earshot (`use earshot::Detector; Detector::default()`, `vad.rs:3,99`) — pure Rust, no model.
- `HOP_SIZE = 256` samples = **16 ms @ 16 kHz** (`vad.rs:7-8`). Leftover samples between chunks are carried (`carry`, cap 256, `:135-167`) so framing is continuous; `vad_score` per chunk = **max frame score** (`:153-163,188`).
- `VadConfig` defaults (`vad.rs:37-51`): `threshold 0.5`, `threshold_end 0.3`, `hangover_ms 240`, `pcen Some(default)`. Validation: thresholds ∈ [0,1], `threshold_end ≤ threshold` (`:68-85`); `hangover_samples = ms·16` floored at `HOP_SIZE` (`:86-88`).
- ⚠️ **The hybrid gate consults Earshot's *raw* `vad_score`, not its hysteresis** (`vad/mod.rs:172`): `earshot_active = vad_score ≥ earshot_activity_threshold (0.35)`, held open `gate_hold_frames` after the last active frame. Earshot's own `threshold`/`threshold_end` start/end state machine is effectively bypassed for segmentation; only PCEN, framing, and the score still feed forward. The legacy `PipelineConfig::speech_threshold` no longer drives the gate at all (`vad/config.rs:from_pipeline`, below).

### 2.2 Stage 2 — Silero neural verify (`vad/silero.rs`)
- Silero VAD v6, run on CPU via `ort`. `SILERO_FRAME = 512` samples = **32 ms @ 16 kHz** (`:23`); `CONTEXT = 64` (`:29`); LSTM state `[2,1,128]`. Input is `[1, CONTEXT + SILERO_FRAME]` per frame.
- `predict(frame)` returns speech probability ∈ [0,1]; a failed call is **fail-safe** — logged and treated as silence (`vad/mod.rs:182-188`). Gate-skipped frames still call `observe(frame)` to advance the rolling context so the next verified frame isn't misaligned (`mod.rs:190`).
- When `silero_enabled` is off, the gate falls back to Earshot-only by collapsing the raw gate to a 0/1 probability into the same span gate (`mod.rs:193-196`).

### 2.3 Stage 3 — contiguous span gate (`vad/chunker.rs`)
Pure and model-free: consumes one Silero probability per 512-sample frame and emits `SpanEvent { Idle, Open, Append, Close { keep } }` (`chunker.rs:step` `:90-130`). **This is the word-fusion fix.**
- Hysteresis: `silero_onset (0.30) > silero_offset (0.20)` so brief dips don't end a span.
- **Contiguous:** while a span is open, *every* frame is appended — voiced **and** sub-`long_gap` silence — so intra-utterance pauses survive to the recognizer. A span closes only on a continuous `long_gap_ms (2000)` trailing-silence run (an intentional stop) or the `max_chunk_ms (20000)` cap. A voiced frame resets the long-gap timer.
- `keep` requires `voiced_frames ≥ min_speech_frames (min_speech_ms 100)` — noise blips drop; long dead-air between spans is discarded (a splice there is harmless, words seconds apart).
- The orchestrator (`mod.rs`) owns the audio: on `Open` it drains the preroll ring into the span (`:269-273`); `close_span` trims the trailing silence to the `hangover_ms (350)` window and emits `GateAction::PushSamples(span)` + `SegmentFinalized` only if kept (`:291-310`). Because ASR decode is batch (`poll_final` at session end), delaying per-span push costs no latency.

### 2.4 PCEN AGC (`lune-audio/src/pcen.rs`)
Per-hop adaptive gain normalizing each Earshot frame **in place before scoring**; diagnostic `energy` is computed on the **raw** input (`vad.rs:130`), independent of PCEN. Tracks signal + noise power envelopes (noise: asymmetric IIR, falls fast / rises slow), bounded by a target output level and a noise ceiling. Defaults (`pcen.rs:60-70`): `target_rms 0.10`, `noise_ceiling_rms 0.02`, `max_gain 12.0`, `min_gain 0.25`, `alpha 0.85`, `tau_signal 250 ms`, `tau_noise_up 5000 ms`, `tau_noise_down 200 ms`, `tau_gain 80 ms`, `warmup_frames 16`, `eps 1e-8`. Toggled by `UserSettings.vad_pcen` (default **true**); when off, `HybridVad::with_config` sets `earshot_config.pcen = None` (`vad/mod.rs`).

### 2.5 VadPipelineConfig & self-tuning (`vad/config.rs`)
`VadPipelineConfig` is the resolved gate config. `DEFAULT_VAD_CONFIG` (`:63-75`): `sample_rate 16000`, `silero_enabled true`, `earshot_prefilter true`, `earshot_activity_threshold 0.35`, `silero_onset 0.30`, `silero_offset 0.20`, `preroll_ms 500`, `hangover_ms 350`, `max_chunk_ms 20000`, `min_speech_ms 100`, `long_gap_ms 2000`. ms↔frame/sample helpers convert against `SILERO_FRAME` (`:90-140`).
- ⚠️ **`from_pipeline` ignores `PipelineConfig`** and returns `DEFAULT_VAD_CONFIG` verbatim (`:80-83`): the gate is self-tuned by `cgs bench TYPE=vad` against end-to-end WER. Mapping the legacy `speech_threshold` in regressed WER (0.50→16% vs 0.35→13.8%), so it is deliberately not wired. The acoustic knobs are WER-tuned; `long_gap_ms` is **not** (WER normalizes spacing away and is blind to the fusion bug) and is set to the intentional-stop threshold.

### 2.6 Earshot calibration (`lune-audio/src/calibrate.rs`)
Pure `recommend(CalibrationInput) → CalibrationResult` over two pre-recorded VAD-score vectors (quiet room, read prompt). Derives `speech_threshold` / `speech_threshold_end` (mapped onto the legacy `VadConfig`) plus `noise_p95`, `speech_p10`, `pause_p20`. Gates: noise ≥ 2 s, speech ≥ 8 s, separation `speech_p10 − noise_p95 ≥ 0.20` else `InsufficientSignal`; end forced below start by ≥ `HYSTERESIS_GAP (0.05)` (`calibrate.rs:8-22,82-104`). ⚠️ Since the hybrid gate is self-tuned and reads the raw score rather than Earshot's hysteresis (§2.1, §2.5), this calibrator's recommended thresholds **no longer steer the live segmentation** — it is effectively orphaned relative to `HybridVad`.

---

## 3. The Capture Loop & Gating (`crates/lune-app/src/capture.rs`)

The capture loop was **extracted out of `session.rs`** into `capture.rs::run_capture` (`:173`). It builds the runtime (audio source + ASR session + `HybridVad`), streams gate-verified audio into the batch ASR backend, and returns a `CaptureResult { segments, raw_text, speech_samples, cancelled, first_asr_error, saw_asr_success, status }` (`:156-164`).

**Loop structure** — `tokio::select!` with `biased` priority (`'capture: loop` at `:215`):
1. `control_rx.recv()` (highest) — Stop/Cancel break promptly even mid-stream.
2. `sleep_until(session_deadline)` — wall-clock guard = `3 × max_speech_seconds` (`:202-203`).
3. `audio_poll_tick.tick()` every 10 ms (Skip on miss, `:205-206`) — drains all available chunks.

**Per chunk:** `hybrid.process_chunk(&chunk)` reframes to Silero's 512-sample frames, runs the two-stage gate, and returns `Vec<GateAction>`. `apply_gate_actions` (`:71-91`) feeds each `PushSamples` to `asr.push_audio` (push errors recorded in `first_asr_error`, loop continues) and ignores `SegmentFinalized` (metrics only). `peak_energy` from the gate drives the level meter (`AudioLevel(min(peak·6,1.0))` via `try_send`, ~15 fps, `level_emit_interval 67ms`). `speech_samples` accumulates pushed samples; break at `max_speech_samples`.

**Pause handling is now implicit:** the old explicit zero-injection is gone. The span gate keeps intra-utterance silence *inside* the span (§2.3) and prepends a 500 ms preroll on onset, so Qwen3 still hears the real rhythm without the loop reconstructing it.

**Guards:** starvation — a live mic delivering no audio for 5 s → `Error("audio device stopped responding")` + `Failed("audio starvation")` (`:295-300`); offline streams exit cleanly on exhaustion (`:303-306`).

**After the loop:** tail-flush the producer; if a span is open, `hybrid.finish()` emits the final span (with the sub-frame tail appended after the trailing-silence trim). Emit `Processing`, then `asr.poll_final`. On cancel, `asr.reset_context` discards the buffer instead.

**Empty-output handling** (`session.rs:416-436`, post-capture): no speech / no segments / empty text — surface a held ASR error as `Failed`, else `Cancelled` (speech but no output) or silent `Completed` (no speech).

---

## 4. ASR — Qwen3-ASR ONNX (`crates/lune-asr/`)

### 4.1 Trait & buffering model
`AsrBackend` (`lune-core/src/asr.rs`) is **batch-only by design**: `poll_final` (`:150`) is the single transcription entry point — the earlier `poll_results` / `supports_streaming` surface is **removed**. Methods: `start_session` (`:142`), `push_audio` (`:144`), `poll_final` (`:150`), `end_session` (`:152`), defaulted `reset_context` (`:158-161`), `segment_capacity_hint` (`:165`). Qwen3 impl (`backend/qwen3.rs`):
- `SessionHandle = Mutex<Qwen3Session>` holding `pending: Vec<f32>`, `total_samples`, `sample_rate`, `next_id`, `forced_language` (`:159-167`).
- `push_audio` **appends raw PCM**; trims oldest beyond `MAX_WINDOW_SAMPLES = 300·16000 = 4_800_000` (5 min) (`:70`, push/trim `:333-340`).
- `poll_final` (`:344`) `mem::take`s the whole buffer and decodes once via `transcribe_pcm`. `reset_context` clears `pending` (`:365`); `end_session` is a no-op (`:371`).

So the session streams audio in but ASR produces **one segment per utterance** at finalize. No partial/incremental text reaches the UI; no dormant streaming machinery behind it.

### 4.2 Models & session options
Three ONNX sessions (`qwen3.rs:172-213`, loads `:186-189`): `qwen3-asr-encoder.onnx`, `qwen3-asr-embed-tokens.onnx`, `qwen3-asr-decoder.onnx` + HF `tokenizer.json`. Model dir `DIR_QWEN3_ASR = "qwen3-asr-0.6b"` (`models.rs:1`). Decoder weights **FP16**. Constants: `VOCAB_SIZE 151_936`, `NUM_LAYERS 28`, `NUM_KV_HEADS 8`, `HEAD_DIM 128` (`:26-29`).

Session build (`ort_utils.rs:~229-278`): `GraphOptimizationLevel::Level3`, 4 intra-threads, optimized-graph caching (`*.onnx.opt`), shared env CUDA allocators. **CUDA EP** only under `feature="cuda"`: heuristic conv search, `error_on_failure()` (no silent in-ORT CPU fallback). Explicit CPU retry one level up only when `use_gpu && !cuda_required`. Arena-shrink run options at prefill + a post-decode "janitor" pass to release KV high-water back to the driver.

### 4.3 Mel front-end (`mel/whisper.rs`)
Whisper-style, **native f16, mel-major**. `N_FFT 400`, `HOP_LENGTH 160`, `WIN_LENGTH 400`, `N_MELS 128`, 16 kHz (`:12-15`). Center reflect-pad by 200; periodic Hann; power spectrum over 201 bins (rustfft); **Slaney mel filterbank** (`fmin 0`, `fmax 8000`, area-normalized); `ln`-floor `1e-10`; Whisper log-norm `log10(v)/4 + 1` then clamp to `peak − 2.0`. Padded to `MEL_MIN_FRAMES = 3000` (30 s, local const at `qwen3.rs:735`) using an analytic silence frame, reshaped `[1,128,3000]` (`:738`), tensor name `"mel"` (`:755`). A boost-only loudness pre-norm toward `NORM_TARGET_RMS 0.08` (≤ `NORM_MAX_GAIN 10×`) runs first (`normalize_loudness` `:85-111`, invoked `:728`).

### 4.4 Decode loop
A **decoder-only multimodal LM**: the encoder produces audio embeddings **scattered into the text-embedding sequence at `<audio_pad>` (151676) positions** (not cross-attention).

Prompt layout (`build_prompt_ids` `:260-307`): ChatML `system` (`SYSTEM_PROMPT` + optional `"\nPreferred terms: …"` boost csv) + `user` block wrapping `AUDIO_START · n×AUDIO_PAD · AUDIO_END` + `assistant` header; if a language is forced, `"language {lang}"` tokens are appended so the model skips its own LID. Position ids = mRoPE `[3,1,seq]`, three identical `0..seq` rows for audio-only (`:566-573`).

**`SYSTEM_PROMPT`** (`:52`), verbatim:
> You are an ASR engine. Transcribe spoken audio exactly in the spoken language. Do not translate. Do not add words not clearly present.

Generation (`:888+`): **greedy argmax only** — no sampling/temperature/top-p (`argmax_on_logits` `:418-546`; two-pass path — argmax + top-k, then a vectorized log-sum-exp — with a fast branch when boost+ban are empty, constrained branch otherwise). KV-cached autoregressive, one token/step. **Stop:** `last ∈ EOS_IDS {151643,151645}` or `tokens ≥ MAX_NEW_TOKENS (512)` (`:901`; consts `:35,:37`).

**Prompt-echo drop** (changed): on low-information audio the model can regurgitate its own instruction. `is_prompt_echo(text)` (`:58-67`) trims leading whitespace and returns true if the text *starts with* `"You are an ASR engine"`, `"You are an automatic speech recognition"`, or `"Transcribe spoken audio"`. When `decode_output` (`:1114-1264`) sees an echo (`:1187`) it **drops the whole segment** (returns empty text → `Ok(vec![])`), rather than the old behavior of stripping a leaked prefix (which targeted a stale string that never matched the real prompt). Otherwise it splits on `<asr_text>` via `parse_auto_asr_output` (`:1269`) → `(detected_language, text)`; `bcp47_to_qwen3_language` (`:1307`) maps BCP-47 → Qwen labels.

### 4.5 Confidence
- **Per token:** `argmax_on_logits` runs two passes over the last position's logits — a max + argmax + top-16 pass over raw f16 logits, then a vectorized log-sum-exp pass using a branch-free `fast_exp_nonpos` (`:404`) that auto-vectorizes under `target-cpu=native` (`MABLE_TOP_K = 16`, `:393`). Token choice uses no `exp`, so it is bit-identical to the old fused loop. Confidence = softmax prob of the chosen token (raw even on the boost/ban path).
- **Per segment:** `mean_conf → AsrSegment.confidence`; `min_conf → min_token_confidence` (`:1191-1193`). The session layer is documented to later apply an isotonic calibrator turning these into `P(correct)` (`asr.rs:38-57`).
- **Per word (HWCE):** word spans built from token boundaries (space / `▁`), then `hwce::estimate` (`:1247`) produces `word_error_risk`; `WordConfidence.confidence = 1 − risk` (`:1258`). No CTC/attention aligner. (HWCE detail in §6.7.)

### 4.6 Repetition / garbage containment
1. **No-repeat-3-gram ban** (`REPEAT_GUARD_ORDER 3`, `:499`): masks tokens that would re-complete a seen 3-block (constrained argmax path; empty in the common case).
2. **Degenerate-cycle trim:** `repeated_cycle_len` (`:551-563`) detects an n-token block (n∈1..8) repeated 4× in the last 4n tokens (≥16 tokens), truncates `len − 3n`, breaks with a warn (`:907-916`).
3. Word-level `has_repetition` feeds HWCE risk.
4. Segment-level `GARBAGE_DROP_THRESHOLD 0.25` drop in the capture stage (`capture.rs:42-51`).

### 4.7 Timestamps & boost
Qwen3 emits **no internal timestamps**: `start_ms = None` (`:1105`), `end_ms = total_samples·1000/sample_rate` (`:350,:1106`). Boost (`set_boost_phrases` `:225-252`): dual-layer — phrases in the system prompt as "Preferred terms" **and** prefix-constrained additive logit bias (entry token at `score` `:517`, continuation at `2×score` on prefix match `:524`). Controlled by `word_boost_enabled` / `word_boost_score` (default off / 1.0 for Qwen3 per registry).

### 4.8 OpenAI-compatible ASR backend (`backend/openai.rs`)
Thin retrying HTTP client. Buffers f32; `poll_final` (`:149`) force-flushes. Encodes 16-bit mono WAV in memory, posts multipart `/audio/transcriptions` with `response_format=verbose_json`; 3 attempts, 2 s doubling backoff on 5xx/429; restores audio on error for retry. Per-segment `start/end` → ms; **no confidence/word signal**. Non-HTTPS rejected unless loopback.

### 4.9 Registry & pipeline defaults (`backend/registry.rs`)
2-variant enum (`Qwen3 | OpenAi`, `:71-73`). `from_config` resolves `use_gpu`/`cuda_required` from `DevicePolicy`. `default_pipeline_config` (`:177-221`):
- **Qwen3Asr:** `max_segment_ms / sentence_close_ms / hard_close_ms = 300_000`, `speech_threshold 0.5`, `speech_threshold_end 0.3`, `min_speech_ms 200`, `word_confidence_threshold Some(0.55)`, **`llm_skip_threshold Some(0.40)`** (the attention band needs ≥2 sub-threshold words; tuned by `benchmarks/tuning/gate_sweep.rs`), `llm_skip_noisy_threshold Some(0.30)`, `max_speech_samples 4_800_000` (300 s), `llm_timeout_secs 30`, `itn_enabled true`, `word_boost_enabled false`. *(These VAD-ish fields are largely vestigial for segmentation — the hybrid gate is self-tuned, §2.5 — but still describe segment-close behavior at the registry level.)*
- **OpenAiCompatible:** `max_segment_ms 30_000`, `sentence_close_ms 800`, `hard_close_ms 2_000`, all confidence/skip thresholds `None` (no logprobs), `max_speech_samples 120·16000`.

---

## 5. Post-Capture Processing (`session.rs::process_and_deliver`, `:564`)

Runs once per session on `(raw_text, segments)`. Order: voice commands → dictionary scan + heuristic correction → formatting → bypass gate → LLM → sentence-cap backstop → deliver + persist. The deterministic middle (heuristic correct → formatting → gate) is now one extracted, unit-tested function, `correct_and_format` (§5.4).

### 5.1 Voice command routing (`crates/lune-dict/src/commands.rs`)
⚠️ **Whole-utterance matching only** (folded in from the former `lune-commands` crate). `router.route_full(&raw_text)` → `RouteResult { commands, remaining }`, where a command fires **only if the entire utterance equals it**; otherwise nothing is stripped and `remaining` is the untouched original text (`commands.rs:81-84,110-118`).
- **Builtins** (`BUILTINS` `:32-39`): `stop dictation`, `stop listening`, `new paragraph`, `scratch that`, `list item`, `new line`. ⚠️ Bare `"stop"` is **deliberately not a command** (it is too common mid-dictation; regression `bare_stop_is_not_a_command` `:196`). Matching is case/punctuation-insensitive via `normalize_words` (`"Stop, dictation!"` == `"stop dictation"`).
- **Parametric prefixes**, longest-first (`:49-54`): `switch preset to `/`switch to ` → `SwitchStyle(arg)`, `expand snippet `/`insert ` → `InsertSnippet(arg)`. The arg is the **whole** normalized remainder (`:97`) — there is no `, . ;` terminator and no mixed command+dictation output.

`SwitchStyle` is applied **live**: the session swaps `style_preset` mid-pipeline (`session.rs:589-616`). Each command is emitted as `SessionEvent::CommandDetected { session_id, payload }` (`:619-622`). If `remaining` is empty after a whole-utterance command, the session finalizes with no delivery (`:625-638`).

### 5.2 Dictionary scan + heuristic correction
**Dictionary** (`lune-dict/src/lib.rs`; modules: `caprate, commands, corrector, edit_calib, formatting, freq, lexicon, ngram, vocab`, `lib.rs:1-10`). `DictionaryManager` holds a priority-DESC-sorted cache of `{phrase, replacement, priority}`.
- `top_relevant(text, 20)` → up to 20 entries (CI substring or Levenshtein-≤1 single-word). Forced-title-case replacements ("Dev", "Server") are **softened to lowercase before the prompt/seed** (`soften_forced_title_case`) so the sentence-cap backstop can re-capitalize correctly (`session.rs` dict block).
- `scan(text)` → `Vec<(matched, replacement)>` without mutating; multi-word = exact CI window match, single-word = Levenshtein-≤1 CI.

**LexicalGate** (`corrector.rs`, a unit struct with no n-gram field, `:225`), `correct_with_risk_threshold(text, word_risk_scores, dict_replacements, lang, flag_threshold, brand_casing)` — note the **6th `brand_casing` arg**:
- ⚠️ **No spelling-candidate generation in the live path.** It does dictionary replacement, canonical-casing, proper-noun/known-word preservation, and *flagging* only — it never rewrites a flagged word to a guess (emits the original verbatim, marks it for the LLM). The dormant `scorer.rs`/`candidate_source.rs` were deleted. The `homophone*` modules are a separate resolver subsystem (§5.5), not invoked by `LexicalGate`.
- Decision order per word (`process_single_word` `:58-200`): digit/empty passthrough → dictionary replacement → known&unflagged passthrough → canonical-casing override (github→GitHub, guarded) → known-word verbatim under `CANONICAL_OVERRIDE_RISK_CEIL 0.75` → proper-noun/acronym/camelCase preservation under `PROPER_NOUN_RISK_CEIL 0.80` → else flag as unresolved if `risk ≥ flag_threshold`.
- Thresholds: `HWCE_FLAG_THRESHOLD 0.40` (`:11`), canonical ceiling `0.75` (`:21`), proper-noun ceiling `0.80` (`:26`).
- Result: `CorrectionResult { corrected_text, word_risk_positions, unresolved_words, all_resolved, dropped_risk_entries }` (`:42-51`). Word↔risk alignment is a greedy forward cursor; non-aligning entries count as `dropped`.

Lexicon (`lexicon.rs`): tiered `lookup` over a frequency dict (`CONFIDENT_FREQ 5000`), curated supplemental `Vocab` (freq floor 10000, dev terms), and English Wordnik membership. 11 bundled frequency dictionaries (en/de/es/fr/it/ja/ko/nl/pt/ru/zh).

### 5.3 Formatting / punctuation (`lune-dict/src/formatting.rs` — renamed from `punctuation.rs`)
`normalize_with(text, fmt: &FormatToggles) → PunctuationOutcome { text, needs_llm }` (`:99`); `normalize(text)` still exists as a default-toggles wrapper (`:87`). `FormatToggles` (`style_llm.rs:74-103`): `keybinds`, `title_abbreviations`, `auto_capitalization`, `brand_casing` — all `bool`, all default **true**.
- Non-Latin → verbatim. Title abbreviations (mister→Mr.) when `title_abbreviations` and followed by a capitalized word.
- **Tier-1 deterministic:** `capitalize_sentence_starts` (when `auto_capitalization`) + append a terminal `.` for **declaratives only**.
- **`needs_llm` = run-on OR (interrogative-lead AND no terminal mark).** Run-on = `≥ RUN_ON_WORD_THRESHOLD (12)` words with no internal separator. Interrogative = WH-lead or AUX-lead + subject-aux inversion. (`brand_casing` is a toggle field but is applied by the corrector / `decase_common_words`, not inside `normalize_with`.)
- The shaping chain also uses `defuse_inconsistent` (de-fuse ASR run-ons like "MakeSure" before casing) and `decase_common_words` (downcase mid-sentence common-word caps), both English-only (§5.4).

### 5.4 `correct_and_format` + the Bypass Gate (`session.rs:455-541`, `crates/lune-app/src/gate.rs`)
The deterministic shaping chain was extracted from `process_and_deliver` into a pure, synchronous, unit-tested function `correct_and_format(text, segments, dict_replacements, heuristic, pipeline, fmt, wants_styling) -> Shaped` (`session.rs:475-541`), wrapped in `tracing::info_span!("correct_format", %session_id)` at the call site (`:674-685`). It runs: word-risk extraction + language detection → `heuristic.correct_with_risk_threshold` → `defuse_inconsistent` (English) → `formatting::normalize_with` → `decase_common_words` (English) → `gate::decide`, returning `Shaped { text_after_heuristic, detected_lang, needs_punct, gate, unresolved_words, dropped_risks }` (`:455-467`). The dictionary scan (an async lock) stays in `process_and_deliver` and passes the pre-scanned replacements in.

**The bypass gate** is now `crate::gate::decide(GateInputs) -> GateOutcome` (`gate.rs:42-84`). Three "needs correction" signals OR-ed:
1. `!all_resolved` — a word the heuristic could neither resolve nor confidently preserve.
2. `needs_punct` — ambiguous punctuation (run-on / interrogative without end mark).
3. ⚠️ **`conf_attention`** — fires only when **≥2 words** fall below `llm_skip_threshold` (`Some(0.40)`) (`gate.rs:57-63`), where a word's confidence is `1 − word_error_risk`. A single low-confidence word no longer escalates a resolved transcript (changed from the prior single-word rule).

Plus a **skip-when-broken veto** (`too_broken`): `avg_conf = 1 − mean_risk < llm_skip_noisy_threshold (0.30)` → too degraded; correction would *degrade* quality (arXiv:2509.25048), so the LLM is skipped (`gate.rs:70-71`). Thresholds come from `PipelineConfig`; if `None`, the gate is a no-op for that signal.

Final: `needs_correction = !all_resolved || needs_punct || conf_attention`; `needs_llm = (needs_correction || wants_styling) && !too_broken` (`gate.rs:72-73`). Styling is orthogonal — a clean transcript still gets a non-`minimal` style pass. A `GateDecision { used_llm, lang, unresolved }` event is emitted for the UI (`session.rs:713-719`). If `!needs_llm`, `text_after_heuristic` is the final output (fast path, `:725-727`).

### 5.5 Homophone resolver + conformal calibration (`lune-dict/{homophone,homophone_calib}.rs`) — built, NOT yet live-wired
A self-evaluating homophone disambiguator exists but is **not called from the live correction path** — `process_and_deliver`/`LexicalGate`/`correct_and_format` never invoke it. Fully built, tested, exercised at startup (calibration computed + logged), pending integration.
- **Resolver** (`homophone.rs`): 15 curated, tokenizer-distinct confusion sets (`CONFUSION_SETS` `:52-98` — e.g. `there/their/they're`, `to/too/two`; apostrophe-colliding pairs like `it's/its` deliberately excluded `:40-42`). `ResolverPolicy` (`:102-112`): `Off`, `RerankArgmax` (ungated — over-corrects), `MarginGate { tau }` (default **2.0**, `max_word_risk 0.85`, `:137-139`). A flip fires only when all gates pass: confusion-set membership, n-gram avg-log-prob margin > τ, trigram context support, word HWCE risk below the ceiling, optional contextual-gain. `best_member_margin` (`:358`) is shared with the calibrator so offline and live decisions match.
- **Calibration** (`homophone_calib.rs`): per-user, per-set self-evaluation over `(raw → final)` history. `calibrate` (`:44`) replays the resolver and sets each set's margin threshold via **Conformal Risk Control** (`conformal_threshold` `:137`; Angelopoulos et al., arXiv:2208.02814) — `DEFAULT_ALPHA = 0.01` (`:24`). **Selective enablement** (reject option): a set is enabled only if net benefit is positive under a Wilson lower bound (`:153`); an **isotonic** margin→P(correct) curve (PAVA, `:119,:167`) is fitted alongside.
- **Wiring today:** `runtime.rs` trains the n-gram on history, calls `calibrate`, and **logs** per-set enable/τ/net decisions at startup. The resolver is not yet placed in the correction path.
- **Benchmarks:** `benchmarks/homophone_tune.rs` (synthetic, validates the conformal α-guarantee) and `benchmarks/homophone_pipeline.rs` (real sqlite DB; compares integration strategies).

---

## 6. LLM Correction & Styling (`crates/lune-llm/`)

Invoked only when the gate sets `needs_llm`. Wrapped in `tokio::time::timeout(llm_timeout_secs)` (default 30 s); on timeout or `Err`, the pipeline **falls back to `text_after_heuristic`** — output is never worse than the heuristic fast path. The LLM input (`PostProcessInput`, `lune-core/src/llm.rs:11-26`) carries `raw_text` (heuristic output), the resolved style preset, up to 20 dictionary entries, detected `language`, and `correction_needed`. ⚠️ There is **no `context` field** — the former cross-utterance rolling-tail was removed end-to-end; the LLM sees only the current utterance.

### 6.1 Two-pass transform (`pipeline.rs:73-145`)
`CorrectionPipeline` wraps any `LlmService`:
1. **Dictionary pre-seed** (`apply_dictionary_seed`, before the LLM) — longest-phrase-first, Unicode-CI, punctuation/whitespace preserved. Replacements arrive at the validator already done, not as LLM edits.
2. **Pass 1 — Correction**, gated on `correction_needed`. `llm.complete(system, user)` → `fix_all_number_spaces` → `contains_forbidden_output` guard → `validate_and_merge`. On `Err`, warns and falls back to seeded raw (styling still runs).
3. **Pass 2 — Styling**, gated on `style_preset.needs_styling()`. `llm.complete("", style_prompt)` (`:113`) — **empty system prompt** (bypasses the prefix-KV cache). Accepted only if non-empty and not forbidden output; **not** run through `validate_and_merge`. Errors keep the corrected text.

### 6.2 Prompts (`config/style_llm.rs`, `backend/ort_llm.rs`, `pipeline.rs`)
- `BASE_SYSTEM_PROMPT` (`style_llm.rs:16-49`): fixed, language-agnostic ASR-corrector instruction ("smallest edits", homophones only when unambiguous, preserve punctuation, no translate/paraphrase/reorder, no censoring, no conversational replies) + **6 inline few-shot examples** (`:38-49`).
- ⚠️ **Dictionary terms are NOT in the system prompt.** `build_system_prompt` (`pipeline.rs:44-52`) returns `BASE_SYSTEM_PROMPT.replace("{lang}", …)` only — the comment is explicit: dictionary terms "must never enter the system prompt, or every dictionary edit would invalidate the prefill cache." (The old "Dictionary terms appended as `- {phrase} → {replacement}`" line is gone.)
- **Correction user prompt:** `user_prompt_with_examples` (`ort_llm.rs:354-385`) → `"Correct the following {Language} transcript:"` + transcript + `"\n\n/no_think"`. ⚠️ The pipeline calls it with **no carrier-sentence examples** (`&[]`, `pipeline.rs:55-60`) — the small model echoed them (`benchmarks/tuning/dict_carrier_ab.rs`). The earlier "+ up to 6 dictionary carrier-sentence examples" and the "Earlier dictation context" block are both gone.
- **Styling user prompt:** self-contained, no system prompt — `build_styling_prompt` (`style_llm.rs:177-202`): "Rewrite … Output only your rewritten version." + style name + rules + few-shot pairs + transcript + `"\n/no_think"`.

### 6.3 ONNX backend (`backend/ort_llm.rs`) — Qwen3-0.6B
- **Model discovery:** quant priority `q4f16 > int8 > fp16 > q4 > quantized > ""` (`:1531`) — q4f16 deliberately outranks int8 (int8 leaks `<think>`/few-shot text and ignores `/no_think`, `:1525-1530`). Split export `decoder_model_merged{suffix}.onnx` + `embed_tokens{suffix}.onnx`, else single `model{suffix}.onnx`.
- **Session:** Level3, 2–8 intra-threads. CUDA EP with `error_on_failure`; CPU fallback only when `use_gpu && !cuda_required`. Deliberately **not** `with_env_allocators` (`:249`) so a TTL unload returns VRAM.
- **ChatML template** pre-fills a **closed empty think block** `"<think>\n\n</think>\n\n"` (`:473`) — the quantized model ignores the soft `/no_think` and would otherwise burn its budget inside an unterminated `<think>`.
- **Decode budget:** `min(user_len + user_len/4 + 48, max_new_tokens 512)` (`:545`, `DEFAULT_MAX_TOKENS :21`). IO-binding loop keeps KV on-device; logits to CPU for the sampler.
- ⚠️ **Sampler is "penalized greedy" in production** (changed from plain greedy). `SamplingParams::default()` (`sampler.rs:23-46`): `temperature 0.0, top_p 1.0, top_k 1, min_p 0.0, repetition_penalty 1.1, frequency_penalty 0.0, presence_penalty 0.5`. `ort_llm.rs` builds the sampler via `..Default::default()`, setting only `eos_token_id`/`stop_token_ids` from the probed model (`:303-307`); the old hardcoded greedy override is removed. Because the penalties are **non-identity**, `penalties_are_identity()` is false (`sampler.rs:140-148`), so the fused-argmax fast path is **not** taken — instead penalties are applied to the logits and, since `temperature ≤ 0`, the token is the post-penalty argmax (`sampler.rs:228-232`). Net: still deterministic, but argmax *after* the repetition/presence penalties demote already-emitted tokens (breaks repeat/echo loops). The softmax/top-p/min-p machinery in `sampler.rs` remains unexercised in production (temp 0 short-circuits before it). Rationale (doc comment `sampler.rs:24-32`): a grid search (`benchmarks/tuning/sampling_tune.rs`) found no sampled config reliably beats greedy (~3× more likely to hurt), and `rep > ~1.1` degrades quality, so 1.1 is the ceiling. *(Verified: `benchmarks/tuning/sampling_ab.rs`, `benchmarks/tuning/sampling_tune.rs`.)*
- **KV cache:** GPU = lossless device-resident frozen system-prefix KV bound by reference. CPU = Hadamard-rotated 4-bit quantized prefix (`sink_tokens 4` unquantized). Prefix invalidated by a hash of the system-prompt text; empty system (styling) skips caching so it never evicts the correction prefix.
- **Output:** `strip_think_tags` (`:1576-1582`) returns text after `</think>` (empty if it starts with an unterminated `<think>`).

### 6.4 Anti-hallucination guards
1. `correction_needed` gate (skip the LLM when predicted no-op).
2. `contains_forbidden_output` (`:1190-1204`) — rejects meta-text (`"here is"`, `"corrected transcript"`, `"the corrected"`, `"i changed"`, `"i fixed"`, `"input:"`, `"output:"`, `"possibly misheard"`) on both passes.
3. **`validate_and_merge`** (correction only) — the core validator:
   - Word-level LCS DP on normalized tokens.
   - **Reject-to-input gates:** `< 55%` of input words survived → return input (`:1255`); word count moved > ~50% either way → return input (`:1258`).
   - **Case preservation:** ASR casing wins for interior-uppercase tokens (GitHub, NASA) and sentence-initial caps unless the LLM legitimately merged sentences (`:1340-1349`).
   - **Substitution control:** English only, only if `is_allowed_sub` (curated homophone/contraction table) or `looks_like_spelling_repair` (Lev ≤1 for ≥4 chars, ≤2 for ≥7, `:1184-1188`). `BLOCKED_SUBS` (`:1153-1154`) hard-blocks `your→youre`, `there→theyre`, `their→theyre`. Non-English → all subs rejected.
   - **Insertion control:** only `SAFE_INSERTIONS` function words (`:1016-1022`), English only, only with following input; negations are not in the safe set.
   - Net: the correction pass **can never output less-trustworthy text than its seeded input** — every failure degrades to passthrough.

### 6.5 Style presets (`style_llm.rs`)
6 builtins (REGISTRY `:130-137`): `minimal` (correction-only, `skip_styling: true` → `needs_styling()` false), `casual`, `professional`, `formal`, `technical`, `writing` (removes fillers 'so'/'just'/'basically'/'like' — `:341`). Each carries 2–3 hardcoded few-shot pairs injected into the styling prompt. ⚠️ There is **no separate Rust filler-removal/capitalization pass**: filler removal is the `writing` preset's prompt instruction only; capitalization is enforced inside `validate_and_merge` and the sentence-cap backstop.

### 6.6 Other backends
`NullPostProcessor` returns raw text verbatim (`complete` always errors) — the default when `llm_backend = Disabled` (the `UserSettings::default()`, though the checked-in TOML uses `Local`). OpenAI backend (`backend/openai.rs:25-27`): chat completion, default `gpt-4o-mini`, `max_tokens 512`, `temperature 0.2`; graceful degradation at the pipeline level.

### 6.7 HWCE — word confidence estimation (`lune-core/src/hwce/`)
Produces the `word_error_risk ∈ [0,1]` consumed by the corrector and gate. Per-token features from the top-16 logits: one softmax, **normalized Shannon entropy** `H/ln(k)`, **Tsallis confidence** (α=1/3), **top1−top2 margin**. Per-word: mean/min token entropy, mean margin, length, rare-casing, proper-noun candidate, nearby entropy spike (>0.35 within ±2 tokens), repetition. Scoring: weighted sum (IR weights summing to 1; proper-noun is *negative* −1/15) → `sigmoid(scale·(raw − 0.28))`, `scale = 4·ln(3) ≈ 4.39`. Length risk is a Gaussian penalty (μ=5, σ=3). Degenerate spans get a fixed risk 0.5.

---

## 7. Finalize & Delivery

### 7.1 Sentence-cap backstop & finalize (`session.rs:752-848`)
After the LLM, `capitalize_sentence_starts` runs as a **deterministic backstop** so any sentence the model left lowercase still gets its initial capital — ⚠️ now **conditional on `fmt.auto_capitalization`** (`:755-759`). Then:
- `SessionEvent::FinalizedText { session_id, text }` emitted (`:791-796`) — the authoritative delivery payload; it carries the originating `session_id` so the delivery path stays correlated end-to-end.
- The `(raw_text, final)` pair is written to the `correction_pairs` table (`:806-814`, best-effort) as labelled calibration data for the homophone subsystem (§5.5, §7.5).
- `finalize_db` awaited **inline** (`:828-848`) so a runtime shutdown can't drop the transcript; the sqlite row gets status/`ended_at`/transcript.
- `Stopped` event + `clear_active`.

(The shared `recent_text` rolling-context buffer that previously lived here was removed alongside the LLM context field.)

### 7.2 DeliveryService (`crates/lune-app/src/delivery.rs`)
⚠️ **Push-based, not event-subscribing.** Owns a dedicated OS thread + an `mpsc` queue. The `SessionEvent::FinalizedText → deliver` wiring lives in `desktop-ui/src-tauri/src/runtime.rs:447-490`, which reads `dictation_mode` and passes the session-start-captured `focus_target`. `SessionEvent::CorrectedText` is **UI-only — explicitly NOT delivered** (`runtime.rs:491-497`).
- **Session correlation (new):** `DeliveryMsg` carries `session_id: Option<Uuid>` (`:18`); `deliver(session_id, text, mode, copy_to_clipboard)` / `deliver_to(…)` / `deliver_owned(…)` take it (`:48,:67,:95`); the delivery thread logs `debug!(session_id = ?msg.session_id, …, "delivery: typing")` (`:156`). The runtime forwarder passes `Some(session_id)` for the dictation `FinalizedText` (`:458,:463,:472`) and for the command-injection newline/bullet deliveries (`:581-608`). So one dictation's `session_id` is traceable from "first audio chunk" through to the OS delivery thread.

**Mode → method** (`delivery_thread`):
- `DictationMode::TypeInPlace` (the **default**) → `type_keys` = **per-key synthetic typing** (or IBus commit, below).
- `DictationMode::Clipboard` / catch-all → `type_text` = clipboard-paste chord (or IBus commit).
- Orthogonal `copy_to_clipboard` toggle (`DeliveryMsg.copy_to_clipboard`): when **off**, the clipboard-paste path snapshots the prior clipboard and restores it byte-for-byte (text or image) after pasting, so dictation never clobbers it; when **on**, the dictated text is kept and the redundant re-copy is skipped. The legacy `Both` mode was removed (it maps to `TypeInPlace` + copy).

**Feedback-loop guard:** on dequeue `busy_until = now + 3600s`; reset to `now + DELIVERY_SETTLE (300ms)` after the driver returns (`:27,:158,:186`). `is_busy()` suppresses new dictation starts (runtime `:300-303,415-418`) so synthetic keystrokes landing on the focused toggle button can't drive a feedback loop.

### 7.3 Delivery backends (Linux: IBus → uinput; macOS/Windows: arboard+enigo)
⚠️ **New: an in-process IBus input-method backend, preferred on Linux** (`delivery.rs:198-272`). `LinuxBackend::Ibus(IbusTyper)` is used when IBus is the active input method (`crate::ibus::is_active_im()` → `IbusTyper::try_new()` succeeds, `:220-226`); it commits Unicode straight into the focused input context via `commit_text` — no terminal detection, no paste chord (fixes GNOME Wayland, where window/terminal detection is blocked). `LinuxBackend::Uinput(UinputTyper)` is the universal fallback (`platform_name` → `"linux-ibus"` / `"linux-uinput"`). `LUNE_DISABLE_IBUS=1` pins uinput. (IBus engine internals live in `crates/lune-app/src/ibus.rs`.)

**uinput typer (`uinput_typer.rs`, Linux fallback):**
- `evdev` `VirtualDevice` named `"lune-voice"`, key codes 1–254; 200 ms post-create settle. `/dev/uinput` access: if not writable, installs a udev `uaccess` rule via `pkexec` (GUI prompt), reloads/triggers udev, re-checks — one-time, then permanent.
- US-QWERTY ASCII keymap; **non-ASCII (accents/CJK/emoji/smart quotes) falls back to the clipboard-paste path** (`is_keystroke_typeable` / `type_str_keys`) so multilingual output types correctly; the raw per-key emitter (logs+skips unmappable chars) is the last resort only when the clipboard is unavailable.
- Per-key path: 30 ms modifier settle, then press/release + 2 ms/char. Paste chord: clipboard set (`wl-copy`/`xclip`, 2 s timeout) timed by an adaptive EMA delay (`clamp(rtt·2, 40, 300) ms`), then Ctrl+V (or Ctrl+Shift+V in terminal mode).

### 7.4 Focus & app identity
`capture_focus_target()` runs at **session start** (off the critical path via `spawn_blocking` — the COSMIC/Wayland probes block ~400-500 ms; `runtime.rs:322`), is stored (`:357-359`), and reused at delivery (`:453`). Probe order: Hyprland → Sway → generic Wayland (`wlr-foreign-toplevel`) → COSMIC (`zcosmic-toplevel-info`) → X11. ⚠️ Generic-Wayland and COSMIC paths carry **no `focus_token`**, so `restore_focus` is a no-op there; delivery relies on focus not changing between session start and finalize. Terminal detection normalizes the app-id/class against ~46 known terminals → drives Ctrl+Shift+V (uinput path; IBus needs no terminal detection).

### 7.5 Persistence (`crates/lune-persist/`)
sqlite-backed `Database`. Session rows written on start (`insert_session` `:284`) and finalize (`update_session` `:402`). A separate **`correction_pairs`** table (migration `00000000000003_add_correction_pairs`) stores `(raw ASR, final delivered)` per session: `insert_correction_pair` (`:423`) caps the table at the newest 5000 rows (`:431`); `list_correction_pairs` (`:452`) feeds the homophone calibrator. Both `delete_session` (`:462-470`) and the day-based `prune_sessions` (`:480-490`) cascade to `correction_pairs`, so deleting/pruning history leaves no transcript text behind. All persist failures during a session are non-fatal `Warning` events. `session_history_days` (default 30) bounds retention.

---

## 8. End-to-End Data Flow (condensed)

```
hotkey
  → SessionManager.start_inner: resolve style, insert DB row, build SessionDeps, spawn run_session
  → capture loop (capture.rs, 10ms tick):
       AudioStream → 1280-sample chunks → HybridVad.process_chunk
         stage1 Earshot (raw vad_score ≥ 0.35) → stage2 Silero (512-frame neural prob)
         → stage3 span gate (onset 0.30 / offset 0.20; keep intra-utterance pauses;
            close on 2 s long-gap or 20 s cap; drop < min_speech)
       → GateAction::PushSamples(span, preroll prepended) → asr.push_audio
       emit AudioLevel @15fps; starvation/deadline/max-speech guards
  → Stop → tail-flush + hybrid.finish() → Processing event → asr.poll_final
  → Qwen3 batch decode (mel[1,128,3000] → encoder → audio-pad inline → greedy KV decode;
       is_prompt_echo drops a regurgitated-prompt segment)
       → AsrSegment{text, mean/min conf, word_error_risk via HWCE, lang}
  → append_segments (drop conf<0.25)
  → process_and_deliver:
       router.route_full (whole-utterance commands; SwitchStyle applied live)
       → correct_and_format [span]: dict.scan + LexicalGate (replace/case/flag; no candidate gen)
            → defuse_inconsistent → formatting::normalize_with (Tier-1 caps+period; needs_llm)
            → decase_common_words → gate::decide
              needs_llm = (unresolved | punct-ambiguous | ≥2 words<0.40 | wants_style) & avg_conf≥0.30
       no  → text_after_heuristic (fast path)
       yes → CorrectionPipeline: dict-seed → correct(validate_and_merge) → style → strip_think
  → capitalize_sentence_starts backstop (if auto_capitalization)
  → FinalizedText{session_id,text} → runtime forwarder → DeliveryService(Some(session_id))
       → IBus commit  (or uinput type / clipboard paste) into focused app
  → insert_correction_pair → finalize_db (status/ended_at/transcript) → Stopped → clear_active
```

---

## 9. Findings & Risks

**Correctness / behavior**
1. **VAD is now a hybrid Earshot→Silero→span gate with a bundled Silero ONNX model** (`vad/silero.rs:20`, `data/silero_vad.onnx` 2.3 MB, `silero_enabled` default true). *This reverses the prior audit's "no learned/ONNX VAD" finding.* `lune-audio` alone stays ONNX-free (Earshot is stage 1). The contiguous-span gate keeps intra-utterance pauses inside a span, fixing the word-fusion ("MakeSure") bug; `long_gap_ms` is intentionally not WER-tuned (WER is blind to the fusion).
2. **The hybrid gate is self-tuned and ignores `PipelineConfig::speech_threshold`** (`vad/config.rs:from_pipeline`); the Earshot calibration subsystem (`calibrate.rs`) therefore no longer steers live segmentation — effectively orphaned.
3. **Qwen3 ASR is fully batch (by design)** — `poll_final` (`qwen3.rs:344`) is the single decode entry point; no partial/incremental text reaches the UI. The capture loop is chunk-by-chunk "streaming-shaped" but decode happens only at finalize.
4. **ASR prompt-echo is dropped, not stripped** — `is_prompt_echo` (`qwen3.rs:58`) discards a whole segment when the model regurgitates `SYSTEM_PROMPT` ("You are an ASR engine…").
5. **No live spelling correction** — `scorer.rs`/`candidate_source.rs` deleted; `LexicalGate` only replaces dictionary terms, fixes casing, and *flags* low-confidence words for the LLM. The homophone resolver (§5.5) targets this gap but is **not yet wired** into the live path.
6. **LLM sampler is penalized greedy** (`sampler.rs:23-46`: temp 0.0, rep 1.1, presence 0.5), not plain greedy — deterministic argmax *after* anti-repeat penalties. Sampling was empirically ~3× more likely to hurt than help (`benchmarks/tuning/sampling_tune.rs`).
7. **`AudioStreamHandle::is_offline()` hardcoded false** (`lib.rs:75-77`) — a handle over an offline stream can't signal drain-complete.

**Robustness**
8. Three silent audio-drop points under load (ring eviction, resample backlog cap, resampler-error clear).
9. Generic-Wayland/COSMIC focus paths have no re-focus token → `restore_focus` is a no-op; uinput delivery relies on focus not changing between session start and finalize. (IBus delivery sidesteps this — it commits to whatever input context is focused at delivery time.)
10. `timestamp_ms` is processing-time, not capture-time.

**Dead / inconsistent code (low risk, hygiene)**
11. `lune-core::AudioCaptureConfig` / `AudioFrame` parallel/legacy types disagree with the live capture path (480 vs 1280 samples, 0.01 vs 0.35/0.5 thresholds).
12. A `pipeline.rs` test comment still says "40% LCS gate" while the code enforces 55% (`ort_llm.rs:1255`).
13. `LlmBackendConfig::default() = Local` vs `UserSettings::default() = Disabled` — divergent defaults reconciled only by the checked-in TOML (`Local`).

**Strengths worth noting**
- End-to-end **`session_id` correlation** now spans the async session task, the Tauri event forwarder, and the OS delivery thread — one dictation is traceable mic→typed-output.
- The bypass gate + skip-when-broken design (arXiv:2509.25048), the ≥2-word attention band, and the layered `validate_and_merge` guards keep the LLM strictly safe — output never degrades below the heuristic fast path.
- The homophone resolver's per-user **conformal calibration** (§5.5) bounds over-correction with a finite-sample guarantee and auto-disables sets that don't help. (Built, not yet live-wired.)
- Cancellation safety is carefully engineered (active-slot assignment last; `stop_and_wait` TOCTOU fix; Stop-vs-Cancel semantics) and regression-tested.
- The contiguous-span VAD preserves real speech rhythm so VAD gating doesn't cost the recognizer its punctuation/word-boundary evidence — proven on the `cgs bench TYPE=vad` corpus (mean WER parity, better p90, ~5→~1.6 segments/clip).

---

## 10. Verification

- Re-grounded against `main` @ `3e8d213` by reading every changed stage in full: the new `crates/lune-app/src/vad/{mod,chunker,config,silero}.rs` hybrid gate; `capture.rs::run_capture`; `session.rs` `SessionDeps` + `correct_and_format` + struct-variant events; `gate.rs::decide`; `sampler.rs::default` (penalized greedy); `qwen3.rs` `SYSTEM_PROMPT`/`is_prompt_echo`; `delivery.rs` IBus backend + `session_id`; `lune-dict/lib.rs` module list; `commands.rs` whole-utterance grammar.
- Independent grep cross-checks confirmed: `data/silero_vad.onnx` present + `include_bytes!`, `silero_enabled: true`; `gate.rs` `>= 2` attention band; `registry.rs` `llm_skip_threshold: Some(0.40)`; `lib.rs` modules (`commands`, `formatting`, no `punctuation`); `delivery.rs` `LinuxBackend::Ibus`; `commands.rs` `bare_stop_is_not_a_command`; `HEAD = 3e8d213` on `main`.
- Stages not re-read line-by-line (mel front-end internals, HWCE math, ort session-builder, uinput keymap, persistence) retain re-grounded anchors from the section-by-section delta pass; deep internal citations may be ±4 lines from intervening edits.

```
VERIFICATION: grep + full-read cross-checks at 3e8d213 — vad/silero.rs:20 (include_bytes silero_vad.onnx), vad/config.rs:63-75 (DEFAULT_VAD_CONFIG), vad/chunker.rs (SpanGate contiguous), capture.rs:173 (run_capture), session.rs:346 (SessionDeps)/:475 (correct_and_format), gate.rs:57-73 (≥2-word band), sampler.rs:23-46 (penalized greedy), ort_llm.rs:303-307 (..Default), qwen3.rs:52 (SYSTEM_PROMPT)/:58 (is_prompt_echo)/:344 (poll_final), delivery.rs:198-272 (IBus)/:18 (session_id), lune-dict/lib.rs:1-10 (modules), commands.rs:32-39 (BUILTINS) → all confirmed.
```
