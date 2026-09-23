// SPDX-License-Identifier: GPL-3.0-or-later

//! Local whisper.cpp ASR backend.
//!
//! The real implementation (feature `whisper-cpp`) wraps [`whisper_rs`], which
//! compiles whisper.cpp. When the feature is off, a stub with the same public
//! surface keeps the registry and the rest of the crate compiling — its
//! constructor returns a clear "rebuild with the feature" error.
//!
//! Design notes (batch decoder, like Qwen3):
//! - The model ([`whisper_rs::WhisperContext`]) is loaded **once** at backend
//!   construction and shared by every session. Each session creates its own
//!   `WhisperState`, which allocates KV caches and compute buffers (hundreds
//!   of MB on the larger models, tens of ms to build).
//! - Pipeline audio is already 16 kHz mono `f32` — exactly whisper's input
//!   format — so `push_audio` is a buffer append with no resample or conversion.
//! - Each session owns its audio buffer; both `poll_final` and `reset_context`
//!   clear it (retaining capacity), so a reused session allocates nothing steady-state.
//! - whisper exposes per-token probabilities, which we fold into the
//!   `confidence` / `min_token_confidence` fields so the existing correction-LLM
//!   routing gate works (the remote OpenAI backend can't do this).

use std::path::PathBuf;

use lune_core::asr::AsrSessionConfig;
use lune_core::error::Result;

/// Construction parameters for [`WhisperCppBackend`].
pub struct WhisperCppConfig {
    /// Path to the GGML model file (`models/<dir>/ggml-*.bin`).
    pub model_path: PathBuf,
    /// Offload to GPU. Only meaningful when built with `whisper-cuda`; on a
    /// CPU-only build whisper-rs ignores it and runs on the CPU.
    pub use_gpu: bool,
    /// Path to the standalone silero VAD ggml model, used to endpoint-trim the
    /// buffer (drop leading/trailing silence) before decode. `None`/missing =>
    /// the full buffer is decoded.
    pub vad_model_path: Option<PathBuf>,
}

/// On-disk magic of a ggml model file: `0x67676d6c` ("ggml") stored
/// little-endian, so the first four bytes read `l m g g`.
const GGML_MAGIC: [u8; 4] = *b"lmgg";

/// Smallest plausible ggml model. Well under any real whisper model, and only
/// there to reject an empty or barely-started file with a clear message.
const MIN_GGML_BYTES: u64 = 4096;

/// Reject a model file whisper.cpp would die on rather than report.
///
/// Its loader raises a C++ exception or trips `GGML_ASSERT` on malformed input.
/// Neither can be caught from Rust — an exception crossing the `extern "C"`
/// boundary is undefined behaviour and the assert calls `abort()` — so the
/// process disappears with no dialog and no error. Validating first is the only
/// lever available.
///
/// This is a cheap sanity check, not an integrity check: the download path
/// already verifies a SHA-256 (`lune-persist`'s hub). What it catches is a file
/// truncated after that — a full disk, a crash mid-write, a hand-placed file.
pub fn check_ggml_file(path: &std::path::Path) -> Result<()> {
    use lune_core::error::Error;
    use std::io::Read;

    let meta = std::fs::metadata(path).map_err(|e| {
        Error::asr(format!(
            "cannot read the model file {}: {e}",
            path.display()
        ))
    })?;
    if meta.len() < MIN_GGML_BYTES {
        return Err(Error::asr(format!(
            "the model file {} is only {} bytes and cannot be a whisper model — \
             it is most likely a partial download; delete it and download the model again",
            path.display(),
            meta.len()
        )));
    }
    let mut head = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut head))
        .map_err(|e| {
            Error::asr(format!(
                "cannot read the model file {}: {e}",
                path.display()
            ))
        })?;
    if head != GGML_MAGIC {
        return Err(Error::asr(format!(
            "the model file {} is not in ggml format — delete it and download the model again",
            path.display()
        )));
    }
    Ok(())
}

/// How a decode runs, chosen once from the device the model runs on.
///
/// On a CPU the fixed 30 s encoder window dominates a short dictation, and
/// fitting it to the clip costs no accuracy. On a GPU the full window is
/// already fast, and flash attention (slower on a CPU) roughly halves encode
/// time and its buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
pub(crate) struct DecodeProfile {
    pub flash_attn: bool,
    /// Size the encoder window to the clip ([`fitted_audio_ctx`]) instead of
    /// whisper's fixed 30 s.
    pub fit_audio_ctx: bool,
}

#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
impl DecodeProfile {
    pub(crate) fn for_device(on_gpu: bool) -> Self {
        Self {
            flash_attn: on_gpu,
            fit_audio_ctx: !on_gpu,
        }
    }
}

/// whisper's full encoder window: 1500 frames of 20 ms, i.e. 30 s.
const AUDIO_CTX_FULL: usize = 1500;

/// Smallest fitted window, about 15 s. Smaller windows made whisper repeat
/// short clips ("Scratch that. Scratch that.") or fall back to its slow
/// temperature retries.
const AUDIO_CTX_MIN: i32 = 768;

/// Frames kept past the end of the clip, about 2.6 s.
const AUDIO_CTX_MARGIN: usize = 128;

/// 16 kHz input samples per 20 ms encoder frame.
const SAMPLES_PER_ENCODER_FRAME: usize = 320;

/// `audio_ctx` for a clip of `samples`: its length plus a margin, in whole
/// blocks of 64 frames, never below [`AUDIO_CTX_MIN`]. `0` (whisper's full
/// window) once that reaches 30 s.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
pub(crate) fn fitted_audio_ctx(samples: usize) -> i32 {
    let frames =
        (samples.div_ceil(SAMPLES_PER_ENCODER_FRAME) + AUDIO_CTX_MARGIN).next_multiple_of(64);
    if frames >= AUDIO_CTX_FULL {
        return 0;
    }
    (frames as i32).max(AUDIO_CTX_MIN)
}

/// Route whisper.cpp + ggml native logs into `tracing` instead of raw stderr,
/// so they obey the global log filter.
///
/// Idempotent, and must run before anything that can make ggml talk —
/// `gpu_backend::load_plugins` above all, whose `dlopen` failures are the only
/// account of why a GPU plugin did not load. On a GUI launch raw stderr goes
/// nowhere, so a diagnostic emitted before this call is lost outright.
#[cfg(feature = "whisper-cpp")]
pub fn install_logging_hooks() {
    static LOG_HOOK: std::sync::Once = std::sync::Once::new();
    LOG_HOOK.call_once(whisper_rs::install_logging_hooks);
}

/// No-op: the stub build links no whisper.cpp, so there is nothing to route.
#[cfg(not(feature = "whisper-cpp"))]
pub fn install_logging_hooks() {}

#[cfg(feature = "whisper-cpp")]
mod imp {
    use super::{
        AsrSessionConfig, DecodeProfile, Result, WhisperCppConfig, check_ggml_file,
        fitted_audio_ctx,
    };

    use std::sync::Mutex;

    use lune_core::asr::{AsrBackend, AsrSegment, WordConfidence};
    use lune_core::error::Error;
    use whisper_rs::{
        FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
        WhisperVadContext, WhisperVadContextParams, WhisperVadParams,
    };

    /// Primes the decoder's preceding-context with a short, well-punctuated
    /// sample so it carries that style into the transcript. whisper infers
    /// punctuation partly from this context; without it, long dictations decode
    /// as run-ons. Generic on purpose so it never leaks into short utterances.
    const PUNCTUATION_PROMPT: &str =
        "Okay, let's begin. Here are my notes, thoughts, and questions:";

    /// Cap decode threads. whisper.cpp scales sub-linearly past a handful of
    /// threads and we don't want to starve the audio/UI threads.
    const MAX_THREADS: usize = 8;

    /// Sample rate of whisper input audio (16 kHz mono f32).
    const SAMPLE_RATE: usize = 16_000;

    /// Temperature step for whisper's decode-fallback ladder. `0.5` allows one
    /// retry (0.0 then 0.5) instead of the upstream default of five (0.2 steps
    /// to 1.0), capping the worst case at two decodes rather than six.
    const TEMPERATURE_FALLBACK_STEP: f32 = 0.5;

    /// Padding kept around the VAD speech span when endpoint-trimming, so soft
    /// onsets/offsets aren't clipped. Asymmetric on purpose: the onset gets more
    /// lead-in than the tail — a bit of silence before the first word lets
    /// whisper anchor the opening phrase, which measurably recovers softly-spoken
    /// opening words (A/B: recovered a dropped opening word across quiet/loud and
    /// with pre-roll lead-in). The tail keeps the original margin; the
    /// producer-side flush in `capture.rs` already guards the very last phonemes.
    const VAD_PAD_LEAD_MS: usize = 400;
    const VAD_PAD_TAIL_MS: usize = 240;

    /// silero VAD probability threshold. Below whisper.cpp's 0.5 default so a
    /// soft first-word onset (fricatives/plosives, or audio still inside the
    /// AGC warm-up) is detected earlier and not clipped mid-word.
    const VAD_THRESHOLD: f32 = 0.35;

    /// Minimum voiced duration for a VAD segment (ms). Below whisper.cpp's 250
    /// default so short commands ("yes", "no", "stop", "undo") aren't discarded
    /// wholesale — that path returns zero segments and drops the whole utterance.
    const VAD_MIN_SPEECH_MS: i32 = 100;

    /// Padding whisper's own VAD keeps around each segment (ms). Raised from the
    /// 30 default for extra onset/offset lead-in; stacks with `VAD_PAD_MS`.
    const VAD_SPEECH_PAD_MS: i32 = 120;

    /// Skip the punctuation-priming prompt below this much decoded audio: the
    /// run-on problem it fixes only appears on long, multi-sentence dictation,
    /// while on a short clip the prompt is a strong bias that can inject a
    /// spurious continuation. 16 kHz mono → 6 s.
    const PROMPT_MIN_SAMPLES: usize = 6 * SAMPLE_RATE;

    /// Push a finished word's confidence — both the weakest token in the word
    /// (so a single garbled token flags the whole word) and the mean across
    /// the same tokens (so a downstream consumer can tell a uniformly-risky
    /// word apart from one outlier token dragging an otherwise-fine word
    /// down, e.g. a glued punctuation token — see `WordConfidence::mean_confidence`)
    /// — and reset the accumulators. Whitespace-only words are dropped.
    fn flush_word(
        words: &mut Vec<WordConfidence>,
        word: &mut String,
        word_min: &mut f32,
        word_sum: &mut f32,
        word_count: &mut u32,
    ) {
        let trimmed = word.trim();
        if !trimmed.is_empty() {
            let confidence = (*word_min).clamp(0.0, 1.0);
            let mean_confidence = if *word_count > 0 {
                (*word_sum / *word_count as f32).clamp(0.0, 1.0)
            } else {
                confidence
            };
            words.push(WordConfidence {
                word: trimmed.to_string(),
                confidence,
                word_error_risk: 1.0 - confidence,
                mean_confidence,
            });
        }
        word.clear();
        *word_min = 1.0;
        *word_sum = 0.0;
        *word_count = 0;
    }

    /// Run CPU/GPU-heavy whisper work without stalling the async runtime.
    ///
    /// Lune Voice runs on Tauri's multi-thread tokio runtime, so `block_in_place`
    /// hands the worker off to the blocking pool and other tasks keep running
    /// during the multi-second decode. There is no `spawn_blocking` here because
    /// the whisper session is `!Send` (it wraps raw C pointers) and cannot move
    /// to another thread. On a current-thread runtime (unit tests) or outside a
    /// runtime the closure runs inline — a current-thread *production* runtime
    /// would stall its single thread for the whole decode, so we warn loudly
    /// rather than freezing silently.
    fn run_blocking<T>(f: impl FnOnce() -> T) -> T {
        use tokio::runtime::{Handle, RuntimeFlavor};
        match Handle::try_current() {
            Ok(h) if h.runtime_flavor() == RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(f)
            }
            Ok(h) if h.runtime_flavor() == RuntimeFlavor::CurrentThread => {
                // Reachable only in tests. A current-thread production runtime
                // must not be used here — it would block its single thread for
                // the whole decode. Warn once so this is visible, then run
                // inline (block_in_place would panic on the current_thread flavor).
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        "whisper decode running on a current-thread runtime blocks the async \
                         runtime; Lune Voice must run on a multi-thread runtime"
                    );
                });
                f()
            }
            _ => f(),
        }
    }

    pub struct WhisperCppBackend {
        ctx: WhisperContext,
        n_threads: i32,
        profile: DecodeProfile,
        /// Standalone silero VAD for endpoint trimming (loaded once, reused).
        /// `None` => decode the full buffer (model absent or failed to load).
        vad: Option<Mutex<WhisperVadContext>>,
    }

    /// Mutable per-session state behind the handle's `Mutex`.
    pub struct WhisperCppSession {
        state: WhisperState,
        /// Reused across utterances — `poll_final`/`reset_context` clear it without freeing.
        audio: Vec<f32>,
        /// BCP-47 short code or `None` for auto-detect.
        language: Option<String>,
        /// Monotonic segment id, unique across `poll_final` calls in a session.
        next_id: u64,
    }

    impl WhisperCppBackend {
        #[cold]
        pub fn new(config: WhisperCppConfig) -> Result<Self> {
            // Normally already done at startup, before the plugin load. Kept
            // here so a caller that builds a backend without going through
            // that path still gets whisper's logs.
            super::install_logging_hooks();

            // Validate before handing the path to whisper.cpp. Its loader
            // signals a malformed file by throwing a C++ exception or tripping
            // `GGML_ASSERT`, neither of which Rust can catch — the process
            // simply dies. A file left truncated by a full disk or a crash
            // mid-write is a normal failure, so it has to be caught here, where
            // it can still be reported.
            check_ggml_file(&config.model_path)?;

            // whisper.cpp falls back to the CPU when asked for a GPU it cannot
            // find, so the profile follows the device it will actually use.
            let on_gpu = config.use_gpu && crate::gpu_backend::gpu_device_registered();
            let profile = DecodeProfile::for_device(on_gpu);
            let mut params = WhisperContextParameters::default();
            params.use_gpu(config.use_gpu);
            // whisper-rs defaults this off, overriding whisper.cpp's own default.
            params.flash_attn(profile.flash_attn);
            let ctx = WhisperContext::new_with_params(&config.model_path, params).map_err(|e| {
                Error::asr(format!(
                    "load whisper model {}: {e}",
                    config.model_path.display()
                ))
            })?;
            let n_threads = std::thread::available_parallelism()
                .map(|n| n.get().min(MAX_THREADS) as i32)
                .unwrap_or(4);
            // Load the standalone silero VAD once for endpoint trimming. A
            // missing/failed model degrades to full-buffer decode (warn once).
            let vad = config.vad_model_path.and_then(|path| {
                // Same loader, same failure mode as the main model: a truncated
                // silero file aborts the process rather than returning an
                // error, so validate before handing over the path. Unlike the
                // main model this degrades instead of failing — losing the
                // endpoint trim is survivable.
                if let Err(e) = check_ggml_file(&path) {
                    tracing::warn!(error = %e, "whisper VAD model unusable; decoding without endpoint trim");
                    return None;
                }
                let mut vp = WhisperVadContextParams::default();
                // Always run the VAD on CPU: silero is tiny (no GPU benefit) and
                // standing up a second CUDA context at init is a crash source.
                vp.set_use_gpu(false);
                vp.set_n_threads(n_threads);
                match WhisperVadContext::new(&path.to_string_lossy(), vp) {
                    Ok(v) => Some(Mutex::new(v)),
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to load whisper VAD; decoding without endpoint trim");
                        None
                    }
                }
            });
            tracing::info!(on_gpu, ?profile, "whisper decode profile");
            Ok(Self {
                ctx,
                n_threads,
                profile,
                vad,
            })
        }
    }

    #[async_trait::async_trait]
    impl AsrBackend for WhisperCppBackend {
        type SessionHandle = Mutex<WhisperCppSession>;

        async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
            // State allocation (KV cache and scratch buffers) is not cheap on
            // the larger models — keep it off the async workers too. A fresh
            // state per session is deliberate: a reused one decodes the same
            // audio differently depending on the previous dictation.
            let state = run_blocking(|| self.ctx.create_state())
                .map_err(|e| Error::asr(format!("whisper create_state: {e}")))?;
            Ok(Mutex::new(WhisperCppSession {
                state,
                // 30 s of 16 kHz audio up front — covers a typical utterance with
                // no realloc; longer dictations grow the buffer once.
                audio: Vec::with_capacity(16_000 * 30),
                language: config.language,
                next_id: 0,
            }))
        }

        async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
            session
                .lock()
                .map_err(|_| Error::asr("whisper session poisoned"))?
                .audio
                .extend_from_slice(pcm);
            Ok(())
        }

        async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
            let mut guard = session
                .lock()
                .map_err(|_| Error::asr("whisper session poisoned"))?;
            let WhisperCppSession {
                state,
                audio,
                language,
                next_id,
            } = &mut *guard;

            if audio.is_empty() {
                return Ok(Vec::new());
            }

            // VAD + decode are seconds of blocking compute; hand the worker
            // off so the UI/event tasks stay responsive during inference.
            run_blocking(|| {
                // BeamSearch is whisper.cpp's accuracy-oriented default (beam 5).
                // Worth the extra CPU here: decoding is one-shot per utterance, and
                // greedy decoding loses words beam search gets right ("ESLint" →
                // "ES and Lint"). A slow machine is better served by a smaller model.
                let mut params = FullParams::new(SamplingStrategy::BeamSearch {
                    beam_size: 5,
                    patience: -1.0,
                });
                params.set_n_threads(self.n_threads);
                // "auto" detects; explicit short codes (en, de, …) match the app's
                // dictation-language values one-to-one.
                params.set_language(Some(language.as_deref().unwrap_or("auto")));
                params.set_print_special(false);
                params.set_print_progress(false);
                params.set_print_realtime(false);
                params.set_print_timestamps(false);
                // Suppress non-speech tokens ("[music]", "(wind)") — pure hallucination
                // residue on quiet/short clips. (`suppress_blank` is already default-true.)
                params.set_suppress_nst(true);
                // Bound the temperature-fallback ladder. whisper.cpp's default
                // increment of 0.2 re-decodes the whole window at 0.0, 0.2 …
                // 1.0 whenever the average log-prob or entropy check fails —
                // up to six full decodes, on exactly the mumbled or noisy
                // utterances that are already the slowest. Dictation is
                // interactive and hotkey-bounded, so a bounded wait beats a
                // marginally better transcript arriving six times later. One
                // retry is kept: it recovers the common single-bad-decode case.
                params.set_temperature_inc(TEMPERATURE_FALLBACK_STEP);

                // Endpoint-trim with whisper's own (standalone) silero VAD: decode
                // only [first speech onset .. last offset] + padding as ONE
                // contiguous slice. Drops leading/trailing silence (less
                // hallucination; anchors the decode at the first word) while
                // preserving every INTERNAL pause — the cue whisper uses for
                // punctuation. We deliberately avoid the in-decode `enable_vad`,
                // which concatenates speech spans and strips those pauses → run-ons.
                enum VadTrim {
                    Speech(usize, usize),
                    Silence,
                    Full,
                }
                let trim = self
                .vad
                .as_ref()
                // Recover from a poisoned lock rather than silently decoding
                // without endpoint trimming for the rest of the process. The
                // VAD context is reset on every use, so a panic in a previous
                // decode leaves nothing behind worth protecting — and losing
                // the trim quietly shows up as hallucinated leading silence,
                // which is very hard to trace back to here.
                .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
                .map(|mut vad| {
                    let mut vad_params = WhisperVadParams::default();
                    vad_params.set_threshold(VAD_THRESHOLD);
                    vad_params.set_min_speech_duration(VAD_MIN_SPEECH_MS);
                    vad_params.set_speech_pad(VAD_SPEECH_PAD_MS);
                    match vad.segments_from_samples(vad_params, audio.as_slice()) {
                        Ok(segs) if segs.num_segments() > 0 => {
                            let last = segs.num_segments() - 1;
                            let start_cs = segs.get_segment_start_timestamp(0).unwrap_or(0.0);
                            let end_cs = segs.get_segment_end_timestamp(last).unwrap_or(0.0);
                            let to_sample = |cs: f32| (cs * SAMPLE_RATE as f32 / 100.0) as isize;
                            let pad_lead = (VAD_PAD_LEAD_MS * SAMPLE_RATE / 1000) as isize;
                            let pad_tail = (VAD_PAD_TAIL_MS * SAMPLE_RATE / 1000) as isize;
                            let start = (to_sample(start_cs) - pad_lead).max(0) as usize;
                            let end =
                                ((to_sample(end_cs) + pad_tail).max(0) as usize).min(audio.len());
                            VadTrim::Speech(start, end)
                        }
                        Ok(_) => VadTrim::Silence,
                        Err(e) => {
                            tracing::warn!(error = %e, "whisper VAD failed; decoding full buffer");
                            VadTrim::Full
                        }
                    }
                })
                .unwrap_or(VadTrim::Full);

                let decode_slice: &[f32] = match trim {
                    VadTrim::Speech(start, end) if start < end => &audio[start..end],
                    VadTrim::Silence => {
                        audio.clear();
                        return Ok(Vec::new());
                    }
                    _ => audio.as_slice(),
                };

                // whisper's segment timestamps are relative to `decode_slice`.
                // Endpoint trimming drops a lead-in (commonly hundreds of ms),
                // so add it back to keep `start_ms`/`end_ms` relative to the
                // original buffer the caller pushed.
                let trim_offset_ms = match trim {
                    VadTrim::Speech(start, end) if start < end => {
                        (start as u64 * 1000) / SAMPLE_RATE as u64
                    }
                    _ => 0,
                };

                // Prime punctuation/casing style only for longer dictation, where
                // whisper otherwise runs sentences together; on a short clip the
                // prompt biases the decode toward a spurious continuation, so skip it.
                if decode_slice.len() >= PROMPT_MIN_SAMPLES {
                    params.set_initial_prompt(PUNCTUATION_PROMPT);
                }
                if self.profile.fit_audio_ctx {
                    params.set_audio_ctx(fitted_audio_ctx(decode_slice.len()));
                }

                state
                    .full(params, decode_slice)
                    .map_err(|e| Error::asr(format!("whisper inference: {e}")))?;
                // Drain the decoded audio so a repeat poll can't re-emit it (matches
                // Qwen3's take-on-poll contract); capacity is retained for reuse.
                audio.clear();

                let detected =
                    whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(str::to_string);

                let n = state.full_n_segments();
                let mut out = Vec::with_capacity(n.max(0) as usize);
                for i in 0..n {
                    let Some(seg) = state.get_segment(i) else {
                        continue;
                    };
                    let text = seg.to_str_lossy().unwrap_or_default().trim().to_string();
                    if text.is_empty() {
                        continue;
                    }

                    // Fold per-token probabilities into the segment confidence AND
                    // group tokens into per-word confidences — whisper marks a new
                    // word with a leading-space token. Per-word values feed the
                    // corrector's risk gating (which protects uncertain words and
                    // drives `unresolved_words`); leaving them empty made that gating
                    // inert. Special tokens ("[_...]", p≈1.0) are skipped so they
                    // don't inflate the mean.
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    let mut min_p = 1.0f32;
                    let mut word_confidences: Vec<WordConfidence> = Vec::new();
                    let mut cur_word = String::new();
                    let mut cur_min = 1.0f32;
                    let mut cur_sum = 0.0f32;
                    let mut cur_count = 0u32;
                    for j in 0..seg.n_tokens() {
                        let Some(tok) = seg.get_token(j) else {
                            continue;
                        };
                        let Ok(piece) = tok.to_str_lossy() else {
                            continue;
                        };
                        if piece.starts_with("[_") {
                            continue;
                        }
                        let p = tok.token_probability();
                        sum += p;
                        min_p = min_p.min(p);
                        count += 1;
                        // A leading space begins a new word — flush the previous one.
                        if piece.starts_with(' ') {
                            flush_word(
                                &mut word_confidences,
                                &mut cur_word,
                                &mut cur_min,
                                &mut cur_sum,
                                &mut cur_count,
                            );
                        }
                        cur_word.push_str(&piece);
                        cur_min = cur_min.min(p);
                        cur_sum += p;
                        cur_count += 1;
                    }
                    flush_word(
                        &mut word_confidences,
                        &mut cur_word,
                        &mut cur_min,
                        &mut cur_sum,
                        &mut cur_count,
                    );
                    let (confidence, min_token_confidence) = if count == 0 {
                        (None, None)
                    } else {
                        (Some(sum / count as f32), Some(min_p))
                    };

                    let id = *next_id;
                    *next_id += 1;
                    out.push(AsrSegment {
                        id,
                        confidence,
                        min_token_confidence,
                        word_confidences,
                        is_final: true,
                        // whisper timestamps are in centiseconds (10 ms units),
                        // relative to `decode_slice` — add back the VAD-trim
                        // offset so callers see original-buffer time.
                        start_ms: Some(trim_offset_ms + (seg.start_timestamp().max(0) as u64) * 10),
                        end_ms: Some(trim_offset_ms + (seg.end_timestamp().max(0) as u64) * 10),
                        text,
                        detected_language: detected.clone(),
                    });
                }
                Ok(out)
            })
        }

        async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
            if let Ok(mut g) = session.lock() {
                g.audio.clear(); // retain capacity for the next utterance
            }
            Ok(())
        }

        async fn end_session(&self, _session: Self::SessionHandle) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(not(feature = "whisper-cpp"))]
mod imp {
    use super::{AsrSessionConfig, Result, WhisperCppConfig};

    use lune_core::asr::{AsrBackend, AsrSegment};
    use lune_core::error::Error;

    const DISABLED: &str =
        "whisper.cpp backend not compiled in — rebuild with --features whisper-cpp";

    /// Stub mirroring the real backend's surface so the registry compiles
    /// without the feature. Every entry point returns the "rebuild" error.
    pub struct WhisperCppBackend;

    impl WhisperCppBackend {
        #[cold]
        pub fn new(_config: WhisperCppConfig) -> Result<Self> {
            Err(Error::config(DISABLED))
        }
    }

    #[async_trait::async_trait]
    impl AsrBackend for WhisperCppBackend {
        type SessionHandle = ();

        async fn start_session(&self, _config: AsrSessionConfig) -> Result<Self::SessionHandle> {
            Err(Error::config(DISABLED))
        }
        async fn push_audio(&self, _session: &Self::SessionHandle, _pcm: &[f32]) -> Result<()> {
            Err(Error::config(DISABLED))
        }
        async fn poll_final(&self, _session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
            Err(Error::config(DISABLED))
        }
        async fn end_session(&self, _session: Self::SessionHandle) -> Result<()> {
            Ok(())
        }
    }
}

pub use imp::WhisperCppBackend;

#[cfg(test)]
mod tests {
    use super::{
        AUDIO_CTX_MIN, DecodeProfile, GGML_MAGIC, MIN_GGML_BYTES, check_ggml_file, fitted_audio_ctx,
    };
    use std::io::Write;

    const SECOND: usize = 16_000;

    #[test]
    fn a_gpu_decode_keeps_the_full_window_and_uses_flash_attention() {
        assert_eq!(
            DecodeProfile::for_device(true),
            DecodeProfile {
                flash_attn: true,
                fit_audio_ctx: false,
            }
        );
    }

    #[test]
    fn a_cpu_decode_fits_the_window_to_the_clip() {
        assert_eq!(
            DecodeProfile::for_device(false),
            DecodeProfile {
                flash_attn: false,
                fit_audio_ctx: true,
            }
        );
    }

    #[test]
    fn a_short_clip_is_encoded_in_the_minimum_window() {
        // Below about 15 s whisper starts repeating itself, so short
        // commands never get a smaller window than this.
        assert_eq!(fitted_audio_ctx(0), AUDIO_CTX_MIN);
        assert_eq!(fitted_audio_ctx(3 * SECOND), AUDIO_CTX_MIN);
    }

    #[test]
    fn a_longer_clip_gets_its_length_plus_a_margin() {
        // 20 s is 1000 encoder frames; the window must cover it with room to
        // spare, in whole blocks of 64 frames.
        let ctx = fitted_audio_ctx(20 * SECOND);
        assert!(ctx >= 1000 + 64, "{ctx} leaves no margin past the clip");
        assert_eq!(ctx % 64, 0);
        assert!(ctx < 1500);
    }

    #[test]
    fn a_clip_near_thirty_seconds_uses_the_full_window() {
        // 0 means whisper's own 30 s window.
        assert_eq!(fitted_audio_ctx(29 * SECOND), 0);
        assert_eq!(fitted_audio_ctx(90 * SECOND), 0);
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("lune-ggml-check-{name}"));
        let mut f = std::fs::File::create(&path).expect("create temp model");
        f.write_all(bytes).expect("write temp model");
        path
    }

    #[test]
    fn a_well_formed_model_passes() {
        let mut bytes = GGML_MAGIC.to_vec();
        bytes.resize(MIN_GGML_BYTES as usize + 1, 0);
        let path = write_temp("ok.bin", &bytes);
        assert!(check_ggml_file(&path).is_ok());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_truncated_download_is_rejected_by_size() {
        // What a full disk or a crash mid-write leaves behind. whisper.cpp
        // would abort the process on this, so it must never reach the loader.
        let path = write_temp("short.bin", &GGML_MAGIC);
        let err = check_ggml_file(&path).expect_err("a 4-byte model must be refused");
        assert!(
            err.to_string().contains("partial download"),
            "the message must tell the user what to do, got: {err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_file_that_is_not_ggml_is_rejected_by_magic() {
        // Right size, wrong contents — an HTML error page saved under the
        // model's name is the realistic version of this.
        let mut bytes = b"<!DOCTYPE html>".to_vec();
        bytes.resize(MIN_GGML_BYTES as usize + 1, b' ');
        let path = write_temp("html.bin", &bytes);
        let err = check_ggml_file(&path).expect_err("a non-ggml file must be refused");
        assert!(err.to_string().contains("not in ggml format"), "got: {err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_missing_model_is_reported_not_panicked() {
        let path = std::env::temp_dir().join("lune-ggml-check-absent.bin");
        let _ = std::fs::remove_file(&path);
        assert!(check_ggml_file(&path).is_err());
    }
}
