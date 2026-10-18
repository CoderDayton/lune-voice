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
    /// What is put in front of the first word before a decode.
    pub lead_in: LeadIn,
}

/// What a decode gets in front of the first word.
///
/// whisper anchors the opening phrase on the audio before it. The endpoint
/// trim keeps what was recorded there, which is nothing when speech starts
/// with the hotkey press.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeadIn {
    /// Only what was recorded before the onset.
    #[default]
    Recorded,
    /// Top a recorded lead-in shorter than `ms` up to `ms`, with noise at the
    /// level of the quietest part of the clip.
    AtLeast { ms: u32 },
    /// `ms` of digital silence ahead of the recorded lead-in.
    Silence { ms: u32 },
}

/// Loudest noise a top-up may use, about -54 dBFS. Keeps a clip with no quiet
/// stretch from being fronted with audible hiss.
const LEAD_NOISE_MAX_RMS: f32 = 0.002;

/// Frame the clip's noise floor is measured over: 20 ms at 16 kHz.
const LEAD_FLOOR_FRAME: usize = 320;

/// RMS of the quietest whole frame of `clip`, capped at [`LEAD_NOISE_MAX_RMS`].
fn noise_floor_rms(clip: &[f32]) -> f32 {
    clip.chunks_exact(LEAD_FLOOR_FRAME)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt())
        .fold(LEAD_NOISE_MAX_RMS, f32::min)
}

/// Samples to put in front of `clip` under `policy`; empty when it needs none.
///
/// `recorded_lead` is how many samples of `clip` come before the speech onset,
/// or `None` when the onset is unknown (no VAD), which leaves a top-up with
/// nothing to measure. The noise is seeded from a constant, so one clip always
/// decodes from the same samples.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
pub(crate) fn lead_in_prefix(
    policy: LeadIn,
    clip: &[f32],
    recorded_lead: Option<usize>,
) -> Vec<f32> {
    const SAMPLES_PER_MS: usize = 16;
    match policy {
        LeadIn::Recorded => Vec::new(),
        LeadIn::Silence { ms } => vec![0.0; ms as usize * SAMPLES_PER_MS],
        LeadIn::AtLeast { ms } => {
            let Some(recorded) = recorded_lead else {
                return Vec::new();
            };
            let missing = (ms as usize * SAMPLES_PER_MS).saturating_sub(recorded);
            let amplitude = noise_floor_rms(clip) * 3f32.sqrt();
            let mut state: u32 = 0x9E37_79B9;
            (0..missing)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
                    (unit * 2.0 - 1.0) * amplitude
                })
                .collect()
        }
    }
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

/// Longest stretch handed to whisper in one decode: 28 s. Past its 30 s window
/// whisper decodes the leftover on its own, and a leftover of a second or less
/// comes back as invented text ("Thank you.").
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const CHUNK_MAX_SAMPLES: usize = 28 * 16_000;

/// Shortest piece a cut may leave on either side: 3 s.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const CHUNK_MIN_SAMPLES: usize = 3 * 16_000;

/// Ranges of a `total`-sample clip to decode one by one, cut at pauses so no
/// piece runs past [`CHUNK_MAX_SAMPLES`] where avoidable — see
/// [`crate::backend::local::plan_chunks_within`] for the rule. `lead` samples
/// are decoded in front of the first piece and count against its window.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
pub(crate) fn plan_chunks(
    speech: &[(usize, usize)],
    total: usize,
    lead: usize,
) -> Vec<(usize, usize)> {
    crate::backend::local::plan_chunks_within(
        speech,
        total,
        lead,
        CHUNK_MAX_SAMPLES,
        CHUNK_MIN_SAMPLES,
    )
}

#[cfg(feature = "whisper-cpp")]
mod imp {
    use super::{
        AsrSessionConfig, DecodeProfile, LeadIn, Result, WhisperCppConfig, fitted_audio_ctx,
        lead_in_prefix, plan_chunks,
    };
    use crate::backend::local::{
        EndpointVad, SAMPLE_RATE, VadTrim, WordConfidences, check_ggml_file, decode_threads,
        ensure_cpu_backend, run_blocking,
    };

    use std::sync::Mutex;

    use lune_core::asr::{AsrBackend, AsrSegment};
    use lune_core::error::Error;
    use whisper_rs::{
        FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    };

    /// Primes the decoder's preceding-context with a short, well-punctuated
    /// sample so it carries that style into the transcript. whisper infers
    /// punctuation partly from this context; without it, long dictations decode
    /// as run-ons. Generic on purpose so it never leaks into short utterances.
    const PUNCTUATION_PROMPT: &str =
        "Okay, let's begin. Here are my notes, thoughts, and questions:";

    /// Temperature step for whisper's decode-fallback ladder. `0.5` allows one
    /// retry (0.0 then 0.5) instead of the upstream default of five (0.2 steps
    /// to 1.0), capping the worst case at two decodes rather than six.
    const TEMPERATURE_FALLBACK_STEP: f32 = 0.5;

    /// Skip the punctuation-priming prompt below this much decoded audio: the
    /// run-on problem it fixes only appears on long, multi-sentence dictation,
    /// while on a short clip the prompt is a strong bias that can inject a
    /// spurious continuation. 16 kHz mono → 6 s.
    const PROMPT_MIN_SAMPLES: usize = 6 * SAMPLE_RATE;

    pub struct WhisperCppBackend {
        ctx: WhisperContext,
        n_threads: i32,
        profile: DecodeProfile,
        lead_in: LeadIn,
        /// Standalone silero VAD for endpoint trimming (loaded once, reused).
        /// `None` => decode the full buffer (model absent or failed to load).
        vad: Option<EndpointVad>,
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
            crate::backend::local::install_logging_hooks();

            // Validate before handing the path to whisper.cpp. Its loader
            // signals a malformed file by throwing a C++ exception or tripping
            // `GGML_ASSERT`, neither of which Rust can catch — the process
            // simply dies. A file left truncated by a full disk or a crash
            // mid-write is a normal failure, so it has to be caught here, where
            // it can still be reported.
            check_ggml_file(&config.model_path)?;
            ensure_cpu_backend()?;

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
            let n_threads = decode_threads();
            // Load the standalone silero VAD once for endpoint trimming. A
            // missing/failed model degrades to full-buffer decode.
            let vad = config
                .vad_model_path
                .and_then(|path| EndpointVad::load(&path, n_threads));
            tracing::info!(on_gpu, ?profile, "whisper decode profile");
            Ok(Self {
                ctx,
                n_threads,
                profile,
                lead_in: config.lead_in,
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
                fn decode_params<'a>(n_threads: i32, lang: &'a str) -> FullParams<'a, 'a> {
                    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
                        beam_size: 5,
                        patience: -1.0,
                    });
                    params.set_n_threads(n_threads);
                    params.set_language(Some(lang));
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
                    params
                }

                // Endpoint-trim with the standalone silero VAD (see
                // `EndpointVad`): decode only the speech, keeping its pauses.
                let trim = self
                    .vad
                    .as_ref()
                    .map_or(VadTrim::Full, |vad| vad.trim(audio.as_slice()));
                let Some(slice) = trim.decode_slice(audio.len()) else {
                    audio.clear();
                    return Ok(Vec::new());
                };
                let decode_slice = &audio[slice.start..slice.end];

                // Endpoint trimming drops a lead-in; segment times add it back so
                // `start_ms`/`end_ms` stay relative to the buffer the caller pushed.
                let trim_offset_ms = slice.offset_ms();

                let recorded_lead = slice.onset.map(|onset| onset.saturating_sub(slice.start));
                let prefix = lead_in_prefix(self.lead_in, decode_slice, recorded_lead);
                let recorded_len = decode_slice.len();
                let prefix_ms = (prefix.len() as u64 * 1000) / SAMPLE_RATE as u64;

                let spans = slice.relative_spans();
                let chunks = plan_chunks(&spans, recorded_len, prefix.len());

                // "auto" detects on the first chunk only; later chunks reuse its answer.
                let wanted = language.as_deref().unwrap_or("auto");
                let mut detected: Option<String> = None;
                let mut out = Vec::new();
                for (index, &(from, to)) in chunks.iter().enumerate() {
                    let lang = match detected.as_deref() {
                        Some(found) if wanted == "auto" => found,
                        _ => wanted,
                    };
                    let mut params = decode_params(self.n_threads, lang);

                    // Prime punctuation/casing style only for longer dictation, where
                    // whisper otherwise runs sentences together; on a short clip the
                    // prompt biases the decode toward a spurious continuation, so skip it.
                    if recorded_len >= PROMPT_MIN_SAMPLES {
                        params.set_initial_prompt(PUNCTUATION_PROMPT);
                    }
                    let padded: Vec<f32>;
                    let piece: &[f32] = if index == 0 && !prefix.is_empty() {
                        padded = [prefix.as_slice(), &decode_slice[from..to]].concat();
                        &padded
                    } else {
                        &decode_slice[from..to]
                    };
                    if self.profile.fit_audio_ctx {
                        params.set_audio_ctx(fitted_audio_ctx(piece.len()));
                    }

                    state
                        .full(params, piece)
                        .map_err(|e| Error::asr(format!("whisper inference: {e}")))?;
                    if detected.is_none() {
                        detected = whisper_rs::get_lang_str(state.full_lang_id_from_state())
                            .map(str::to_string);
                    }

                    let piece_ms = trim_offset_ms + (from as u64 * 1000) / SAMPLE_RATE as u64;
                    let lead_ms = if index == 0 { prefix_ms } else { 0 };
                    let n = state.full_n_segments();
                    for i in 0..n {
                        let Some(seg) = state.get_segment(i) else {
                            continue;
                        };
                        let text = seg.to_str_lossy().unwrap_or_default().trim().to_string();
                        if text.is_empty() {
                            continue;
                        }

                        // Fold per-token probabilities into the segment confidence
                        // and per-word confidences (`WordConfidences`) — whisper marks
                        // a new word with a leading-space token. Special tokens
                        // ("[_...]", p≈1.0) are skipped so they don't inflate the mean.
                        let mut acc = WordConfidences::new();
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
                            acc.push(&piece, tok.token_probability(), piece.starts_with(' '));
                        }
                        let (word_confidences, confidence, min_token_confidence) = acc.finish();

                        let id = *next_id;
                        *next_id += 1;
                        out.push(AsrSegment {
                            id,
                            confidence,
                            min_token_confidence,
                            word_confidences,
                            is_final: true,
                            // whisper timestamps are in centiseconds (10 ms units),
                            // relative to the piece it decoded — add back that piece's
                            // place in the buffer so callers see original-buffer time.
                            start_ms: Some(
                                (piece_ms + (seg.start_timestamp().max(0) as u64) * 10)
                                    .saturating_sub(lead_ms),
                            ),
                            end_ms: Some(
                                (piece_ms + (seg.end_timestamp().max(0) as u64) * 10)
                                    .saturating_sub(lead_ms),
                            ),
                            text,
                            detected_language: detected.clone(),
                        });
                    }
                }
                // Drain the decoded audio so a repeat poll can't re-emit it (matches
                // Qwen3's take-on-poll contract); capacity is retained for reuse.
                audio.clear();
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
        AUDIO_CTX_MIN, CHUNK_MAX_SAMPLES, CHUNK_MIN_SAMPLES, DecodeProfile, LEAD_NOISE_MAX_RMS,
        LeadIn, fitted_audio_ctx, lead_in_prefix, plan_chunks,
    };

    const SECOND: usize = 16_000;
    const MS: usize = SECOND / 1000;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// A clip with a quiet room-tone stretch followed by a loud one.
    fn clip_with_floor(floor: f32) -> Vec<f32> {
        let mut clip: Vec<f32> = (0..SECOND / 2)
            .map(|i| if i % 2 == 0 { floor } else { -floor })
            .collect();
        clip.extend((0..SECOND).map(|i| if i % 2 == 0 { 0.3 } else { -0.3 }));
        clip
    }

    #[test]
    fn the_recorded_lead_in_is_decoded_as_is_by_default() {
        assert_eq!(LeadIn::default(), LeadIn::Recorded);
        let clip = clip_with_floor(0.001);
        assert!(lead_in_prefix(LeadIn::Recorded, &clip, Some(0)).is_empty());
    }

    #[test]
    fn a_long_enough_recorded_lead_in_gets_no_top_up() {
        let clip = clip_with_floor(0.001);
        let policy = LeadIn::AtLeast { ms: 400 };
        assert!(lead_in_prefix(policy, &clip, Some(400 * MS)).is_empty());
        assert!(lead_in_prefix(policy, &clip, Some(900 * MS)).is_empty());
    }

    #[test]
    fn a_short_recorded_lead_in_is_topped_up_to_the_minimum() {
        let clip = clip_with_floor(0.001);
        let policy = LeadIn::AtLeast { ms: 400 };
        assert_eq!(lead_in_prefix(policy, &clip, Some(0)).len(), 400 * MS);
        assert_eq!(
            lead_in_prefix(policy, &clip, Some(150 * MS)).len(),
            250 * MS
        );
    }

    #[test]
    fn a_top_up_without_a_known_onset_adds_nothing() {
        // No VAD means no onset, so there is no lead-in to measure.
        let clip = clip_with_floor(0.001);
        assert!(lead_in_prefix(LeadIn::AtLeast { ms: 400 }, &clip, None).is_empty());
    }

    #[test]
    fn the_top_up_is_noise_at_the_level_of_the_quietest_part_of_the_clip() {
        let clip = clip_with_floor(0.001);
        let prefix = lead_in_prefix(LeadIn::AtLeast { ms: 400 }, &clip, Some(0));
        let level = rms(&prefix);
        assert!((0.0007..0.0013).contains(&level), "level {level}");
        // Digital silence would put a step at the join.
        assert!(prefix.iter().any(|s| *s != 0.0));
    }

    #[test]
    fn the_top_up_never_gets_louder_than_the_noise_cap() {
        // A clip with no quiet stretch must not be fronted with loud noise.
        let clip = clip_with_floor(0.2);
        let prefix = lead_in_prefix(LeadIn::AtLeast { ms: 400 }, &clip, Some(0));
        assert!(rms(&prefix) <= LEAD_NOISE_MAX_RMS * 1.05);
    }

    #[test]
    fn the_top_up_is_the_same_on_every_decode() {
        let clip = clip_with_floor(0.001);
        let policy = LeadIn::AtLeast { ms: 400 };
        assert_eq!(
            lead_in_prefix(policy, &clip, Some(0)),
            lead_in_prefix(policy, &clip, Some(0))
        );
    }

    #[test]
    fn fixed_silence_is_added_whatever_the_recorded_lead_in() {
        let clip = clip_with_floor(0.001);
        let policy = LeadIn::Silence { ms: 3000 };
        for lead in [None, Some(0), Some(900 * MS)] {
            let prefix = lead_in_prefix(policy, &clip, lead);
            assert_eq!(prefix.len(), 3 * SECOND);
            assert!(prefix.iter().all(|s| *s == 0.0));
        }
    }

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

    /// Speech spans of `len` seconds each, separated by the given pauses (ms).
    fn speech_with_pauses(len: usize, pauses_ms: &[usize]) -> (Vec<(usize, usize)>, usize) {
        let mut spans = Vec::new();
        let mut at = 0;
        for pause in pauses_ms {
            spans.push((at, at + len * SECOND));
            at += len * SECOND + pause * MS;
        }
        spans.push((at, at + len * SECOND));
        (spans, at + len * SECOND)
    }

    fn assert_covers(chunks: &[(usize, usize)], total: usize) {
        assert_eq!(chunks.first().map(|c| c.0), Some(0));
        assert_eq!(chunks.last().map(|c| c.1), Some(total));
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "chunks must not overlap or skip");
        }
    }

    #[test]
    fn a_clip_that_fits_one_window_is_decoded_whole() {
        let (speech, total) = speech_with_pauses(5, &[300, 300, 300]);
        assert!(total <= CHUNK_MAX_SAMPLES);
        assert_eq!(plan_chunks(&speech, total, 0), vec![(0, total)]);
    }

    #[test]
    fn a_clip_just_past_the_window_is_cut_at_a_pause_with_no_sliver() {
        // 30.2 s: whisper alone would decode 30 s and invent text for the rest.
        let mut speech = vec![(0, 10 * SECOND), (10 * SECOND + 400 * MS, 20 * SECOND)];
        speech.push((20 * SECOND + 400 * MS, 30 * SECOND + 200 * MS));
        let total = 30 * SECOND + 200 * MS;
        let chunks = plan_chunks(&speech, total, 0);
        assert_eq!(chunks.len(), 2);
        assert_covers(&chunks, total);
        for (start, end) in &chunks {
            assert!(end - start <= CHUNK_MAX_SAMPLES);
            assert!(end - start >= CHUNK_MIN_SAMPLES);
        }
        // The cut sits in the middle of the pause, clear of both words.
        assert_eq!(chunks[0].1, 20 * SECOND + 200 * MS);
    }

    #[test]
    fn the_longest_pause_in_the_back_half_is_preferred() {
        // Pauses end at 6.1, 12.4, 19.3 and 25.4 s; the 900 ms one is a
        // sentence break, the later 100 ms one is a gap between words.
        let (speech, total) = speech_with_pauses(6, &[100, 300, 900, 100, 100]);
        let chunks = plan_chunks(&speech, total, 0);
        assert_covers(&chunks, total);
        let long_pause_mid = 18 * SECOND + 400 * MS + 450 * MS;
        assert_eq!(chunks[0].1, long_pause_mid);
    }

    #[test]
    fn a_long_dictation_is_cut_into_windows_that_all_fit() {
        let (speech, total) = speech_with_pauses(4, &[300; 20]);
        assert!(total > 80 * SECOND);
        let chunks = plan_chunks(&speech, total, 0);
        assert!(chunks.len() >= 3);
        assert_covers(&chunks, total);
        for (start, end) in &chunks {
            assert!(end - start <= CHUNK_MAX_SAMPLES);
            assert!(end - start >= CHUNK_MIN_SAMPLES);
        }
    }

    #[test]
    fn spans_that_touch_still_mark_a_pause_to_cut_at() {
        // The VAD pads each span, so a pause shorter than both pads leaves
        // two spans that meet with no gap between them.
        let total = 30 * SECOND + 400 * MS;
        let speech = [
            (0, 4 * SECOND),
            (4 * SECOND, 14 * SECOND),
            (14 * SECOND, total),
        ];
        assert_eq!(
            plan_chunks(&speech, total, 0),
            vec![(0, 14 * SECOND), (14 * SECOND, total)]
        );
    }

    #[test]
    fn a_lead_in_in_front_of_the_first_piece_counts_against_its_window() {
        let speech = [
            (0, 10 * SECOND),
            (10 * SECOND + 400 * MS, 27 * SECOND + 500 * MS),
            (27 * SECOND + 900 * MS, 40 * SECOND),
        ];
        let total = 40 * SECOND;
        let lead = 3 * SECOND;
        let chunks = plan_chunks(&speech, total, lead);
        assert_covers(&chunks, total);
        assert!(chunks[0].1 - chunks[0].0 + lead <= CHUNK_MAX_SAMPLES);
        for (start, end) in &chunks[1..] {
            assert!(end - start <= CHUNK_MAX_SAMPLES);
        }
    }

    #[test]
    fn speech_with_no_pause_is_left_to_whisper_in_one_piece() {
        let total = 40 * SECOND;
        assert_eq!(plan_chunks(&[(0, total)], total, 0), vec![(0, total)]);
    }
}
