// SPDX-License-Identifier: GPL-3.0-or-later

//! Local NVIDIA Parakeet TDT ASR backend.
//!
//! Runs whisper.cpp's `parakeet` library — the Parakeet TDT runtime that ships
//! in the same whisper.cpp tree as whisper itself and links the same ggml — so
//! it inherits the downloadable CUDA plugin, the CPU backend variants and the
//! log routing with no second native stack. `whisper-rs` does not wrap it, so
//! this module calls the raw bindings (`whisper_rs::whisper_rs_sys`) and keeps
//! every `unsafe` block behind a small owned-handle type.
//!
//! With the `whisper-cpp` feature off, a stub with the same public surface
//! keeps the registry compiling; its constructor returns a clear "rebuild with
//! the feature" error.
//!
//! How it differs from the whisper backend:
//! - **One decoder state, reused.** `parakeet_init_state` allocates the
//!   decoder state and its buffers, so the backend builds it once at load
//!   (the encoder graph is still re-planned whenever the clip length changes,
//!   inside the library). Reuse is
//!   safe because every decode runs with `no_context`, which clears the
//!   prediction network's LSTM state and the decoded tokens first — the same
//!   audio decodes the same way whatever was dictated before it.
//! - **Chunked at pauses, inside the model's window.** The model declares an
//!   encoder window (`n_audio_ctx`: 5000 mel frames, 50 s, for v3). Past it
//!   whisper.cpp switches to a dynamic encoder graph that silently drops
//!   speech — a 55 s clip lost its second half in testing — and keeps full,
//!   quadratic attention up to ~655 s, so a 5-minute single pass would also
//!   need 1–2 GB. Long dictations are therefore cut in the middle of pauses
//!   into pieces that fit the window ([`piece_limit`]); a stretch with no
//!   pause at all is split evenly as a last resort ([`split_oversized`]).
//! - **No language hint.** v3 identifies which of its 25 European languages is
//!   spoken and cannot be steered. A dictation language outside that set is
//!   refused at session start instead of decoding to the wrong language.
//! - **Greedy only.** TDT decoding has no beam search in whisper.cpp, and does
//!   not need the temperature-fallback ladder whisper uses.
//!
//! Shared with whisper ([`super::local`]): the VAD endpoint trim, the ggml file
//! sanity check, decode-thread count and the per-word confidence rule.

use std::path::PathBuf;

use lune_core::asr::AsrSessionConfig;
use lune_core::config::ParakeetModel;
use lune_core::error::{Error, Result};

/// Construction parameters for [`ParakeetCppBackend`].
pub struct ParakeetCppConfig {
    /// Which model this is; decides the languages it accepts.
    pub model: ParakeetModel,
    /// Path to the GGML model file (`models/<dir>/ggml-parakeet-*.bin`).
    pub model_path: PathBuf,
    /// Offload to GPU when a CUDA device is registered; otherwise the CPU.
    pub use_gpu: bool,
    /// Path to the standalone silero VAD ggml model, used to endpoint-trim the
    /// buffer before decode. `None`/missing => the full buffer is decoded.
    pub vad_model_path: Option<PathBuf>,
}

/// SentencePiece's word-boundary marker, U+2581 LOWER ONE EIGHTH BLOCK.
const SPM_SPACE: char = '\u{2581}';

/// The text of one decoded SentencePiece piece, with the word-boundary marker
/// as a leading space the way whisper spells it. `None` for control pieces,
/// which carry no text.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn piece_text(piece: &str) -> Option<String> {
    if matches!(piece, "<unk>" | "<s>" | "</s>" | "[BLANK]") || piece.is_empty() {
        return None;
    }
    Some(piece.replace(SPM_SPACE, " "))
}

/// Milliseconds per Parakeet token-timestamp unit: one mel frame, a 160-sample
/// hop at 16 kHz.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const MS_PER_FRAME: u64 = 10;

/// Upper bound on one decode: 60 s. Full attention over that is a few tens of
/// MB; over the 300 s session cap it would be well over a gigabyte. The real
/// limit is usually the model's own encoder window — see [`piece_limit`].
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const CHUNK_MAX_SAMPLES: usize = 60 * 16_000;

/// 16 kHz samples per mel frame (a 160-sample hop).
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const SAMPLES_PER_FRAME: usize = 160;

/// Mel frames kept free below the model's window, for the frame the STFT
/// padding adds and any rounding.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const WINDOW_MARGIN_FRAMES: usize = 32;

/// Longest piece, in samples, that decodes inside a model whose encoder window
/// is `n_audio_ctx` mel frames — 5000, i.e. 50 s, for parakeet-tdt-0.6b-v3.
///
/// Past its window whisper.cpp's Parakeet switches to a "dynamic" encoder
/// graph, and that path silently drops speech: a 55 s clip came back with its
/// second half missing. Every piece is therefore kept inside the window, and
/// never above [`CHUNK_MAX_SAMPLES`]. An unknown window falls back to that cap.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn piece_limit(n_audio_ctx: i32) -> usize {
    let frames = usize::try_from(n_audio_ctx).unwrap_or(0);
    if frames <= WINDOW_MARGIN_FRAMES {
        return CHUNK_MAX_SAMPLES;
    }
    ((frames - WINDOW_MARGIN_FRAMES) * SAMPLES_PER_FRAME).min(CHUNK_MAX_SAMPLES)
}

/// Split any piece longer than `limit` into equal parts that fit.
///
/// The pause planner leaves a stretch with no usable pause whole. For whisper
/// that is the lesser evil; here a piece past the window loses whole sentences
/// (see [`piece_limit`]), while an even cut costs at most the one word it
/// lands in. Only reached for continuous speech with no pause in ~50 s.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn split_oversized(chunks: &[(usize, usize)], limit: usize) -> Vec<(usize, usize)> {
    let limit = limit.max(1);
    let mut out = Vec::with_capacity(chunks.len());
    for &(from, to) in chunks {
        let len = to - from;
        let parts = len.div_ceil(limit).max(1);
        for k in 0..parts {
            out.push((from + len * k / parts, from + len * (k + 1) / parts));
        }
    }
    out
}

/// Shortest piece a cut may leave on either side: 3 s, so a cut never strands
/// a fragment too short to give the encoder context.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const CHUNK_MIN_SAMPLES: usize = 3 * 16_000;

/// Shortest clip handed to the encoder: 0.5 s. Its convolutional subsampling
/// reduces time 8×, so a clip of a few frames encodes to nothing and asks the
/// decoder for zero steps. A short "yes" is padded with trailing silence up to
/// this rather than risk that, and silence decodes to no tokens.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
const MIN_DECODE_SAMPLES: usize = 8_000;

/// Refuse a session whose dictation language this model cannot transcribe.
///
/// Parakeet cannot be told the language: asked for Japanese it would still
/// transcribe, as whichever European language the sounds resemble, and the
/// user would get confident nonsense. Refusing names the fix instead.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn check_language(model: ParakeetModel, language: Option<&str>) -> Result<()> {
    let Some(tag) = language else {
        return Ok(());
    };
    if model.supports_language(tag) {
        return Ok(());
    }
    let name = lune_core::lang::display_name(tag).unwrap_or(tag);
    Err(Error::config(format!(
        "Parakeet cannot transcribe {name}. It covers 25 European languages. \
         Set the dictation language to Auto-detect or one it supports, or switch \
         the engine to Whisper in Settings → Transcription."
    )))
}

#[cfg(feature = "whisper-cpp")]
mod imp {
    use super::{
        AsrSessionConfig, CHUNK_MIN_SAMPLES, MIN_DECODE_SAMPLES, MS_PER_FRAME, ParakeetCppConfig,
        Result, check_language, piece_limit, piece_text, split_oversized,
    };
    use crate::backend::local::{
        EndpointVad, VadTrim, WordConfidences, check_ggml_file, decode_threads, ensure_cpu_backend,
        plan_chunks_within, run_blocking,
    };

    use std::ffi::{CStr, CString};
    use std::ptr::NonNull;
    use std::sync::Mutex;

    use lune_core::asr::{AsrBackend, AsrSegment};
    use lune_core::config::ParakeetModel;
    use lune_core::error::Error;
    use whisper_rs::whisper_rs_sys as sys;

    /// A loaded model plus the one decoder state built for it.
    ///
    /// Owns both handles and frees them in the only safe order — the state
    /// first, since it references the context's weights.
    struct Engine {
        ctx: NonNull<sys::parakeet_context>,
        state: NonNull<sys::parakeet_state>,
    }

    // SAFETY: both handles are plain heap objects with no thread affinity —
    // ggml backends are not tied to the thread that created them — and an
    // `Engine` is only ever reached through the backend's `Mutex`, so no two
    // threads touch them at once.
    unsafe impl Send for Engine {}

    impl Engine {
        fn load(path: &std::path::Path, use_gpu: bool) -> Result<Self> {
            let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
                Error::asr(format!(
                    "the Parakeet model path {} contains a NUL byte",
                    path.display()
                ))
            })?;
            // SAFETY: returns a plain struct by value.
            let mut params = unsafe { sys::parakeet_context_default_params() };
            params.use_gpu = use_gpu;
            params.gpu_device = 0;
            // SAFETY: `c_path` is a valid NUL-terminated string for the call's
            // duration. The loader catches its own C++ exceptions and returns
            // null on any failure; `check_ggml_file` has already rejected the
            // truncated and non-ggml files that would trip an assert instead.
            let ctx = unsafe {
                sys::parakeet_init_from_file_with_params_no_state(c_path.as_ptr(), params)
            };
            let ctx = NonNull::new(ctx).ok_or_else(|| {
                Error::asr(format!(
                    "could not load the Parakeet model {} — it may be damaged or from \
                     another whisper.cpp version; delete it and download the model again",
                    path.display()
                ))
            })?;
            // SAFETY: `ctx` is a live context returned just above.
            let state = unsafe { sys::parakeet_init_state(ctx.as_ptr()) };
            let Some(state) = NonNull::new(state) else {
                // SAFETY: `ctx` is live and owned here; nothing else holds it.
                unsafe { sys::parakeet_free(ctx.as_ptr()) };
                return Err(Error::asr(
                    "could not allocate Parakeet's decoder state — the device may be out of memory",
                ));
            };
            Ok(Self { ctx, state })
        }

        /// The model's encoder window, in mel frames.
        fn n_audio_ctx(&self) -> i32 {
            // SAFETY: `ctx` is live; this reads a field of the loaded model.
            unsafe { sys::parakeet_model_n_audio_ctx(self.ctx.as_ptr()) }
        }

        /// Decode `pcm` and read back each segment's text and tokens.
        fn decode(&mut self, pcm: &[f32], n_threads: i32) -> Result<Vec<Decoded>> {
            let n_samples = i32::try_from(pcm.len())
                .map_err(|_| Error::asr("dictation too long for one Parakeet decode"))?;
            // SAFETY: returns a plain struct by value.
            let mut params = unsafe {
                sys::parakeet_full_default_params(
                    sys::parakeet_sampling_strategy_PARAKEET_SAMPLING_GREEDY,
                )
            };
            params.n_threads = n_threads;
            // Clear the LSTM state and the previous dictation's tokens, so the
            // reused state decodes exactly as a fresh one would.
            params.no_context = true;
            // SAFETY: `ctx`/`state` are live and exclusively borrowed through
            // `&mut self`; `pcm` outlives the call and holds `n_samples`
            // samples. Every callback in `params` is null.
            let rc = unsafe {
                sys::parakeet_full_with_state(
                    self.ctx.as_ptr(),
                    self.state.as_ptr(),
                    params,
                    pcm.as_ptr(),
                    n_samples,
                )
            };
            if rc != 0 {
                return Err(Error::asr(format!("Parakeet inference failed (code {rc})")));
            }

            let state = self.state.as_ptr();
            // SAFETY: reads of results the decode above just wrote into
            // `state`. Indices stay below the counts the library reports, and
            // returned strings are copied out before the next call.
            let n_segments = unsafe { sys::parakeet_full_n_segments_from_state(state) };
            let mut out = Vec::with_capacity(n_segments.max(0) as usize);
            for i in 0..n_segments {
                let text =
                    unsafe { c_text(sys::parakeet_full_get_segment_text_from_state(state, i)) };
                let n_tokens = unsafe { sys::parakeet_full_n_tokens_from_state(state, i) };
                let mut tokens = Vec::with_capacity(n_tokens.max(0) as usize);
                for j in 0..n_tokens {
                    let data = unsafe { sys::parakeet_full_get_token_data_from_state(state, i, j) };
                    let piece = unsafe {
                        c_text(sys::parakeet_full_get_token_text_from_state(
                            self.ctx.as_ptr(),
                            state,
                            i,
                            j,
                        ))
                    };
                    tokens.push(Token {
                        piece,
                        p: data.p,
                        t0: data.t0,
                        t1: data.t1,
                        word_start: data.is_word_start,
                    });
                }
                out.push(Decoded { text, tokens });
            }
            Ok(out)
        }
    }

    impl Drop for Engine {
        fn drop(&mut self) {
            // SAFETY: both handles are live and owned here; the state goes
            // first because it references the context's weights.
            unsafe {
                sys::parakeet_free_state(self.state.as_ptr());
                sys::parakeet_free(self.ctx.as_ptr());
            }
        }
    }

    /// Copy a C string the library returned; null reads as empty.
    ///
    /// # Safety
    /// `ptr` is null or a NUL-terminated string valid for this call.
    unsafe fn c_text(ptr: *const std::ffi::c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: guaranteed by the caller.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    /// One decoded segment, copied out of the native state.
    struct Decoded {
        text: String,
        tokens: Vec<Token>,
    }

    struct Token {
        /// Raw SentencePiece piece (`▁And`, `ell`, `,`).
        piece: String,
        p: f32,
        /// Start and end, in mel frames from the start of the decoded clip.
        t0: i64,
        t1: i64,
        word_start: bool,
    }

    pub struct ParakeetCppBackend {
        model: ParakeetModel,
        engine: Mutex<Engine>,
        n_threads: i32,
        /// Longest piece one decode may take, from the model's encoder window.
        piece_limit: usize,
        /// Standalone silero VAD for endpoint trimming (loaded once, reused).
        /// `None` => decode the full buffer (model absent or failed to load).
        vad: Option<EndpointVad>,
    }

    /// Mutable per-session state behind the handle's `Mutex`.
    pub struct ParakeetCppSession {
        /// Reused across utterances — `poll_final`/`reset_context` clear it
        /// without freeing.
        audio: Vec<f32>,
        /// The dictation language, when one is set. Parakeet does not report
        /// the language it heard, so this is the best answer downstream
        /// formatting gets; `None` leaves it to its default.
        language: Option<String>,
        /// Monotonic segment id, unique across `poll_final` calls in a session.
        next_id: u64,
    }

    impl ParakeetCppBackend {
        #[cold]
        pub fn new(config: ParakeetCppConfig) -> Result<Self> {
            // Normally already done at startup, before the plugin load. Kept
            // here so a backend built any other way still gets the logs —
            // and so ggml's log callback, which loading Parakeet re-points,
            // still lands in `tracing`.
            crate::backend::local::install_logging_hooks();

            // A malformed file kills the process inside the native loader
            // rather than returning an error; catch the ordinary causes first.
            check_ggml_file(&config.model_path)?;
            ensure_cpu_backend()?;

            let on_gpu = config.use_gpu && crate::gpu_backend::gpu_device_registered();
            let engine = Engine::load(&config.model_path, on_gpu)?;
            let piece_limit = piece_limit(engine.n_audio_ctx());
            let n_threads = decode_threads();
            let vad = config
                .vad_model_path
                .and_then(|path| EndpointVad::load(&path, n_threads));
            tracing::info!(
                on_gpu,
                n_threads,
                piece_limit_ms = piece_limit / 16,
                model = ?config.model,
                "parakeet backend loaded"
            );
            Ok(Self {
                model: config.model,
                engine: Mutex::new(engine),
                n_threads,
                piece_limit,
                vad,
            })
        }
    }

    #[async_trait::async_trait]
    impl AsrBackend for ParakeetCppBackend {
        type SessionHandle = Mutex<ParakeetCppSession>;

        async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
            check_language(self.model, config.language.as_deref())?;
            Ok(Mutex::new(ParakeetCppSession {
                // 30 s of 16 kHz audio up front — covers a typical utterance
                // with no realloc; longer dictations grow the buffer once.
                audio: Vec::with_capacity(16_000 * 30),
                language: config.language,
                next_id: 0,
            }))
        }

        async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
            session
                .lock()
                .map_err(|_| Error::asr("parakeet session poisoned"))?
                .audio
                .extend_from_slice(pcm);
            Ok(())
        }

        async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
            let mut guard = session
                .lock()
                .map_err(|_| Error::asr("parakeet session poisoned"))?;
            let ParakeetCppSession {
                audio,
                language,
                next_id,
            } = &mut *guard;

            if audio.is_empty() {
                return Ok(Vec::new());
            }

            // VAD + decode are seconds of blocking compute; hand the worker
            // off so the UI/event tasks stay responsive during inference.
            let result = run_blocking(|| {
                let trim = self
                    .vad
                    .as_ref()
                    .map_or(VadTrim::Full, |vad| vad.trim(audio.as_slice()));
                let Some(slice) = trim.decode_slice(audio.len()) else {
                    return Ok(Vec::new());
                };
                let speech = &audio[slice.start..slice.end];
                let chunks = split_oversized(
                    &plan_chunks_within(
                        &slice.relative_spans(),
                        speech.len(),
                        0,
                        self.piece_limit,
                        CHUNK_MIN_SAMPLES,
                    ),
                    self.piece_limit,
                );

                let mut out = Vec::new();
                for &(from, to) in &chunks {
                    let piece = &speech[from..to];
                    let padded: Vec<f32>;
                    let pcm: &[f32] = if piece.len() < MIN_DECODE_SAMPLES {
                        padded = piece
                            .iter()
                            .copied()
                            .chain(std::iter::repeat_n(0.0, MIN_DECODE_SAMPLES - piece.len()))
                            .collect();
                        &padded
                    } else {
                        piece
                    };

                    // A panic mid-decode leaves the native state in whatever
                    // shape it was in; `no_context` resets everything the next
                    // decode reads, so recovering the lock is safe.
                    let decoded = self
                        .engine
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .decode(pcm, self.n_threads)?;

                    // Token times are mel frames from the start of this piece;
                    // add back where the piece sits in the buffer so callers
                    // see original-buffer time.
                    let piece_ms = slice.offset_ms() + (from as u64 * 1000) / 16_000;
                    for segment in decoded {
                        let text = segment.text.trim();
                        if text.is_empty() {
                            continue;
                        }
                        let mut acc = WordConfidences::new();
                        let mut first_frame = None;
                        let mut last_frame = None;
                        for token in &segment.tokens {
                            let Some(text) = piece_text(&token.piece) else {
                                continue;
                            };
                            acc.push(&text, token.p, token.word_start);
                            first_frame.get_or_insert(token.t0);
                            last_frame = Some(token.t1.max(token.t0));
                        }
                        let (word_confidences, confidence, min_token_confidence) = acc.finish();
                        let at = |frame: Option<i64>| {
                            frame.map(|f| piece_ms + f.max(0) as u64 * MS_PER_FRAME)
                        };
                        let id = *next_id;
                        *next_id += 1;
                        out.push(AsrSegment {
                            id,
                            confidence,
                            min_token_confidence,
                            word_confidences,
                            is_final: true,
                            start_ms: at(first_frame),
                            end_ms: at(last_frame),
                            text: text.to_string(),
                            detected_language: language.clone(),
                        });
                    }
                }
                Ok(out)
            });
            // Drain the decoded audio whatever the outcome, so a repeat poll
            // can't re-emit it; capacity is retained for reuse.
            audio.clear();
            result
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
    use super::{AsrSessionConfig, ParakeetCppConfig, Result};

    use lune_core::asr::{AsrBackend, AsrSegment};
    use lune_core::error::Error;

    const DISABLED: &str = "Parakeet backend not compiled in — rebuild with --features whisper-cpp";

    /// Stub mirroring the real backend's surface so the registry compiles
    /// without the feature. Every entry point returns the "rebuild" error.
    pub struct ParakeetCppBackend;

    impl ParakeetCppBackend {
        #[cold]
        pub fn new(_config: ParakeetCppConfig) -> Result<Self> {
            Err(Error::config(DISABLED))
        }
    }

    #[async_trait::async_trait]
    impl AsrBackend for ParakeetCppBackend {
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

pub use imp::ParakeetCppBackend;

#[cfg(test)]
mod tests {
    use super::{
        CHUNK_MAX_SAMPLES, ParakeetModel, SAMPLES_PER_FRAME, WINDOW_MARGIN_FRAMES, check_language,
        piece_limit, piece_text, split_oversized,
    };

    #[test]
    fn pieces_stay_inside_the_model_window_and_the_cap() {
        // v3: 5000 frames is 50 s; the margin keeps a piece just under it.
        let v3 = piece_limit(5000);
        assert_eq!(v3, (5000 - WINDOW_MARGIN_FRAMES) * SAMPLES_PER_FRAME);
        assert!(v3 < 50 * 16_000);
        // A bigger window is still capped for memory; a missing one falls back.
        assert_eq!(piece_limit(20_000), CHUNK_MAX_SAMPLES);
        assert_eq!(piece_limit(0), CHUNK_MAX_SAMPLES);
        assert_eq!(piece_limit(-1), CHUNK_MAX_SAMPLES);
    }

    #[test]
    fn a_piece_with_no_pause_is_split_evenly_to_fit() {
        let limit = 50;
        assert_eq!(split_oversized(&[(0, 40)], limit), vec![(0, 40)]);
        assert_eq!(
            split_oversized(&[(0, 30), (30, 160)], limit),
            vec![(0, 30), (30, 73), (73, 116), (116, 160)]
        );
        for (from, to) in split_oversized(&[(10, 1_000_010)], limit) {
            assert!(to - from <= limit);
        }
    }

    #[test]
    fn a_word_start_piece_reads_as_a_leading_space() {
        assert_eq!(piece_text("\u{2581}And").as_deref(), Some(" And"));
        assert_eq!(piece_text("ell").as_deref(), Some("ell"));
        assert_eq!(piece_text(",").as_deref(), Some(","));
    }

    #[test]
    fn control_pieces_carry_no_text() {
        for piece in ["<unk>", "<s>", "</s>", "[BLANK]", ""] {
            assert_eq!(piece_text(piece), None, "{piece}");
        }
    }

    #[test]
    fn a_supported_or_absent_language_starts_a_session() {
        let model = ParakeetModel::TdtV3Q8;
        assert!(check_language(model, None).is_ok());
        assert!(check_language(model, Some("de")).is_ok());
        assert!(check_language(model, Some("en-GB")).is_ok());
    }

    #[test]
    fn an_unsupported_language_is_refused_with_the_fix() {
        let err = check_language(ParakeetModel::TdtV3Q8, Some("ja"))
            .expect_err("Japanese is not a Parakeet v3 language");
        let msg = err.to_string();
        assert!(msg.contains("Japanese"), "names the language: {msg}");
        assert!(msg.contains("Whisper"), "names the way out: {msg}");
    }

    #[test]
    fn an_unknown_tag_is_named_as_given() {
        let err = check_language(ParakeetModel::TdtV3Q8, Some("tlh")).expect_err("not supported");
        assert!(err.to_string().contains("tlh"));
    }
}
