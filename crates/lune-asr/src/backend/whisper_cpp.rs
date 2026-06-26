//! Local whisper.cpp ASR backend.
//!
//! The real implementation (feature `whisper-cpp`) wraps [`whisper_rs`], which
//! compiles whisper.cpp. When the feature is off, a stub with the same public
//! surface keeps the registry and the rest of the crate compiling — its
//! constructor returns a clear "rebuild with the feature" error.
//!
//! Design notes (batch decoder, like Qwen3):
//! - The model ([`whisper_rs::WhisperContext`]) is loaded **once** at backend
//!   construction and shared by every session; per-session `WhisperState` is cheap.
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
}

#[cfg(feature = "whisper-cpp")]
mod imp {
    use super::{AsrSessionConfig, Result, WhisperCppConfig};

    use std::sync::Mutex;

    use lune_core::asr::{AsrBackend, AsrSegment};
    use lune_core::error::Error;
    use whisper_rs::{
        FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    };

    /// Cap decode threads. whisper.cpp scales sub-linearly past a handful of
    /// threads and we don't want to starve the audio/UI threads.
    const MAX_THREADS: usize = 8;

    pub struct WhisperCppBackend {
        ctx: WhisperContext,
        n_threads: i32,
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
            let mut params = WhisperContextParameters::default();
            params.use_gpu(config.use_gpu);
            let ctx = WhisperContext::new_with_params(&config.model_path, params).map_err(|e| {
                Error::asr(format!(
                    "load whisper model {}: {e}",
                    config.model_path.display()
                ))
            })?;
            let n_threads = std::thread::available_parallelism()
                .map(|n| n.get().min(MAX_THREADS) as i32)
                .unwrap_or(4);
            Ok(Self { ctx, n_threads })
        }
    }

    #[async_trait::async_trait]
    impl AsrBackend for WhisperCppBackend {
        type SessionHandle = Mutex<WhisperCppSession>;

        async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
            let state = self
                .ctx
                .create_state()
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

            // BeamSearch is whisper.cpp's accuracy-oriented default (beam 5).
            // Worth the extra CPU here: decoding is one-shot per utterance, and
            // accuracy is the whole reason for offering the larger models.
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

            state
                .full(params, audio.as_slice())
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

                // Fold per-token probabilities into the gate's confidence fields.
                // Skip whisper's special tokens (rendered "[_...]"): their p≈1.0
                // would inflate the mean.
                let mut sum = 0.0f32;
                let mut count = 0u32;
                let mut min_p = 1.0f32;
                for j in 0..seg.n_tokens() {
                    let Some(tok) = seg.get_token(j) else {
                        continue;
                    };
                    if matches!(tok.to_str_lossy(), Ok(s) if s.starts_with("[_")) {
                        continue;
                    }
                    let p = tok.token_probability();
                    sum += p;
                    min_p = min_p.min(p);
                    count += 1;
                }
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
                    word_confidences: Vec::new(),
                    is_final: true,
                    // whisper timestamps are in centiseconds (10 ms units).
                    start_ms: Some((seg.start_timestamp().max(0) as u64) * 10),
                    end_ms: Some((seg.end_timestamp().max(0) as u64) * 10),
                    text,
                    detected_language: detected.clone(),
                });
            }
            Ok(out)
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
