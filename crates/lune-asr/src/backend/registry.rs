//! Runtime-dispatched ASR backend registry.
//!
//! [`AsrRegistry`] wraps all available ASR backends behind a single enum that
//! implements [`AsrBackend`]. Construct one via [`AsrRegistry::from_config`]
//! using the user's [`AsrBackendConfig`] and [`DevicePolicy`].

use std::path::Path;
use std::time::Duration;

use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSegment, AsrSessionConfig};
use lune_core::config::{AsrBackendConfig, DevicePolicy, PipelineConfig};
use lune_core::error::{Error, Result};

use crate::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, Qwen3AsrBackend, Qwen3AsrConfig};

// ── Dispatch helpers ────────────────────────────────────────────────────────
//
// Two macros cover every `AsrBackend` method on `AsrRegistry`:
//
//   * `dispatch!`     — calls a method on an existing session handle.
//                       Works whether `$session` is `&SessionHandle` or
//                       owned `SessionHandle`; the match arm binds `s`
//                       with the same ownership the caller passed in.
//   * `dispatch_new!` — calls a backend constructor (no session input)
//                       and wraps the returned handle into the matching
//                       `AsrSessionHandle` variant.
//
// Adding a backend variant requires editing both macros plus the two
// enums below. Compile-time exhaustiveness checking catches misses.

const MISMATCH: &str = "backend/session type mismatch";

#[cold]
#[inline(never)]
fn mismatch_err() -> Error {
    Error::asr(MISMATCH)
}

/// Dispatch a method call on a session handle to the matching backend.
///
/// `$session` may be `&SessionHandle` or owned; ownership flows through.
macro_rules! dispatch {
    ($self:ident, $session:ident, $method:ident $(, $arg:expr)*) => {
        match $self {
            AsrRegistry::Qwen3(b) => match $session {
                AsrSessionHandle::Qwen3(s) => b.$method(s $(, $arg)*).await,
                _ => Err(mismatch_err()),
            },
            AsrRegistry::OpenAi(b) => match $session {
                AsrSessionHandle::OpenAi(s) => b.$method(s $(, $arg)*).await,
                _ => Err(mismatch_err()),
            },
        }
    };
}

/// Dispatch a constructor (e.g. `start_session`) and wrap the returned
/// session handle into the matching `AsrSessionHandle` variant.
macro_rules! dispatch_new {
    ($self:ident, $method:ident $(, $arg:expr)*) => {
        match $self {
            AsrRegistry::Qwen3(b) => Ok(AsrSessionHandle::Qwen3(b.$method($($arg),*).await?)),
            AsrRegistry::OpenAi(b) => Ok(AsrSessionHandle::OpenAi(b.$method($($arg),*).await?)),
        }
    };
}

// ── Registry enum ───────────────────────────────────────────────────────────

/// Runtime-dispatched ASR backend. Construct via [`AsrRegistry::from_config`].
#[allow(clippy::large_enum_variant)]
pub enum AsrRegistry {
    Qwen3(Qwen3AsrBackend),
    OpenAi(OpenAiCompatibleAsrBackend),
}

/// Runtime-dispatched ASR session handle.
#[allow(clippy::large_enum_variant)]
pub enum AsrSessionHandle {
    Qwen3(<Qwen3AsrBackend as AsrBackend>::SessionHandle),
    OpenAi(<OpenAiCompatibleAsrBackend as AsrBackend>::SessionHandle),
}

// ── AsrBackend impl ─────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl AsrBackend for AsrRegistry {
    type SessionHandle = AsrSessionHandle;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        dispatch_new!(self, start_session, config)
    }

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        dispatch!(self, session, push_audio, pcm)
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        dispatch!(self, session, poll_final)
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        dispatch!(self, session, end_session)
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        dispatch!(self, session, reset_context)
    }
}

// ── Constructor ─────────────────────────────────────────────────────────────

impl AsrRegistry {
    /// Build the appropriate ASR backend from user configuration.
    ///
    /// Cold path: called once at startup.
    #[cold]
    pub fn from_config(
        config: &AsrBackendConfig,
        device: DevicePolicy,
        models_dir: &Path,
    ) -> Result<Self> {
        let use_gpu = device.use_gpu(cfg!(feature = "cuda"));
        let cuda_required = device.strict_cuda();

        match config {
            AsrBackendConfig::Qwen3Asr { .. } => {
                let backend = Qwen3AsrBackend::new(Qwen3AsrConfig {
                    model_dir: models_dir.join(lune_core::models::DIR_QWEN3_ASR),
                    use_gpu,
                    cuda_required,
                })?;
                Ok(Self::Qwen3(backend))
            }
            AsrBackendConfig::OpenAiCompatible {
                base_url,
                api_key,
                model,
                ..
            } => {
                let backend = OpenAiCompatibleAsrBackend::new(OpenAiAsrConfig {
                    base_url: base_url.clone(),
                    api_key: api_key.clone(),
                    model: model.clone(),
                    timeout: Duration::from_secs(30),
                    min_samples_before_request: 8000,
                })?;
                Ok(Self::OpenAi(backend))
            }
            #[allow(unreachable_patterns)]
            _ => Err(Error::config("unknown ASR backend type")),
        }
    }

    /// Returns the backend kind for this registry entry.
    pub fn kind(&self) -> AsrBackendKind {
        match self {
            Self::Qwen3(_) => AsrBackendKind::Qwen3Asr,
            Self::OpenAi(_) => AsrBackendKind::OpenAiCompatible,
        }
    }

    /// Returns the recommended pipeline configuration for this backend.
    ///
    /// Each backend provides tuned defaults for VAD timing, confidence
    /// thresholds, and segment limits. Callers should merge user overrides
    /// on top via [`PipelineConfig::with_overrides`].
    pub fn pipeline_config(&self) -> PipelineConfig {
        Self::default_pipeline_config(self.kind())
    }

    /// Backend-specific defaults addressable without instantiating a backend.
    ///
    /// Used by callers (e.g. settings UI) that need the same tuned defaults
    /// the runtime would apply, but cannot construct the heavy backend.
    pub fn default_pipeline_config(kind: AsrBackendKind) -> PipelineConfig {
        match kind {
            AsrBackendKind::Qwen3Asr => PipelineConfig {
                // Qwen3 is a batch decoder — one complete utterance per decode.
                // Flushing is controlled by the dictation toggle/hold release,
                // not by silence. Set silence thresholds extremely high so the
                // VAD never triggers a mid-utterance flush.
                max_segment_ms: 300_000,
                sentence_close_ms: 300_000,
                hard_close_ms: 300_000,
                speech_threshold: 0.5,
                speech_threshold_end: 0.3,
                min_speech_ms: 200,
                adaptive_vad: false,
                itn_enabled: true,
                word_boost_enabled: false,
                word_boost_score: 1.0,
                word_confidence_threshold: Some(0.55),
                // Attention band requires >=2 sub-threshold words; 0.40 matched
                // oracle routing on the domain corpus (benchmarks/gate_sweep.rs).
                llm_skip_threshold: Some(0.40),
                llm_skip_noisy_threshold: Some(0.30),
                max_speech_samples: 300 * 16_000,
                llm_timeout_secs: 30,
            },
            AsrBackendKind::OpenAiCompatible => PipelineConfig {
                max_segment_ms: 30_000,
                sentence_close_ms: 800,
                hard_close_ms: 2_000,
                speech_threshold: 0.5,
                speech_threshold_end: 0.3,
                min_speech_ms: 200,
                adaptive_vad: false,
                itn_enabled: true,
                word_boost_enabled: false,
                word_boost_score: 1.0,
                // No per-token log-probs from remote API — disable all
                // confidence-based gating.
                word_confidence_threshold: None,
                llm_skip_threshold: None,
                llm_skip_noisy_threshold: None,
                max_speech_samples: 120 * 16_000,
                llm_timeout_secs: 30,
            },
            // Fallback for any future variants — start from PipelineConfig::default().
            _ => PipelineConfig::default(),
        }
    }

    /// Per-transcription sub-stage profile (mel / encoder / prefill / decode
    /// timing and decode-step count) recorded by the most recent `poll_final`.
    /// Only the Qwen3 backend instruments this; other backends return `None`.
    pub fn last_profile(&self) -> Option<crate::Qwen3AsrProfile> {
        match self {
            Self::Qwen3(b) => b.last_profile(),
            Self::OpenAi(_) => None,
        }
    }

    /// Set word boost phrases on the underlying ASR backend.
    /// No-op for backends that don't support token-level boosting.
    pub fn set_boost_phrases(&self, phrases: &[String], score: f32) {
        match self {
            Self::Qwen3(b) => b.set_boost_phrases(phrases, score),
            Self::OpenAi(_) => {}
        }
    }
}
