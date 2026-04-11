//! Runtime-dispatched ASR backend registry.
//!
//! [`AsrRegistry`] wraps all available ASR backends behind a single enum that
//! implements [`AsrBackend`]. Construct one via [`AsrRegistry::from_config`]
//! using the user's [`AsrBackendConfig`] and [`DevicePolicy`].

use std::time::Duration;

use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSegment, AsrSessionConfig};
use lune_core::config::{AsrBackendConfig, DevicePolicy, PipelineConfig};
use lune_core::error::{Error, Result};

use crate::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, ParakeetTdtBackend, ParakeetTdtConfig};
#[cfg(feature = "ort-qwen")]
use crate::{OrtQwenAsrBackend, OrtQwenAsrConfig};

// ── Dispatch helpers ────────────────────────────────────────────────────────

const MISMATCH: &str = "backend/session type mismatch";

#[cold]
#[inline(never)]
fn mismatch_err() -> Error {
    Error::Asr(MISMATCH.into())
}

/// Dispatch a `&session` method call to the matching backend variant.
macro_rules! dispatch_ref {
    ($self:ident, $session:ident, $method:ident $(, $arg:expr)*) => {
        match $self {
            AsrRegistry::Parakeet(b) => match $session {
                AsrSessionHandle::Parakeet(s) => b.$method(s $(, $arg)*).await,
                _ => Err(mismatch_err()),
            },
            #[cfg(feature = "ort-qwen")]
            AsrRegistry::OrtQwen(b) => match $session {
                AsrSessionHandle::OrtQwen(s) => b.$method(s $(, $arg)*).await,
                _ => Err(mismatch_err()),
            },
            AsrRegistry::OpenAi(b) => match $session {
                AsrSessionHandle::OpenAi(s) => b.$method(s $(, $arg)*).await,
                _ => Err(mismatch_err()),
            },
        }
    };
}

// ── Registry enum ───────────────────────────────────────────────────────────

/// Runtime-dispatched ASR backend. Construct via [`AsrRegistry::from_config`].
#[allow(clippy::large_enum_variant)]
pub enum AsrRegistry {
    Parakeet(ParakeetTdtBackend),
    #[cfg(feature = "ort-qwen")]
    OrtQwen(OrtQwenAsrBackend),
    OpenAi(OpenAiCompatibleAsrBackend),
}

/// Runtime-dispatched ASR session handle.
#[allow(clippy::large_enum_variant)]
pub enum AsrSessionHandle {
    Parakeet(<ParakeetTdtBackend as AsrBackend>::SessionHandle),
    #[cfg(feature = "ort-qwen")]
    OrtQwen(<OrtQwenAsrBackend as AsrBackend>::SessionHandle),
    OpenAi(<OpenAiCompatibleAsrBackend as AsrBackend>::SessionHandle),
}

// ── AsrBackend impl ─────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl AsrBackend for AsrRegistry {
    type SessionHandle = AsrSessionHandle;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        match self {
            Self::Parakeet(b) => Ok(AsrSessionHandle::Parakeet(b.start_session(config).await?)),
            #[cfg(feature = "ort-qwen")]
            Self::OrtQwen(b) => Ok(AsrSessionHandle::OrtQwen(b.start_session(config).await?)),
            Self::OpenAi(b) => Ok(AsrSessionHandle::OpenAi(b.start_session(config).await?)),
        }
    }

    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        dispatch_ref!(self, session, push_audio, pcm)
    }

    #[inline]
    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        dispatch_ref!(self, session, poll_results)
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        dispatch_ref!(self, session, poll_final)
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        match self {
            Self::Parakeet(b) => match session {
                AsrSessionHandle::Parakeet(s) => b.end_session(s).await,
                _ => Err(mismatch_err()),
            },
            #[cfg(feature = "ort-qwen")]
            Self::OrtQwen(b) => match session {
                AsrSessionHandle::OrtQwen(s) => b.end_session(s).await,
                _ => Err(mismatch_err()),
            },
            Self::OpenAi(b) => match session {
                AsrSessionHandle::OpenAi(s) => b.end_session(s).await,
                _ => Err(mismatch_err()),
            },
        }
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        dispatch_ref!(self, session, reset_context)
    }

    fn supports_streaming(&self) -> bool {
        match self {
            Self::Parakeet(_) => false,
            #[cfg(feature = "ort-qwen")]
            Self::OrtQwen(_) => true,
            Self::OpenAi(_) => false,
        }
    }
}

// ── Constructor ─────────────────────────────────────────────────────────────

impl AsrRegistry {
    /// Build the appropriate ASR backend from user configuration.
    ///
    /// Cold path: called once at startup.
    #[cold]
    pub fn from_config(config: &AsrBackendConfig, device: DevicePolicy) -> Result<Self> {
        let use_gpu = device.use_gpu(cfg!(feature = "cuda"));

        match config {
            AsrBackendConfig::ParakeetTdt { model_dir } => {
                let backend = ParakeetTdtBackend::new(ParakeetTdtConfig {
                    model_dir: model_dir.clone(),
                    use_gpu,
                })?;
                Ok(Self::Parakeet(backend))
            }
            #[cfg(feature = "ort-qwen")]
            AsrBackendConfig::OrtQwenAsr { model_dir } => {
                let backend = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
                    model_dir: model_dir.clone(),
                    use_gpu,
                })?;
                Ok(Self::OrtQwen(backend))
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
            _ => Err(Error::Config("unknown ASR backend type".into())),
        }
    }

    /// Returns the backend kind for this registry entry.
    pub fn kind(&self) -> AsrBackendKind {
        match self {
            Self::Parakeet(_) => AsrBackendKind::ParakeetTdt,
            #[cfg(feature = "ort-qwen")]
            Self::OrtQwen(_) => AsrBackendKind::OrtQwenAsr,
            Self::OpenAi(_) => AsrBackendKind::OpenAiCompatible,
        }
    }

    /// Returns the recommended pipeline configuration for this backend.
    ///
    /// Each backend provides tuned defaults for VAD timing, confidence
    /// thresholds, and segment limits. Callers should merge user overrides
    /// on top via [`PipelineConfig::with_overrides`].
    pub fn pipeline_config(&self) -> PipelineConfig {
        match self {
            Self::Parakeet(_) => PipelineConfig {
                max_segment_ms: 30_000,
                sentence_close_ms: 776,
                hard_close_ms: 1_552,
                speech_threshold: 0.5,
                min_speech_ms: 150,
                adaptive_vad: false,
                // Thresholds are interpreted on the CALIBRATED probability
                // scale (P(correct | raw)) after SessionManager runs the
                // isotonic calibrator. Parakeet's raw softmax concentrates
                // 86% of tokens in the [0.9, 1.0) bin, so anything below
                // 0.65 raw is effectively unreachable — 0.85 / 0.95 on the
                // calibrated scale surfaces the bins where acoustic errors
                // actually live (raw 0.5–0.8 → calibrated 0.70–0.94).
                word_confidence_threshold: Some(0.85),
                llm_skip_threshold: Some(0.95),
                max_speech_samples: 60 * 16_000,
            },
            #[cfg(feature = "ort-qwen")]
            Self::OrtQwen(_) => PipelineConfig {
                // Qwen3-ASR offline inference is whole-clip up to the export
                // ceiling, not an 8s streaming-style decode loop.
                max_segment_ms: 30_000,
                sentence_close_ms: 800,
                hard_close_ms: 2_000,
                speech_threshold: 0.5,
                min_speech_ms: 150,
                adaptive_vad: false,
                // Qwen is even more concentrated in [0.9, 1.0) (93% of
                // tokens per fixture calibration data). Same calibrated
                // thresholds as Parakeet — raising them gets the LLM
                // corrector off the sidelines on genuinely noisy audio.
                word_confidence_threshold: Some(0.85),
                llm_skip_threshold: Some(0.95),
                max_speech_samples: 60 * 16_000,
            },
            Self::OpenAi(_) => PipelineConfig {
                max_segment_ms: 30_000,
                sentence_close_ms: 800,
                hard_close_ms: 2_000,
                speech_threshold: 0.5,
                min_speech_ms: 200,
                adaptive_vad: false,
                // No per-token log-probs from remote API — disable both
                // marking and skip-when-clean gating.
                word_confidence_threshold: None,
                llm_skip_threshold: None,
                max_speech_samples: 120 * 16_000,
            },
        }
    }

    /// Returns `true` if this registry holds an OrtQwen backend.
    #[cfg(feature = "ort-qwen")]
    pub fn as_ort_qwen(&self) -> Option<&OrtQwenAsrBackend> {
        match self {
            Self::OrtQwen(b) => Some(b),
            _ => None,
        }
    }
}
