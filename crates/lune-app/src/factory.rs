use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use lune_asr::{
    OpenAiAsrConfig, OpenAiCompatibleAsrBackend, OrtQwenAsrBackend, OrtQwenAsrConfig,
    ParakeetTdtBackend, ParakeetTdtConfig,
};
#[cfg(feature = "legacy-qwen")]
use lune_asr::{QwenAsrBackend, QwenAsrConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::{AsrBackendConfig, LlmBackendConfig, UserSettings};
use lune_core::error::{Error, Result};
use lune_core::llm::TextPostProcessor;
use lune_persist::ResolvedModels;

// Deduplicated, static — no heap alloc on the cold mismatch branch.
const MISMATCH: &str = "backend/session type mismatch";

// Keeps String allocation off the hot path; cold hint guides branch predictor.
#[cold]
#[inline(never)]
fn mismatch_err() -> Error {
    Error::Asr(MISMATCH.into())
}

// ── Runtime dispatch enum ────────────────────────────────────────────────────

/// Runtime-dispatched ASR backend (Parakeet local or OpenAI-compatible API).
#[allow(clippy::large_enum_variant)]
pub enum AsrBackendDispatch {
    Parakeet(ParakeetTdtBackend),
    #[cfg(feature = "legacy-qwen")]
    Qwen(QwenAsrBackend),
    OrtQwen(OrtQwenAsrBackend),
    OpenAi(OpenAiCompatibleAsrBackend),
}

/// Runtime-dispatched ASR session handle.
#[allow(clippy::large_enum_variant)]
pub enum AsrSessionDispatch {
    Parakeet(<ParakeetTdtBackend as AsrBackend>::SessionHandle),
    #[cfg(feature = "legacy-qwen")]
    Qwen(<QwenAsrBackend as AsrBackend>::SessionHandle),
    OrtQwen(<OrtQwenAsrBackend as AsrBackend>::SessionHandle),
    OpenAi(<OpenAiCompatibleAsrBackend as AsrBackend>::SessionHandle),
}

#[async_trait::async_trait]
impl AsrBackend for AsrBackendDispatch {
    type SessionHandle = AsrSessionDispatch;

    #[inline]
    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        match self {
            Self::Parakeet(b) => Ok(AsrSessionDispatch::Parakeet(b.start_session(config).await?)),
            #[cfg(feature = "legacy-qwen")]
            Self::Qwen(b) => Ok(AsrSessionDispatch::Qwen(b.start_session(config).await?)),
            Self::OrtQwen(b) => Ok(AsrSessionDispatch::OrtQwen(b.start_session(config).await?)),
            Self::OpenAi(b) => Ok(AsrSessionDispatch::OpenAi(b.start_session(config).await?)),
        }
    }

    /// Hot path: called per audio frame.
    ///
    /// Nested match avoids constructing a temporary `(self, session)` tuple on
    /// each call. The outer branch on `self` is the only one that varies at
    /// steady-state; the inner `if let` is trivially predicted-taken.
    /// The cold `mismatch_err()` call stays out of the I-cache success path.
    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        match self {
            Self::Parakeet(b) => {
                if let AsrSessionDispatch::Parakeet(s) = session {
                    b.push_audio(s, pcm).await
                } else {
                    Err(mismatch_err())
                }
            }
            #[cfg(feature = "legacy-qwen")]
            Self::Qwen(b) => {
                if let AsrSessionDispatch::Qwen(s) = session {
                    b.push_audio(s, pcm).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OrtQwen(b) => {
                if let AsrSessionDispatch::OrtQwen(s) = session {
                    b.push_audio(s, pcm).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OpenAi(b) => {
                if let AsrSessionDispatch::OpenAi(s) = session {
                    b.push_audio(s, pcm).await
                } else {
                    Err(mismatch_err())
                }
            }
        }
    }

    /// Hot path: polled every decode cycle.
    #[inline]
    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        match self {
            Self::Parakeet(b) => {
                if let AsrSessionDispatch::Parakeet(s) = session {
                    b.poll_results(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            #[cfg(feature = "legacy-qwen")]
            Self::Qwen(b) => {
                if let AsrSessionDispatch::Qwen(s) = session {
                    b.poll_results(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OrtQwen(b) => {
                if let AsrSessionDispatch::OrtQwen(s) = session {
                    b.poll_results(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OpenAi(b) => {
                if let AsrSessionDispatch::OpenAi(s) = session {
                    b.poll_results(s).await
                } else {
                    Err(mismatch_err())
                }
            }
        }
    }

    #[inline]
    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        match self {
            Self::Parakeet(b) => {
                if let AsrSessionDispatch::Parakeet(s) = session {
                    b.end_session(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            #[cfg(feature = "legacy-qwen")]
            Self::Qwen(b) => {
                if let AsrSessionDispatch::Qwen(s) = session {
                    b.end_session(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OrtQwen(b) => {
                if let AsrSessionDispatch::OrtQwen(s) = session {
                    b.end_session(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OpenAi(b) => {
                if let AsrSessionDispatch::OpenAi(s) = session {
                    b.end_session(s).await
                } else {
                    Err(mismatch_err())
                }
            }
        }
    }

    #[inline]
    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        match self {
            Self::Parakeet(b) => {
                if let AsrSessionDispatch::Parakeet(s) = session {
                    b.reset_context(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            #[cfg(feature = "legacy-qwen")]
            Self::Qwen(b) => {
                if let AsrSessionDispatch::Qwen(s) = session {
                    b.reset_context(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OrtQwen(b) => {
                if let AsrSessionDispatch::OrtQwen(s) = session {
                    b.reset_context(s).await
                } else {
                    Err(mismatch_err())
                }
            }
            Self::OpenAi(b) => {
                if let AsrSessionDispatch::OpenAi(s) = session {
                    b.reset_context(s).await
                } else {
                    Err(mismatch_err())
                }
            }
        }
    }
}

// ── Global ORT initialization ────────────────────────────────────────────────

/// Initialize the global ORT environment with execution providers.
///
/// Must be called exactly once, before any ORT `Session` is created.
/// Registers CUDA EP when `settings.device` requests GPU.
/// All sessions share a single CUDA memory arena.  Falls back to CPU
/// when the `cuda` feature is not enabled or CUDA init fails.
#[cold]
pub fn init_ort_runtime(settings: &UserSettings) {
    let wants_gpu = settings.device.use_gpu(cfg!(feature = "cuda"));

    if wants_gpu {
        #[cfg(feature = "cuda")]
        {
            if ort::init()
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .commit()
            {
                tracing::info!("ORT initialized with CUDA EP");
                return;
            } else {
                tracing::warn!("ORT CUDA init failed, falling back to CPU");
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::info!("GPU requested but cuda feature not enabled, using CPU");
        }
    }

    // CPU-only fallback.
    if ort::init().commit() {
        tracing::info!("ORT initialized (CPU only)");
    } else {
        tracing::warn!("ORT init failed or already initialized");
    }
}

// ── Factory functions ────────────────────────────────────────────────────────

/// Patch empty model paths in `settings` from `resolved` hub paths.
///
/// The default `UserSettings` ships with empty `PathBuf`s for model paths.
/// After `ensure_default_models` resolves the actual paths, this fills them in.
#[inline]
pub fn patch_model_paths(settings: &mut UserSettings, resolved: &ResolvedModels) {
    match &mut settings.asr_backend {
        AsrBackendConfig::ParakeetTdt { model_dir, .. } if model_dir.as_os_str().is_empty() => {
            *model_dir = resolved.asr_model_dir.clone();
        }
        AsrBackendConfig::QwenAsr { model_dir, .. } if model_dir.as_os_str().is_empty() => {
            *model_dir = resolved.qwen_asr_model_dir.clone();
        }
        AsrBackendConfig::OrtQwenAsr { model_dir, .. } if model_dir.as_os_str().is_empty() => {
            *model_dir = resolved.ort_qwen_asr_model_dir.clone();
        }
        _ => {}
    }

    match &mut settings.llm_backend {
        LlmBackendConfig::OrtLocal { model_dir, .. } if model_dir.as_os_str().is_empty() => {
            *model_dir = resolved.ort_llm_model_dir.clone();
        }
        _ => {}
    }
}

/// Construct the ASR backend from settings.
///
/// Cold path: called once at startup. `#[cold]` signals the linker to place
/// this function outside hot text sections, reducing I-cache pressure at
/// steady-state call sites that branch to factory functions.
#[cold]
pub fn build_asr_backend(settings: &UserSettings) -> Result<AsrBackendDispatch> {
    match &settings.asr_backend {
        AsrBackendConfig::ParakeetTdt { model_dir } => {
            let backend = ParakeetTdtBackend::new(ParakeetTdtConfig {
                model_dir: model_dir.clone(),
                use_gpu: settings.device.use_gpu(cfg!(feature = "cuda")),
            })?;
            Ok(AsrBackendDispatch::Parakeet(backend))
        }
        #[cfg(feature = "legacy-qwen")]
        AsrBackendConfig::QwenAsr {
            model_dir,
            stream_chunk_sec,
            stream_rollback,
            stream_unfixed_chunks,
            stream_max_new_tokens,
            past_text_conditioning,
        } => {
            let backend = QwenAsrBackend::new(QwenAsrConfig {
                model_dir: model_dir.clone(),
                stream_chunk_sec: *stream_chunk_sec,
                stream_rollback: *stream_rollback,
                stream_unfixed_chunks: *stream_unfixed_chunks,
                stream_max_new_tokens: *stream_max_new_tokens,
                past_text_conditioning: *past_text_conditioning,
            })?;
            Ok(AsrBackendDispatch::Qwen(backend))
        }
        #[cfg(not(feature = "legacy-qwen"))]
        AsrBackendConfig::QwenAsr { .. } => Err(Error::Config(
            "QwenAsr backend requires legacy-qwen feature".into(),
        )),
        AsrBackendConfig::OrtQwenAsr {
            model_dir,
            stream_chunk_sec,
            stream_rollback,
            stream_unfixed_chunks,
            stream_max_new_tokens,
            past_text_conditioning,
            max_encoder_sec,
        } => {
            let backend = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
                model_dir: model_dir.clone(),
                use_gpu: settings.device.use_gpu(cfg!(feature = "cuda")),
                stream_chunk_sec: *stream_chunk_sec,
                stream_rollback: *stream_rollback,
                stream_unfixed_chunks: *stream_unfixed_chunks,
                stream_max_new_tokens: *stream_max_new_tokens,
                past_text_conditioning: *past_text_conditioning,
                max_encoder_sec: *max_encoder_sec,
            })?;
            Ok(AsrBackendDispatch::OrtQwen(backend))
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
            Ok(AsrBackendDispatch::OpenAi(backend))
        }
        _ => Err(Error::Config("unknown ASR backend type".into())),
    }
}

/// Construct the LLM post-processor from settings.
///
/// Cold path: called once at startup.
#[cold]
pub fn build_post_processor(settings: &UserSettings) -> Result<Arc<dyn TextPostProcessor>> {
    match &settings.llm_backend {
        LlmBackendConfig::OrtLocal {
            model_dir,
            n_ctx,
            max_output_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            repeat_last_n,
        } => {
            let backend = lune_llm::OrtLlmBackend::new(lune_llm::OrtLlmConfig {
                model_dir: model_dir.clone(),
                use_gpu: settings.device.use_gpu(cfg!(feature = "cuda")),
                n_ctx: *n_ctx,
                max_output_tokens: *max_output_tokens,
                temperature: *temperature,
                top_p: *top_p,
                top_k: *top_k,
                repeat_penalty: *repeat_penalty,
                repeat_last_n: *repeat_last_n,
            })?;
            Ok(Arc::new(backend))
        }
        LlmBackendConfig::OpenAiCompatible { .. } => {
            Err(Error::Config("OpenAI LLM not yet supported in v0.1".into()))
        }
        _ => Err(Error::Config("unknown LLM backend type".into())),
    }
}

/// Build a database path, creating parent dirs if needed.
///
/// Cold path: called once at startup.
#[cold]
pub fn ensure_db_path(db_path: &Path) -> Result<()> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Config(format!("create db dir: {e}")))?;
    }
    Ok(())
}
