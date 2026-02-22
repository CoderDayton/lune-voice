use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use lune_asr::{
    OpenAiAsrConfig, OpenAiCompatibleAsrBackend, ParakeetTdtBackend, ParakeetTdtConfig,
};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::{AsrBackendConfig, LlmBackendConfig, UserSettings};
use lune_core::error::{Error, Result};
use lune_core::llm::TextPostProcessor;
use lune_llm::LocalLlmBackend;
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
pub enum AsrBackendDispatch {
    Parakeet(ParakeetTdtBackend),
    OpenAi(OpenAiCompatibleAsrBackend),
}

/// Runtime-dispatched ASR session handle.
pub enum AsrSessionDispatch {
    Parakeet(<ParakeetTdtBackend as AsrBackend>::SessionHandle),
    OpenAi(<OpenAiCompatibleAsrBackend as AsrBackend>::SessionHandle),
}

#[async_trait::async_trait]
impl AsrBackend for AsrBackendDispatch {
    type SessionHandle = AsrSessionDispatch;

    #[inline]
    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        match self {
            Self::Parakeet(b) => Ok(AsrSessionDispatch::Parakeet(b.start_session(config).await?)),
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
            Self::OpenAi(b) => {
                if let AsrSessionDispatch::OpenAi(s) = session {
                    b.end_session(s).await
                } else {
                    Err(mismatch_err())
                }
            }
        }
    }
}

// ── Factory functions ────────────────────────────────────────────────────────

/// Patch empty model paths in `settings` from `resolved` hub paths.
///
/// The default `UserSettings` ships with empty `PathBuf`s for model paths.
/// After `ensure_default_models` resolves the actual paths, this fills them in.
#[inline]
pub fn patch_model_paths(settings: &mut UserSettings, resolved: &ResolvedModels) {
    if let AsrBackendConfig::ParakeetTdt { model_dir, .. } = &mut settings.asr_backend
        && model_dir.as_os_str().is_empty()
    {
        *model_dir = resolved.asr_model_dir.clone();
    }

    if let LlmBackendConfig::Local { model_path, .. } = &mut settings.llm_backend
        && model_path.as_os_str().is_empty()
    {
        *model_path = resolved.llm_model_path.clone();
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
        AsrBackendConfig::ParakeetTdt {
            model_dir,
            use_gpu,
            context_secs,
        } => {
            let backend = ParakeetTdtBackend::new(ParakeetTdtConfig {
                model_dir: model_dir.clone(),
                use_gpu: *use_gpu,
                context_secs: *context_secs,
            })?;
            Ok(AsrBackendDispatch::Parakeet(backend))
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
        LlmBackendConfig::Local {
            model_path,
            use_gpu,
            n_ctx,
            max_output_tokens,
        } => {
            let backend = LocalLlmBackend::new(lune_llm::LocalLlmConfig {
                model_path: model_path.clone(),
                use_gpu: *use_gpu,
                n_ctx: *n_ctx,
                max_output_tokens: *max_output_tokens,
                ..Default::default()
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
