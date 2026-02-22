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

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        match self {
            Self::Parakeet(b) => Ok(AsrSessionDispatch::Parakeet(b.start_session(config).await?)),
            Self::OpenAi(b) => Ok(AsrSessionDispatch::OpenAi(b.start_session(config).await?)),
        }
    }

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        match (self, session) {
            (Self::Parakeet(b), AsrSessionDispatch::Parakeet(s)) => b.push_audio(s, pcm).await,
            (Self::OpenAi(b), AsrSessionDispatch::OpenAi(s)) => b.push_audio(s, pcm).await,
            _ => Err(Error::Asr("backend/session type mismatch".into())),
        }
    }

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        match (self, session) {
            (Self::Parakeet(b), AsrSessionDispatch::Parakeet(s)) => b.poll_results(s).await,
            (Self::OpenAi(b), AsrSessionDispatch::OpenAi(s)) => b.poll_results(s).await,
            _ => Err(Error::Asr("backend/session type mismatch".into())),
        }
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        match (self, session) {
            (Self::Parakeet(b), AsrSessionDispatch::Parakeet(s)) => b.end_session(s).await,
            (Self::OpenAi(b), AsrSessionDispatch::OpenAi(s)) => b.end_session(s).await,
            _ => Err(Error::Asr("backend/session type mismatch".into())),
        }
    }
}

// ── Factory functions ────────────────────────────────────────────────────────

/// Patch empty model paths in `settings` from `resolved` hub paths.
///
/// The default `UserSettings` ships with empty `PathBuf`s for model paths.
/// After `ensure_default_models` resolves the actual paths, this fills them in.
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
pub fn build_post_processor(settings: &UserSettings) -> Result<Arc<dyn TextPostProcessor>> {
    match &settings.llm_backend {
        LlmBackendConfig::Local {
            model_path,
            use_gpu,
            n_ctx,
        } => {
            let backend = LocalLlmBackend::new(lune_llm::LocalLlmConfig {
                model_path: model_path.clone(),
                use_gpu: *use_gpu,
                n_ctx: *n_ctx,
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
pub fn ensure_db_path(db_path: &Path) -> Result<()> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Config(format!("create db dir: {e}")))?;
    }
    Ok(())
}
