use std::path::Path;
use std::sync::Arc;

use lune_asr::AsrRegistry;
use lune_core::config::{AsrBackendConfig, UserSettings};
use lune_core::error::{Error, Result};
use lune_core::llm::TextPostProcessor;
use lune_persist::ResolvedModels;

// ── Global ORT initialization ────────────────────────────────────────────────

/// Initialize the global ORT environment with execution providers.
///
/// Must be called exactly once, before any ORT `Session` is created.
/// All sessions share a single CUDA memory arena when GPU is enabled.
///
/// Resolution order:
/// 1. `settings.device == Cpu` → CPU only, skip CUDA attempt.
/// 2. `settings.device == Cuda | Auto` + cuda feature compiled in
///    → try CUDA EP; fall back to CPU on runtime failure.
/// 3. `settings.device == Cuda | Auto` + cuda feature NOT compiled
///    → log warning, use CPU (handles `--no-default-features` builds).
///
/// The `cuda` feature is enabled by default, so `Auto` is equivalent to
/// "try CUDA, fall back to CPU" on standard builds.
#[cold]
pub fn init_ort_runtime() {
    // CPU-only global init. CUDA EP is registered per-session by
    // `build_ort_session` for ASR models — keeps VAD and forced
    // aligner on CPU automatically.
    if ort::init().commit() {
        tracing::info!("ORT initialized (CPU global, CUDA per-session for ASR)");
    } else {
        tracing::info!("ORT already initialized");
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
            resolved.asr_model_dir.clone_into(model_dir);
        }

        _ => {}
    }
}

/// Construct the ASR backend from settings via [`AsrRegistry`].
///
/// Cold path: called once at startup.
#[cold]
pub fn build_asr_backend(settings: &UserSettings) -> Result<AsrRegistry> {
    AsrRegistry::from_config(&settings.asr_backend, settings.device)
}

/// Construct the post-processor for the session pipeline.
///
/// All local correction now flows through the Nemotron cross-encoder
/// reranker, which is wired separately via `SessionManager::with_reranker`.
/// This function returns a `NullPostProcessor` (passthrough) so the
/// session's `TextPostProcessor` slot is filled — the trait stays for
/// the OpenAI remote correction path, which is not yet wired in v0.1.
///
/// Cold path: called once at startup.
#[cold]
pub async fn build_post_processor(_settings: &UserSettings) -> Result<Arc<dyn TextPostProcessor>> {
    Ok(Arc::new(lune_llm::NullPostProcessor::new()))
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
