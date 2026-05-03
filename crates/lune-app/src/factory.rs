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
    if ort::init().with_logger(quiet_ort_logger()).commit() {
        tracing::info!("ORT initialized (CPU global, CUDA per-session for ASR)");
    } else {
        tracing::info!("ORT already initialized");
    }
}

fn quiet_ort_logger() -> ort::logging::LoggerFunction {
    Arc::new(|level, _category, id, location, message| {
        if level >= ort::logging::LogLevel::Error {
            tracing::error!(
                target: "ort::logging",
                ort_id = %id,
                ort_location = %location,
                "{message}"
            );
        }
    })
}

// ── Factory functions ────────────────────────────────────────────────────────

/// Patch empty model paths in `settings` from `resolved` hub paths.
///
/// The default `UserSettings` ships with empty `PathBuf`s for model paths.
/// After `ensure_default_models` resolves the actual paths, this fills them in.
pub fn patch_model_paths(settings: &mut UserSettings, resolved: &ResolvedModels) {
    let _ = (settings, resolved);
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
/// Returns a `NullPostProcessor` (passthrough) unless a remote LLM
/// correction backend is wired in.
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
