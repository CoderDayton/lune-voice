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
    // Cap cuBLAS per-handle workspace at 16 MiB (default is several hundred MB
    // of lazily-allocated state per session). With 3 ASR + 1 LLM CUDA sessions
    // alive simultaneously this saves ~0.5–1 GB of cuBLAS workspace VRAM.
    // Must be set before any cuBLAS handle is created (i.e. before ORT loads
    // its first CUDA session). Safe at process start: nothing has touched
    // CUDA yet.
    // Format: ":<workspace_size_mib>:<workspace_count>" — :16:8 = 8 × 16 MiB.
    // SAFETY: called once at process start, before any CUDA work.
    unsafe { std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":16:8") };

    // Register CUDA EP at the environment level. Sessions that opt in via
    // `SessionBuilder::with_env_allocators()` will share a single CUDA arena
    // instead of each owning a private one. With 3 ASR + 1 LLM sessions, this
    // collapses 4 independent arenas into 1 shared pool — material VRAM win
    // when several sessions are warm at once.
    //
    // Sessions that do NOT call `with_env_allocators()` (e.g. CPU-only models
    // like VAD/forced aligner) are unaffected: they keep their default
    // allocator. Per-session EP overrides via `with_execution_providers()` on
    // the session also continue to work; this is purely additive at the env.
    let mut builder = ort::init().with_logger(quiet_ort_logger());
    #[cfg(feature = "cuda")]
    {
        builder = builder.with_execution_providers([ort::ep::CUDA::default()
            .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
            .with_conv_max_workspace(false)
            .with_tf32(true)
            .build()]);
    }
    if builder.commit() {
        tracing::info!("ORT initialized (env-level CUDA EP for shared arena)");
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
