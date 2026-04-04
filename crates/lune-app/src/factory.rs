use std::path::Path;
use std::sync::Arc;

use lune_asr::AsrRegistry;
use lune_core::config::{AsrBackendConfig, LlmBackendConfig, UserSettings};
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
        // Reached only on --no-default-features builds with device != Cpu.
        #[cfg(not(feature = "cuda"))]
        {
            tracing::info!("GPU requested but cuda feature not compiled, using CPU");
        }
    }

    // CPU-only path: either device == Cpu, or CUDA init above failed.
    // The second call to ort::init() is a no-op if already initialized.
    if ort::init().commit() {
        tracing::info!("ORT initialized (CPU only)");
    } else {
        tracing::info!("ORT already initialized (CPU fallback from failed CUDA attempt)");
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
        AsrBackendConfig::OrtQwenAsr { model_dir, .. } if model_dir.as_os_str().is_empty() => {
            resolved.ort_qwen_asr_model_dir.clone_into(model_dir);
        }
        _ => {}
    }

    match &mut settings.llm_backend {
        LlmBackendConfig::LlamaCppLocal { model_path, .. } if model_path.as_os_str().is_empty() => {
            resolved.llm_model_path.clone_into(model_path);
        }
        _ => {}
    }

    if let (Some(aligner), Some(model_dir)) = (&mut settings.aligner, &resolved.aligner_model_dir)
        && aligner.model_dir.as_os_str().is_empty()
    {
        model_dir.clone_into(&mut aligner.model_dir);
    }
}

/// Construct the ASR backend from settings via [`AsrRegistry`].
///
/// Cold path: called once at startup.
#[cold]
pub fn build_asr_backend(settings: &UserSettings) -> Result<AsrRegistry> {
    AsrRegistry::from_config(&settings.asr_backend, settings.device)
}

/// Construct the LLM post-processor from settings.
///
/// Cold path: called once at startup. Async because mistralrs model loading is async.
#[cold]
pub async fn build_post_processor(settings: &UserSettings) -> Result<Arc<dyn TextPostProcessor>> {
    match &settings.llm_backend {
        LlmBackendConfig::LlamaCppLocal {
            model_path,
            n_gpu_layers,
            n_ctx,
            max_output_tokens,
        } => {
            let backend = lune_llm::LlamaCppBackend::new(lune_llm::LlamaCppConfig {
                model_path: model_path.clone(),
                n_gpu_layers: *n_gpu_layers,
                n_ctx: *n_ctx,
                n_threads: 4,
                max_output_tokens: *max_output_tokens,
            })
            .await?;
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
