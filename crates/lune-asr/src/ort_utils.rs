//! Shared ORT session construction utilities.
//!
//! Extracted from `tdt_decoder::build_session` so both Parakeet TDT and the
//! new ORT-based Qwen ASR/LLM backends share a single code path for session
//! creation, EP selection, and optimization settings.

use std::path::Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use lune_core::error::{Error, Result};

/// Build an ORT session from an ONNX model file.
///
/// Applies Level3 graph optimization and 4 intra-op threads.
/// When `use_gpu` is true and the `cuda` feature is enabled, registers the
/// CUDA EP with CPU fallback.  Without the feature, logs a warning and
/// falls back to CPU-only.
///
/// The global `ort::init()` call (if any) supplies the default EP and memory
/// arena.  Per-session EPs registered here override the global default for
/// this session only.
pub fn build_ort_session(path: &Path, use_gpu: bool) -> Result<Session> {
    let builder = Session::builder()
        .map_err(|e| Error::Asr(format!("session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Asr(format!("optimization level: {e}")))?
        .with_intra_threads(4)
        .map_err(|e| Error::Asr(format!("intra threads: {e}")))?;

    let builder = if use_gpu {
        #[cfg(feature = "cuda")]
        {
            builder
                .with_execution_providers([
                    ort::ep::CUDA::default().build(),
                    ort::ep::CPU::default().build().error_on_failure(),
                ])
                .map_err(|e| Error::Asr(format!("CUDA provider: {e}")))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::warn!("use_gpu=true but cuda feature not enabled, falling back to CPU");
            builder
        }
    } else {
        builder
    };

    builder
        .commit_from_file(path)
        .map_err(|e| Error::Asr(format!("load model {}: {e}", path.display())))
}
