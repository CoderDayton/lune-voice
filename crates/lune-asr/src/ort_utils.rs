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
/// Execution providers are inherited from the global `ort::init()` call
/// (see `factory::init_ort_runtime`). All sessions share the same CUDA
/// memory arena when GPU is enabled — do NOT register per-session EPs.
pub fn build_ort_session(path: &Path, _use_gpu: bool) -> Result<Session> {
    Session::builder()
        .map_err(|e| Error::Asr(format!("session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Asr(format!("optimization level: {e}")))?
        .with_intra_threads(4)
        .map_err(|e| Error::Asr(format!("intra threads: {e}")))?
        .commit_from_file(path)
        .map_err(|e| Error::Asr(format!("load model {}: {e}", path.display())))
}
