//! Shared ORT model loading and session construction.
//!
//! `OrtModelLoader` centralizes model directory resolution, GPU policy,
//! and session creation for ORT-backed ASR. CUDA EP is registered per-session
//! when `use_gpu` is true.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ort::logging::{LogLevel, LoggerFunction};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use lune_core::error::{Error, Result};

fn quiet_ort_logger() -> LoggerFunction {
    Arc::new(|level, _category, id, location, message| {
        if level >= LogLevel::Error {
            tracing::error!(
                target: "ort::logging",
                ort_id = %id,
                ort_location = %location,
                "{message}"
            );
        }
    })
}

/// Shared ORT model loading context.
#[derive(Debug, Clone)]
pub struct OrtModelLoader {
    pub model_dir: PathBuf,
    pub use_gpu: bool,
}

impl OrtModelLoader {
    pub fn new(model_dir: PathBuf, use_gpu: bool) -> Self {
        Self { model_dir, use_gpu }
    }

    /// Build an ORT session for a model file within the model directory.
    pub fn load(&self, filename: &str) -> Result<Session> {
        let path = self.model_dir.join(filename);
        build_ort_session(&path, self.use_gpu)
    }

    /// Build an ORT session from an arbitrary path.
    pub fn load_path(&self, path: &Path) -> Result<Session> {
        build_ort_session(path, self.use_gpu)
    }

    /// Find the first existing model file from a list of candidates.
    pub fn find(&self, candidates: &[&str], kind: &str) -> Result<PathBuf> {
        for name in candidates {
            let p = self.model_dir.join(name);
            if p.exists() {
                return Ok(p);
            }
        }
        if let Ok(entries) = std::fs::read_dir(&self.model_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str())
                    && name.starts_with(kind)
                    && name.ends_with(".onnx")
                {
                    return Ok(p);
                }
            }
        }
        Err(Error::Asr(format!(
            "no {kind} model found in {}",
            self.model_dir.display()
        )))
    }

    /// Build an ORT session for the first matching candidate.
    pub fn load_first(&self, candidates: &[&str], kind: &str) -> Result<Session> {
        let path = self.find(candidates, kind)?;
        build_ort_session(&path, self.use_gpu)
    }

    /// Build ORT session with pre-loaded external data shared across sessions.
    pub fn load_first_with_external_data(
        &self,
        candidates: &[&str],
        kind: &str,
        ext_filename: &str,
        ext_data: std::borrow::Cow<'static, [u8]>,
    ) -> Result<Session> {
        let path = self.find(candidates, kind)?;
        build_ort_session_with_external_data(&path, self.use_gpu, ext_filename, ext_data)
    }
}

/// Build an ORT session from an ONNX model file.
///
/// Applies Level3 graph optimization and 4 intra-op threads.
/// When `use_gpu` is true, registers a per-session CUDA EP.
/// Saves the graph-optimized model to `<path>.opt` on first load.
pub fn build_ort_session(path: &Path, use_gpu: bool) -> Result<Session> {
    make_builder(path, use_gpu)?
        .commit_from_file(path)
        .map_err(|e| Error::Asr(format!("load model {}: {e}", path.display())))
}

/// Build ORT session with external data pre-loaded in memory (shared across sessions).
pub fn build_ort_session_with_external_data(
    path: &Path,
    use_gpu: bool,
    ext_filename: &str,
    ext_data: std::borrow::Cow<'static, [u8]>,
) -> Result<Session> {
    make_builder(path, use_gpu)?
        .with_external_initializer_file_in_memory(ext_filename, ext_data)
        .map_err(|e| Error::Asr(format!("external data: {e}")))?
        .commit_from_file(path)
        .map_err(|e| Error::Asr(format!("load model {}: {e}", path.display())))
}

fn make_builder(path: &Path, use_gpu: bool) -> Result<ort::session::builder::SessionBuilder> {
    let opt_path = path.with_extension("onnx.opt");

    #[allow(unused_mut)]
    let mut builder = Session::builder()
        .map_err(|e| Error::Asr(format!("session builder: {e}")))?
        .with_logger(quiet_ort_logger())
        .map_err(|e| Error::Asr(format!("logger: {e}")))?
        .with_log_level(LogLevel::Error)
        .map_err(|e| Error::Asr(format!("log level: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Asr(format!("optimization level: {e}")))?
        .with_intra_threads(4)
        .map_err(|e| Error::Asr(format!("intra threads: {e}")))?
        .with_optimized_model_path(&opt_path)
        .map_err(|e| Error::Asr(format!("optimized model path: {e}")))?;

    #[cfg(feature = "cuda")]
    if use_gpu {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default()
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                .with_tf32(true)
                .build()])
            .map_err(|e| Error::Asr(format!("CUDA EP: {e}")))?;
    }

    #[cfg(not(feature = "cuda"))]
    let _ = use_gpu;

    Ok(builder)
}
