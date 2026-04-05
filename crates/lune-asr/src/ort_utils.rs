//! Shared ORT model loading and session construction.
//!
//! `OrtModelLoader` centralizes model directory resolution, GPU policy,
//! and session creation so both Parakeet and Qwen backends share a single
//! code path. All sessions inherit the global CUDA EP from `init_ort_runtime`.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use lune_core::error::{Error, Result};

/// Shared ORT model loading context.
///
/// Owns the model directory and GPU preference. Both ASR backends
/// construct from this, ensuring consistent session configuration.
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

    /// Build an ORT session from an arbitrary path (e.g. forced aligner).
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
        // Fallback: glob for `{kind}*.onnx`
        if let Ok(entries) = std::fs::read_dir(&self.model_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|s| s.to_str())
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
}

/// Build an ORT session from an ONNX model file.
///
/// Applies Level3 graph optimization and 4 intra-op threads.
/// Execution providers are inherited from the global `ort::init()` call.
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
