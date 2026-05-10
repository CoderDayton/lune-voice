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

/// Helper trait for concise error context wrapping in ORT operations.
///
/// Chains ORT errors with a descriptive label via `map_err`, converting them
/// to `Error::Asr` with formatted context. Eliminates repetitive `.map_err(|e| Error::Asr(format!(...)))?`.
pub trait OrtContext<T> {
    fn ctx(self, label: &str) -> Result<T>;
}

impl<T, E: std::fmt::Display> OrtContext<T> for std::result::Result<T, E> {
    fn ctx(self, label: &str) -> Result<T> {
        self.map_err(|e| Error::Asr(format!("{label}: {e}")))
    }
}

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
    /// When true, CUDA EP registration failure is treated as a hard error
    /// (mirrors `DevicePolicy::Cuda`). When false, EP registration silently
    /// falls back to CPU on failure (mirrors `DevicePolicy::Auto`).
    pub cuda_required: bool,
}

impl OrtModelLoader {
    pub fn new(model_dir: PathBuf, use_gpu: bool, cuda_required: bool) -> Self {
        Self {
            model_dir,
            use_gpu,
            cuda_required,
        }
    }

    /// Build an ORT session for a model file within the model directory.
    pub fn load(&self, filename: &str) -> Result<Session> {
        let path = self.model_dir.join(filename);
        build_ort_session(&path, self.use_gpu, self.cuda_required)
    }

    /// Build an ORT session from an arbitrary path.
    pub fn load_path(&self, path: &Path) -> Result<Session> {
        build_ort_session(path, self.use_gpu, self.cuda_required)
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
        build_ort_session(&path, self.use_gpu, self.cuda_required)
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
        build_ort_session_with_external_data(
            &path,
            self.use_gpu,
            self.cuda_required,
            ext_filename,
            ext_data,
        )
    }
}

/// Build an ORT session from an ONNX model file.
///
/// Applies Level3 graph optimization and 4 intra-op threads.
/// When `use_gpu` is true, registers a per-session CUDA EP. When
/// `cuda_required` is also true, EP registration failure is hard-errored
/// instead of silently falling back to CPU.
/// Saves the graph-optimized model to `<path>.opt` on first load.
pub fn build_ort_session(path: &Path, use_gpu: bool, cuda_required: bool) -> Result<Session> {
    let session = make_builder(path, use_gpu, cuda_required)?
        .commit_from_file(path)
        .map_err(|e| Error::Asr(format!("load model {}: {e}", path.display())))?;
    log_session_loaded(path, use_gpu, None);
    Ok(session)
}

/// Build ORT session with external data pre-loaded in memory (shared across sessions).
pub fn build_ort_session_with_external_data(
    path: &Path,
    use_gpu: bool,
    cuda_required: bool,
    ext_filename: &str,
    ext_data: std::borrow::Cow<'static, [u8]>,
) -> Result<Session> {
    let ext_len = ext_data.len();
    let session = make_builder(path, use_gpu, cuda_required)?
        .with_external_initializer_file_in_memory(ext_filename, ext_data)
        .map_err(|e| Error::Asr(format!("external data: {e}")))?
        .commit_from_file(path)
        .map_err(|e| Error::Asr(format!("load model {}: {e}", path.display())))?;
    log_session_loaded(path, use_gpu, Some(ext_len as u64));
    Ok(session)
}

fn log_session_loaded(path: &Path, use_gpu: bool, ext_data_bytes: Option<u64>) {
    let onnx_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let data_path = path.with_extension("onnx_data");
    let data_bytes = ext_data_bytes
        .or_else(|| std::fs::metadata(&data_path).ok().map(|m| m.len()))
        .unwrap_or(0);
    let total_mib = (onnx_bytes + data_bytes) as f64 / (1024.0 * 1024.0);
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");
    tracing::info!(
        target: "vram",
        model = name,
        device = if use_gpu { "cuda" } else { "cpu" },
        weights_mib = format_args!("{total_mib:.1}"),
        "ASR session loaded"
    );
}

fn make_builder(
    path: &Path,
    use_gpu: bool,
    cuda_required: bool,
) -> Result<ort::session::builder::SessionBuilder> {
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
        .map_err(|e| Error::Asr(format!("optimized model path: {e}")))?
        // Share the env-registered CUDA arena across all ASR sessions.
        .with_env_allocators()
        .map_err(|e| Error::Asr(format!("env allocators: {e}")))?;

    #[cfg(feature = "cuda")]
    if use_gpu {
        // Arena strategy: use the ORT default (kNextPowerOfTwo). The IO-binding
        // decode loop allocates KV outputs of slightly different sizes each step
        // (sequence length grows by 1 token). Power-of-two buckets coalesce
        // those into a small set of pools that reuse across dictations.
        // SameAsRequested fragments the arena and grew VRAM ~1 GB per dictation.
        let cuda_ep = ort::ep::CUDA::default()
            .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
            .with_conv_max_workspace(false)
            .with_tf32(true)
            .build();
        let cuda_ep = if cuda_required {
            cuda_ep.error_on_failure()
        } else {
            cuda_ep
        };
        builder = builder
            .with_execution_providers([cuda_ep])
            .map_err(|e| Error::Asr(format!("CUDA EP: {e}")))?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = use_gpu;
        let _ = cuda_required;
    }

    Ok(builder)
}
