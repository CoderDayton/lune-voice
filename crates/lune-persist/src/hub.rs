use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use lune_core::config::AsrBackendConfig;
use tracing::info;

use crate::error::{PersistError, PersistResult};

/// What to download from a HuggingFace repository.
pub enum ModelSpec {
    /// A directory of files (ONNX models, configs, tokenizers).
    Directory {
        repo_id: String,
        /// Files that must exist (e.g. `vocab.txt`).
        required_files: Vec<String>,
        /// Suffix patterns to match (e.g. `.onnx`).
        suffix_patterns: Vec<String>,
        /// Local directory name under `models/` (e.g. `parakeet-tdt-0.6b`).
        local_name: String,
    },
}

/// Resolved local model paths.
pub struct ResolvedModels {
    /// Parakeet TDT model directory.
    pub asr_model_dir: PathBuf,
    /// Qwen3-ASR ONNX model directory (for ORT backend).
    pub ort_qwen_asr_model_dir: PathBuf,
    /// LLM GGUF model file path.
    pub llm_model_path: PathBuf,
}

impl ModelSpec {
    /// Parakeet TDT 0.6B v3 ASR: INT8 ONNX models and vocab.
    #[inline]
    pub fn parakeet_tdt() -> Self {
        Self::Directory {
            repo_id: "istupakov/parakeet-tdt-0.6b-v3-onnx".into(),
            required_files: vec![
                "vocab.txt".into(),
                "encoder-model.int8.onnx".into(),
                "decoder_joint-model.int8.onnx".into(),
            ],
            suffix_patterns: vec![],
            local_name: "parakeet-tdt-0.6b".into(),
        }
    }

    /// Qwen3-ASR 0.6B ONNX: encoder, embed_tokens, decoder models + tokenizer.
    #[inline]
    pub fn qwen3_asr_onnx() -> Self {
        Self::Directory {
            repo_id: "lune-voice/qwen3-asr-0.6b-onnx".into(),
            required_files: vec![
                "qwen3-asr-encoder.onnx".into(),
                "qwen3-asr-decoder.onnx".into(),
                "qwen3-asr-embed-tokens.onnx".into(),
                "tokenizer.json".into(),
            ],
            suffix_patterns: vec![".onnx".into()],
            local_name: "qwen3-asr-0.6b-onnx".into(),
        }
    }

    /// LLM corrector GGUF model (Qwen2.5-1.5B-Instruct Q4_K_M).
    #[inline]
    pub fn llm_corrector() -> Self {
        Self::Directory {
            repo_id: "bartowski/Qwen2.5-1.5B-Instruct-GGUF".into(),
            required_files: vec!["Qwen2.5-1.5B-Instruct-Q4_K_M.gguf".into()],
            suffix_patterns: vec!["Q4_K_M.gguf".into()],
            local_name: "qwen2.5-1.5b-instruct-gguf".into(),
        }
    }
}

/// Thin wrapper around `hf_hub::api::sync::Api` with a hidden cache directory.
///
/// Files are downloaded via hf_hub into `.hub-cache/`, then copied into a
/// flat local directory under `models/`. We prefer `hard_link` first (zero
/// copy, same inode) and fall back to `fs::copy` only on cross-device moves.
struct ModelHub {
    api: Api,
}

impl ModelHub {
    fn new(hub_cache_dir: &Path) -> PersistResult<Self> {
        std::fs::create_dir_all(hub_cache_dir)?;
        let api = ApiBuilder::new()
            .with_cache_dir(hub_cache_dir.to_path_buf())
            .with_progress(true)
            .build()
            .map_err(|e| PersistError::ModelDownload(format!("init hub api: {e}")))?;
        Ok(Self { api })
    }

    /// Download matching files from HF Hub directly into `local_dir`.
    ///
    /// Uses hard links where possible (free on same filesystem, avoids data
    /// duplication). Falls back to `fs::copy` on EXDEV. Already-complete
    /// files are skipped via `is_file()` check.
    fn download_directory(
        &self,
        repo_id: &str,
        required_files: &[String],
        suffix_patterns: &[String],
        local_dir: &Path,
    ) -> PersistResult<()> {
        let repo = self.api.model(repo_id.to_string());
        info!(repo_id, "discovering repository files");

        let file_list = self.list_repo_files(&repo, repo_id)?;

        // Validate required files exist in remote listing first.
        for req in required_files {
            if !file_list.iter().any(|f| f == req) {
                return Err(PersistError::ModelDownload(format!(
                    "{repo_id}: required file `{req}` not found in repository"
                )));
            }
        }

        // Build download list: required files union suffix-matched files.
        // Pre-allocate to avoid repeated reallocation.
        let mut to_download: Vec<&str> = Vec::with_capacity(file_list.len());
        for f in &file_list {
            if required_files.iter().any(|r| f == r)
                || suffix_patterns.iter().any(|s| f.ends_with(s.as_str()))
            {
                to_download.push(f.as_str());
            }
        }

        if to_download.is_empty() {
            return Err(PersistError::ModelDownload(format!(
                "{repo_id}: no files matched filters"
            )));
        }

        std::fs::create_dir_all(local_dir)
            .map_err(|e| PersistError::ModelDownload(format!("create dir: {e}")))?;

        info!(
            repo_id,
            count = to_download.len(),
            "downloading model files"
        );

        for filename in &to_download {
            let dst = local_dir.join(filename);
            if dst.is_file() {
                continue;
            }
            let hub_path = repo
                .get(filename)
                .map_err(|e| PersistError::ModelDownload(format!("{repo_id}/{filename}: {e}")))?;
            let real_src = hub_path.canonicalize().map_err(|e| {
                PersistError::ModelDownload(format!("{filename}: resolve blob: {e}"))
            })?;
            // Hard-link first (zero-copy, same inode on same fs).
            // EXDEV (cross-device) → fall back to full copy.
            link_or_copy(&real_src, &dst)
                .map_err(|e| PersistError::ModelDownload(format!("{filename}: write: {e}")))?;
        }

        Ok(())
    }

    #[inline]
    fn list_repo_files(&self, repo: &ApiRepo, repo_id: &str) -> PersistResult<Vec<String>> {
        let info = repo
            .info()
            .map_err(|e| PersistError::ModelDownload(format!("{repo_id}: repo info: {e}")))?;
        let siblings = info.siblings;
        let mut files = Vec::with_capacity(siblings.len());
        files.extend(siblings.into_iter().map(|s| s.rfilename));
        Ok(files)
    }
}

/// Hard-link `src` → `dst`; on EXDEV fall back to byte copy.
///
/// Hard links are instantaneous and consume no additional disk space when
/// source and destination are on the same filesystem (the common case when
/// hf_hub cache lives under `models_dir`).
#[inline]
fn link_or_copy(src: &Path, dst: &Path) -> std::io::Result<()> {
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if e.raw_os_error() == Some(18) => {
            // EXDEV: cross-device link
            std::fs::copy(src, dst).map(|_| ())
        }
        Err(e) => Err(e),
    }
}

/// Download default models if not already present.
///
/// Only the ASR model selected by `asr_backend` is downloaded.
///
/// Layout after completion:
/// ```text
/// models_dir/
///   parakeet-tdt-0.6b/
///     encoder-model.int8.onnx
///     decoder_joint-model.int8.onnx
///     vocab.txt
///   qwen3-asr-0.6b/
///     model.safetensors
///     vocab.json
///     merges.txt
///   Qwen3-0.6B-Q4_K_M.gguf
///   .hub-cache/          (hf_hub internal cache)
/// ```
pub fn ensure_default_models(
    models_dir: &Path,
    asr_backend: &AsrBackendConfig,
    llm_backend: &lune_core::config::LlmBackendConfig,
) -> PersistResult<ResolvedModels> {
    let hub_cache = models_dir.join(".hub-cache");
    let hub = ModelHub::new(&hub_cache)?;

    let parakeet_dir = models_dir.join("parakeet-tdt-0.6b");
    let ort_qwen_asr_dir = models_dir.join("qwen3-asr-0.6b-onnx");

    let (asr_model_dir, ort_qwen_asr_model_dir) = match asr_backend {
        AsrBackendConfig::ParakeetTdt { .. } => (
            ensure_directory_model(&hub, &ModelSpec::parakeet_tdt(), models_dir)?,
            ort_qwen_asr_dir,
        ),
        AsrBackendConfig::OrtQwenAsr { .. } => (
            parakeet_dir,
            ensure_directory_model(&hub, &ModelSpec::qwen3_asr_onnx(), models_dir)?,
        ),
        _ => (parakeet_dir, ort_qwen_asr_dir),
    };

    let llm_model_path = match llm_backend {
        lune_core::config::LlmBackendConfig::LlamaCppLocal { .. } => {
            let dir = ensure_directory_model(&hub, &ModelSpec::llm_corrector(), models_dir)?;
            dir.join("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
        }
        _ => models_dir.join("qwen2.5-1.5b-instruct-gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
    };

    Ok(ResolvedModels {
        asr_model_dir,
        ort_qwen_asr_model_dir,
        llm_model_path,
    })
}

/// Ensure a directory-type model is present at `models_dir/<local_name>/`.
fn ensure_directory_model(
    hub: &ModelHub,
    spec: &ModelSpec,
    models_dir: &Path,
) -> PersistResult<PathBuf> {
    let ModelSpec::Directory {
        repo_id,
        required_files,
        suffix_patterns,
        local_name,
    } = spec;

    let local_dir = models_dir.join(local_name);

    if local_dir.exists() {
        // Fast path: all required files present → skip download entirely.
        if required_files.iter().all(|f| local_dir.join(f).is_file()) {
            info!(path = %local_dir.display(), "model directory already present");
            return Ok(local_dir);
        }
        info!(path = %local_dir.display(), "model directory incomplete, re-downloading");
    }

    hub.download_directory(repo_id, required_files, suffix_patterns, &local_dir)?;

    info!(path = %local_dir.display(), "model directory ready");
    Ok(local_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_spec_has_expected_repo() {
        let spec = ModelSpec::parakeet_tdt();
        let ModelSpec::Directory {
            repo_id,
            required_files,
            suffix_patterns,
            local_name,
        } = &spec;
        assert_eq!(repo_id, "istupakov/parakeet-tdt-0.6b-v3-onnx");
        assert!(required_files.contains(&"vocab.txt".to_string()));
        assert!(required_files.contains(&"encoder-model.int8.onnx".to_string()));
        assert!(suffix_patterns.is_empty());
        assert_eq!(local_name, "parakeet-tdt-0.6b");
    }

    #[test]
    fn llm_corrector_spec_has_expected_repo() {
        let spec = ModelSpec::llm_corrector();
        match &spec {
            ModelSpec::Directory {
                repo_id,
                required_files,
                local_name,
                ..
            } => {
                assert_eq!(repo_id, "bartowski/Qwen2.5-1.5B-Instruct-GGUF");
                assert!(required_files.contains(&"Qwen2.5-1.5B-Instruct-Q4_K_M.gguf".to_string()));
                assert_eq!(local_name, "qwen2.5-1.5b-instruct-gguf");
            }
        }
    }

    #[test]
    fn hub_constructs_with_valid_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let hub_cache = tmp.path().join(".hub-cache");
        let hub = ModelHub::new(&hub_cache);
        assert!(hub.is_ok());
    }

    #[test]
    #[ignore] // Requires network — run with `cargo test -- --ignored`
    fn integration_download_ort_llm() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path();
        let defaults = lune_core::config::UserSettings::default();
        let asr_backend = &defaults.asr_backend;
        let llm_backend = &defaults.llm_backend;

        let resolved = ensure_default_models(models_dir, asr_backend, llm_backend).unwrap();

        assert!(resolved.llm_model_path.exists());

        // Second call should be instant (already present).
        let resolved2 = ensure_default_models(models_dir, asr_backend, llm_backend).unwrap();
        assert_eq!(resolved.llm_model_path, resolved2.llm_model_path);
    }
}
