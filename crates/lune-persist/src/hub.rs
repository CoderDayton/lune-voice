use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
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
    /// A single file from a repository.
    SingleFile { repo_id: String, filename: String },
}

/// Resolved paths for both default models.
pub struct ResolvedModels {
    pub asr_model_dir: PathBuf,
    pub llm_model_path: PathBuf,
}

impl ModelSpec {
    /// Parakeet TDT 0.6B v3 ASR: INT8 ONNX models and vocab.
    pub fn parakeet_tdt() -> Self {
        Self::Directory {
            repo_id: "istupakov/parakeet-tdt-0.6b-v3-onnx".into(),
            required_files: vec!["vocab.txt".into()],
            suffix_patterns: vec![".int8.onnx".into()],
            local_name: "parakeet-tdt-0.6b".into(),
        }
    }

    /// Qwen3 LLM: single GGUF file.
    pub fn qwen3_llm() -> Self {
        Self::SingleFile {
            repo_id: "unsloth/Qwen3-0.6B-GGUF".into(),
            filename: "Qwen3-0.6B-Q4_K_M.gguf".into(),
        }
    }
}

/// Thin wrapper around `hf_hub::api::sync::Api` with a hidden cache directory.
///
/// Files are downloaded via hf_hub into `.hub-cache/`, then hardlinked (or
/// copied as fallback) into a flat local directory under `models/`.
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
    /// Each file is written as a real file (no symlinks). Already-present
    /// complete files are skipped.
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

        for req in required_files {
            if !file_list.iter().any(|f| f == req) {
                return Err(PersistError::ModelDownload(format!(
                    "{repo_id}: required file `{req}` not found in repository"
                )));
            }
        }

        let to_download: Vec<&str> = file_list
            .iter()
            .filter(|f| {
                required_files.iter().any(|r| *f == r)
                    || suffix_patterns.iter().any(|s| f.ends_with(s.as_str()))
            })
            .map(String::as_str)
            .collect();

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
            std::fs::copy(&real_src, &dst)
                .map_err(|e| PersistError::ModelDownload(format!("{filename}: write: {e}")))?;
        }

        Ok(())
    }

    /// Download a single file from HF Hub directly to `local_path`.
    fn download_single_file(
        &self,
        repo_id: &str,
        filename: &str,
        local_path: &Path,
    ) -> PersistResult<()> {
        let repo = self.api.model(repo_id.to_string());
        info!(repo_id, filename, "downloading model file");
        let hub_path = repo
            .get(filename)
            .map_err(|e| PersistError::ModelDownload(format!("{repo_id}/{filename}: {e}")))?;
        let real_src = hub_path
            .canonicalize()
            .map_err(|e| PersistError::ModelDownload(format!("{filename}: resolve blob: {e}")))?;
        std::fs::copy(&real_src, local_path)
            .map_err(|e| PersistError::ModelDownload(format!("{filename}: write: {e}")))?;
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

/// Download both default models if not already present.
///
/// Layout after completion:
/// ```text
/// models_dir/
///   parakeet-tdt-0.6b/
///     encoder-model.int8.onnx
///     decoder_joint-model.int8.onnx
///     vocab.txt
///   Qwen3-0.6B-Q4_K_M.gguf
///   .hub-cache/          (hf_hub internal cache)
/// ```
pub fn ensure_default_models(models_dir: &Path) -> PersistResult<ResolvedModels> {
    let hub_cache = models_dir.join(".hub-cache");
    let hub = ModelHub::new(&hub_cache)?;

    let asr_model_dir = ensure_directory_model(&hub, &ModelSpec::parakeet_tdt(), models_dir)?;

    let llm_model_path = ensure_single_file_model(&hub, &ModelSpec::qwen3_llm(), models_dir)?;

    Ok(ResolvedModels {
        asr_model_dir,
        llm_model_path,
    })
}

/// Ensure a directory-type model is present at `models_dir/<local_name>/`.
fn ensure_directory_model(
    hub: &ModelHub,
    spec: &ModelSpec,
    models_dir: &Path,
) -> PersistResult<PathBuf> {
    let (repo_id, required_files, suffix_patterns, local_name) = match spec {
        ModelSpec::Directory {
            repo_id,
            required_files,
            suffix_patterns,
            local_name,
        } => (repo_id, required_files, suffix_patterns, local_name),
        _ => {
            return Err(PersistError::ModelDownload(
                "expected Directory spec".into(),
            ));
        }
    };

    let local_dir = models_dir.join(local_name);

    // Already present — verify required files exist.
    if local_dir.exists() {
        let all_present = required_files.iter().all(|f| local_dir.join(f).is_file());
        if all_present {
            info!(path = %local_dir.display(), "model directory already present");
            return Ok(local_dir);
        }
        info!(path = %local_dir.display(), "model directory incomplete, re-downloading");
    }

    hub.download_directory(repo_id, required_files, suffix_patterns, &local_dir)?;

    info!(path = %local_dir.display(), "model directory ready");
    Ok(local_dir)
}

/// Ensure a single-file model is present at `models_dir/<filename>`.
fn ensure_single_file_model(
    hub: &ModelHub,
    spec: &ModelSpec,
    models_dir: &Path,
) -> PersistResult<PathBuf> {
    let (repo_id, filename) = match spec {
        ModelSpec::SingleFile { repo_id, filename } => (repo_id, filename),
        _ => {
            return Err(PersistError::ModelDownload(
                "expected SingleFile spec".into(),
            ));
        }
    };

    let local_path = models_dir.join(filename);

    if local_path.is_file() {
        info!(path = %local_path.display(), "model file already present");
        return Ok(local_path);
    }

    hub.download_single_file(repo_id, filename, &local_path)?;

    info!(path = %local_path.display(), "model file ready");
    Ok(local_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_spec_has_expected_repo() {
        let spec = ModelSpec::parakeet_tdt();
        match &spec {
            ModelSpec::Directory {
                repo_id,
                required_files,
                suffix_patterns,
                local_name,
            } => {
                assert_eq!(repo_id, "istupakov/parakeet-tdt-0.6b-v3-onnx");
                assert!(required_files.contains(&"vocab.txt".to_string()));
                assert!(suffix_patterns.contains(&".int8.onnx".to_string()));
                assert_eq!(local_name, "parakeet-tdt-0.6b");
            }
            _ => panic!("expected Directory variant"),
        }
    }

    #[test]
    fn qwen3_spec_has_expected_repo() {
        let spec = ModelSpec::qwen3_llm();
        match &spec {
            ModelSpec::SingleFile { repo_id, filename } => {
                assert_eq!(repo_id, "unsloth/Qwen3-0.6B-GGUF");
                assert_eq!(filename, "Qwen3-0.6B-Q4_K_M.gguf");
            }
            _ => panic!("expected SingleFile variant"),
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
    fn integration_download_qwen3() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path();

        let resolved = ensure_default_models(models_dir).unwrap();

        // LLM file should be at models_dir/Qwen3-0.6B-Q5_K_M.gguf
        assert!(resolved.llm_model_path.exists());
        assert_eq!(
            resolved.llm_model_path.file_name().unwrap(),
            "Qwen3-0.6B-Q4_K_M.gguf"
        );
        assert_eq!(resolved.llm_model_path.parent().unwrap(), models_dir);

        // Second call should be instant (already present).
        let resolved2 = ensure_default_models(models_dir).unwrap();
        assert_eq!(resolved.llm_model_path, resolved2.llm_model_path);
    }
}
