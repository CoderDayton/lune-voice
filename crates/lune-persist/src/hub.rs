use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use tracing::info;

use crate::error::{PersistError, PersistResult};

/// What to download from a HuggingFace repository.
pub enum ModelSpec {
    /// A directory of files (safetensors, configs, tokenizers).
    Directory {
        repo_id: String,
        /// Files that must exist (e.g. `config.json`).
        required_files: Vec<String>,
        /// Suffix patterns to match (e.g. `.safetensors`).
        suffix_patterns: Vec<String>,
    },
    /// A single file from a repository.
    SingleFile {
        repo_id: String,
        filename: String,
    },
}

/// Resolved local path after download.
pub enum ResolvedModel {
    Directory(PathBuf),
    File(PathBuf),
}

impl ResolvedModel {
    pub fn path(&self) -> &Path {
        match self {
            Self::Directory(p) | Self::File(p) => p,
        }
    }
}

/// Resolved paths for both default models.
pub struct ResolvedModels {
    pub asr_model_dir: PathBuf,
    pub llm_model_path: PathBuf,
}

impl ModelSpec {
    /// Voxtral ASR: directory with config, tokenizer, and safetensors.
    pub fn voxtral_asr() -> Self {
        Self::Directory {
            repo_id: "mistralai/Voxtral-Mini-4B-Realtime-2602".into(),
            required_files: vec!["config.json".into(), "tekken.json".into()],
            suffix_patterns: vec![".safetensors".into()],
        }
    }

    /// Qwen3 LLM: single GGUF file.
    pub fn qwen3_llm() -> Self {
        Self::SingleFile {
            repo_id: "unsloth/Qwen3-0.6B-GGUF".into(),
            filename: "Qwen3-0.6B-Q5_K_M.gguf".into(),
        }
    }
}

/// Thin wrapper around `hf_hub::api::sync::Api` with a fixed cache directory.
pub struct ModelHub {
    api: Api,
}

impl ModelHub {
    pub fn new(cache_dir: &Path) -> PersistResult<Self> {
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .with_progress(true)
            .build()
            .map_err(|e| PersistError::ModelDownload(format!("init hub api: {e}")))?;
        Ok(Self { api })
    }

    /// Ensure a model is present locally, downloading if needed.
    /// Idempotent — returns cached paths on subsequent calls.
    pub fn ensure(&self, spec: &ModelSpec) -> PersistResult<ResolvedModel> {
        match spec {
            ModelSpec::SingleFile { repo_id, filename } => {
                self.ensure_single_file(repo_id, filename)
            }
            ModelSpec::Directory {
                repo_id,
                required_files,
                suffix_patterns,
            } => self.ensure_directory(repo_id, required_files, suffix_patterns),
        }
    }

    fn ensure_single_file(
        &self,
        repo_id: &str,
        filename: &str,
    ) -> PersistResult<ResolvedModel> {
        let repo = self.api.model(repo_id.to_string());
        info!(repo_id, filename, "ensuring single model file");
        let path = repo.get(filename).map_err(|e| {
            PersistError::ModelDownload(format!("{repo_id}/{filename}: {e}"))
        })?;
        info!(path = %path.display(), "model file ready");
        Ok(ResolvedModel::File(path))
    }

    fn ensure_directory(
        &self,
        repo_id: &str,
        required_files: &[String],
        suffix_patterns: &[String],
    ) -> PersistResult<ResolvedModel> {
        let repo = self.api.model(repo_id.to_string());
        info!(repo_id, "discovering repository files");

        let file_list = self.list_repo_files(&repo, repo_id)?;

        let to_download: Vec<&str> = file_list
            .iter()
            .filter(|f| {
                required_files.iter().any(|r| *f == r)
                    || suffix_patterns.iter().any(|s| f.ends_with(s.as_str()))
            })
            .map(String::as_str)
            .collect();

        // Verify all required files are present in the repo.
        for req in required_files {
            if !file_list.iter().any(|f| f == req) {
                return Err(PersistError::ModelDownload(format!(
                    "{repo_id}: required file `{req}` not found in repository"
                )));
            }
        }

        if to_download.is_empty() {
            return Err(PersistError::ModelDownload(format!(
                "{repo_id}: no files matched filters"
            )));
        }

        info!(repo_id, count = to_download.len(), "downloading model files");

        let mut first_path: Option<PathBuf> = None;
        for filename in &to_download {
            let path = repo.get(filename).map_err(|e| {
                PersistError::ModelDownload(format!("{repo_id}/{filename}: {e}"))
            })?;
            if first_path.is_none() {
                first_path = Some(path);
            }
        }

        // All files land in the same snapshot directory.
        let model_dir = first_path
            .as_ref()
            .and_then(|p| p.parent())
            .ok_or_else(|| {
                PersistError::ModelDownload(format!(
                    "{repo_id}: could not determine model directory"
                ))
            })?
            .to_path_buf();

        info!(path = %model_dir.display(), "model directory ready");
        Ok(ResolvedModel::Directory(model_dir))
    }

    fn list_repo_files(
        &self,
        repo: &ApiRepo,
        repo_id: &str,
    ) -> PersistResult<Vec<String>> {
        let info = repo.info().map_err(|e| {
            PersistError::ModelDownload(format!("{repo_id}: repo info: {e}"))
        })?;
        Ok(info
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .collect())
    }
}

/// Download both default models if not already cached.
pub fn ensure_default_models(cache_dir: &Path) -> PersistResult<ResolvedModels> {
    let hub = ModelHub::new(cache_dir)?;

    let asr = hub.ensure(&ModelSpec::voxtral_asr())?;
    let llm = hub.ensure(&ModelSpec::qwen3_llm())?;

    let asr_model_dir = match asr {
        ResolvedModel::Directory(p) => p,
        ResolvedModel::File(_) => unreachable!("voxtral spec is Directory"),
    };
    let llm_model_path = match llm {
        ResolvedModel::File(p) => p,
        ResolvedModel::Directory(_) => unreachable!("qwen3 spec is SingleFile"),
    };

    Ok(ResolvedModels {
        asr_model_dir,
        llm_model_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxtral_spec_has_expected_repo() {
        let spec = ModelSpec::voxtral_asr();
        match &spec {
            ModelSpec::Directory {
                repo_id,
                required_files,
                suffix_patterns,
            } => {
                assert_eq!(repo_id, "mistralai/Voxtral-Mini-4B-Realtime-2602");
                assert!(required_files.contains(&"config.json".to_string()));
                assert!(required_files.contains(&"tekken.json".to_string()));
                assert!(suffix_patterns.contains(&".safetensors".to_string()));
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
                assert_eq!(filename, "Qwen3-0.6B-Q5_K_M.gguf");
            }
            _ => panic!("expected SingleFile variant"),
        }
    }

    #[test]
    fn hub_constructs_with_valid_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = ModelHub::new(tmp.path());
        assert!(hub.is_ok());
    }

    #[test]
    #[ignore] // Requires network — run with `cargo test -- --ignored`
    fn integration_download_qwen3() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = ModelHub::new(tmp.path()).unwrap();
        let spec = ModelSpec::qwen3_llm();

        let resolved = hub.ensure(&spec).unwrap();
        assert!(matches!(resolved, ResolvedModel::File(_)));
        assert!(resolved.path().exists());

        // Second call should be instant (cached).
        let resolved2 = hub.ensure(&spec).unwrap();
        assert_eq!(resolved.path(), resolved2.path());
    }
}
