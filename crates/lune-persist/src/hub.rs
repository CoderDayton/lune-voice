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

/// Status of a single model: present locally or needs download.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    /// Human-readable name (e.g. "Qwen3 ASR 0.6B")
    pub name: String,
    /// HuggingFace repo ID
    pub repo_id: String,
    /// Local directory name
    pub local_name: String,
    /// Whether the model is fully present locally
    pub ready: bool,
}

/// Progress update during model download.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub model_name: String,
    pub file_name: String,
    pub file_index: usize,
    pub file_count: usize,
    pub done: bool,
}

/// Resolved local model paths.
pub struct ResolvedModels {
    /// Parakeet TDT model directory.
    pub asr_model_dir: PathBuf,
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

    /// Human-readable display name for this model.
    pub fn display_name(&self) -> &str {
        match self {
            Self::Directory { local_name, .. } => match local_name.as_str() {
                "parakeet-tdt-0.6b" => "Parakeet TDT 0.6B",
                _ => local_name,
            },
        }
    }

    /// Check if this model is fully present locally (no download needed).
    pub fn is_ready(&self, models_dir: &Path) -> bool {
        match self {
            Self::Directory {
                required_files,
                local_name,
                ..
            } => {
                let dir = models_dir.join(local_name);
                dir.exists() && required_files.iter().all(|f| dir.join(f).is_file())
            }
        }
    }

    fn repo_id(&self) -> &str {
        match self {
            Self::Directory { repo_id, .. } => repo_id,
        }
    }

    fn local_name(&self) -> &str {
        match self {
            Self::Directory { local_name, .. } => local_name,
        }
    }
}

/// Check which models are needed for the current config. Returns status of each.
/// Does NOT download anything.
pub fn check_model_status(
    models_dir: &Path,
    asr_backend: &AsrBackendConfig,
    _llm_backend: &lune_core::config::LlmBackendConfig,
) -> Vec<ModelStatus> {
    let mut statuses = Vec::new();

    let asr_spec = match asr_backend {
        AsrBackendConfig::ParakeetTdt { .. } => Some(ModelSpec::parakeet_tdt()),
        _ => None,
    };

    if let Some(spec) = &asr_spec {
        statuses.push(ModelStatus {
            name: spec.display_name().to_string(),
            repo_id: spec.repo_id().to_string(),
            local_name: spec.local_name().to_string(),
            ready: spec.is_ready(models_dir),
        });
    }

    statuses
}

/// Download a single model by local_name. Calls `on_progress` for each file.
/// Blocking — run on a background thread.
pub fn download_model(
    models_dir: &Path,
    local_name: &str,
    on_progress: impl Fn(DownloadProgress),
) -> PersistResult<PathBuf> {
    let spec = match local_name {
        "parakeet-tdt-0.6b" => ModelSpec::parakeet_tdt(),
        _ => {
            return Err(PersistError::ModelDownload(format!(
                "unknown model: {local_name}"
            )));
        }
    };

    let hub_cache = models_dir.join(".hub-cache");
    let hub = ModelHub::new(&hub_cache)?;

    let name = spec.display_name().to_string();
    on_progress(DownloadProgress {
        model_name: name.clone(),
        file_name: String::new(),
        file_index: 0,
        file_count: 0,
        done: false,
    });

    let path =
        ensure_directory_model_with_progress(&hub, &spec, models_dir, |file, idx, total| {
            on_progress(DownloadProgress {
                model_name: name.clone(),
                file_name: file.to_string(),
                file_index: idx,
                file_count: total,
                done: false,
            });
        })?;

    on_progress(DownloadProgress {
        model_name: name,
        file_name: String::new(),
        file_index: 0,
        file_count: 0,
        done: true,
    });

    Ok(path)
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
    _asr_backend: &AsrBackendConfig,
) -> PersistResult<ResolvedModels> {
    let hub_cache = models_dir.join(".hub-cache");
    let hub = ModelHub::new(&hub_cache)?;

    let asr_model_dir = ensure_directory_model(&hub, &ModelSpec::parakeet_tdt(), models_dir)?;

    Ok(ResolvedModels { asr_model_dir })
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

/// Like `ensure_directory_model` but calls `on_file` for each file downloaded.
fn ensure_directory_model_with_progress(
    hub: &ModelHub,
    spec: &ModelSpec,
    models_dir: &Path,
    on_file: impl Fn(&str, usize, usize),
) -> PersistResult<PathBuf> {
    let ModelSpec::Directory {
        repo_id,
        required_files,
        suffix_patterns,
        local_name,
    } = spec;

    let local_dir = models_dir.join(local_name);

    if local_dir.exists() && required_files.iter().all(|f| local_dir.join(f).is_file()) {
        info!(path = %local_dir.display(), "model directory already present");
        return Ok(local_dir);
    }

    // Determine which files to download
    let repo = hub.api.repo(hf_hub::Repo::model(repo_id.clone()));
    let all_files = hub.list_repo_files(&repo, repo_id)?;
    let files_to_download: Vec<&str> = all_files
        .iter()
        .filter(|f| {
            required_files.contains(f)
                || suffix_patterns.iter().any(|pat| f.ends_with(pat.as_str()))
        })
        .map(String::as_str)
        .collect();

    let total = files_to_download.len();
    std::fs::create_dir_all(&local_dir)
        .map_err(|e| PersistError::ModelDownload(format!("create dir: {e}")))?;

    for (idx, filename) in files_to_download.iter().enumerate() {
        on_file(filename, idx, total);

        let hub_path = repo
            .get(filename)
            .map_err(|e| PersistError::ModelDownload(format!("{repo_id}/{filename}: {e}")))?;
        let real_src = hub_path
            .canonicalize()
            .map_err(|e| PersistError::ModelDownload(format!("{filename}: resolve: {e}")))?;
        let dst = local_dir.join(filename);
        link_or_copy(&real_src, &dst)
            .map_err(|e| PersistError::ModelDownload(format!("{filename}: write: {e}")))?;
    }

    info!(path = %local_dir.display(), "model directory ready");
    Ok(local_dir)
}

/// Fill in empty model paths with their expected local locations.
///
/// Does **not** download anything or touch the network — just resolves
/// where each model *would* live under `models_dir/`. Safe to call at
/// startup even when models have not been downloaded yet.
pub fn resolve_local_model_paths(
    models_dir: &Path,
    settings: &mut lune_core::config::UserSettings,
) {
    use lune_core::config::AsrBackendConfig;

    match &mut settings.asr_backend {
        AsrBackendConfig::ParakeetTdt { model_dir } if model_dir.as_os_str().is_empty() => {
            *model_dir = models_dir.join("parakeet-tdt-0.6b");
        }
        _ => {}
    }
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
    fn hub_constructs_with_valid_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let hub_cache = tmp.path().join(".hub-cache");
        let hub = ModelHub::new(&hub_cache);
        assert!(hub.is_ok());
    }

    #[test]
    #[ignore] // Requires network — run with `cargo test -- --ignored`
    fn integration_download_parakeet() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path();
        let defaults = lune_core::config::UserSettings::default();

        let resolved = ensure_default_models(models_dir, &defaults.asr_backend).unwrap();
        assert!(resolved.asr_model_dir.exists());

        // Second call should be instant (already present).
        let resolved2 = ensure_default_models(models_dir, &defaults.asr_backend).unwrap();
        assert_eq!(resolved.asr_model_dir, resolved2.asr_model_dir);
    }
}
