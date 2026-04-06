use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use lune_core::config::{AlignerConfig, AsrBackendConfig};
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
    /// Qwen3-ASR ONNX model directory (for ORT backend).
    pub ort_qwen_asr_model_dir: PathBuf,
    /// LLM GGUF model file path.
    pub llm_model_path: PathBuf,
    /// Optional forced-aligner model directory.
    pub aligner_model_dir: Option<PathBuf>,
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

    /// Human-readable display name for this model.
    pub fn display_name(&self) -> &str {
        match self {
            Self::Directory { local_name, .. } => match local_name.as_str() {
                "parakeet-tdt-0.6b" => "Parakeet TDT 0.6B",
                "qwen3-asr-0.6b-onnx" => "Qwen3 ASR 0.6B",
                "qwen2.5-1.5b-instruct-gguf" => "Qwen2.5 1.5B LLM",
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
    llm_backend: &lune_core::config::LlmBackendConfig,
) -> Vec<ModelStatus> {
    let mut statuses = Vec::new();

    let asr_spec = match asr_backend {
        AsrBackendConfig::ParakeetTdt { .. } => Some(ModelSpec::parakeet_tdt()),
        AsrBackendConfig::OrtQwenAsr { .. } => Some(ModelSpec::qwen3_asr_onnx()),
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

    if matches!(
        llm_backend,
        lune_core::config::LlmBackendConfig::LlamaCppLocal { .. }
    ) {
        let spec = ModelSpec::llm_corrector();
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
        "qwen3-asr-0.6b-onnx" => ModelSpec::qwen3_asr_onnx(),
        "qwen2.5-1.5b-instruct-gguf" => ModelSpec::llm_corrector(),
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
    asr_backend: &AsrBackendConfig,
    llm_backend: &lune_core::config::LlmBackendConfig,
    aligner: Option<&AlignerConfig>,
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

    let aligner_model_dir = resolve_aligner_model_dir(models_dir, aligner);

    Ok(ResolvedModels {
        asr_model_dir,
        ort_qwen_asr_model_dir,
        llm_model_path,
        aligner_model_dir,
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
    use lune_core::config::{AsrBackendConfig, LlmBackendConfig};

    match &mut settings.asr_backend {
        AsrBackendConfig::ParakeetTdt { model_dir } if model_dir.as_os_str().is_empty() => {
            *model_dir = models_dir.join("parakeet-tdt-0.6b");
        }
        AsrBackendConfig::OrtQwenAsr { model_dir } if model_dir.as_os_str().is_empty() => {
            *model_dir = models_dir.join("qwen3-asr-0.6b-onnx");
        }
        _ => {}
    }

    match &mut settings.llm_backend {
        LlmBackendConfig::LlamaCppLocal { model_path, .. } if model_path.as_os_str().is_empty() => {
            *model_path = models_dir
                .join("qwen2.5-1.5b-instruct-gguf")
                .join("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf");
        }
        _ => {}
    }

    if let Some(aligner) = &mut settings.aligner
        && aligner.enabled
        && aligner.model_dir.as_os_str().is_empty()
    {
        aligner.model_dir = models_dir.join("wav2vec2-fa");
    }
}

/// Resolve the aligner model directory when the feature is enabled.
///
/// Returns the default `models_dir/wav2vec2-fa` path when enabled with
/// an empty `model_dir`. Models are provisioned during the setup flow,
/// so the directory is expected to exist by the time this is called.
fn resolve_aligner_model_dir(
    models_dir: &Path,
    aligner: Option<&AlignerConfig>,
) -> Option<PathBuf> {
    match aligner {
        Some(cfg) if cfg.enabled && cfg.model_dir.as_os_str().is_empty() => {
            Some(models_dir.join("wav2vec2-fa"))
        }
        _ => None,
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
    fn aligner_resolution_disabled_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        let disabled = AlignerConfig {
            enabled: false,
            model_dir: PathBuf::new(),
        };

        let resolved = resolve_aligner_model_dir(tmp.path(), Some(&disabled));
        assert!(resolved.is_none());
    }

    #[test]
    fn aligner_resolution_enabled_empty_uses_default_path() {
        let tmp = tempfile::tempdir().unwrap();
        let enabled = AlignerConfig {
            enabled: true,
            model_dir: PathBuf::new(),
        };

        let resolved = resolve_aligner_model_dir(tmp.path(), Some(&enabled));
        assert_eq!(resolved, Some(tmp.path().join("wav2vec2-fa")));
    }

    #[test]
    fn aligner_resolution_enabled_custom_path_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let enabled = AlignerConfig {
            enabled: true,
            model_dir: PathBuf::from("/tmp/custom-fa"),
        };

        let resolved = resolve_aligner_model_dir(tmp.path(), Some(&enabled));
        assert!(resolved.is_none());
    }

    #[test]
    #[ignore] // Requires network — run with `cargo test -- --ignored`
    fn integration_download_ort_llm() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path();
        let defaults = lune_core::config::UserSettings::default();
        let asr_backend = &defaults.asr_backend;
        let llm_backend = &defaults.llm_backend;

        let resolved = ensure_default_models(
            models_dir,
            asr_backend,
            llm_backend,
            defaults.aligner.as_ref(),
        )
        .unwrap();

        assert!(resolved.llm_model_path.exists());

        // Second call should be instant (already present).
        let resolved2 = ensure_default_models(
            models_dir,
            asr_backend,
            llm_backend,
            defaults.aligner.as_ref(),
        )
        .unwrap();
        assert_eq!(resolved.llm_model_path, resolved2.llm_model_path);
    }
}
