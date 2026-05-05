use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use lune_core::config::AsrBackendConfig;
use tracing::info;

use lune_core::models as model_dirs;

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
        /// Human-readable description of this model's role.
        description: &'static str,
        /// Whether the core pipeline requires this model.
        required: bool,
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
    /// Human-readable description of what this model does
    pub description: String,
    /// Whether the pipeline requires this model to function
    pub required: bool,
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
    pub asr_model_dir: PathBuf,
    pub style_llm_dir: PathBuf,
}

impl ModelSpec {
    pub fn parakeet_tdt() -> Self {
        Self::Directory {
            repo_id: "istupakov/parakeet-tdt-0.6b-v3-onnx".into(),
            required_files: vec![
                "vocab.txt".into(),
                "encoder-model.int8.onnx".into(),
                "decoder_joint-model.int8.onnx".into(),
            ],
            suffix_patterns: vec![],
            local_name: model_dirs::DIR_PARAKEET.into(),
            description: "Speech-to-text engine. CTC-based ONNX model.",
            required: true,
        }
    }

    pub fn qwen3_asr() -> Self {
        Self::Directory {
            repo_id: "Qwen/Qwen3-ASR-0.6B".into(),
            required_files: vec![
                "tokenizer.json".into(),
                "qwen3-asr-encoder.onnx".into(),
                "qwen3-asr-embed-tokens.onnx".into(),
                "qwen3-asr-decoder.onnx".into(),
            ],
            suffix_patterns: vec![".onnx_data".into()],
            local_name: model_dirs::DIR_QWEN3_ASR.into(),
            description: "Speech-to-text engine. Encoder-decoder ONNX model with SmoothQuant.",
            required: true,
        }
    }

    pub fn style_llm() -> Self {
        Self::Directory {
            repo_id: "onnx-community/Qwen3-0.6B-ONNX".into(),
            required_files: vec!["tokenizer.json".into(), "model_q4f16.onnx".into()],
            suffix_patterns: vec![],
            local_name: model_dirs::DIR_STYLE_LLM.into(),
            description: "Style & correction LLM. Fixes grammar and reformats transcripts.",
            required: false,
        }
    }

    /// Human-readable display name for this model.
    pub fn display_name(&self) -> &str {
        let Self::Directory { local_name, .. } = self;
        match local_name.as_str() {
            model_dirs::DIR_PARAKEET => "Parakeet TDT 0.6B",
            model_dirs::DIR_QWEN3_ASR => "Qwen3-ASR 0.6B ONNX",
            model_dirs::DIR_STYLE_LLM => "Qwen3 0.6B ONNX",
            _ => local_name,
        }
    }

    /// Check if this model is fully present locally (no download needed).
    pub fn is_ready(&self, models_dir: &Path) -> bool {
        let Self::Directory {
            required_files,
            local_name,
            ..
        } = self;
        let dir = models_dir.join(local_name);
        dir.exists() && required_files.iter().all(|f| dir.join(f).is_file())
    }

    fn repo_id(&self) -> &str {
        let Self::Directory { repo_id, .. } = self;
        repo_id
    }

    pub fn local_name(&self) -> &str {
        let Self::Directory { local_name, .. } = self;
        local_name
    }

    fn description(&self) -> &str {
        let Self::Directory { description, .. } = self;
        description
    }

    fn required(&self) -> bool {
        let Self::Directory { required, .. } = self;
        *required
    }

    pub fn all() -> Vec<Self> {
        vec![Self::qwen3_asr(), Self::style_llm()]
    }

    fn to_status(&self, models_dir: &Path) -> ModelStatus {
        ModelStatus {
            name: self.display_name().to_string(),
            repo_id: self.repo_id().to_string(),
            local_name: self.local_name().to_string(),
            ready: self.is_ready(models_dir),
            description: self.description().to_string(),
            required: self.required(),
        }
    }
}

/// Returns true if every model marked `required` is fully present locally.
pub fn has_required_models(models_dir: &Path) -> bool {
    ModelSpec::all()
        .iter()
        .filter(|s| s.required())
        .all(|s| s.is_ready(models_dir))
}

/// Check which models are needed for the current config. Returns status of each.
/// Does NOT download anything.
pub fn check_model_status(
    models_dir: &Path,
    _asr_backend: &AsrBackendConfig,
    _llm_backend: &lune_core::config::LlmBackendConfig,
) -> Vec<ModelStatus> {
    ModelSpec::all()
        .iter()
        .map(|spec| spec.to_status(models_dir))
        .collect()
}

/// Delete a downloaded model by local_name. Returns Ok(true) if removed,
/// Ok(false) if it wasn't present.
pub fn delete_model(models_dir: &Path, local_name: &str) -> PersistResult<bool> {
    let known = ModelSpec::all();
    if !known.iter().any(|s| s.local_name() == local_name) {
        return Err(PersistError::ModelDownload(format!(
            "unknown model name: {local_name}"
        )));
    }
    let dir = models_dir.join(local_name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir)?;
        info!(local_name, "model deleted");
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Download a single model by local_name. Calls `on_progress` for each file.
/// Blocking — run on a background thread.
pub fn download_model(
    models_dir: &Path,
    local_name: &str,
    on_progress: impl Fn(DownloadProgress),
) -> PersistResult<PathBuf> {
    let spec = ModelSpec::all()
        .into_iter()
        .find(|s| s.local_name() == local_name)
        .ok_or_else(|| PersistError::ModelDownload(format!("unknown model: {local_name}")))?;

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

/// Direct HTTPS download from HuggingFace CDN. Fallback for Xet-backed
/// repos where the `hf_hub` crate's `repo.get()` fails.
///
/// Uses the resolve endpoint which redirects to the actual blob URL.
/// Streams to a temp file, then renames atomically to avoid partial files.
/// Reads `HF_TOKEN` from env for gated/private repos.
fn download_file_direct(repo_id: &str, filename: &str, dst: &Path) -> PersistResult<()> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
    let mut req = ureq::get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let resp = req.call().map_err(|_| {
        PersistError::ModelDownload(format!("{repo_id}/{filename}: HTTP request failed"))
    })?;

    let tmp = dst.with_extension("dl.tmp");
    {
        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| PersistError::ModelDownload(format!("create tmp: {e}")))?;
        std::io::copy(&mut resp.into_body().as_reader(), &mut file)
            .map_err(|e| PersistError::ModelDownload(format!("{filename}: stream: {e}")))?;
    }
    std::fs::rename(&tmp, dst)
        .map_err(|e| PersistError::ModelDownload(format!("{filename}: rename: {e}")))?;
    info!(repo_id, filename, "direct download complete");
    Ok(())
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
) -> PersistResult<ResolvedModels> {
    let hub_cache = models_dir.join(".hub-cache");
    let hub = ModelHub::new(&hub_cache)?;

    let asr_spec = match asr_backend {
        AsrBackendConfig::Qwen3Asr { .. } => ModelSpec::qwen3_asr(),
        AsrBackendConfig::OpenAiCompatible { .. } | _ => ModelSpec::qwen3_asr(),
    };
    let asr_model_dir = ensure_directory_model(&hub, &asr_spec, models_dir)?;
    let style_llm_dir = ensure_directory_model(&hub, &ModelSpec::style_llm(), models_dir)?;

    Ok(ResolvedModels {
        asr_model_dir,
        style_llm_dir,
    })
}

/// Ensure a directory-type model is present at `models_dir/<local_name>/`.
fn ensure_directory_model(
    hub: &ModelHub,
    spec: &ModelSpec,
    models_dir: &Path,
) -> PersistResult<PathBuf> {
    ensure_directory_model_with_progress(hub, spec, models_dir, |_, _, _| {})
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
        ..
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
        // Flatten nested paths: "onnx/model.onnx" → "model.onnx" locally
        let local_name = Path::new(filename)
            .file_name()
            .ok_or_else(|| PersistError::ModelDownload(format!("invalid filename: {filename}")))?;
        let dst = local_dir.join(local_name);
        if dst.is_file() && dst.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            on_file(filename, idx, total);
            continue;
        }
        on_file(filename, idx, total);

        // Try hf_hub cache; fall back to direct HTTPS for Xet-backed repos
        let ok = repo
            .get(filename)
            .ok()
            .and_then(|p| p.canonicalize().ok())
            .filter(|src| src.metadata().map(|m| m.len() > 0).unwrap_or(false))
            .and_then(|src| link_or_copy(&src, &dst).ok())
            .is_some();
        if !ok {
            let _ = std::fs::remove_file(&dst);
            download_file_direct(repo_id, filename, &dst)?;
        }
    }

    let missing: Vec<&str> = required_files
        .iter()
        .map(String::as_str)
        .filter(|filename| {
            let local_name = Path::new(filename)
                .file_name()
                .is_some_and(|name| local_dir.join(name).is_file());
            !local_name
        })
        .collect();
    if !missing.is_empty() {
        return Err(PersistError::ModelDownload(format!(
            "{} is missing required files after download: {}",
            local_dir.display(),
            missing.join(", ")
        )));
    }

    info!(path = %local_dir.display(), "model directory ready");
    Ok(local_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

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
