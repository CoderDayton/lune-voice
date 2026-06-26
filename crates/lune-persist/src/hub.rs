use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use lune_core::config::{AsrBackendConfig, WhisperModel};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

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
        /// Local directory name under `models/` (e.g. `qwen3-asr-0.6b`).
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

impl ModelSpec {
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

    /// One whisper.cpp GGML model — a single `ggml-*.bin` file. Reuses the
    /// directory download path with exactly one required file.
    pub fn whisper(model: WhisperModel) -> Self {
        Self::Directory {
            repo_id: WhisperModel::REPO_ID.into(),
            required_files: vec![model.ggml_file().into()],
            suffix_patterns: vec![],
            local_name: model.dir().into(),
            description: "On-device Whisper speech-to-text (whisper.cpp). \
                          Audio never leaves your machine.",
            required: false,
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
            model_dirs::DIR_QWEN3_ASR => "Qwen3-ASR 0.6B ONNX",
            model_dirs::DIR_STYLE_LLM => "Qwen3 0.6B ONNX",
            model_dirs::DIR_WHISPER_SMALL => WhisperModel::Small.display_name(),
            model_dirs::DIR_WHISPER_MEDIUM => WhisperModel::Medium.display_name(),
            model_dirs::DIR_WHISPER_LARGE_V3_TURBO => WhisperModel::LargeV3Turbo.display_name(),
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
        let mut specs = vec![Self::qwen3_asr(), Self::style_llm()];
        specs.extend(WhisperModel::ALL.map(Self::whisper));
        specs
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

/// Returns true if every local model the active config depends on is present.
///
/// Remote backends (e.g. OpenAI-compatible) need no local files, so a fully-remote
/// configuration trivially satisfies this check even with an empty models directory.
pub fn has_required_models(
    models_dir: &Path,
    asr_backend: &AsrBackendConfig,
    llm_backend: &lune_core::config::LlmBackendConfig,
) -> bool {
    ModelSpec::all()
        .iter()
        .filter(|s| spec_needed(s, asr_backend, llm_backend))
        .all(|s| s.is_ready(models_dir))
}

/// Check which models are needed for the current config. Returns status of each.
/// Does NOT download anything. Specs whose backend is remote (or disabled) are
/// excluded from the result so the UI doesn't push users to download models they
/// won't use.
pub fn check_model_status(
    models_dir: &Path,
    asr_backend: &AsrBackendConfig,
    llm_backend: &lune_core::config::LlmBackendConfig,
) -> Vec<ModelStatus> {
    ModelSpec::all()
        .into_iter()
        .filter(|spec| spec_needed(spec, asr_backend, llm_backend))
        .map(|spec| spec.to_status(models_dir))
        .collect()
}

/// Whether `spec` corresponds to a local model the active backend selection uses.
fn spec_needed(
    spec: &ModelSpec,
    asr: &AsrBackendConfig,
    llm: &lune_core::config::LlmBackendConfig,
) -> bool {
    use lune_core::config::LlmBackendConfig;

    match spec.local_name() {
        n if n == model_dirs::DIR_QWEN3_ASR => matches!(asr, AsrBackendConfig::Qwen3Asr { .. }),
        n if n == model_dirs::DIR_STYLE_LLM => matches!(llm, LlmBackendConfig::Local { .. }),
        // A whisper spec is needed only when the WhisperCpp backend is active AND
        // this is the selected model — so the UI offers exactly one whisper card.
        n => {
            matches!(asr, AsrBackendConfig::WhisperCpp { model } if model.dir() == n)
                || spec.required()
        }
    }
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

/// What we know about a remote file before/after download. `size` is the
/// authoritative byte count from the HF tree API; `sha256` is the LFS content
/// hash when the file is LFS-tracked (typically the large model files we most
/// care about validating). Both are optional — non-LFS small files like
/// `config.json` are reported by HF with `size` but no sha256.
#[derive(Debug, Default, Clone)]
struct FileMetadata {
    size: Option<u64>,
    sha256: Option<String>,
}

/// True when `path` looks like a complete download of `meta`.
///
/// - With a known size, the file must match exactly (no partial-write tolerance).
/// - With a known sha256, the file's hash must match.
/// - With neither, accept any non-empty file (legacy behaviour for callers
///   that don't have HF metadata yet).
fn local_file_complete(path: &Path, meta: &FileMetadata) -> bool {
    let Ok(file_meta) = path.metadata() else {
        return false;
    };
    if !file_meta.is_file() {
        return false;
    }
    let len = file_meta.len();
    match (meta.size, meta.sha256.as_deref()) {
        (Some(expected), _) if len != expected => false,
        (None, None) if len == 0 => false,
        (_, Some(expected)) => sha256_of_file(path)
            .map(|got| got.eq_ignore_ascii_case(expected))
            .unwrap_or(false),
        _ => true,
    }
}

/// Stream-hash a file with SHA-256; returns lowercase hex.
fn sha256_of_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Fetch per-file size + LFS sha256 from the HF tree API.
///
/// Returns an empty map on any failure — callers fall back to the size-less
/// behaviour rather than blocking downloads when HF is unreachable or rate-limits
/// us. We log the failure once per repo at WARN so it's visible in support logs.
fn fetch_repo_file_metadata(repo_id: &str) -> HashMap<String, FileMetadata> {
    let url = format!("https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true");
    let mut req = ureq::get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let entries: Vec<TreeEntry> = match req.call().and_then(|mut r| r.body_mut().read_json()) {
        Ok(v) => v,
        Err(e) => {
            warn!(repo_id, error = %e, "could not fetch repo tree metadata; falling back to size-less integrity check");
            return HashMap::new();
        }
    };
    entries
        .into_iter()
        .filter(|e| e.entry_type == "file")
        .map(|e| {
            let sha = e.lfs.as_ref().map(|l| l.oid.clone());
            (
                e.path,
                FileMetadata {
                    size: Some(e.size),
                    sha256: sha,
                },
            )
        })
        .collect()
}

#[derive(serde::Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    size: u64,
    #[serde(default)]
    lfs: Option<TreeEntryLfs>,
}

#[derive(serde::Deserialize)]
struct TreeEntryLfs {
    oid: String,
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
        Err(e) if e.kind() == std::io::ErrorKind::CrossesDevices => {
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
    let resp = req
        .call()
        .map_err(|e| PersistError::ModelDownload(format!("{repo_id}/{filename}: {e}")))?;

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

/// Ensure a directory-type model is present at `models_dir/<local_name>/`,
/// downloading any missing files and reporting per-file progress.
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

    // Fetch authoritative size + LFS sha256 once per ensure call. Empty on failure.
    let remote_meta = fetch_repo_file_metadata(repo_id);

    if local_dir.exists()
        && required_files.iter().all(|f| {
            local_file_complete(&local_dir.join(file_basename(f)), meta_for(&remote_meta, f))
        })
    {
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
        // Flatten nested paths: "onnx/model.onnx" → "model.onnx" locally.
        let basename = file_basename(filename);
        let dst = local_dir.join(basename);
        let meta = meta_for(&remote_meta, filename);
        if local_file_complete(&dst, meta) {
            on_file(filename, idx, total);
            continue;
        }
        // Any stale partial from a prior run is dropped before re-downloading.
        let _ = std::fs::remove_file(&dst);
        on_file(filename, idx, total);

        // Try hf_hub cache; fall back to direct HTTPS for Xet-backed repos.
        let linked = repo
            .get(filename)
            .ok()
            .and_then(|p| p.canonicalize().ok())
            .filter(|src| local_file_complete(src, meta))
            .and_then(|src| link_or_copy(&src, &dst).ok())
            .is_some();
        if !linked {
            let _ = std::fs::remove_file(&dst);
            download_file_direct(repo_id, filename, &dst)?;
        }

        // Post-download verification — if we have remote metadata, the final file
        // must match. Mismatch indicates a corrupted CDN response or a mid-stream
        // failure that left bytes on disk; refuse rather than ship a broken model.
        if !local_file_complete(&dst, meta) {
            let _ = std::fs::remove_file(&dst);
            return Err(PersistError::ModelDownload(format!(
                "{repo_id}/{filename}: downloaded file failed integrity check (size/sha mismatch)"
            )));
        }
    }

    let missing: Vec<&str> = required_files
        .iter()
        .map(String::as_str)
        .filter(|filename| !local_dir.join(file_basename(filename)).is_file())
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

/// Flatten a possibly-nested HF repo path (`onnx/model.onnx`) to the last segment.
/// Falls back to the original string if it ends in a separator (shouldn't happen
/// for file entries, but keeps the call infallible at the use sites).
fn file_basename(filename: &str) -> &str {
    Path::new(filename)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(filename)
}

fn meta_for<'a>(map: &'a HashMap<String, FileMetadata>, filename: &str) -> &'a FileMetadata {
    static EMPTY: FileMetadata = FileMetadata {
        size: None,
        sha256: None,
    };
    map.get(filename).unwrap_or(&EMPTY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::config::{AsrBackendConfig, LlmBackendConfig};

    fn fake_openai_asr() -> AsrBackendConfig {
        AsrBackendConfig::OpenAiCompatible {
            base_url: "https://example.com/v1".parse().unwrap(),
            api_key: lune_core::config::RedactedString::new("test-key"),
            model: "whisper".into(),
            streaming: lune_core::config::StreamingMode::HttpPolling,
        }
    }

    #[test]
    fn check_model_status_returns_nothing_when_all_backends_are_remote() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses =
            check_model_status(tmp.path(), &fake_openai_asr(), &LlmBackendConfig::Disabled);
        assert!(
            statuses.is_empty(),
            "no local models needed when ASR is remote and LLM is disabled, got {statuses:?}"
        );
    }

    #[test]
    fn check_model_status_surfaces_only_the_selected_whisper_model() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses = check_model_status(
            tmp.path(),
            &AsrBackendConfig::WhisperCpp {
                model: WhisperModel::Medium,
            },
            &LlmBackendConfig::Disabled,
        );
        let names: Vec<&str> = statuses.iter().map(|s| s.local_name.as_str()).collect();
        assert_eq!(
            names,
            vec![WhisperModel::Medium.dir()],
            "only the selected whisper model is offered, not the other sizes or the Qwen3 ASR model: {names:?}"
        );
    }

    #[test]
    fn check_model_status_hides_whisper_models_when_asr_is_qwen3() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses = check_model_status(
            tmp.path(),
            &AsrBackendConfig::Qwen3Asr {},
            &LlmBackendConfig::Disabled,
        );
        let names: Vec<&str> = statuses.iter().map(|s| s.local_name.as_str()).collect();
        assert!(
            names.iter().all(|n| !n.starts_with("whisper-")),
            "no whisper model should be offered under the Qwen3 backend: {names:?}"
        );
    }

    #[test]
    fn check_model_status_includes_only_asr_when_llm_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses = check_model_status(
            tmp.path(),
            &AsrBackendConfig::Qwen3Asr {},
            &LlmBackendConfig::Disabled,
        );
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].local_name, lune_core::models::DIR_QWEN3_ASR);
    }

    #[test]
    fn check_model_status_includes_both_when_local_asr_and_local_llm() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses = check_model_status(
            tmp.path(),
            &AsrBackendConfig::Qwen3Asr {},
            &LlmBackendConfig::Local {},
        );
        let names: Vec<&str> = statuses.iter().map(|s| s.local_name.as_str()).collect();
        assert!(
            names.contains(&lune_core::models::DIR_QWEN3_ASR),
            "{names:?}"
        );
        assert!(
            names.contains(&lune_core::models::DIR_STYLE_LLM),
            "{names:?}"
        );
    }

    #[test]
    fn local_file_complete_accepts_matching_size() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"hello").unwrap();
        let meta = FileMetadata {
            size: Some(5),
            sha256: None,
        };
        assert!(local_file_complete(&p, &meta));
    }

    #[test]
    fn local_file_complete_rejects_size_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"hello").unwrap();
        let meta = FileMetadata {
            size: Some(99),
            sha256: None,
        };
        assert!(
            !local_file_complete(&p, &meta),
            "wrong-sized file must be rejected"
        );
    }

    #[test]
    fn local_file_complete_validates_sha256_when_present() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"hello").unwrap();
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let meta = FileMetadata {
            size: Some(5),
            sha256: Some("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824".into()),
        };
        assert!(local_file_complete(&p, &meta));

        let wrong = FileMetadata {
            size: Some(5),
            sha256: Some("0000000000000000000000000000000000000000000000000000000000000000".into()),
        };
        assert!(
            !local_file_complete(&p, &wrong),
            "wrong sha must be rejected even when size matches"
        );
    }

    #[test]
    fn local_file_complete_falls_back_to_size_only_when_no_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"hello").unwrap();
        let meta = FileMetadata {
            size: None,
            sha256: None,
        };
        // No metadata → accept any non-empty file (legacy behaviour).
        assert!(local_file_complete(&p, &meta));

        let empty = tmp.path().join("empty.bin");
        std::fs::write(&empty, b"").unwrap();
        assert!(!local_file_complete(&empty, &meta), "empty file rejected");
    }

    #[test]
    fn hub_constructs_with_valid_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let hub_cache = tmp.path().join(".hub-cache");
        let hub = ModelHub::new(&hub_cache);
        assert!(hub.is_ok());
    }
}
