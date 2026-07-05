use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use lune_core::config::{AsrBackendConfig, WhisperModel};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use lune_core::models as model_dirs;

use crate::error::{PersistError, PersistResult};

/// Which part of the pipeline a model serves. Drives how the model library
/// groups entries in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelRole {
    /// Whisper speech-to-text models and their bundled VAD.
    SpeechToText,
    /// The style/correction LLM.
    Styling,
}

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
        /// Which pipeline stage this model serves.
        role: ModelRole,
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
    /// Whether the active backend selection currently uses this model.
    pub in_use: bool,
    /// Which pipeline stage this model serves (for library grouping).
    pub role: ModelRole,
    /// Actual on-disk directory size in bytes when installed; `None` otherwise.
    pub size_bytes: Option<u64>,
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
    /// One whisper.cpp GGML model — a single `ggml-*.bin` file. Reuses the
    /// directory download path with exactly one required file.
    pub fn whisper(model: WhisperModel) -> Self {
        Self::Directory {
            repo_id: WhisperModel::REPO_ID.into(),
            required_files: vec![model.ggml_file().into()],
            suffix_patterns: vec![],
            local_name: model.dir().into(),
            description: model.description(),
            required: false,
            role: ModelRole::SpeechToText,
        }
    }

    pub fn style_llm() -> Self {
        Self::Directory {
            repo_id: "unsloth/Qwen3.5-0.8B-GGUF".into(),
            required_files: vec!["Qwen3.5-0.8B-UD-Q5_K_XL.gguf".into()],
            suffix_patterns: vec![],
            local_name: model_dirs::DIR_STYLE_LLM.into(),
            description: "Style LLM (llama.cpp). Rewrites transcripts in your chosen style.",
            required: false,
            role: ModelRole::Styling,
        }
    }

    /// whisper.cpp's built-in VAD model (silero ggml). The local whisper backend
    /// uses it to gate speech before decoding; it degrades to raw decode when
    /// absent, so it's optional.
    pub fn vad_model() -> Self {
        Self::Directory {
            repo_id: "ggml-org/whisper-vad".into(),
            required_files: vec![model_dirs::WHISPER_VAD_GGML.into()],
            suffix_patterns: vec![],
            local_name: model_dirs::DIR_WHISPER_VAD.into(),
            description: "Voice-activity detector for the local Whisper backend.",
            required: false,
            role: ModelRole::SpeechToText,
        }
    }

    /// Human-readable display name for this model.
    pub fn display_name(&self) -> &str {
        let Self::Directory { local_name, .. } = self;
        match local_name.as_str() {
            model_dirs::DIR_STYLE_LLM => "Qwen3.5 0.8B (style LLM)",
            model_dirs::DIR_WHISPER_VAD => "Whisper VAD (silero)",
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

    /// Which pipeline stage this model serves.
    pub fn role(&self) -> ModelRole {
        let Self::Directory { role, .. } = self;
        *role
    }

    pub fn all() -> Vec<Self> {
        let mut specs = vec![Self::style_llm(), Self::vad_model()];
        specs.extend(WhisperModel::ALL.map(Self::whisper));
        specs
    }

    fn to_status(&self, models_dir: &Path, in_use: bool) -> ModelStatus {
        let ready = self.is_ready(models_dir);
        ModelStatus {
            name: self.display_name().to_string(),
            repo_id: self.repo_id().to_string(),
            local_name: self.local_name().to_string(),
            ready,
            description: self.description().to_string(),
            required: self.required(),
            in_use,
            role: self.role(),
            size_bytes: if ready {
                installed_size_bytes(models_dir, self.local_name())
            } else {
                None
            },
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

/// What the user still needs to set up before the styling LLM can serve a
/// request. Each variant maps to a concrete setup step surfaced by the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmUnavailable {
    /// Styling is turned off — no engine is selected.
    NoProvider,
    /// The local styling engine is selected but its model isn't downloaded.
    ModelMissing,
    /// A remote styling provider is selected but not fully configured.
    ProviderNotConfigured,
}

impl LlmUnavailable {
    /// A short, user-facing explanation of the setup step still required.
    pub fn user_message(self) -> &'static str {
        match self {
            Self::NoProvider => {
                "Styling is turned off. Enable a styling engine in Settings before generating examples."
            }
            Self::ModelMissing => {
                "The styling model isn't downloaded yet. Download it in Settings → Models before generating examples."
            }
            Self::ProviderNotConfigured => {
                "Your styling provider isn't fully configured. Add its API key and model in Settings before generating examples."
            }
        }
    }
}

/// Whether the configured styling LLM is ready to serve a request. `Ok(())`
/// means a request can proceed; `Err(reason)` names the setup step the user
/// still needs. This is the guard the app runs before asking the LLM to
/// (re)generate few-shot examples — the local engine needs its model on disk;
/// a remote engine needs an API key and model configured.
pub fn styling_llm_readiness(
    models_dir: &Path,
    llm_backend: &lune_core::config::LlmBackendConfig,
) -> Result<(), LlmUnavailable> {
    use lune_core::config::LlmBackendConfig;
    match llm_backend {
        LlmBackendConfig::Local { .. } => {
            let ready = ModelSpec::all()
                .iter()
                .find(|s| s.local_name() == model_dirs::DIR_STYLE_LLM)
                .is_some_and(|s| s.is_ready(models_dir));
            if ready {
                Ok(())
            } else {
                Err(LlmUnavailable::ModelMissing)
            }
        }
        LlmBackendConfig::OpenAiCompatible { api_key, model, .. } => {
            if api_key.expose().trim().is_empty() || model.trim().is_empty() {
                Err(LlmUnavailable::ProviderNotConfigured)
            } else {
                Ok(())
            }
        }
        _ => Err(LlmUnavailable::NoProvider),
    }
}

/// Status of the full installable model library. Returns every spec EXCEPT the
/// VAD model — that one is bundled with any Whisper size and never surfaced as
/// its own entry. Each entry carries `ready`, `in_use` (whether the active
/// backend selection uses it), `role`, and on-disk `size_bytes`. Does NOT
/// download anything.
pub fn check_model_status(
    models_dir: &Path,
    asr_backend: &AsrBackendConfig,
    llm_backend: &lune_core::config::LlmBackendConfig,
) -> Vec<ModelStatus> {
    ModelSpec::all()
        .into_iter()
        .filter(|spec| spec.local_name() != model_dirs::DIR_WHISPER_VAD)
        .map(|spec| {
            let in_use = spec_needed(&spec, asr_backend, llm_backend);
            spec.to_status(models_dir, in_use)
        })
        .collect()
}

/// Sum the byte sizes of the files directly under `models_dir/<local_name>`.
/// Model directories are flat (download paths are flattened to basenames), so a
/// single read of the directory is enough. Returns `None` when the directory is
/// absent or unreadable.
fn installed_size_bytes(models_dir: &Path, local_name: &str) -> Option<u64> {
    let dir = models_dir.join(local_name);
    let entries = std::fs::read_dir(&dir).ok()?;
    let mut total = 0u64;
    for entry in entries.flatten() {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                total += meta.len();
            }
        }
    }
    Some(total)
}

/// Whether `spec` corresponds to a local model the active backend selection uses.
fn spec_needed(
    spec: &ModelSpec,
    asr: &AsrBackendConfig,
    llm: &lune_core::config::LlmBackendConfig,
) -> bool {
    use lune_core::config::LlmBackendConfig;

    match spec.local_name() {
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

    // Bundle the VAD model with any Whisper size: the local whisper backend
    // gates speech with it, but it's never offered as its own library entry.
    // Fetched first (when absent) so the final `done` event below belongs to the
    // requested model and the UI clears progress on the right name.
    if spec.role() == ModelRole::SpeechToText && spec.local_name() != model_dirs::DIR_WHISPER_VAD {
        let vad = ModelSpec::vad_model();
        if !vad.is_ready(models_dir) {
            let vad_name = vad.display_name().to_string();
            ensure_directory_model_with_progress(&hub, &vad, models_dir, |file, idx, total| {
                on_progress(DownloadProgress {
                    model_name: vad_name.clone(),
                    file_name: file.to_string(),
                    file_index: idx,
                    file_count: total,
                    done: false,
                });
            })?;
        }
    }

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
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// `Write` adapter that counts bytes and sha256-hashes them on the way to the
/// inner writer, so a direct download can be verified without re-reading the
/// multi-GB file afterwards.
struct HashingWriter<W: std::io::Write> {
    inner: W,
    hasher: Sha256,
    written: u64,
}

impl<W: std::io::Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            written: 0,
        }
    }

    /// Total bytes written and their lowercase-hex sha256.
    fn finish(self) -> (u64, String) {
        (self.written, format!("{:x}", self.hasher.finalize()))
    }
}

impl<W: std::io::Write> std::io::Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Per-model integrity manifest: local basename → last verified size/sha,
/// written next to the model files after every successful ensure. It keeps
/// integrity checks meaningful when the HF tree fetch fails (offline,
/// rate-limited) — without it, one failed metadata fetch used to downgrade
/// verification to "any non-empty file passes", letting a truncated model be
/// cached as complete forever.
const INTEGRITY_MANIFEST: &str = ".integrity.json";

fn load_integrity_manifest(dir: &Path) -> HashMap<String, FileMetadata> {
    let path = dir.join(INTEGRITY_MANIFEST);
    let Ok(bytes) = std::fs::read(&path) else {
        return HashMap::new();
    };
    match serde_json::from_slice(&bytes) {
        Ok(map) => map,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "unreadable integrity manifest; ignoring");
            HashMap::new()
        }
    }
}

/// Best-effort atomic write (tmp + rename). A failure only degrades future
/// offline verification, so it warns instead of failing the download.
fn save_integrity_manifest(dir: &Path, manifest: &HashMap<String, FileMetadata>) {
    let path = dir.join(INTEGRITY_MANIFEST);
    let tmp = dir.join(".integrity.json.tmp");
    let Ok(json) = serde_json::to_vec_pretty(manifest) else {
        return;
    };
    let result = std::fs::write(&tmp, &json).and_then(|()| std::fs::rename(&tmp, &path));
    if let Err(e) = result {
        warn!(path = %path.display(), error = %e, "could not persist integrity manifest");
    }
}

/// Expected metadata for `filename`, in trust order: the HF tree API
/// (authoritative, repo-path keyed) → the persisted manifest from the last
/// verified download (basename keyed) → empty (legacy non-empty acceptance,
/// only for files that have never been verified).
fn effective_meta(
    remote: &HashMap<String, FileMetadata>,
    manifest: &HashMap<String, FileMetadata>,
    filename: &str,
) -> FileMetadata {
    remote
        .get(filename)
        .or_else(|| manifest.get(file_basename(filename)))
        .cloned()
        .unwrap_or_default()
}

/// Copy newly-learned authoritative values into the manifest so a
/// pre-manifest install becomes offline-verifiable once HF is reachable.
/// Returns whether anything changed.
fn refresh_manifest(
    manifest: &mut HashMap<String, FileMetadata>,
    remote: &HashMap<String, FileMetadata>,
    files: &[String],
) -> bool {
    let mut changed = false;
    for f in files {
        if let Some(meta) = remote.get(f.as_str()) {
            let key = file_basename(f);
            if manifest.get(key) != Some(meta) {
                manifest.insert(key.to_owned(), meta.clone());
                changed = true;
            }
        }
    }
    changed
}

/// Record a now-verified file into the manifest: prefer the expected
/// (authoritative) values, fall back to what the direct download observed
/// (length checked against Content-Length, sha computed in-stream), and
/// finally to the on-disk length.
fn record_verified(
    manifest: &mut HashMap<String, FileMetadata>,
    basename: &str,
    expected: &FileMetadata,
    dst: &Path,
    observed: Option<FileMetadata>,
) {
    let fallback = observed.unwrap_or_default();
    let size = expected
        .size
        .or(fallback.size)
        .or_else(|| dst.metadata().ok().map(|m| m.len()));
    let sha256 = expected.sha256.clone().or(fallback.sha256);
    manifest.insert(basename.to_owned(), FileMetadata { size, sha256 });
}

/// Fetch per-file size + LFS sha256 from the HF tree API.
///
/// Returns an empty map on any failure — callers then fall back to the
/// persisted integrity manifest (values from the last verified download)
/// rather than blocking downloads when HF is unreachable or rate-limits us.
/// We log the failure once per repo at WARN so it's visible in support logs.
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
/// The stream is length-checked against the response's Content-Length and
/// sha256-hashed as it is written, so a truncated body (a cleanly-closed
/// short response) is rejected here even when the HF tree metadata was
/// unavailable; the observed values are returned for the caller's integrity
/// manifest. Reads `HF_TOKEN` from env for gated/private repos.
fn download_file_direct(repo_id: &str, filename: &str, dst: &Path) -> PersistResult<FileMetadata> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
    let mut req = ureq::get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let resp = req
        .call()
        .map_err(|e| PersistError::ModelDownload(format!("{repo_id}/{filename}: {e}")))?;

    // Content-Length only equals the bytes we write when the body isn't
    // transparently decompressed — ureq's default `gzip` feature decodes
    // Content-Encoding: gzip on the fly (possible on small text files; CDN
    // model blobs are served identity). Skip the check in that case.
    let content_encoded = resp
        .headers()
        .get("content-encoding")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"));
    let expected_len: Option<u64> = if content_encoded {
        None
    } else {
        resp.headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
    };

    let tmp = dst.with_extension("dl.tmp");
    let (written, sha256) = {
        let file = std::fs::File::create(&tmp)
            .map_err(|e| PersistError::ModelDownload(format!("create tmp: {e}")))?;
        let mut writer = HashingWriter::new(file);
        std::io::copy(&mut resp.into_body().as_reader(), &mut writer).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            PersistError::ModelDownload(format!("{filename}: stream: {e}"))
        })?;
        writer.finish()
    };
    if let Some(expected) = expected_len {
        if written != expected {
            let _ = std::fs::remove_file(&tmp);
            return Err(PersistError::ModelDownload(format!(
                "{repo_id}/{filename}: truncated download ({written} of {expected} bytes)"
            )));
        }
    }
    std::fs::rename(&tmp, dst)
        .map_err(|e| PersistError::ModelDownload(format!("{filename}: rename: {e}")))?;
    info!(
        repo_id,
        filename,
        bytes = written,
        "direct download complete"
    );
    Ok(FileMetadata {
        size: Some(written),
        sha256: Some(sha256),
    })
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

    // Expected size/sha come from two sources, in trust order: the HF tree API
    // (authoritative, refreshed every ensure; empty on failure) and the
    // persisted per-model manifest from the last verified download. The
    // manifest keeps verification meaningful offline and across HF outages —
    // a truncated file can't be accepted just because the metadata fetch
    // failed this time.
    let mut manifest = load_integrity_manifest(&local_dir);
    let remote_meta = fetch_repo_file_metadata(repo_id);

    if local_dir.exists()
        && required_files.iter().all(|f| {
            local_file_complete(
                &local_dir.join(file_basename(f)),
                &effective_meta(&remote_meta, &manifest, f),
            )
        })
    {
        if refresh_manifest(&mut manifest, &remote_meta, required_files) {
            save_integrity_manifest(&local_dir, &manifest);
        }
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
        let meta = effective_meta(&remote_meta, &manifest, filename);
        if local_file_complete(&dst, &meta) {
            record_verified(&mut manifest, basename, &meta, &dst, None);
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
            .filter(|src| local_file_complete(src, &meta))
            .and_then(|src| link_or_copy(&src, &dst).ok())
            .is_some();
        let observed = if linked {
            None
        } else {
            let _ = std::fs::remove_file(&dst);
            Some(download_file_direct(repo_id, filename, &dst)?)
        };

        // Post-download verification — if we have expected metadata, the final
        // file must match. Mismatch indicates a corrupted CDN response or a
        // mid-stream failure that left bytes on disk; refuse rather than ship a
        // broken model. (Without metadata, a truncated direct download has
        // already been rejected against the response's Content-Length.)
        if !local_file_complete(&dst, &meta) {
            let _ = std::fs::remove_file(&dst);
            return Err(PersistError::ModelDownload(format!(
                "{repo_id}/{filename}: downloaded file failed integrity check (size/sha mismatch)"
            )));
        }
        record_verified(&mut manifest, basename, &meta, &dst, observed);
    }

    // Persist what was just verified so later runs (possibly offline) check
    // against real expectations instead of the size-less fallback.
    save_integrity_manifest(&local_dir, &manifest);

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

    /// The library always lists every Whisper size plus the styling LLM, and
    /// never the bundled VAD model.
    fn assert_full_library(statuses: &[ModelStatus]) {
        let names: Vec<&str> = statuses.iter().map(|s| s.local_name.as_str()).collect();
        assert!(names.contains(&WhisperModel::Small.dir()), "{names:?}");
        assert!(names.contains(&WhisperModel::Medium.dir()), "{names:?}");
        assert!(
            names.contains(&WhisperModel::LargeV3Turbo.dir()),
            "{names:?}"
        );
        assert!(
            names.contains(&lune_core::models::DIR_STYLE_LLM),
            "{names:?}"
        );
        assert!(
            !names.contains(&lune_core::models::DIR_WHISPER_VAD),
            "VAD must stay out of the library: {names:?}"
        );
        // Whisper sizes (3) + styling LLM = 4 entries, no more.
        assert_eq!(statuses.len(), 4, "{names:?}");
    }

    fn in_use_names(statuses: &[ModelStatus]) -> Vec<&str> {
        statuses
            .iter()
            .filter(|s| s.in_use)
            .map(|s| s.local_name.as_str())
            .collect()
    }

    #[test]
    fn check_model_status_lists_full_library_with_nothing_in_use_when_remote() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses =
            check_model_status(tmp.path(), &fake_openai_asr(), &LlmBackendConfig::Disabled);
        assert_full_library(&statuses);
        assert!(
            in_use_names(&statuses).is_empty(),
            "remote ASR + disabled LLM use no local models: {statuses:?}"
        );
    }

    #[test]
    fn check_model_status_marks_only_the_selected_whisper_in_use() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses = check_model_status(
            tmp.path(),
            &AsrBackendConfig::WhisperCpp {
                model: WhisperModel::Medium,
            },
            &LlmBackendConfig::Disabled,
        );
        assert_full_library(&statuses);
        assert_eq!(
            in_use_names(&statuses),
            vec![WhisperModel::Medium.dir()],
            "only the selected whisper size is in use when the LLM is disabled"
        );
    }

    #[test]
    fn check_model_status_marks_whisper_and_llm_in_use_when_both_local() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses = check_model_status(
            tmp.path(),
            &AsrBackendConfig::WhisperCpp {
                model: WhisperModel::Medium,
            },
            &LlmBackendConfig::Local {},
        );
        assert_full_library(&statuses);
        let used = in_use_names(&statuses);
        assert!(used.contains(&WhisperModel::Medium.dir()), "{used:?}");
        assert!(used.contains(&lune_core::models::DIR_STYLE_LLM), "{used:?}");
        assert_eq!(
            used.len(),
            2,
            "exactly the selected whisper + LLM: {used:?}"
        );
    }

    #[test]
    fn check_model_status_reports_role_and_size() {
        let tmp = tempfile::tempdir().unwrap();
        let statuses = check_model_status(
            tmp.path(),
            &AsrBackendConfig::WhisperCpp {
                model: WhisperModel::Small,
            },
            &LlmBackendConfig::Local {},
        );
        for s in &statuses {
            // Nothing is installed in an empty models dir.
            assert!(!s.ready, "{}", s.local_name);
            assert!(s.size_bytes.is_none(), "{}", s.local_name);
            let expected_role = if s.local_name == lune_core::models::DIR_STYLE_LLM {
                ModelRole::Styling
            } else {
                ModelRole::SpeechToText
            };
            assert_eq!(s.role, expected_role, "{}", s.local_name);
        }
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
    fn hashing_writer_reports_len_and_sha() {
        let mut out = Vec::new();
        let mut w = HashingWriter::new(&mut out);
        std::io::Write::write_all(&mut w, b"hello").unwrap();
        let (n, sha) = w.finish();
        assert_eq!(n, 5);
        assert_eq!(
            sha,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert_eq!(out, b"hello");
    }

    #[test]
    fn integrity_manifest_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let mut m = HashMap::new();
        m.insert(
            "a.bin".to_string(),
            FileMetadata {
                size: Some(5),
                sha256: Some("abc".into()),
            },
        );
        save_integrity_manifest(tmp.path(), &m);
        assert_eq!(load_integrity_manifest(tmp.path()), m);
    }

    #[test]
    fn corrupt_manifest_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(INTEGRITY_MANIFEST), b"{not json").unwrap();
        assert!(load_integrity_manifest(tmp.path()).is_empty());
    }

    #[test]
    fn effective_meta_prefers_remote_then_manifest() {
        let mut remote = HashMap::new();
        remote.insert(
            "onnx/model.onnx".to_string(),
            FileMetadata {
                size: Some(10),
                sha256: None,
            },
        );
        let mut manifest = HashMap::new();
        manifest.insert(
            "model.onnx".to_string(),
            FileMetadata {
                size: Some(20),
                sha256: None,
            },
        );
        // Remote (repo-path keyed) wins when present.
        assert_eq!(
            effective_meta(&remote, &manifest, "onnx/model.onnx").size,
            Some(10)
        );
        // Manifest (basename keyed) fills in when the remote fetch failed.
        assert_eq!(
            effective_meta(&HashMap::new(), &manifest, "onnx/model.onnx").size,
            Some(20)
        );
        // Neither → default (legacy non-empty acceptance).
        assert_eq!(
            effective_meta(&HashMap::new(), &HashMap::new(), "other").size,
            None
        );
    }

    #[test]
    fn manifest_pins_size_after_metadata_loss() {
        // A file verified once (manifest knows its size) must be rejected when
        // truncated later, even with no remote metadata available.
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("model.bin");
        std::fs::write(&p, b"0123456789").unwrap();
        let mut manifest = HashMap::new();
        manifest.insert(
            "model.bin".to_string(),
            FileMetadata {
                size: Some(10),
                sha256: None,
            },
        );
        let no_remote = HashMap::new();
        assert!(local_file_complete(
            &p,
            &effective_meta(&no_remote, &manifest, "model.bin")
        ));
        std::fs::write(&p, b"01234").unwrap(); // truncated
        assert!(!local_file_complete(
            &p,
            &effective_meta(&no_remote, &manifest, "model.bin")
        ));
    }

    #[test]
    fn record_verified_prefers_expected_then_observed_then_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.bin");
        std::fs::write(&p, b"hello").unwrap();
        let mut manifest = HashMap::new();

        // Expected values win.
        record_verified(
            &mut manifest,
            "a.bin",
            &FileMetadata {
                size: Some(5),
                sha256: Some("expected".into()),
            },
            &p,
            Some(FileMetadata {
                size: Some(99),
                sha256: Some("observed".into()),
            }),
        );
        assert_eq!(manifest["a.bin"].size, Some(5));
        assert_eq!(manifest["a.bin"].sha256.as_deref(), Some("expected"));

        // Observed fills in when nothing was expected.
        record_verified(
            &mut manifest,
            "a.bin",
            &FileMetadata::default(),
            &p,
            Some(FileMetadata {
                size: Some(5),
                sha256: Some("observed".into()),
            }),
        );
        assert_eq!(manifest["a.bin"].size, Some(5));
        assert_eq!(manifest["a.bin"].sha256.as_deref(), Some("observed"));

        // Disk length is the last resort.
        record_verified(&mut manifest, "a.bin", &FileMetadata::default(), &p, None);
        assert_eq!(manifest["a.bin"].size, Some(5));
        assert_eq!(manifest["a.bin"].sha256, None);
    }

    /// Real-network end-to-end: `download_model` writes the integrity
    /// manifest, and a truncated model file is detected and re-downloaded on
    /// the next run. Opt-in (network + ~1 MB):
    /// `cargo test -p lune-persist -- --ignored network_`
    #[test]
    #[ignore = "downloads ~1 MB from huggingface.co"]
    fn network_download_writes_manifest_and_heals_truncation() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path();

        let path = download_model(models_dir, lune_core::models::DIR_WHISPER_VAD, |_| {}).unwrap();
        let file = path.join(lune_core::models::WHISPER_VAD_GGML);
        let full_len = file.metadata().unwrap().len();
        assert!(full_len > 100_000, "real model expected, got {full_len}B");

        // Manifest pins the verified size.
        let manifest = load_integrity_manifest(&path);
        assert_eq!(
            manifest[lune_core::models::WHISPER_VAD_GGML].size,
            Some(full_len)
        );

        // Truncate → the next ensure must reject the file and re-download it.
        let half = std::fs::read(&file).unwrap()[..(full_len / 2) as usize].to_vec();
        std::fs::write(&file, half).unwrap();
        download_model(models_dir, lune_core::models::DIR_WHISPER_VAD, |_| {}).unwrap();
        assert_eq!(
            file.metadata().unwrap().len(),
            full_len,
            "truncated file must be healed"
        );
    }

    #[test]
    fn hub_constructs_with_valid_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let hub_cache = tmp.path().join(".hub-cache");
        let hub = ModelHub::new(&hub_cache);
        assert!(hub.is_ok());
    }
}
