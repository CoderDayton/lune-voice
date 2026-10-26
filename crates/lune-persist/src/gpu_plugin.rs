// SPDX-License-Identifier: GPL-3.0-or-later

//! Integrity is not optional here: this unpacks shared objects that the app
//! will later `dlopen`. A tampered archive is arbitrary code execution, so the
//! download is rejected unless its SHA-256 matches the pinned digest exactly.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

use crate::error::{PersistError, PersistResult};

/// A published GPU backend archive.
pub struct GpuPluginRelease {
    /// Direct download URL for the `.tar.gz`.
    pub url: &'static str,
    /// Lowercase hex SHA-256 of the archive, pinned at build time.
    pub sha256: &'static str,
    /// Compressed size of the `.tar.gz`, for the progress bar and as the
    /// fallback total when the server sends no `Content-Length`.
    pub download_bytes: u64,
}

impl GpuPluginRelease {
    /// Whether this release can actually be downloaded. False while `sha256` is
    /// still [`UNPUBLISHED`]: [`install`] refuses in that state, so offering the
    /// download would be a button that cannot succeed.
    pub fn is_published(&self) -> bool {
        self.sha256 != UNPUBLISHED
    }
}

/// ggml derives this name itself (`backend_filename_prefix` /
/// `backend_filename_extension` in `ggml-backend-reg.cpp`), which is why the
/// `lib` prefix is absent on Windows rather than merely conventional.
#[cfg(target_os = "windows")]
const PLUGIN_FILE: &str = "ggml-vulkan.dll";
#[cfg(not(target_os = "windows"))]
const PLUGIN_FILE: &str = "libggml-vulkan.so";

/// Text file in the archive naming the ggml version the plugin was compiled
/// against. Mirrors `lune_asr::gpu_backend::GGML_VERSION_FILE`, which refuses
/// to load a plugin whose stamp is missing or names another version.
const GGML_VERSION_FILE: &str = "ggml-version";

/// Digest placeholder for a release that has not been published yet. Kept as a
/// recognisable sentinel so [`install`] can fail with an actionable message
/// instead of downloading something it cannot verify.
pub const UNPUBLISHED: &str = "UNPUBLISHED";

pub const VULKAN_LINUX_X86_64: GpuPluginRelease = GpuPluginRelease {
    url: "https://github.com/CoderDayton/lune-voice/releases/download/gpu-v1/\
          lune-voice-gpu-vulkan-linux-x86_64.tar.gz",
    sha256: "a512a543b835585f8380ba6769af711bba4c4abb937300eabf414a796460dcfa",
    download_bytes: 17_448_154,
};

pub const VULKAN_WINDOWS_X86_64: GpuPluginRelease = GpuPluginRelease {
    url: "https://github.com/CoderDayton/lune-voice/releases/download/gpu-v1/\
          lune-voice-gpu-vulkan-windows-x86_64.tar.gz",
    sha256: "be7a6489850fa324b778b0f30b910c0db8c33d51c97ae74a0c5005eb56f10469",
    download_bytes: 17_290_674,
};

/// The release this build can install, if the platform has one.
pub const CURRENT: Option<&GpuPluginRelease> = {
    #[cfg(target_os = "linux")]
    {
        Some(&VULKAN_LINUX_X86_64)
    }
    #[cfg(target_os = "windows")]
    {
        Some(&VULKAN_WINDOWS_X86_64)
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
};

/// Progress of a GPU plugin install, emitted as the download streams.
#[derive(Debug, Clone, Copy)]
pub struct GpuInstallProgress {
    pub downloaded_bytes: u64,
    /// `0` when the server sends no `Content-Length`.
    pub total_bytes: u64,
    pub done: bool,
}

/// Files the archive is allowed to contain. Anything else is refused rather
/// than written: the archive is unpacked into a directory the app later
/// `dlopen`s, so an unexpected member is a reason to stop, not to shrug.
fn is_permitted_member(name: &Path) -> bool {
    // `tar -czf … -C "$STAGE" .` (what `cgs gpu-plugin` runs) stores members as
    // `./libfoo.so`, so a leading `.` is normal and must not be mistaken for a
    // nested path. Everything else has to be a single plain file name: no
    // directories, no `..`, no absolute paths.
    let mut parts = name
        .components()
        .filter(|c| !matches!(c, Component::CurDir));
    let Some(Component::Normal(file)) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    let Some(file) = file.to_str() else {
        return false;
    };
    permitted_file_name(file)
}

fn permitted_file_name(file: &str) -> bool {
    file == GGML_VERSION_FILE || file == PLUGIN_FILE
}

/// Set while an [`install`] runs. Every install writes the same
/// `.gpu-plugin.tar.gz.part` and unpacks into the same directory, so a second
/// one would interleave its bytes with the first and fail both digests.
static INSTALL_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// Holds [`INSTALL_IN_FLIGHT`] and clears it on drop, which also covers an
/// install whose future is dropped mid-download.
struct InstallGuard;

impl InstallGuard {
    fn acquire() -> Option<Self> {
        INSTALL_IN_FLIGHT
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        INSTALL_IN_FLIGHT.store(false, Ordering::Release);
    }
}

/// Download, verify, and unpack the GPU backend into `dest_dir`.
///
/// One install runs at a time; a call made while another is in flight is
/// refused.
///
/// The archive is streamed to a temporary file next to the destination and
/// hashed as it arrives, so a corrupt or tampered download never reaches the
/// plugin directory. Extraction happens only after the digest matches.
pub async fn install<F>(
    release: &GpuPluginRelease,
    dest_dir: &Path,
    mut on_progress: F,
) -> PersistResult<Vec<PathBuf>>
where
    F: FnMut(GpuInstallProgress),
{
    if release.sha256 == UNPUBLISHED {
        return Err(PersistError::Io(std::io::Error::other(
            "no published GPU backend for this platform yet: build one with `cgs gpu-plugin`, \
             upload it to the GitHub release, and pin its SHA-256 in gpu_plugin.rs",
        )));
    }
    let Some(_in_flight) = InstallGuard::acquire() else {
        return Err(PersistError::GpuPlugin(
            "GPU support is already being installed — wait for that download to finish."
                .to_string(),
        ));
    };
    std::fs::create_dir_all(dest_dir)?;

    let archive = dest_dir.join(".gpu-plugin.tar.gz.part");
    // A failed download must not leave hundreds of megabytes of partial archive
    // sitting in the plugin directory forever.
    let digest = match stream_to_file(release, &archive, &mut on_progress).await {
        Ok(digest) => digest,
        Err(e) => {
            let _ = std::fs::remove_file(&archive);
            return Err(e);
        }
    };

    if !digest.eq_ignore_ascii_case(release.sha256) {
        let _ = std::fs::remove_file(&archive);
        tracing::warn!(
            url = release.url,
            expected = release.sha256,
            got = %digest,
            "GPU backend archive failed digest verification; refusing to install"
        );
        return Err(PersistError::GpuPlugin(
            "The GPU download did not match what was expected and has been discarded. \
             This usually means the transfer was corrupted — please try again."
                .to_string(),
        ));
    }

    let archive_path = archive.clone();
    let dest = dest_dir.to_path_buf();
    let written = tokio::task::spawn_blocking(move || unpack(&archive_path, &dest))
        .await
        .map_err(|e| PersistError::Io(std::io::Error::other(format!("GPU backend unpack: {e}"))));
    let _ = std::fs::remove_file(&archive);
    let written = written??;

    on_progress(GpuInstallProgress {
        downloaded_bytes: release.download_bytes,
        total_bytes: release.download_bytes,
        done: true,
    });
    Ok(written)
}

/// Turn a failed archive fetch into copy the user can act on.
///
/// Both frontends render this verbatim — Settings › Models and the onboarding
/// GPU step — so it carries no reqwest wording, no status line and no URL; the
/// technical detail goes to the log instead. `None` means the request never
/// got far enough to have a status (DNS, TLS, connect timeout).
///
/// Every branch says transcription still works, because it does: the GPU is an
/// optional accelerator and a failure here is not a broken install.
fn download_failure_message(status: Option<reqwest::StatusCode>) -> String {
    match status {
        None => "Could not connect to the download server. Check your internet \
                 connection and try again — transcription keeps running on the CPU."
            .to_string(),
        // The archive for this version was never uploaded. Nothing the user can
        // retry their way out of, so do not invite one.
        Some(s) if s == reqwest::StatusCode::NOT_FOUND || s == reqwest::StatusCode::GONE => {
            "GPU support is not available to download yet — the package for this \
             version has not been published. Transcription keeps running on the CPU."
                .to_string()
        }
        Some(s) if s.is_server_error() => "The download server is having trouble right now. \
             Please try again later — transcription keeps running on the CPU."
            .to_string(),
        Some(_) => "The download server refused the request. Please try again later — \
             transcription keeps running on the CPU."
            .to_string(),
    }
}

/// How long to wait for the TCP/TLS handshake before giving up.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// How long a single read may stall before the download is abandoned.
///
/// Deliberately a *read* timeout rather than a total-request timeout: this is a
/// ~500 MB transfer, and any whole-request deadline generous enough for a slow
/// connection is too generous to catch a server that accepts the connection and
/// then goes silent — which is the failure this exists to bound.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Headroom over the pinned size before a download is treated as hostile.
/// The archive is a fixed, known artifact; anything past this is a server
/// streaming us bytes we will never accept, so stop rather than fill the disk.
const SIZE_TOLERANCE_PERCENT: u64 = 10;

/// Ceiling on bytes written to disk for `release`.
fn max_download_bytes(release: &GpuPluginRelease) -> u64 {
    // Checked, not saturating: `saturating_mul` would clamp to `u64::MAX` and
    // the following `/ 100` would then yield a ceiling *below* the release size,
    // rejecting every legitimate byte. On overflow there is no meaningful cap to
    // impose, and the SHA-256 pin still gates what gets installed.
    release
        .download_bytes
        .checked_mul(100 + SIZE_TOLERANCE_PERCENT)
        .map_or(u64::MAX, |scaled| scaled / 100)
}

/// How much has to arrive before another progress event is emitted. Every
/// event crosses the IPC boundary, and a several-hundred-megabyte download
/// yields tens of thousands of chunks — reporting each one floods the channel
/// and janks the UI it is meant to inform.
const PROGRESS_STEP_BYTES: u64 = 4 * 1024 * 1024;

/// Stream the archive to `path`, hashing as it goes. Returns the hex digest.
async fn stream_to_file<F>(
    release: &GpuPluginRelease,
    path: &Path,
    on_progress: &mut F,
) -> PersistResult<String>
where
    F: FnMut(GpuInstallProgress),
{
    use futures_util::StreamExt;
    use std::io::Write;

    // A default client has no timeouts at all, so a server that accepts the
    // connection and then stalls would hang this `await` forever — with the UI
    // parked on a progress bar that never moves and no way to cancel.
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .map_err(|e| {
            PersistError::Io(std::io::Error::other(format!(
                "GPU backend http client: {e}"
            )))
        })?;

    let response = client
        .get(release.url)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(url = release.url, error = %e, "GPU backend fetch failed");
            PersistError::GpuPlugin(download_failure_message(e.status()))
        })?
        .error_for_status()
        .map_err(|e| {
            tracing::warn!(url = release.url, error = %e, "GPU backend fetch rejected");
            PersistError::GpuPlugin(download_failure_message(e.status()))
        })?;

    let declared = response.content_length();
    let total = declared.unwrap_or(release.download_bytes);
    let max_bytes = max_download_bytes(release);
    // Reject an oversized body from the advertised length before transferring
    // any of it, when the server is honest enough to declare one.
    if let Some(len) = declared
        && len > max_bytes
    {
        return Err(PersistError::Io(std::io::Error::other(format!(
            "GPU backend archive declares {len} bytes, more than the {max_bytes} expected for \
             this release; refusing to download it"
        ))));
    }
    let mut file = std::fs::File::create(path)?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0u64;
    let mut reported = 0u64;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            tracing::warn!(url = release.url, error = %e, "GPU backend download interrupted");
            PersistError::GpuPlugin(
                "The GPU download stopped before it finished. Check your internet \
                 connection and try again — transcription keeps running on the CPU."
                    .to_string(),
            )
        })?;
        downloaded += chunk.len() as u64;
        // The digest is only checked once the whole body has arrived, so
        // without this a server could stream unbounded bytes into the user's
        // data directory before verification ever runs.
        if downloaded > max_bytes {
            return Err(PersistError::Io(std::io::Error::other(format!(
                "GPU backend archive exceeded the {max_bytes} bytes expected for this release; \
                 aborting download"
            ))));
        }
        hasher.update(&chunk);
        file.write_all(&chunk)?;
        if downloaded - reported >= PROGRESS_STEP_BYTES {
            reported = downloaded;
            on_progress(GpuInstallProgress {
                downloaded_bytes: downloaded,
                total_bytes: total,
                done: false,
            });
        }
    }
    // The last partial step still has to be shown, or the bar stops short of
    // the end while the (long) unpack runs.
    if downloaded != reported {
        on_progress(GpuInstallProgress {
            downloaded_bytes: downloaded,
            total_bytes: total,
            done: false,
        });
    }
    // Flushing ~500 MB can take seconds of pure disk I/O. `stream_to_file` is
    // awaited from a Tauri command, so doing it inline would park an async
    // worker — and the download UI with it — for the whole flush.
    tokio::task::spawn_blocking(move || file.sync_all())
        .await
        .map_err(|e| PersistError::Io(std::io::Error::other(format!("archive flush: {e}"))))??;
    Ok(hex(&hasher.finalize()))
}

/// Suffix each member is written under before the set is published. Anything
/// left with this suffix is debris from a failed install, never a library the
/// app would load: ggml only ever `dlopen`s files ending in `.so`.
const PART_SUFFIX: &str = ".lune-part";

/// Unpack the verified archive, writing only permitted members.
fn unpack(archive: &Path, dest_dir: &Path) -> PersistResult<Vec<PathBuf>> {
    let mut staged: Vec<(PathBuf, PathBuf)> = Vec::new();
    if let Err(e) = extract_members(archive, dest_dir, &mut staged) {
        for (part, _) in &staged {
            let _ = std::fs::remove_file(part);
        }
        return Err(e);
    }

    if staged.is_empty() {
        return Err(PersistError::Io(std::io::Error::other(
            "GPU backend archive contained no libraries",
        )));
    }

    // Record what was installed before anything is published. The pinned
    // archive digest proves the download; this proves the files still on disk
    // are the ones that came out of it, which is a different question and the
    // one that matters at `dlopen` time — see `lune_asr::gpu_backend`.
    let manifest = staged
        .iter()
        .map(|(part, out)| {
            let name = out
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| {
                    PersistError::Io(std::io::Error::other("plugin file name is not UTF-8"))
                })?
                .to_owned();
            Ok((name, sha256_of_file(part)?))
        })
        .collect::<PersistResult<BTreeMap<String, String>>>()?;
    write_plugin_manifest(dest_dir, &manifest)?;

    // The plugin file is the installation marker, so publish it last.
    staged.sort_by_key(|(_, out)| out.file_name().is_some_and(|n| n == PLUGIN_FILE));

    let mut written = Vec::with_capacity(staged.len());
    for (part, out) in staged {
        std::fs::rename(&part, &out)?;
        written.push(out);
    }
    Ok(written)
}

/// Name of the per-file digest manifest written beside the installed plugin.
///
/// Read back by `lune_asr::gpu_backend` before any of these libraries is
/// `dlopen`ed — the two names must stay in step.
pub const PLUGIN_MANIFEST: &str = ".integrity.json";

/// Stream-hash a file with SHA-256; returns lowercase hex.
fn sha256_of_file(path: &Path) -> PersistResult<String> {
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
    Ok(hex::encode(hasher.finalize()))
}

fn write_plugin_manifest(dir: &Path, manifest: &BTreeMap<String, String>) -> PersistResult<()> {
    let json = serde_json::to_vec_pretty(manifest)
        .map_err(|e| PersistError::Io(std::io::Error::other(e)))?;
    std::fs::write(dir.join(PLUGIN_MANIFEST), json)?;
    Ok(())
}

/// Extract every permitted member to its `.lune-part` staging name, recording
/// `(staged, final)` pairs in `staged` as it goes so a failure can clean up
/// exactly what it created.
fn extract_members(
    archive: &Path,
    dest_dir: &Path,
    staged: &mut Vec<(PathBuf, PathBuf)>,
) -> PersistResult<()> {
    let file = std::fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));

    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if !entry.header().entry_type().is_file() {
            continue;
        }
        if !is_permitted_member(&path) {
            return Err(PersistError::Io(std::io::Error::other(format!(
                "GPU backend archive contains an unexpected member: {}",
                path.display()
            ))));
        }
        let name = path
            .file_name()
            .expect("permitted members have a file name")
            .to_owned();
        let out = dest_dir.join(&name);
        let mut part_name = name;
        part_name.push(PART_SUFFIX);
        let part = dest_dir.join(part_name);

        let mut out_file = std::fs::File::create(&part)?;
        staged.push((part.clone(), out));
        std::io::copy(&mut entry, &mut out_file)?;
        out_file.sync_all()?;
        drop(out_file);
        set_executable(&part)?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn remove_legacy_plugin(backends_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(backends_dir) else {
        return Vec::new();
    };
    let mut removed = Vec::new();
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if !(kind.is_file() || kind.is_symlink()) {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let name = name.strip_suffix(PART_SUFFIX).unwrap_or(name);
        if is_legacy_member(name) && std::fs::remove_file(entry.path()).is_ok() {
            removed.push(entry.path());
        }
    }
    removed
}

fn is_legacy_member(name: &str) -> bool {
    if [
        GGML_VERSION_FILE,
        PLUGIN_MANIFEST,
        "libggml-cuda.so",
        "ggml-cuda.dll",
        ".gpu-plugin.tar.gz.part",
    ]
    .contains(&name)
    {
        return true;
    }
    let soname = |stem: &str| {
        name.strip_prefix(stem).is_some_and(|rest| {
            rest.is_empty()
                || rest.strip_prefix('.').is_some_and(|v| {
                    !v.is_empty() && v.chars().all(|c| c.is_ascii_digit() || c == '.')
                })
        })
    };
    let dll = |stem: &str| {
        name.strip_prefix(stem)
            .and_then(|rest| rest.strip_prefix("64_"))
            .and_then(|rest| rest.strip_suffix(".dll"))
            .is_some_and(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()))
    };
    ["libcudart.so", "libcublas.so", "libcublasLt.so"]
        .into_iter()
        .any(soname)
        || ["cudart", "cublas", "cublasLt"].into_iter().any(dll)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAMP_MEMBER: &str = "./ggml-version";

    #[test]
    fn the_old_cuda_install_is_removed_and_nothing_else() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let backends = tmp.path();
        let legacy = [
            "libggml-cuda.so",
            "libcudart.so.13",
            "libcublas.so.13",
            "libcublasLt.so.13.1.0",
            "ggml-cuda.dll",
            "cudart64_13.dll",
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "ggml-version",
            ".integrity.json",
            "libcublasLt.so.13.lune-part",
            "libcudart.so",
            ".gpu-plugin.tar.gz.part",
        ];
        for name in legacy {
            std::fs::write(backends.join(name), b"x").expect("write legacy file");
        }
        let kept = ["notes.txt", "libggml-cpu-haswell.so", "libcudart.so.evil"];
        for name in kept {
            std::fs::write(backends.join(name), b"x").expect("write kept file");
        }
        std::fs::create_dir_all(backends.join("vulkan")).expect("create vulkan dir");
        std::fs::write(backends.join("vulkan").join("ggml-version"), b"0.23.0\n")
            .expect("write vulkan stamp");

        let mut removed = remove_legacy_plugin(backends);
        removed.sort();
        let mut expected: Vec<PathBuf> = legacy.iter().map(|n| backends.join(n)).collect();
        expected.sort();
        assert_eq!(removed, expected);
        for name in legacy {
            assert!(!backends.join(name).exists(), "{name} should be gone");
        }
        for name in kept {
            assert!(backends.join(name).is_file(), "{name} should stay");
        }
        assert!(backends.join("vulkan").join("ggml-version").is_file());
    }

    #[test]
    fn a_missing_backends_dir_is_nothing_to_clean() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(remove_legacy_plugin(&tmp.path().join("absent")).is_empty());
    }

    #[test]
    fn permitted_members_are_the_vulkan_plugin_only() {
        #[cfg(not(target_os = "windows"))]
        let names = ["libggml-vulkan.so", "./libggml-vulkan.so"];
        #[cfg(target_os = "windows")]
        let names = ["ggml-vulkan.dll", "./ggml-vulkan.dll"];
        for ok in names {
            assert!(is_permitted_member(Path::new(ok)), "{ok} should be allowed");
        }
    }

    #[test]
    fn the_ggml_version_stamp_is_a_permitted_member() {
        // The app refuses a plugin without it, so the installer must keep it.
        for ok in ["ggml-version", "./ggml-version"] {
            assert!(is_permitted_member(Path::new(ok)), "{ok} should be allowed");
        }
        for bad in ["ggml-version.txt", "nested/ggml-version", "../ggml-version"] {
            assert!(
                !is_permitted_member(Path::new(bad)),
                "{bad} should be refused"
            );
        }
    }

    #[test]
    fn unrelated_or_nested_members_are_refused() {
        // Shape violations are platform-independent: a nested or escaping
        // path is refused before any name matcher sees it. The trailing
        // entries are this platform's "permitted name plus junk" cases.
        #[cfg(not(target_os = "windows"))]
        let names = [
            "evil.so",
            "run.sh",
            "nested/libggml-vulkan.so",
            "./nested/libggml-vulkan.so",
            "../libggml-vulkan.so",
            "/libggml-vulkan.so",
            "libvulkan.so.1",
            "libggml-vulkan.soevil",
            "libggml-vulkan.so.1",
            "libggml-cuda.so",
            "libcudart.so.13",
        ];
        #[cfg(target_os = "windows")]
        let names = [
            "evil.dll",
            "run.bat",
            "nested/ggml-vulkan.dll",
            "./nested/ggml-vulkan.dll",
            "../ggml-vulkan.dll",
            "/ggml-vulkan.dll",
            "vulkan-1.dll",
            "ggml-vulkan.dllevil",
            "ggml-cuda.dll",
            "cudart64_13.dll",
        ];
        for bad in names {
            assert!(
                !is_permitted_member(Path::new(bad)),
                "{bad} should be refused"
            );
        }
    }

    #[test]
    fn a_release_is_unpublished_until_its_digest_is_pinned() {
        let mut release = GpuPluginRelease {
            url: "https://example.invalid/plugin.tar.gz",
            sha256: UNPUBLISHED,
            download_bytes: 1,
        };
        assert!(!release.is_published());
        release.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(release.is_published());
    }

    /// Build a gzipped tar of `(name, bytes)` members at `path`.
    fn write_archive(path: &Path, members: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).expect("create archive");
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::none(),
        ));
        for (name, bytes) in members {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, name, *bytes)
                .expect("append member");
        }
        builder
            .into_inner()
            .expect("finish tar")
            .finish()
            .expect("finish gzip");
    }

    #[test]
    fn a_rejected_member_leaves_no_half_installation_behind() {
        // The plugin file is extracted before the bad member is reached. If it
        // were published as it went, `plugin_installed` would report an install
        // that cannot load — and the UI offers no way back from that state.
        let tmp = tempfile::tempdir().expect("tempdir");
        let archive = tmp.path().join("gpu.tar.gz");
        let plugin_name = format!("./{PLUGIN_FILE}");
        write_archive(
            &archive,
            &[
                (plugin_name.as_str(), b"plugin" as &[u8]),
                (STAMP_MEMBER, b"0.23.0\n"),
                ("./evil.so", b"nope"),
            ],
        );

        let err = unpack(&archive, tmp.path()).expect_err("must refuse the archive");
        assert!(err.to_string().contains("unexpected member"), "{err}");
        assert!(!plugin_installed_at(tmp.path()));
        let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
            .expect("read dir")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "gpu.tar.gz")
            .collect();
        assert!(
            leftovers.is_empty(),
            "staged files were left behind: {leftovers:?}"
        );
    }

    #[test]
    fn a_clean_archive_publishes_every_member() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let archive = tmp.path().join("gpu.tar.gz");
        let plugin_name = format!("./{PLUGIN_FILE}");
        write_archive(
            &archive,
            &[
                (STAMP_MEMBER, b"0.23.0\n" as &[u8]),
                (plugin_name.as_str(), b"plugin"),
            ],
        );

        let written = unpack(&archive, tmp.path()).expect("unpack");
        assert_eq!(written.len(), 2);
        assert!(plugin_installed_at(tmp.path()));
        assert!(tmp.path().join(GGML_VERSION_FILE).is_file());
        assert!(tmp.path().join(PLUGIN_MANIFEST).is_file());
    }

    /// `plugin_installed` resolves `<data dir>/backends`; these tests unpack
    /// straight into a directory, so check the marker file directly.
    fn plugin_installed_at(dir: &Path) -> bool {
        dir.join(PLUGIN_FILE).is_file()
    }

    #[test]
    fn the_download_ceiling_allows_headroom_but_not_a_runaway_body() {
        let release = GpuPluginRelease {
            url: "https://example.invalid/plugin.tar.gz",
            sha256: UNPUBLISHED,
            download_bytes: 1_000_000,
        };
        let max = max_download_bytes(&release);
        // A little over the pinned size is fine — servers and recompression
        // wobble — but an order of magnitude is not.
        assert_eq!(max, 1_100_000);
        assert!(max > release.download_bytes);
        assert!(max < release.download_bytes * 2);
    }

    #[test]
    fn the_download_ceiling_does_not_overflow_on_an_absurd_release_size() {
        let release = GpuPluginRelease {
            url: "https://example.invalid/plugin.tar.gz",
            sha256: UNPUBLISHED,
            download_bytes: u64::MAX,
        };
        // Saturating rather than wrapping: a wrapped ceiling would be tiny and
        // would reject every legitimate byte.
        assert!(max_download_bytes(&release) >= release.download_bytes / 2);
    }

    #[tokio::test]
    async fn an_unpublished_release_refuses_to_install() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let release = GpuPluginRelease {
            url: "https://example.invalid/unpublished.tar.gz",
            sha256: UNPUBLISHED,
            download_bytes: 1,
        };
        let err = install(&release, tmp.path(), |_| {})
            .await
            .expect_err("must refuse an unpinned digest");
        assert!(
            err.to_string().contains("cgs gpu-plugin"),
            "error should say how to publish one: {err}"
        );
    }

    #[tokio::test]
    async fn a_second_install_is_refused_while_one_is_running() {
        // A server that accepts the download and never answers, so the first
        // install stays in flight for as long as the test needs.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (accepted_tx, mut accepted_rx) = tokio::sync::mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                held.push(stream);
                let _ = accepted_tx.send(());
            }
        });

        let url: &'static str = Box::leak(format!("http://{addr}/plugin.tar.gz").into_boxed_str());
        let release: &'static GpuPluginRelease = Box::leak(Box::new(GpuPluginRelease {
            url,
            sha256: "00",
            download_bytes: 1,
        }));
        let tmp = tempfile::tempdir().expect("tempdir");
        let dest = tmp.path().to_path_buf();

        let first = tokio::spawn({
            let dest = dest.clone();
            async move { install(release, &dest, |_| {}).await }
        });
        accepted_rx.recv().await.expect("first install connects");

        let second = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            install(release, &dest, |_| {}),
        )
        .await
        .expect("the second install must not wait on the first");
        let err = second.expect_err("a concurrent install must be refused");
        assert!(
            err.to_string().contains("already"),
            "error should say an install is running: {err}"
        );

        first.abort();
        let _ = first.await;
        server.abort();
    }

    /// Restoring a digest without its byte size is a silent footgun:
    /// [`max_download_bytes`] scales `download_bytes`, so a published release
    /// left at `0` caps the transfer at zero and refuses every legitimate byte
    /// with a size complaint. Both halves go back together or neither does.
    #[test]
    fn a_published_release_must_declare_its_size() {
        for release in [VULKAN_LINUX_X86_64, VULKAN_WINDOWS_X86_64] {
            if release.is_published() {
                assert!(
                    release.download_bytes > 0,
                    "{} pins a digest but no size — restore both",
                    release.url
                );
            }
        }
    }

    /// The platform selector must agree with the platform this test runs on;
    /// a mismatch would offer Linux bytes to a Windows install.
    #[test]
    fn the_current_release_matches_the_build_target() {
        let want = if cfg!(target_os = "linux") {
            Some("linux")
        } else if cfg!(target_os = "windows") {
            Some("windows")
        } else {
            None
        };
        match (CURRENT, want) {
            (Some(r), Some(os)) => assert!(
                r.url.contains(os) && r.url.contains("vulkan"),
                "selected a {os} build but got {}",
                r.url
            ),
            (None, None) => {}
            (Some(r), None) => panic!("no GPU plugin exists here, yet {} was offered", r.url),
            (None, Some(os)) => panic!("{os} has a GPU plugin but none was selected"),
        }
    }

    /// Every download failure is rendered verbatim under the GPU card in
    /// Settings and in onboarding, so none of them may leak reqwest's wording,
    /// the `io error:` prefix or the release URL at the user.
    fn assert_user_facing(msg: &str) {
        for leak in [
            "io error",
            "HTTP status",
            "github.com",
            "reqwest",
            "error sending request",
        ] {
            assert!(
                !msg.to_ascii_lowercase()
                    .contains(&leak.to_ascii_lowercase()),
                "developer detail {leak:?} leaked into user copy: {msg}"
            );
        }
        assert!(
            msg.chars().next().is_some_and(char::is_uppercase),
            "user copy should read as a sentence: {msg}"
        );
    }

    #[test]
    fn a_missing_archive_reads_as_not_yet_available() {
        let msg = download_failure_message(Some(reqwest::StatusCode::NOT_FOUND));
        assert_user_facing(&msg);
        assert!(
            msg.contains("not") && msg.to_ascii_lowercase().contains("available"),
            "a 404 means the archive was never published, not that the user did \
             something wrong: {msg}"
        );
        assert!(
            msg.to_ascii_lowercase().contains("cpu"),
            "the user should be told transcription still works: {msg}"
        );
    }

    #[test]
    fn a_server_error_invites_a_retry() {
        let msg = download_failure_message(Some(reqwest::StatusCode::SERVICE_UNAVAILABLE));
        assert_user_facing(&msg);
        assert!(
            msg.to_ascii_lowercase().contains("again"),
            "a 5xx is transient — say so: {msg}"
        );
    }

    /// A 404 and an outage are the same `PersistError` variant today and are
    /// told apart only by prose, so they must not read alike.
    #[test]
    fn an_unreachable_server_is_distinct_from_a_missing_archive() {
        let offline = download_failure_message(None);
        let missing = download_failure_message(Some(reqwest::StatusCode::NOT_FOUND));
        assert_user_facing(&offline);
        assert!(
            offline.to_ascii_lowercase().contains("connect")
                || offline.to_ascii_lowercase().contains("connection"),
            "an unreachable server should point at the network: {offline}"
        );
        assert_ne!(
            offline, missing,
            "an outage and an unpublished archive need different advice"
        );
    }

    /// `PersistError::Io` renders with an `io error:` prefix, which is why the
    /// raw download failures read like a stack trace. The GPU variant must
    /// render bare so the copy above survives the trip to the UI.
    #[test]
    fn the_gpu_error_variant_renders_without_a_prefix() {
        let e = PersistError::GpuPlugin("Something the user can act on.".into());
        assert_eq!(e.to_string(), "Something the user can act on.");
    }
}
