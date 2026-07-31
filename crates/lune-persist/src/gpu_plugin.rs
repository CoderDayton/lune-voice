//! Downloading and installing the optional CUDA compute backend.
//!
//! The app ships without any NVIDIA code: the executable has no `DT_NEEDED` on
//! the CUDA runtime, and the CPU backends bundled beside it are enough to run.
//! GPU acceleration is this archive, fetched on request and unpacked into
//! `<data dir>/backends/`, where `lune_asr::gpu_backend` picks it up on the
//! next launch.
//!
//! What the archive may contain is constrained by licensing, not convenience.
//! `libcudart`/`libcublas`/`libcublasLt` are redistributable under NVIDIA's
//! CUDA EULA; `libcuda.so.1` ships with the driver and is not redistributable,
//! so a machine without a driver cannot be fixed by downloading. That is why
//! the caller checks `nvidia_driver_present()` before offering this.
//!
//! Integrity is not optional here: this unpacks shared objects that the app
//! will later `dlopen`. A tampered archive is arbitrary code execution, so the
//! download is rejected unless its SHA-256 matches the pinned digest exactly.

use std::path::{Component, Path, PathBuf};

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
    /// CUDA runtime major version the plugin links against, for display.
    pub cuda_major: u32,
}

/// Digest placeholder for a release that has not been published yet. Kept as a
/// recognisable sentinel so [`install`] can fail with an actionable message
/// instead of downloading something it cannot verify.
pub const UNPUBLISHED: &str = "UNPUBLISHED";

/// The CUDA backend for Linux x86-64.
///
/// `sha256` is [`UNPUBLISHED`] until the archive built by `cgs gpu-plugin` is
/// uploaded to the GitHub release; fill in both fields from the generated
/// `.sha256` file at that point.
pub const CUDA_LINUX_X86_64: GpuPluginRelease = GpuPluginRelease {
    url: "https://github.com/CoderDayton/lune-voice/releases/download/gpu-cuda13-v1/\
          lune-voice-gpu-cuda13-linux-x86_64.tar.gz",
    sha256: UNPUBLISHED,
    // Measured from `cgs gpu-plugin` on CUDA 13.1, covering sm_75/86/89/120a
    // with compute_75 PTX. Unpacked it is 664 MB, three quarters of which is
    // NVIDIA's cuBLASLt, and that part is irreducible: ggml calls no cuBLASLt symbol
    // directly, but libcublas declares it as DT_NEEDED and imports 76 symbols
    // from it, so it cannot be dropped or stubbed. `nvprune` only rewrites
    // relocatable device code, and static linking with --gc-sections was
    // measured at 541 MB for the ten cuBLAS entry points ggml uses — the
    // dispatcher reaches every kernel object, so nothing is prunable.
    download_bytes: 493_568_081,
    cuda_major: 13,
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
    file.starts_with("libggml-cuda.so")
        || file.starts_with("libcudart.so")
        || file.starts_with("libcublas.so")
        || file.starts_with("libcublasLt.so")
}

/// Download, verify, and unpack the GPU backend into `dest_dir`.
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
        return Err(PersistError::Io(std::io::Error::other(format!(
            "GPU backend archive failed verification (expected {}, got {digest}); refusing to \
             install shared libraries that will be loaded into this process",
            release.sha256
        ))));
    }

    let written = unpack(&archive, dest_dir);
    let _ = std::fs::remove_file(&archive);
    let written = written?;

    on_progress(GpuInstallProgress {
        downloaded_bytes: release.download_bytes,
        total_bytes: release.download_bytes,
        done: true,
    });
    Ok(written)
}

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

    let response = reqwest::get(release.url)
        .await
        .map_err(|e| PersistError::Io(std::io::Error::other(format!("GPU backend fetch: {e}"))))?
        .error_for_status()
        .map_err(|e| PersistError::Io(std::io::Error::other(format!("GPU backend fetch: {e}"))))?;

    let total = response.content_length().unwrap_or(release.download_bytes);
    let mut file = std::fs::File::create(path)?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            PersistError::Io(std::io::Error::other(format!("GPU backend download: {e}")))
        })?;
        hasher.update(&chunk);
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        on_progress(GpuInstallProgress {
            downloaded_bytes: downloaded,
            total_bytes: total,
            done: false,
        });
    }
    file.sync_all()?;
    Ok(hex(&hasher.finalize()))
}

/// Unpack the verified archive, writing only permitted members.
fn unpack(archive: &Path, dest_dir: &Path) -> PersistResult<Vec<PathBuf>> {
    let file = std::fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let mut written = Vec::new();

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
        let out = dest_dir.join(
            path.file_name()
                .expect("permitted members have a file name"),
        );
        // Streamed rather than buffered: cuBLASLt alone unpacks to ~479 MB, and
        // reading it into a `Vec` first would double the peak memory for no gain.
        let mut out_file = std::fs::File::create(&out)?;
        std::io::copy(&mut entry, &mut out_file)?;
        out_file.sync_all()?;
        drop(out_file);
        set_executable(&out)?;
        written.push(out);
    }

    if written.is_empty() {
        return Err(PersistError::Io(std::io::Error::other(
            "GPU backend archive contained no libraries",
        )));
    }
    Ok(written)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permitted_members_are_the_cuda_libraries_only() {
        for ok in [
            "libggml-cuda.so",
            // How GNU tar names members of an archive built with `-C dir .`.
            "./libggml-cuda.so",
            "./libcublasLt.so.13",
            "libcudart.so.13",
            "libcublas.so.13",
            "libcublasLt.so.13",
        ] {
            assert!(is_permitted_member(Path::new(ok)), "{ok} should be allowed");
        }
    }

    #[test]
    fn unrelated_or_nested_members_are_refused() {
        for bad in [
            "evil.so",
            "run.sh",
            "nested/libggml-cuda.so",
            "./nested/libggml-cuda.so",
            "../libggml-cuda.so",
            "/libggml-cuda.so",
            "libcuda.so.1",
        ] {
            assert!(
                !is_permitted_member(Path::new(bad)),
                "{bad} should be refused"
            );
        }
    }

    #[tokio::test]
    async fn an_unpublished_release_refuses_to_install() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = install(&CUDA_LINUX_X86_64, tmp.path(), |_| {})
            .await
            .expect_err("must refuse an unpinned digest");
        assert!(
            err.to_string().contains("cgs gpu-plugin"),
            "error should say how to publish one: {err}"
        );
    }
}
