//! dlopen-able ggml compute backends.
//!
//! The shipped binary is built with `GGML_BACKEND_DL=ON`, so ggml's compute
//! backends are plugins rather than libraries linked into the executable. Two
//! directories hold them, and both are scanned at startup:
//!
//! - **Bundled** (`<exe dir>/backends/`): the CPU backends, shipped with the
//!   app. Without these there is no compute backend at all, so they are not
//!   optional and never downloaded.
//! - **Downloaded** (`<data dir>/backends/`): the CUDA plugin and the NVIDIA
//!   runtime libraries it needs. Absent until the user opts in.
//!
//! The practical win is that the executable carries no `DT_NEEDED` on
//! `libcudart`/`libcublas`, so it starts on a machine with no NVIDIA driver at
//! all and GPU support becomes an optional download rather than a separate
//! build of the whole application.
//!
//! Why not the obvious approach: setting `LD_LIBRARY_PATH` from inside the
//! process does not work. glibc's loader reads it once at startup into a cached
//! search list, so a later `setenv` changes nothing for subsequent `dlopen`
//! calls. Each plugin instead carries `RUNPATH=$ORIGIN`, applied when it is
//! built or packaged, so it resolves its own dependencies from its own
//! directory.
//!
//! Ordering matters: [`load_plugins`] must run before the first
//! [`whisper_rs::WhisperContext`] is created, because ggml selects a device
//! from whatever is in the backend registry at context-creation time.

use std::path::{Path, PathBuf};

use lune_core::models::DIR_GPU_BACKEND;

/// Shared-object name of the CUDA compute backend, as emitted by ggml's build.
/// Used to detect an installed plugin and to validate a downloaded archive.
pub const CUDA_PLUGIN_FILE: &str = "libggml-cuda.so";

/// The NVIDIA driver library. Ships with the driver, not the CUDA toolkit, and
/// is deliberately not redistributable — so its presence is the one
/// precondition a download cannot satisfy on the user's behalf.
const NVIDIA_DRIVER_SONAME: &str = "libcuda.so.1";

/// Plugin directory shipped inside the application bundle, resolved relative to
/// the running executable. `None` when the executable path is unavailable.
pub fn bundled_plugin_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join(DIR_GPU_BACKEND))
}

/// Plugin directory the GPU download installs into.
pub fn downloaded_plugin_dir(data_dir: &Path) -> PathBuf {
    data_dir.join(DIR_GPU_BACKEND)
}

/// Whether the downloadable CUDA plugin is present *and* intact.
///
/// Presence alone is not enough: an interrupted download can leave a zero-byte
/// `libggml-cuda.so` behind, which then reports as installed, fails to load,
/// and gets misdiagnosed as an out-of-date driver. A failed check reads as "not
/// installed", which is the state that offers the download again.
pub fn plugin_installed(data_dir: &Path) -> bool {
    let dir = downloaded_plugin_dir(data_dir);
    dir.join(CUDA_PLUGIN_FILE).is_file() && installed_plugin_is_intact(&dir)
}

/// Name of the digest manifest `lune_persist::gpu_plugin` writes at install
/// time. The two names must stay in step.
const PLUGIN_MANIFEST: &str = ".integrity.json";

/// Whether every file the installer recorded is still byte-for-byte what it
/// installed.
///
/// The archive's pinned SHA-256 proves what was *downloaded*; this proves what
/// is on disk now, which is the question that matters at `dlopen` time. These
/// libraries execute their constructors inside a process that holds an
/// exclusive grab on every keyboard, so anything that can drop a file into the
/// plugin directory would otherwise get code execution here. A missing or
/// unreadable manifest counts as a failure — the installer always writes one.
fn installed_plugin_is_intact(dir: &Path) -> bool {
    use sha2::{Digest, Sha256};

    let Ok(bytes) = std::fs::read(dir.join(PLUGIN_MANIFEST)) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_slice::<std::collections::BTreeMap<String, String>>(&bytes)
    else {
        return false;
    };
    if manifest.is_empty() {
        return false;
    }
    manifest.iter().all(|(name, expected)| {
        let Ok(mut file) = std::fs::File::open(dir.join(name)) else {
            return false;
        };
        let mut hasher = Sha256::new();
        if std::io::copy(&mut file, &mut hasher).is_err() {
            return false;
        }
        format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(expected)
    })
}

/// What GPU acceleration can do on this machine right now. Each variant maps to
/// exactly one thing the setup UI can say or offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuPluginState {
    /// This build has no plugin support compiled in (stub ASR backend).
    Unsupported,
    /// No NVIDIA driver here. Downloading would not help: the driver is not
    /// ours to redistribute, so the user has to install it themselves.
    DriverMissing,
    /// Driver present, plugin not downloaded yet. This is the offer state.
    NotInstalled,
    /// Plugin loaded, carrying the compute devices ggml registered so the UI
    /// can name the GPU instead of claiming success generically.
    Ready { devices: Vec<String> },
    /// Plugin present but no usable device appeared. Either it was installed
    /// after this process built its backend registry (restart needed), or the
    /// driver is older than the plugin's CUDA runtime. `reason` says which.
    Failed { reason: String },
}

/// Whether an NVIDIA driver is installed, tested the way the loader will test
/// it: `dlopen` the soname and let the dynamic linker search its normal paths.
/// Scanning `/usr/local/cuda` and friends would both miss distro packages that
/// put the driver elsewhere and find a *toolkit* that has no driver behind it.
#[cfg(unix)]
pub fn nvidia_driver_present() -> bool {
    use std::ffi::CString;

    let Ok(name) = CString::new(NVIDIA_DRIVER_SONAME) else {
        return false;
    };
    // SAFETY: `name` is a valid NUL-terminated C string that outlives the call.
    // `dlopen`/`dlclose` are thread-safe, and the handle is closed on success.
    unsafe {
        let handle = libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        if handle.is_null() {
            return false;
        }
        libc::dlclose(handle);
        true
    }
}

#[cfg(not(unix))]
pub fn nvidia_driver_present() -> bool {
    false
}

#[cfg(feature = "whisper-cpp")]
mod imp {
    use super::{
        CUDA_PLUGIN_FILE, GpuPluginState, bundled_plugin_dir, downloaded_plugin_dir,
        nvidia_driver_present,
    };

    use std::ffi::{CStr, CString};
    use std::path::Path;
    use std::sync::OnceLock;

    use tracing::{debug, info, warn};

    /// Result of the one load attempt this process makes. ggml's backend
    /// registry is process-global with no unload, so loading twice would
    /// register duplicate devices — hence the `OnceLock`.
    struct Loaded {
        /// Every compute device in ggml's registry after the load. Not just the
        /// ones the load added: a statically linked CUDA build registers its
        /// device before we run, and it is just as real.
        devices: Vec<String>,
        /// Whether the downloaded plugin was on disk when the registry was
        /// built. A plugin that arrived later cannot register until restart,
        /// which is what separates "restart me" from "this is broken".
        plugin_at_load: bool,
    }

    static LOADED: OnceLock<Loaded> = OnceLock::new();

    /// Load every ggml backend plugin from the bundled and downloaded plugin
    /// directories, returning the compute devices that registered as a result.
    ///
    /// Idempotent: the first call does the work, later calls replay its result.
    /// A directory that does not exist is skipped, which is exactly the
    /// CPU-only and GPU-not-installed cases.
    pub fn load_plugins(data_dir: &Path) -> Vec<String> {
        loaded(data_dir).devices.clone()
    }

    fn loaded(data_dir: &Path) -> &'static Loaded {
        LOADED.get_or_init(|| {
            // Verifying the downloaded plugin re-hashes every installed
            // library — hundreds of megabytes. Do it once and share the answer
            // with the load below, which asks the same question.
            let plugin_at_load = super::plugin_installed(data_dir);
            Loaded {
                plugin_at_load,
                devices: load_once(data_dir, plugin_at_load),
            }
        })
    }

    fn load_once(data_dir: &Path, downloaded_is_intact: bool) -> Vec<String> {
        let downloaded = downloaded_plugin_dir(data_dir);
        for dir in bundled_plugin_dir()
            .into_iter()
            .chain(std::iter::once(downloaded.clone()))
        {
            if !dir.is_dir() {
                debug!(dir = %dir.display(), "no plugin directory; skipping");
                continue;
            }
            // The bundled directory ships inside the installed application and
            // is only writable by whoever installed it. The downloaded one sits
            // in the user's data directory, so its contents are re-verified
            // against the manifest the installer wrote before anything here is
            // `dlopen`ed into this process.
            if dir == downloaded && !downloaded_is_intact {
                warn!(
                    dir = %dir.display(),
                    "downloaded GPU backend does not match what was installed; refusing to load it"
                );
                continue;
            }
            let Ok(path) = CString::new(dir.as_os_str().as_encoded_bytes()) else {
                warn!(dir = %dir.display(), "plugin path is not representable as a C string");
                continue;
            };
            // SAFETY: `path` is a valid NUL-terminated C string that outlives
            // the call. ggml logs and skips any plugin it cannot load, so a
            // broken or ABI-mismatched `.so` does not abort the process.
            unsafe { whisper_rs::whisper_rs_sys::ggml_backend_load_all_from_path(path.as_ptr()) };
            debug!(dir = %dir.display(), "scanned for ggml backend plugins");
        }

        let devices = device_names();
        info!(?devices, "ggml compute devices available");
        devices
    }

    /// Names of every device currently in ggml's backend registry.
    fn device_names() -> Vec<String> {
        // SAFETY: the registry accessors take no arguments and return either a
        // count or a valid device handle for an index below that count. The
        // returned name is a static C string owned by the backend.
        unsafe {
            let count = whisper_rs::whisper_rs_sys::ggml_backend_dev_count();
            (0..count)
                .filter_map(|i| {
                    let dev = whisper_rs::whisper_rs_sys::ggml_backend_dev_get(i);
                    if dev.is_null() {
                        return None;
                    }
                    let name = whisper_rs::whisper_rs_sys::ggml_backend_dev_name(dev);
                    if name.is_null() {
                        return None;
                    }
                    Some(CStr::from_ptr(name).to_string_lossy().into_owned())
                })
                .collect()
        }
    }

    /// Current GPU state for the setup UI. Loading is a side effect when a
    /// plugin is installed, which is what makes "installed" and "actually
    /// working" distinguishable rather than assumed.
    pub fn state(data_dir: &Path) -> GpuPluginState {
        let loaded = loaded(data_dir);

        // A registered CUDA device is the only proof acceleration actually
        // works, and it is the right test for both shapes of build: the
        // downloaded plugin *and* a statically linked `whisper-cuda` binary,
        // which has no plugin file on disk yet needs no download either. The
        // bundled CPU backends are always registered, so presence of *a* device
        // proves nothing — the "CUDA0", "CUDA1", … prefix is what matters.
        let cuda: Vec<String> = loaded
            .devices
            .iter()
            .filter(|d| d.starts_with("CUDA"))
            .cloned()
            .collect();
        if !cuda.is_empty() {
            return GpuPluginState::Ready { devices: cuda };
        }

        if !super::plugin_installed(data_dir) {
            return if nvidia_driver_present() {
                GpuPluginState::NotInstalled
            } else {
                GpuPluginState::DriverMissing
            };
        }

        // Installed, but no CUDA device. If the plugin was not there when the
        // registry was built, that is expected and temporary — ggml has no
        // unload, so it can only appear on the next launch. Reporting the
        // driver-too-old guess here would be a false alarm.
        if !loaded.plugin_at_load {
            return GpuPluginState::Failed {
                reason: "Installed. Restart Lune Voice to start using your GPU.".to_owned(),
            };
        }
        GpuPluginState::Failed {
            reason: format!(
                "{CUDA_PLUGIN_FILE} is installed but registered no CUDA device. Your NVIDIA \
                 driver is most likely older than the CUDA runtime it needs."
            ),
        }
    }
}

#[cfg(not(feature = "whisper-cpp"))]
mod imp {
    use super::GpuPluginState;
    use std::path::Path;

    pub fn load_plugins(_data_dir: &Path) -> Vec<String> {
        Vec::new()
    }

    pub fn state(_data_dir: &Path) -> GpuPluginState {
        GpuPluginState::Unsupported
    }
}

pub use imp::{load_plugins, state};

/// Whether CUDA can actually be used for a decode right now.
///
/// This is a *runtime* question, and it has to stay one: the CUDA backend ships
/// as a downloadable ggml plugin, so a binary built with only `whisper-cpp` can
/// still acquire a working GPU after the user installs it. Gating on
/// `cfg!(feature = "whisper-cuda")` instead would make the plugin unreachable
/// in exactly the builds it exists for. [`state`] answers from ggml's registered
/// device list, which covers both the plugin and a static `whisper-cuda` build.
pub fn cuda_available(data_dir: &Path) -> bool {
    matches!(state(data_dir), GpuPluginState::Ready { .. })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_dir_is_under_the_data_dir() {
        assert_eq!(
            downloaded_plugin_dir(Path::new("/data")),
            Path::new("/data").join(DIR_GPU_BACKEND)
        );
    }

    #[test]
    fn bundled_dir_sits_next_to_the_executable() {
        let dir = bundled_plugin_dir().expect("current_exe resolves under test");
        let exe = std::env::current_exe().expect("current_exe");
        assert_eq!(dir.parent(), exe.parent());
        assert_eq!(dir.file_name().unwrap(), DIR_GPU_BACKEND);
    }

    #[test]
    fn cuda_is_available_only_when_a_device_actually_registered() {
        // `cuda_available` is the input to `AsrRegistry::from_config`'s GPU
        // decision, so it must agree with `state` and nothing else — in
        // particular it must never be answered by a build feature.
        let dir = std::env::temp_dir().join("lune-gpu-availability-probe");
        let expected = matches!(state(&dir), GpuPluginState::Ready { .. });
        assert_eq!(cuda_available(&dir), expected);
    }

    #[test]
    fn a_plugin_without_a_manifest_is_not_trusted() {
        // Every install writes one. Its absence means these libraries did not
        // come from the installer, and they are about to be dlopen'd into a
        // process holding an exclusive grab on the keyboard.
        let dir = std::env::temp_dir().join("lune-plugin-no-manifest");
        std::fs::create_dir_all(&dir).expect("create temp plugin dir");
        let _ = std::fs::remove_file(dir.join(PLUGIN_MANIFEST));
        assert!(!installed_plugin_is_intact(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_modified_plugin_file_fails_verification() {
        let dir = std::env::temp_dir().join("lune-plugin-tampered");
        std::fs::create_dir_all(&dir).expect("create temp plugin dir");
        std::fs::write(dir.join("libggml-cuda.so"), b"not what was installed")
            .expect("write library");
        // A digest that cannot match anything.
        std::fs::write(
            dir.join(PLUGIN_MANIFEST),
            br#"{"libggml-cuda.so":"0000000000000000000000000000000000000000000000000000000000000000"}"#,
        )
        .expect("write manifest");
        assert!(!installed_plugin_is_intact(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_file_matching_its_recorded_digest_verifies() {
        use sha2::{Digest, Sha256};
        let dir = std::env::temp_dir().join("lune-plugin-intact");
        std::fs::create_dir_all(&dir).expect("create temp plugin dir");
        let body = b"pretend shared object";
        std::fs::write(dir.join("libggml-cuda.so"), body).expect("write library");
        let digest = format!("{:x}", Sha256::digest(body));
        std::fs::write(
            dir.join(PLUGIN_MANIFEST),
            format!(r#"{{"libggml-cuda.so":"{digest}"}}"#),
        )
        .expect("write manifest");
        assert!(installed_plugin_is_intact(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn absent_plugin_file_reads_as_not_installed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(!plugin_installed(tmp.path()));
    }

    #[test]
    fn a_plugin_file_alone_is_not_an_installation() {
        use sha2::{Digest, Sha256};
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = downloaded_plugin_dir(tmp.path());
        std::fs::create_dir_all(&dir).expect("create plugin dir");
        assert!(!plugin_installed(tmp.path()), "an empty directory");

        // A zero-byte file is what an interrupted download leaves behind. It
        // used to read as installed, which turned into a bogus "your driver is
        // too old" once the load inevitably failed.
        std::fs::write(dir.join(CUDA_PLUGIN_FILE), b"").expect("write plugin");
        assert!(
            !plugin_installed(tmp.path()),
            "a truncated download is not an installation"
        );

        // With the installer's manifest beside it, the same file is installed.
        let body = b"pretend shared object";
        std::fs::write(dir.join(CUDA_PLUGIN_FILE), body).expect("write plugin");
        let digest = format!("{:x}", Sha256::digest(body));
        std::fs::write(
            dir.join(PLUGIN_MANIFEST),
            format!(r#"{{"{CUDA_PLUGIN_FILE}":"{digest}"}}"#),
        )
        .expect("write manifest");
        assert!(plugin_installed(tmp.path()));
    }
}
