// SPDX-License-Identifier: GPL-3.0-or-later

//! dlopen-able ggml compute backends.
//!
//! Ordering matters: [`load_plugins`] must run before the first
//! [`whisper_rs::WhisperContext`] is created, because ggml selects a device
//! from whatever is in the backend registry at context-creation time.

use std::path::{Path, PathBuf};

use lune_core::models::DIR_GPU_BACKEND;

/// ggml derives the name itself (`backend_filename_prefix` /
/// `backend_filename_extension`), which is why Windows drops the `lib` prefix.
#[cfg(target_os = "windows")]
pub const PLUGIN_FILE: &str = "ggml-vulkan.dll";
#[cfg(not(target_os = "windows"))]
pub const PLUGIN_FILE: &str = "libggml-vulkan.so";

/// Tauri's resource directory name on Linux: the `productName` in
/// `desktop-ui/src-tauri/tauri.conf.json`. Installed bundles keep the ggml
/// libraries and CPU backends there, as `<prefix>/lib/<productName>`; the
/// executable's run path in `desktop-ui/src-tauri/build.rs` names it too.
const LINUX_RESOURCE_DIR: &str = "LuneVoice";

/// Plugin directory shipped inside the application bundle, resolved relative to
/// the running executable. `None` when the executable path is unavailable.
pub fn bundled_plugin_dir() -> Option<PathBuf> {
    bundled_plugin_dir_for(&std::env::current_exe().ok()?)
}

/// [`bundled_plugin_dir`] for an executable at `exe`: `<exe dir>/backends` (a
/// build tree, the Windows install directory) unless only the Linux bundle's
/// resource directory has one.
fn bundled_plugin_dir_for(exe: &Path) -> Option<PathBuf> {
    let bin = exe.parent()?;
    let beside = bin.join(DIR_GPU_BACKEND);
    if !beside.is_dir()
        && let Some(prefix) = bin.parent()
    {
        let resources = prefix
            .join("lib")
            .join(LINUX_RESOURCE_DIR)
            .join(DIR_GPU_BACKEND);
        if resources.is_dir() {
            return Some(resources);
        }
    }
    Some(beside)
}

/// Plugin directories to scan, bundled first. None for a statically linked
/// ggml: every plugin links the shared libggml-base, which would put a second
/// ggml in the process, and ggml aborts when that copy is unloaded and loaded
/// again.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn plugin_dirs(backend_dl: bool, bundled: Option<PathBuf>, downloaded: PathBuf) -> Vec<PathBuf> {
    if !backend_dl {
        return Vec::new();
    }
    bundled
        .into_iter()
        .chain(std::iter::once(downloaded))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Igpu,
    Other,
}

#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn discrete_gpus(registered: &[(String, DeviceKind)]) -> Vec<String> {
    let mut gpus: Vec<String> = Vec::new();
    for (name, _) in registered.iter().filter(|(_, k)| *k == DeviceKind::Gpu) {
        if !gpus.contains(name) {
            gpus.push(name.clone());
        }
    }
    gpus
}

#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn display_name(name: &str, description: Option<&str>) -> String {
    description
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .unwrap_or(name)
        .to_owned()
}

#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn decode_device_index(kinds: &[DeviceKind]) -> Option<i32> {
    kinds
        .iter()
        .filter(|k| matches!(k, DeviceKind::Gpu | DeviceKind::Igpu))
        .position(|k| *k == DeviceKind::Gpu)
        .and_then(|i| i32::try_from(i).ok())
}

/// Plugin directory the GPU download installs into.
pub fn downloaded_plugin_dir(data_dir: &Path) -> PathBuf {
    data_dir.join(DIR_GPU_BACKEND).join("vulkan")
}

pub fn plugin_installed(data_dir: &Path) -> bool {
    let dir = downloaded_plugin_dir(data_dir);
    dir.join(PLUGIN_FILE).is_file() && installed_plugin_is_intact(&dir)
}

/// Latching only the `true` answer is what makes this sound: within one process
/// an installed plugin cannot become uninstalled by anything but the user
/// deleting the directory, and even then the reported state is already
/// [`GpuPluginState::NeedsRestart`] until the next launch. A `false` answer is
/// never cached, so a plugin installed *after* startup is still noticed.
// Only the native backend has a `state` to answer; the stub build reports
// `Unsupported` without ever asking.
#[cfg(feature = "whisper-cpp")]
static PLUGIN_VERIFIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`plugin_installed`] for a plugin built against the ggml this process
/// linked, paid at most once per process. The stamp is read first: it costs
/// one small file, where the integrity check re-hashes every library.
#[cfg(feature = "whisper-cpp")]
fn plugin_installed_cached(data_dir: &Path) -> bool {
    use std::sync::atomic::Ordering;
    if PLUGIN_VERIFIED.load(Ordering::Relaxed) {
        return true;
    }
    let installed = plugin_built_for(
        &downloaded_plugin_dir(data_dir),
        &imp::linked_ggml_version(),
    ) && plugin_installed(data_dir);
    if installed {
        PLUGIN_VERIFIED.store(true, Ordering::Relaxed);
    }
    installed
}

/// Text file beside the downloaded plugin naming the ggml version it was
/// compiled against, written by `cgs gpu-plugin` and covered by the install
/// manifest. Mirrored by `lune_persist::gpu_plugin::GGML_VERSION_FILE`.
const GGML_VERSION_FILE: &str = "ggml-version";

/// The ggml version the plugin in `dir` says it was built against, if it says.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn plugin_ggml_version(dir: &Path) -> Option<String> {
    let stamp = std::fs::read_to_string(dir.join(GGML_VERSION_FILE)).ok()?;
    Some(stamp.trim().to_owned())
}

/// Whether the plugin in `dir` was built against ggml `version`.
///
/// ggml checks only its backend API version when loading a plugin, and that
/// does not change when the structs shared across the plugin boundary do. A
/// plugin from another ggml therefore loads, registers its device, and crashes
/// the process on first use. A missing stamp counts as a mismatch.
#[cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]
fn plugin_built_for(dir: &Path, version: &str) -> bool {
    plugin_ggml_version(dir).as_deref() == Some(version)
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
    use std::io::Read;

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
        let mut buf = [0u8; 64 * 1024];
        loop {
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => hasher.update(&buf[..n]),
                Err(_) => return false,
            }
        }
        hex::encode(hasher.finalize()).eq_ignore_ascii_case(expected)
    })
}

/// What GPU acceleration can do on this machine right now. Each variant maps to
/// exactly one thing the setup UI can say or offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuPluginState {
    /// This build has no plugin support compiled in (stub ASR backend).
    Unsupported,
    DriverMissing,
    DriverOutdated,
    /// Driver present, plugin not downloaded yet. This is the offer state.
    NotInstalled,
    /// Plugin loaded, carrying the compute devices ggml registered so the UI
    /// can name the GPU instead of claiming success generically.
    Ready {
        devices: Vec<String>,
    },
    /// Plugin installed after this process built its backend registry. ggml has
    /// no unload, so the device can only appear on the next launch. Nothing is
    /// wrong — kept apart from [`Self::Failed`] so the UI can say "restart" in
    /// the user's own language instead of surfacing a failure they don't have.
    NeedsRestart,
    Failed {
        reason: String,
    },
}

pub fn gpu_driver_present() -> bool {
    crate::vulkan_probe::probe() == crate::vulkan_probe::Probe::Ready
}

#[cfg(feature = "whisper-cpp")]
mod imp {
    use super::{
        DeviceKind, GpuPluginState, PLUGIN_FILE, bundled_plugin_dir, downloaded_plugin_dir,
    };
    use crate::vulkan_probe::Probe;

    use std::ffi::{CStr, CString};
    use std::path::Path;
    use std::sync::OnceLock;

    use tracing::{debug, info, warn};

    /// Result of the one load attempt this process makes. ggml's backend
    /// registry is process-global with no unload, so loading twice would
    /// register duplicate devices — hence the `OnceLock`.
    struct Loaded {
        /// Every compute device in ggml's registry after the load. Not just the
        /// ones the load added: a statically linked GPU build registers its
        /// device before we run, and it is just as real.
        devices: Vec<(String, DeviceKind)>,
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
        loaded(data_dir)
            .devices
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn loaded(data_dir: &Path) -> &'static Loaded {
        LOADED.get_or_init(|| {
            let plugin_at_load = super::plugin_installed_cached(data_dir);
            Loaded {
                plugin_at_load,
                devices: load_once(data_dir, plugin_at_load),
            }
        })
    }

    fn load_once(data_dir: &Path, downloaded_is_intact: bool) -> Vec<(String, DeviceKind)> {
        let downloaded = downloaded_plugin_dir(data_dir);
        let backend_dl = whisper_rs::whisper_rs_sys::GGML_BACKEND_DL;
        if !backend_dl {
            debug!("ggml is linked statically; not loading backend plugins");
        }
        for dir in super::plugin_dirs(backend_dl, bundled_plugin_dir(), downloaded.clone()) {
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
                let linked = linked_ggml_version();
                if super::plugin_built_for(&dir, &linked) {
                    warn!(
                        dir = %dir.display(),
                        "downloaded GPU backend does not match what was installed; refusing to load it"
                    );
                } else {
                    warn!(
                        dir = %dir.display(),
                        plugin_ggml = super::plugin_ggml_version(&dir).as_deref().unwrap_or("unknown"),
                        app_ggml = %linked,
                        "downloaded GPU backend was built against another ggml; not loading it, decoding on the CPU"
                    );
                }
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

        let devices = registered();
        info!(?devices, "ggml compute devices available");
        devices
    }

    /// The ggml version this process linked, as `ggml_version()` reports it.
    pub(super) fn linked_ggml_version() -> String {
        // SAFETY: `ggml_version` takes no arguments and returns a static
        // NUL-terminated string compiled into ggml-base.
        unsafe { CStr::from_ptr(whisper_rs::whisper_rs_sys::ggml_version()) }
            .to_string_lossy()
            .into_owned()
    }

    fn registered() -> Vec<(String, DeviceKind)> {
        use whisper_rs::whisper_rs_sys as sys;
        // SAFETY: the registry accessors take no arguments and return either a
        // count or a valid device handle for an index below that count. The
        // returned name is a static C string owned by the backend.
        unsafe {
            let count = sys::ggml_backend_dev_count();
            (0..count)
                .filter_map(|i| {
                    let dev = sys::ggml_backend_dev_get(i);
                    if dev.is_null() {
                        return None;
                    }
                    let name = sys::ggml_backend_dev_name(dev);
                    if name.is_null() {
                        return None;
                    }
                    let kind = match sys::ggml_backend_dev_type(dev) {
                        sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU => DeviceKind::Cpu,
                        sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => DeviceKind::Gpu,
                        sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU => {
                            DeviceKind::Igpu
                        }
                        _ => DeviceKind::Other,
                    };
                    let description = sys::ggml_backend_dev_description(dev);
                    let description = (!description.is_null())
                        .then(|| CStr::from_ptr(description).to_string_lossy());
                    let name = CStr::from_ptr(name).to_string_lossy();
                    Some((super::display_name(&name, description.as_deref()), kind))
                })
                .collect()
        }
    }

    pub fn gpu_decode_device() -> Option<i32> {
        let kinds: Vec<DeviceKind> = registered().into_iter().map(|(_, k)| k).collect();
        super::decode_device_index(&kinds)
    }

    /// Whether ggml's registry holds a CPU device.
    ///
    /// Always true for a static ggml, which links its CPU backend in. Under
    /// `GGML_BACKEND_DL` the CPU backend is itself a plugin, so a bundle whose
    /// `backends/` directory went missing has none — and both whisper.cpp and
    /// Parakeet answer that by throwing a C++ exception out of their state
    /// setup, which aborts the process instead of reporting anything.
    pub fn cpu_device_registered() -> bool {
        use whisper_rs::whisper_rs_sys as sys;
        // SAFETY: as in `device_names`: a count, then handles for indices
        // below it.
        unsafe {
            (0..sys::ggml_backend_dev_count()).any(|i| {
                let dev = sys::ggml_backend_dev_get(i);
                !dev.is_null()
                    && sys::ggml_backend_dev_type(dev)
                        == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU
            })
        }
    }

    /// Current GPU state for the setup UI. Loading is a side effect when a
    /// plugin is installed, which is what makes "installed" and "actually
    /// working" distinguishable rather than assumed.
    pub fn state(data_dir: &Path) -> GpuPluginState {
        let loaded = loaded(data_dir);

        let gpus = super::discrete_gpus(&loaded.devices);
        if !gpus.is_empty() {
            return GpuPluginState::Ready { devices: gpus };
        }
        // A static ggml never loads the plugin, so there is nothing to offer.
        if !whisper_rs::whisper_rs_sys::GGML_BACKEND_DL {
            return GpuPluginState::Unsupported;
        }

        if !super::plugin_installed_cached(data_dir) {
            return match crate::vulkan_probe::probe() {
                Probe::Ready => GpuPluginState::NotInstalled,
                Probe::NoDiscreteGpu => GpuPluginState::Unsupported,
                Probe::NoDriver => GpuPluginState::DriverMissing,
                Probe::OutdatedDriver => GpuPluginState::DriverOutdated,
            };
        }

        if !loaded.plugin_at_load {
            return GpuPluginState::NeedsRestart;
        }
        if loaded.devices.iter().any(|(_, k)| *k == DeviceKind::Igpu) {
            return GpuPluginState::Failed {
                reason: format!(
                    "{PLUGIN_FILE} found only an integrated GPU, so transcription stays on the \
                     CPU, which is usually faster."
                ),
            };
        }
        GpuPluginState::Failed {
            reason: format!(
                "{PLUGIN_FILE} is installed but registered no Vulkan device. Your graphics \
                 driver most likely lacks Vulkan 1.2 support."
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

#[cfg(feature = "whisper-cpp")]
pub(crate) use imp::{cpu_device_registered, gpu_decode_device};

pub fn gpu_available(data_dir: &Path) -> bool {
    matches!(state(data_dir), GpuPluginState::Ready { .. })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(name: &str, kind: DeviceKind) -> (String, DeviceKind) {
        (name.to_owned(), kind)
    }

    #[test]
    fn only_discrete_gpus_are_listed_once_each() {
        let registered = [
            dev("CPU", DeviceKind::Cpu),
            dev("Vulkan0", DeviceKind::Igpu),
            dev("Vulkan1", DeviceKind::Gpu),
            dev("Vulkan1", DeviceKind::Gpu),
            dev("CUDA0", DeviceKind::Gpu),
        ];
        assert_eq!(discrete_gpus(&registered), ["Vulkan1", "CUDA0"]);
    }

    #[test]
    fn a_device_is_shown_by_its_product_name_when_the_backend_gives_one() {
        assert_eq!(
            display_name("Vulkan0", Some("NVIDIA GeForce RTX 4090")),
            "NVIDIA GeForce RTX 4090"
        );
        assert_eq!(display_name("CUDA0", Some("  ")), "CUDA0");
        assert_eq!(display_name("CPU", None), "CPU");
    }

    #[test]
    fn decoding_uses_the_first_discrete_gpu_counted_the_way_whisper_counts() {
        use DeviceKind::{Cpu, Gpu, Igpu, Other};
        assert_eq!(decode_device_index(&[Cpu, Gpu]), Some(0));
        assert_eq!(decode_device_index(&[Cpu, Igpu, Gpu]), Some(1));
        assert_eq!(decode_device_index(&[Igpu, Other, Igpu, Gpu, Gpu]), Some(2));
        assert_eq!(decode_device_index(&[Cpu, Igpu]), None);
        assert_eq!(decode_device_index(&[]), None);
    }

    #[test]
    fn download_dir_is_its_own_folder_under_the_data_dir() {
        assert_eq!(
            downloaded_plugin_dir(Path::new("/data")),
            Path::new("/data").join(DIR_GPU_BACKEND).join("vulkan")
        );
    }

    #[test]
    fn the_plugin_is_ggmls_vulkan_backend() {
        #[cfg(target_os = "windows")]
        assert_eq!(PLUGIN_FILE, "ggml-vulkan.dll");
        #[cfg(not(target_os = "windows"))]
        assert_eq!(PLUGIN_FILE, "libggml-vulkan.so");
    }

    #[test]
    fn an_installed_linux_bundle_keeps_its_plugins_in_the_resource_dir() {
        // deb, rpm, AppImage and the tarball put the binary in `<prefix>/bin`
        // and Tauri's resources in `<prefix>/lib/<productName>`.
        let prefix = std::env::temp_dir().join("lune-bundle-layout/usr");
        let backends = prefix
            .join("lib")
            .join(LINUX_RESOURCE_DIR)
            .join(DIR_GPU_BACKEND);
        std::fs::create_dir_all(&backends).expect("create bundle layout");
        assert_eq!(
            bundled_plugin_dir_for(&prefix.join("bin/lune-voice")),
            Some(backends)
        );
    }

    #[test]
    fn without_a_resource_dir_the_plugins_sit_next_to_the_executable() {
        // A build tree, `cgs stage-runtime`, and the Windows install directory.
        let bin = std::env::temp_dir().join("lune-build-tree-layout/release");
        assert_eq!(
            bundled_plugin_dir_for(&bin.join("lune-voice")),
            Some(bin.join(DIR_GPU_BACKEND))
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
    fn gpu_is_available_only_when_a_device_actually_registered() {
        let dir = std::env::temp_dir().join("lune-gpu-availability-probe");
        let expected = matches!(state(&dir), GpuPluginState::Ready { .. });
        assert_eq!(gpu_available(&dir), expected);
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
        std::fs::write(dir.join(PLUGIN_FILE), b"not what was installed").expect("write library");
        // A digest that cannot match anything.
        std::fs::write(
            dir.join(PLUGIN_MANIFEST),
            format!(r#"{{"{PLUGIN_FILE}":"{}"}}"#, "0".repeat(64)),
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
        std::fs::write(dir.join(PLUGIN_FILE), body).expect("write library");
        let digest = hex::encode(Sha256::digest(body));
        std::fs::write(
            dir.join(PLUGIN_MANIFEST),
            format!(r#"{{"{PLUGIN_FILE}":"{digest}"}}"#),
        )
        .expect("write manifest");
        assert!(installed_plugin_is_intact(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_statically_linked_ggml_scans_no_plugin_directory() {
        // Any plugin would bring the shared libggml-base in beside the static
        // copy, and ggml aborts when it is unloaded and loaded again.
        let dirs = plugin_dirs(
            false,
            Some(PathBuf::from("/app/backends")),
            PathBuf::from("/data/backends"),
        );
        assert!(dirs.is_empty(), "{dirs:?}");
    }

    #[test]
    fn a_plugin_build_scans_the_bundled_then_the_downloaded_directory() {
        assert_eq!(
            plugin_dirs(
                true,
                Some(PathBuf::from("/app/backends")),
                PathBuf::from("/data/backends")
            ),
            [
                PathBuf::from("/app/backends"),
                PathBuf::from("/data/backends")
            ]
        );
        assert_eq!(
            plugin_dirs(true, None, PathBuf::from("/data/backends")),
            [PathBuf::from("/data/backends")]
        );
    }

    #[test]
    fn a_plugin_without_a_ggml_stamp_is_not_built_for_this_ggml() {
        // Plugins packaged before the stamp existed are exactly the ones built
        // against an older ggml.
        let tmp = tempfile::tempdir().expect("tempdir");
        assert_eq!(plugin_ggml_version(tmp.path()), None);
        assert!(!plugin_built_for(tmp.path(), "0.23.0"));
    }

    #[test]
    fn a_plugin_stamped_with_another_ggml_is_refused() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(GGML_VERSION_FILE), "0.9.4\n").expect("write stamp");
        assert_eq!(plugin_ggml_version(tmp.path()).as_deref(), Some("0.9.4"));
        assert!(!plugin_built_for(tmp.path(), "0.23.0"));
    }

    #[test]
    fn a_plugin_stamped_with_this_ggml_is_accepted() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(GGML_VERSION_FILE), "0.23.0\n").expect("write stamp");
        assert!(plugin_built_for(tmp.path(), "0.23.0"));
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
        std::fs::write(dir.join(PLUGIN_FILE), b"").expect("write plugin");
        assert!(
            !plugin_installed(tmp.path()),
            "a truncated download is not an installation"
        );

        // With the installer's manifest beside it, the same file is installed.
        let body = b"pretend shared object";
        std::fs::write(dir.join(PLUGIN_FILE), body).expect("write plugin");
        let digest = hex::encode(Sha256::digest(body));
        std::fs::write(
            dir.join(PLUGIN_MANIFEST),
            format!(r#"{{"{PLUGIN_FILE}":"{digest}"}}"#),
        )
        .expect("write manifest");
        assert!(plugin_installed(tmp.path()));
    }
}
