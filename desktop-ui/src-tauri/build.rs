// SPDX-License-Identifier: GPL-3.0-or-later

fn main() {
    // ggml's compute backends are built as dlopen-able plugins
    // (`GGML_BACKEND_DL`), which also makes libwhisper/libggml shared objects
    // rather than static archives. They ship next to the executable, so the
    // binary needs `$ORIGIN` on its run path or it cannot start outside the
    // build tree. `$ORIGIN` is expanded by the loader, so the literal dollar
    // sign has to survive both make and the shell: `-Wl,-z,origin` marks the
    // object as needing that expansion. Installed bundles (deb, rpm, AppImage,
    // tarball) keep them in Tauri's resource directory instead,
    // `<prefix>/lib/<productName>` beside `<prefix>/bin`; that name is also
    // `LINUX_RESOURCE_DIR` in `crates/lune-asr/src/gpu_backend.rs`.
    //
    // Keyed off `CARGO_CFG_TARGET_OS`, not `cfg!(target_os = ..)`: inside a
    // build script the `cfg` macro describes the *host*, so cross-building for
    // Windows from Linux would otherwise feed ELF rpath flags to a PE link.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-z,origin");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/LuneVoice");
    }

    // `tauri_build` puts its app manifest in a resource linked into binaries
    // only, so test executables start without Common Controls v6 and exit
    // with STATUS_ENTRYPOINT_NOT_FOUND. On MSVC the linker embeds the manifest
    // into every binary instead, tests included. The GNU link has no such
    // flag and keeps tauri_build's resource manifest.
    let windows = if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let manifest =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("windows-app-manifest.xml");
        println!("cargo:rerun-if-changed={}", manifest.display());
        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg=/MANIFESTINPUT:{}", manifest.display());
        tauri_build::WindowsAttributes::new_without_app_manifest()
    } else {
        tauri_build::WindowsAttributes::new()
    };
    tauri_build::try_build(tauri_build::Attributes::new().windows_attributes(windows))
        .expect("failed to run tauri_build");
}
