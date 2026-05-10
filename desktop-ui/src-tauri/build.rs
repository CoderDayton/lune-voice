use std::path::PathBuf;
use std::{env, fs};

fn main() {
    #[cfg(feature = "cuda")]
    stage_onnx_provider_libs();
    tauri_build::build();
}

// ORT statically links the onnxruntime core into the binary, but the CUDA /
// TensorRT execution providers are separate `.so` files that ORT dlopens at
// runtime by short name. They never appear in the binary's NEEDED list, so
// linuxdeploy doesn't bundle them — when the AppImage runs on the user's
// machine, `dlopen("libonnxruntime_providers_cuda.so")` fails and ORT silently
// falls back to CPU. Stage the libs at a stable, repo-relative path so
// tauri.conf.json's bundle.linux.{appimage,deb}.files entries can copy them
// into the bundle's /usr/lib/ — RUNPATH=$ORIGIN/../lib then resolves them via
// DT_RUNPATH.
#[cfg(feature = "cuda")]
fn stage_onnx_provider_libs() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let staging_dir = manifest_dir.join("onnx-libs");
    fs::create_dir_all(&staging_dir).expect("create onnx-libs dir");

    let target_dir = locate_target_dir();
    let libs = [
        "libonnxruntime_providers_cuda.so",
        "libonnxruntime_providers_shared.so",
    ];

    for lib in libs {
        let src = target_dir.join(lib);
        if !src.exists() {
            // ort's copy-dylibs feature places the libs here during the build,
            // but ordering is not guaranteed across all cargo configurations.
            // A subsequent build will pick them up.
            println!(
                "cargo:warning=ORT provider lib not yet staged: {}",
                src.display()
            );
            continue;
        }
        let dst = staging_dir.join(lib);
        fs::copy(&src, &dst).unwrap_or_else(|e| panic!("copy {}: {e}", src.display()));
        println!("cargo:rerun-if-changed={}", src.display());
    }
}

// Walk up from OUT_DIR (target/<profile>/build/<crate>-<hash>/out) to
// target/<profile>/, where ort's copy-dylibs feature drops the provider libs.
#[cfg(feature = "cuda")]
fn locate_target_dir() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR has fewer than 3 ancestors")
        .to_path_buf()
}
