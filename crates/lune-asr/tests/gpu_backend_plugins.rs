//! End-to-end check that ggml's dlopen-able backends actually register.
//!
//! Only meaningful for a `GGML_BACKEND_DL=ON` build whose plugins have been
//! staged next to the test binary, so the test skips itself otherwise rather
//! than failing on a plain static build. Run it after `cgs stage-runtime`.

#![cfg(feature = "whisper-cpp")]

use lune_asr::gpu_backend::{bundled_plugin_dir, load_plugins};

#[test]
fn bundled_plugins_register_a_compute_device() {
    let Some(dir) = bundled_plugin_dir() else {
        eprintln!("skip: cannot resolve the executable directory");
        return;
    };
    let staged = dir.is_dir()
        && std::fs::read_dir(&dir).is_ok_and(|mut e| {
            e.any(|f| f.is_ok_and(|f| f.file_name().to_string_lossy().starts_with("libggml-")))
        });
    if !staged {
        eprintln!(
            "skip: no ggml plugins staged at {} — static build, or run `cgs stage-runtime`",
            dir.display()
        );
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let devices = load_plugins(tmp.path());

    assert!(
        !devices.is_empty(),
        "plugins are staged at {} but ggml registered no compute device; \
         the backend registry would fall back to nothing and decoding would fail",
        dir.display()
    );
    eprintln!("registered devices: {devices:?}");
}
