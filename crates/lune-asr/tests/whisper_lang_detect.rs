// SPDX-License-Identifier: GPL-3.0-or-later

//! Language auto-detection must not cost a second encoder pass.
//!
//! With `language = "auto"`, whisper.cpp detects the language by encoding the
//! first window, then transcribes. The vendored fork sizes that detection
//! encode to the requested `audio_ctx` and reuses it for the first window, so a
//! clip that fits one window is encoded once. The count comes from whisper's
//! own timing report.
//!
//! Ignored by default (needs a ggml model on disk + the C++ build). Run with:
//!   LIBCLANG_PATH=/lib/x86_64-linux-gnu \
//!   LUNE_TEST_WHISPER_GGML=$HOME/.local/share/lune-voice/models/whisper-base-q8_0/ggml-base-q8_0.bin \
//!   cargo test -p lune-asr --features whisper-cpp --test whisper_lang_detect -- --ignored

#![cfg(feature = "whisper-cpp")]

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Mutex;

use whisper_rs::whisper_rs_sys as sys;

static LOG: Mutex<String> = Mutex::new(String::new());

unsafe extern "C" fn capture_log(_level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    if text.is_null() {
        return;
    }
    // SAFETY: whisper.cpp passes a NUL-terminated string valid for the call.
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    LOG.lock().unwrap().push_str(&text);
}

/// Encoder runs from `whisper_print_timings`:
/// `encode time =   123.45 ms /     1 runs (...)`.
fn encoder_runs(log: &str) -> Option<u32> {
    let line = log.lines().find(|l| l.contains("encode time"))?;
    line.split('/')
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[test]
#[ignore = "needs a ggml whisper model (LUNE_TEST_WHISPER_GGML)"]
fn auto_language_encodes_a_one_window_clip_once() {
    let Some(model) = std::env::var("LUNE_TEST_WHISPER_GGML")
        .ok()
        .filter(|p| std::path::Path::new(p).exists())
    else {
        eprintln!("LUNE_TEST_WHISPER_GGML not set; skipping");
        return;
    };
    let model = CString::new(model).unwrap();
    let auto = CString::new("auto").unwrap();
    let audio = vec![0.0f32; 16_000 * 3];

    // SAFETY: plain whisper.cpp C API use; every pointer outlives the calls.
    let runs = unsafe {
        sys::whisper_log_set(Some(capture_log), std::ptr::null_mut());

        let mut cparams = sys::whisper_context_default_params();
        cparams.use_gpu = false;
        let ctx = sys::whisper_init_from_file_with_params(model.as_ptr(), cparams);
        assert!(!ctx.is_null(), "model failed to load");

        let mut params = sys::whisper_full_default_params(
            sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY,
        );
        params.language = auto.as_ptr();
        // The fitted window the CPU decode profile uses for a short clip.
        params.audio_ctx = 768;
        params.n_threads = 4;
        let rc = sys::whisper_full(ctx, params, audio.as_ptr(), audio.len() as i32);
        assert_eq!(rc, 0, "whisper_full failed");

        LOG.lock().unwrap().clear();
        sys::whisper_print_timings(ctx);
        let runs = encoder_runs(&LOG.lock().unwrap());
        sys::whisper_free(ctx);
        runs
    };

    assert_eq!(
        runs,
        Some(1),
        "detection and transcription should share one encode"
    );
}
