// SPDX-License-Identifier: GPL-3.0-or-later

//! Device-memory probe for the ASR backend lifecycle.
//!
//! Answers two questions that only a running GPU can answer, and that a leak
//! reads very differently from a spike on:
//!
//! 1. **Does toggling dictation grow VRAM?** Every dictation builds a fresh
//!    `WhisperState` (KV caches, compute buffers) and drops it on the way out,
//!    while the model itself is loaded once and shared. If any of that teardown
//!    is incomplete, the floor rises one toggle at a time and only shows up
//!    after a long session.
//!
//! 2. **What does swapping the model peak at?** `LivePipeline::take_reload_plan`
//!    releases the outgoing model *before* the replacement is loaded. Holding
//!    it across the load instead puts two whole models in VRAM at once, which
//!    OOMs a card sized for one and reports as "the new model failed to load".
//!    Both orders are measured here so the difference is a number.
//!
//! Run: `cgs bench --env TYPE=vram` (add `--env WHISPER_MODEL=small` to pin the
//! toggle-loop model, `--env CYCLES=n` for a longer loop). Needs an NVIDIA GPU
//! and a CUDA-enabled build; on CPU it says so and exits.

// `bench_utils` is shared by every benchmark, so most of it is unused here.
#[path = "../bench_utils.rs"]
#[allow(dead_code)]
mod bench_utils;

use std::path::Path;

use bench_utils::{load_fixtures, models_dir, vram_mb, vram_str, workspace_root};
use lune_asr::AsrRegistry;
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use lune_core::config::{AsrBackendConfig, DevicePolicy, WhisperModel};

/// Toggle cycles in the leak loop. Enough that a per-toggle leak of even a few
/// MB clears the sampling noise, short enough to finish in a couple of minutes.
const DEFAULT_CYCLES: usize = 20;

fn load(model: WhisperModel, cuda: bool) -> AsrRegistry {
    AsrRegistry::from_config(
        &AsrBackendConfig::WhisperCpp { model },
        DevicePolicy::Auto,
        &models_dir(),
        cuda,
    )
    .unwrap_or_else(|e| panic!("load {}: {e}", model.display_name()))
}

fn on_disk(model: WhisperModel) -> bool {
    models_dir()
        .join(model.dir())
        .join(model.ggml_file())
        .is_file()
}

/// One dictation's worth of backend work: the exact call sequence
/// `run_capture` makes per hotkey toggle.
async fn one_toggle(asr: &AsrRegistry, audio: &[f32]) {
    let session = asr
        .start_session(AsrSessionConfig {
            language: Some("en".into()),
            ..Default::default()
        })
        .await
        .expect("start_session");
    asr.push_audio(&session, audio).await.expect("push_audio");
    let _ = asr.poll_final(&session).await.expect("poll_final");
    asr.end_session(session).await.expect("end_session");
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let root = workspace_root();
    let cuda = lune_asr::gpu_backend::gpu_available(
        lune_persist::AppPaths::resolve()
            .expect("resolve XDG paths")
            .data_dir(),
    );

    println!("\n=== ASR device-memory probe ===\n");
    if !cuda {
        println!(
            "No CUDA device registered — there is no VRAM to measure.\n\
             Build with the whisper-cuda feature on a machine with an NVIDIA GPU."
        );
        return;
    }
    if vram_mb().is_some() {
        // Before any CUDA allocation this process has no compute context, so it
        // is absent from the listing rather than listed at zero.
        println!("  [warn] this process already holds device memory before loading anything");
    }

    let audio = pick_audio(&root);
    let cycles: usize = std::env::var("CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CYCLES);

    toggle_loop(cuda, &audio, cycles).await;
    swap_peaks(cuda);
}

/// The longest fixture available, so each decode is substantial enough that an
/// incomplete teardown has something to leave behind.
fn pick_audio(root: &Path) -> Vec<f32> {
    let mut fixtures = load_fixtures(root);
    fixtures.sort_by_key(|f| f.pcm.len());
    let fixture = fixtures
        .pop()
        .expect("no audio fixtures under fixtures/audio");
    println!("  fixture: {} ({:.1}s)\n", fixture.name, fixture.audio_secs);
    fixture.pcm
}

/// Question 1: does repeated dictation raise the floor?
///
/// The model is loaded once, exactly as the app does it, and only the
/// per-session state churns. A flat trace across cycles means the churn is
/// balanced; a rising one is a leak, and the slope is its size per toggle.
async fn toggle_loop(cuda: bool, audio: &[f32], cycles: usize) {
    println!("── Toggle loop ─────────────────────────────────────────────");
    let model = std::env::var("WHISPER_MODEL")
        .ok()
        .and_then(|v| match v.as_str() {
            "base" => Some(WhisperModel::Base),
            "small" => Some(WhisperModel::Small),
            "medium" => Some(WhisperModel::Medium),
            "large" => Some(WhisperModel::LargeV3Turbo),
            _ => None,
        })
        .unwrap_or(WhisperModel::Medium);
    println!(
        "model {} loaded once, {cycles} toggles\n",
        model.display_name()
    );

    let asr = load(model, cuda);
    let after_load = vram_mb();
    println!("  after model load      {}", vram_str());

    let mut first = None;
    let mut last = 0.0;
    let mut peak: f64 = 0.0;
    for i in 1..=cycles {
        one_toggle(&asr, audio).await;
        let now = vram_mb().unwrap_or(0.0);
        peak = peak.max(now);
        last = now;
        if first.is_none() {
            first = Some(now);
        }
        // Only the ends and a sparse middle: 20 identical lines hide the trend.
        if i == 1 || i == cycles || i % 5 == 0 {
            println!("  after toggle {i:>3}       {now:>7.0} MB");
        }
    }

    let first = first.unwrap_or(0.0);
    println!("\n  peak                  {peak:>7.0} MB");
    println!(
        "  drift over {cycles} toggles {:>+7.0} MB   ({:+.1} MB per toggle)",
        last - first,
        (last - first) / cycles as f64
    );
    if let Some(loaded) = after_load {
        println!("  per-session overhead  {:>7.0} MB", peak - loaded);
    }
    println!();
    drop(asr);
}

/// Question 2: what does a model swap peak at, held versus released?
///
/// `hold` is the shape the reload path had before `take_reload_plan`: the old
/// backend stays reachable for the whole of the new model's allocation. Its
/// peak is the sum; the released order's peak is the larger of the two.
fn swap_peaks(cuda: bool) {
    println!("── Model swap ──────────────────────────────────────────────");
    let (a, b) = (WhisperModel::Small, WhisperModel::Medium);
    for m in [a, b] {
        if !on_disk(m) {
            println!(
                "skipped: needs both {} and {} on disk\n",
                a.display_name(),
                b.display_name()
            );
            return;
        }
    }
    println!(
        "{} -> {}, each order measured from an empty start\n",
        a.display_name(),
        b.display_name()
    );

    // Hold: the outgoing model is still reachable while the incoming one is
    // allocated, so both are resident at the peak.
    let old = load(a, cuda);
    let before = vram_mb().unwrap_or(0.0);
    let new = load(b, cuda);
    let held_peak = vram_mb().unwrap_or(0.0);
    drop(new);
    drop(old);
    println!("  hold-then-load    {before:>7.0} MB -> {held_peak:>7.0} MB peak");

    // Release: the outgoing model is dropped first, so the peak is whichever
    // single model is larger — never the sum.
    let old = load(a, cuda);
    let before = vram_mb().unwrap_or(0.0);
    drop(old);
    let new = load(b, cuda);
    let released_peak = vram_mb().unwrap_or(0.0);
    drop(new);
    println!("  release-then-load {before:>7.0} MB -> {released_peak:>7.0} MB peak");
    println!(
        "\n  peak avoided      {:>7.0} MB\n",
        held_peak - released_peak
    );
}
