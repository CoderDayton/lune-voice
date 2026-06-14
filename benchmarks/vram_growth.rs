//! VRAM growth benchmark — measures per-dictation GPU memory delta.
//!
//! Run: cargo run --release --features cuda --example vram_growth -p lune-app
//! Iterations: LUNE_VRAM_ITERS=10 (default 10)
//! Fixture:    LUNE_BENCH_FIXTURE=10s (default — picks the shortest match)
//!
//! What it does:
//!   1. Snapshots VRAM at process start (before any model load).
//!   2. Builds the BenchmarkHarness with ASR + LLM (matches production wiring).
//!   3. Snapshots VRAM after model load.
//!   4. Runs N transcriptions on the same fixture through SessionManager
//!      (the real pipeline path: VAD → ASR → LLM correction).
//!   5. Snapshots VRAM after each iteration; prints a growth table.
//!
//! Success criteria: per-iteration delta should plateau within ±50 MiB after
//! the first 1–2 dictations. If it keeps growing ~500–1000 MiB per dictation,
//! the arena fragmentation issue is still present and Layer 3 (static KV) is
//! needed.
//!
//! Requires: nvidia-smi on $PATH and a CUDA-capable GPU.

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::process::Command;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use bench_utils::{BenchmarkHarness, Fixture};
use lune_app::SessionManager;
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::config::UserSettings;
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use tokio::sync::{Mutex, RwLock, mpsc};

const SETTLE_MS: u64 = 250;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    VRAM GROWTH BENCHMARK                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let pid = std::process::id();
    println!("  PID: {pid}");

    if !nvidia_smi_available() {
        println!("  [skip] nvidia-smi not on PATH; cannot measure VRAM");
        return;
    }

    let iters: usize = std::env::var("LUNE_VRAM_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("  Iterations: {iters}");
    println!();

    // ── Baseline: before any model load ──────────────────────────────────
    sleep(Duration::from_millis(SETTLE_MS));
    let pre_load = vram_for_pid(pid);
    println!("  Baseline (pre-load):           {} MiB", fmt_mib(pre_load));

    // ── Harness build (loads ASR + LLM) ──────────────────────────────────
    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .with_llm()
        .build();

    if !harness.has_gpu {
        println!("  [skip] CUDA not available; nothing to measure");
        return;
    }

    sleep(Duration::from_millis(SETTLE_MS));
    let post_load = vram_for_pid(pid);
    println!(
        "  After model load:              {} MiB  (Δ {} MiB)",
        fmt_mib(post_load),
        fmt_signed_mib(post_load.saturating_sub(pre_load) as i64),
    );
    println!();

    // ── Pick fixture ─────────────────────────────────────────────────────
    let fixture_idx = match pick_fixture_idx(&harness) {
        Some(i) => i,
        None => {
            println!("  [skip] no fixture available");
            return;
        }
    };
    let fixture = &harness.fixtures[fixture_idx];
    println!(
        "  Fixture: {} ({:.1}s audio)",
        fixture.name, fixture.audio_secs
    );
    println!();

    // ── Header ───────────────────────────────────────────────────────────
    println!("  ┌──────┬────────────┬────────────┬───────────┬─────────────┐");
    println!("  │ Iter │  Pre (MiB) │ Post (MiB) │  Δ (MiB)  │ Wall (s)    │");
    println!("  ├──────┼────────────┼────────────┼───────────┼─────────────┤");

    // ── Run iterations ───────────────────────────────────────────────────
    // Track post-warmup growth: iter 1 typically pre-sizes ORT's
    // kNextPowerOfTwo arena (one-time jump); iters 2..N must plateau.
    let mut last_vram = post_load;
    let mut iter1_post: Option<u64> = None;
    for i in 1..=iters {
        sleep(Duration::from_millis(SETTLE_MS));
        let pre = vram_for_pid(pid);

        let start = Instant::now();
        let result = harness.rt.block_on(run_full_pipeline(
            &harness.fixtures[fixture_idx],
            Arc::clone(&harness.asr),
            Arc::clone(&harness.llm),
            harness.asr.pipeline_config(),
        ));
        let wall = start.elapsed().as_secs_f64();

        // Drop transient runtime state and let CUDA settle before sampling.
        drop(result);
        sleep(Duration::from_millis(SETTLE_MS));
        let post = vram_for_pid(pid);

        let delta = post as i64 - pre as i64;
        println!(
            "  │  {:>3} │  {:>9} │  {:>9} │  {:>+8} │ {:>10.3}  │",
            i,
            fmt_mib(pre),
            fmt_mib(post),
            delta,
            wall,
        );
        last_vram = post;
        if i == 1 {
            iter1_post = Some(post);
        }
    }
    println!("  └──────┴────────────┴────────────┴───────────┴─────────────┘");
    println!();

    let warmup_delta = iter1_post.map(|v| v as i64 - post_load as i64).unwrap_or(0);
    let postwarmup_delta = iter1_post.map(|v| last_vram as i64 - v as i64).unwrap_or(0);
    let postwarmup_iters = iters.saturating_sub(1);

    println!(
        "  Iter 1 warmup (one-time arena pre-size): {:+} MiB",
        warmup_delta
    );
    println!(
        "  Post-warmup growth (iters 2..{iters}):     {:+} MiB total",
        postwarmup_delta
    );

    let avg = if postwarmup_iters > 0 {
        postwarmup_delta as f64 / postwarmup_iters as f64
    } else {
        0.0
    };
    println!("  Average per post-warmup dictation:       {avg:+.1} MiB");
    println!();

    if postwarmup_iters == 0 {
        println!(
            "  ⓘ Need at least 2 iterations to assess plateau. Re-run with LUNE_VRAM_ITERS=10."
        );
    } else if avg.abs() < 50.0 {
        println!("  ✓ Plateau achieved (|avg| < 50 MiB/iter). Arena reuse is healthy.");
    } else if avg > 0.0 {
        println!("  ⚠ Arena keeps growing across dictations.");
        println!("    Consider Layer 3 (static KV cache via ONNX re-export).");
    } else {
        println!("  ⚠ VRAM is shrinking — possible measurement noise or transient allocations.");
    }
    println!();
}

// ── nvidia-smi helpers ──────────────────────────────────────────────────────

fn nvidia_smi_available() -> bool {
    Command::new("nvidia-smi")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Returns the GPU memory used by the given PID, in MiB. Returns 0 if the PID
/// has no GPU allocations yet (model not loaded).
fn vram_for_pid(pid: u32) -> u64 {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let stdout = match out {
        Ok(o) if o.status.success() => o.stdout,
        _ => return 0,
    };
    let s = String::from_utf8_lossy(&stdout);
    for line in s.lines() {
        let mut parts = line.split(',').map(|p| p.trim());
        let p = parts.next().and_then(|v| v.parse::<u32>().ok());
        let mib = parts.next().and_then(|v| v.parse::<u64>().ok());
        if let (Some(p), Some(mib)) = (p, mib)
            && p == pid
        {
            return mib;
        }
    }
    0
}

fn fmt_mib(mib: u64) -> String {
    format!("{mib} MiB")
}

fn fmt_signed_mib(d: i64) -> String {
    if d >= 0 {
        format!("+{d}")
    } else {
        format!("{d}")
    }
}

// ── Fixture selection ───────────────────────────────────────────────────────

fn pick_fixture_idx(harness: &BenchmarkHarness) -> Option<usize> {
    if harness.fixtures.is_empty() {
        return None;
    }
    let needle = std::env::var("LUNE_BENCH_FIXTURE").unwrap_or_else(|_| "10s".into());
    harness
        .fixtures
        .iter()
        .position(|f| f.name.contains(&needle))
        .or(Some(0))
}

// ── Pipeline driver (mirror of pipeline_real.rs::run_full_pipeline) ────────

async fn run_full_pipeline(
    fixture: &Fixture,
    asr: Arc<AsrRegistry>,
    llm: Arc<dyn TextPostProcessor>,
    pipeline: lune_core::config::PipelineConfig,
) -> Option<(String, f64)> {
    let audio_stream = AudioStream::from_samples(&fixture.pcm, AudioConfig::default()).unwrap();

    let db = lune_persist::Database::open_in_memory().unwrap();
    let dict = lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
        .unwrap();

    let config = lune_app::SessionManagerConfig {
        asr_backend_kind: asr.kind(),
        pipeline,
        ..Default::default()
    };

    let user_settings = UserSettings {
        language: BenchmarkHarness::language_for_fixture(fixture.name).into(),
        ..Default::default()
    };
    let settings = Arc::new(RwLock::new(user_settings));

    let mgr = SessionManager::new(
        asr,
        Arc::new(lune_commands::VoiceCommandRouter::new()),
        llm,
        Arc::new(Mutex::new(dict)),
        Arc::new(Mutex::new(db)),
        Arc::new(lune_dict::corrector::LexicalGate::new()),
        config,
        settings,
    );
    let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(256);

    let start = Instant::now();

    tokio::task::LocalSet::new()
        .run_until(async {
            let _id = mgr
                .start_with_audio(event_tx, audio_stream)
                .await
                .expect("start session");

            let mut final_text = String::new();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(120);

            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(SessionEvent::FinalizedText(t))) => final_text.push_str(&t),
                    Ok(Some(SessionEvent::Stopped)) => break,
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("  [warn] 120s safety timeout reached");
                        break;
                    }
                }
            }
            let wall = start.elapsed().as_secs_f64();
            if final_text.is_empty() {
                None
            } else {
                Some((final_text, wall))
            }
        })
        .await
}
