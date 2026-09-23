// SPDX-License-Identifier: GPL-3.0-or-later

//! Per-stage pipeline latency benchmark.
//!
//! Decomposes the dictation pipeline into measured stages
//!
//!   session create → push audio → ASR poll_final
//!
//! and reports which stage dominates. whisper.cpp does not expose the internal
//! mel/encoder/decode sub-stage timings the old ONNX Qwen3 backend recorded, so
//! the ASR cost is reported as a single `poll_final` stage. LLM styling is not
//! measured here — production skips it by default (no style selected), and
//! its cost belongs in a dedicated styling benchmark, not the ASR latency one.
//!
//! Each fixture runs `WARMUP` discarded passes then `ITERS` timed passes, so
//! the reported P50/P95 reflect steady state rather than first-call cost.
//!
//! Env:
//!   ITERS=N         timed iterations per fixture           (default 3)
//!   WARMUP=N        discarded warmup iterations per fixture (default 1)
//!   PROFILE=1       profiling mode: tight full-pipeline loop over every
//!                   fixture for PROFILE_SECS, minimal output — feed a
//!                   sampling profiler this build.
//!   PROFILE_SECS=N  duration of the profiling loop          (default 25)
//!
//! Run:   cargo run --release --example pipeline_stages -p lune-voice --features whisper-cpp
//! Flame: PROFILE=1 samply record -- target/release/examples/pipeline_stages

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, Fixture, char_error_rate, percentile, process_memory_usage, word_error_rate,
};
use lune_asr::AsrRegistry;
use lune_core::asr::{AsrBackend, AsrSessionConfig};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

/// One timed pass over a single fixture: every measurable stage.
#[derive(Clone, Default)]
struct Sample {
    audio_secs: f64,
    sess_ms: f64,
    push_ms: f64,
    poll_ms: f64,
    total_ms: f64,
    wer: f64,
    cer: f64,
}

/// Run the full pipeline once for `fixture`. When `score` is false the text
/// metrics are skipped (profiling loop doesn't need them).
fn run_once(
    harness: &BenchmarkHarness,
    asr: &AsrRegistry,
    fixture: &Fixture,
    score: bool,
) -> Sample {
    let rt = &harness.rt;
    let total_t0 = Instant::now();

    // Stage 1 — session create.
    let t0 = Instant::now();
    // Auto-detect, as the app's default `dictation_language = "auto"` does.
    let session = rt
        .block_on(asr.start_session(AsrSessionConfig::default()))
        .expect("start session");
    let sess_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Stage 2 — push audio (buffer fill).
    let t0 = Instant::now();
    rt.block_on(asr.push_audio(&session, &fixture.pcm))
        .expect("push audio");
    let push_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Stage 3 — ASR poll_final (whole whisper decode incl. native VAD).
    let t0 = Instant::now();
    let segments = rt.block_on(asr.poll_final(&session)).expect("poll final");
    let poll_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let raw_text: String = segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");

    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    let _ = rt.block_on(asr.end_session(session));

    let (wer, cer) = if score {
        (
            word_error_rate(&raw_text, &fixture.reference),
            char_error_rate(&raw_text, &fixture.reference),
        )
    } else {
        (0.0, 0.0)
    };

    Sample {
        audio_secs: fixture.audio_secs,
        sess_ms,
        push_ms,
        poll_ms,
        total_ms,
        wer,
        cer,
    }
}

fn col(values: &[f64], p: f64) -> f64 {
    percentile(values, p)
}

fn print_table(samples: &[(&'static str, Sample)]) {
    println!(
        "\n  {:18}{:>6}{:>7}{:>7}{:>9}{:>8}{:>7}",
        "Fixture", "Audio", "Sess", "Push", "ASRpoll", "Total", "RTF"
    );
    println!("  {}", "─".repeat(78));
    for (name, s) in samples {
        let rtf = (s.total_ms / 1000.0) / s.audio_secs.max(1e-9);
        println!(
            "  {:18}{:5.1}s{:7.1}{:7.1}{:9.1}{:8.1}{:7.3}",
            truncate(name, 18),
            s.audio_secs,
            s.sess_ms,
            s.push_ms,
            s.poll_ms,
            s.total_ms,
            rtf,
        );
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}

fn print_aggregate(samples: &[Sample]) {
    if samples.is_empty() {
        return;
    }
    let pick = |f: &dyn Fn(&Sample) -> f64| -> Vec<f64> { samples.iter().map(f).collect() };

    let polls = pick(&|s| s.poll_ms);
    let totals = pick(&|s| s.total_ms);
    let pushes = pick(&|s| s.push_ms);
    let sessions = pick(&|s| s.sess_ms);
    let wers = pick(&|s| s.wer * 100.0);
    let cers = pick(&|s| s.cer * 100.0);
    let rtfs = pick(&|s| (s.total_ms / 1000.0) / s.audio_secs.max(1e-9));

    println!("\n  ══ Aggregate over {} timed samples ══", samples.len());
    let row = |label: &str, v: &[f64]| {
        println!(
            "  {:14}  P50 {:8.2}  P95 {:8.2}  max {:8.2}",
            label,
            col(v, 50.0),
            col(v, 95.0),
            col(v, 100.0),
        );
    };
    row("session ms", &sessions);
    row("push ms", &pushes);
    row("ASR poll ms", &polls);
    row("total ms", &totals);
    println!(
        "  {:14}  P50 {:8.4}  P95 {:8.4}                 (wall/audio)",
        "RTF",
        col(&rtfs, 50.0),
        col(&rtfs, 95.0)
    );
    println!(
        "  {:14}  P50 {:8.2}  P95 {:8.2}",
        "WER %",
        col(&wers, 50.0),
        col(&wers, 95.0)
    );
    println!(
        "  {:14}  P50 {:8.2}  P95 {:8.2}",
        "CER %",
        col(&cers, 50.0),
        col(&cers, 95.0)
    );

    // Time distribution — where the wall clock actually goes.
    let sum = |f: &dyn Fn(&Sample) -> f64| -> f64 { samples.iter().map(f).sum() };
    let s_poll = sum(&|s| s.poll_ms);
    let s_sess = sum(&|s| s.sess_ms);
    let s_push = sum(&|s| s.push_ms);
    let grand = sum(&|s| s.total_ms);

    if grand > 0.0 {
        println!("\n  Time distribution (share of total wall clock):");
        let pct = |x: f64| x / grand * 100.0;
        println!(
            "    ASR poll {:5.1}%   session {:5.1}%   push {:5.1}%   other {:5.1}%",
            pct(s_poll),
            pct(s_sess),
            pct(s_push),
            pct(grand - s_poll - s_sess - s_push).max(0.0),
        );
    }
}

/// Tight full-pipeline loop for a sampling profiler. Runs every fixture in a
/// round-robin until the time budget is spent, then reports throughput.
fn profile_loop(harness: &BenchmarkHarness, asr: &AsrRegistry, secs: f64) {
    println!("\n  ══ PROFILE mode — looping pipeline for {secs:.0}s ══");
    println!("  Attach a sampler to this process (samply record -- <bin>).");
    // One warm pass per fixture so we never profile cold-start.
    for fx in &harness.fixtures {
        let _ = run_once(harness, asr, fx, false);
    }
    let start = Instant::now();
    let mut iters = 0usize;
    let mut audio_secs = 0.0f64;
    while start.elapsed().as_secs_f64() < secs {
        for fx in &harness.fixtures {
            let _ = run_once(harness, asr, fx, false);
            iters += 1;
            audio_secs += fx.audio_secs;
            if start.elapsed().as_secs_f64() >= secs {
                break;
            }
        }
    }
    let wall = start.elapsed().as_secs_f64();
    println!(
        "  Completed {iters} pipeline runs in {wall:.1}s — {:.2} runs/s, {:.1}× realtime",
        iters as f64 / wall,
        audio_secs / wall,
    );
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  PIPELINE STAGE LATENCY                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let harness = BenchmarkHarness::builder().with_tracing().build();

    if harness.fixtures.is_empty() {
        eprintln!("  [skip] No fixtures found in fixtures/audio/");
        return;
    }

    // The harness already loaded the ASR backend (no second load here).
    let asr: &AsrRegistry = &harness.asr;

    println!(
        "  Device: {}   Fixtures: {}",
        if harness.has_gpu { "CUDA" } else { "CPU" },
        harness.fixtures.len(),
    );
    if let Some(mem) = process_memory_usage() {
        println!("  Memory after load: {}", mem.display());
    }

    if env_flag("PROFILE") {
        let secs = env_usize("PROFILE_SECS", 25) as f64;
        profile_loop(&harness, asr, secs);
        if let Some(mem) = process_memory_usage() {
            println!("\n  Final: {}", mem.display());
        }
        println!();
        return;
    }

    let warmup = env_usize("WARMUP", 1);
    let iters = env_usize("ITERS", 3);
    println!("  Warmup: {warmup}   Timed iters: {iters}");

    let mut all_samples: Vec<Sample> = Vec::new();
    let mut last_per_fixture: Vec<(&'static str, Sample)> = Vec::new();

    for fx in &harness.fixtures {
        for _ in 0..warmup {
            let _ = run_once(&harness, asr, fx, false);
        }
        let mut last = Sample::default();
        for _ in 0..iters {
            let s = run_once(&harness, asr, fx, true);
            last = s.clone();
            all_samples.push(s);
        }
        last_per_fixture.push((fx.name, last));
    }

    print_table(&last_per_fixture);
    print_aggregate(&all_samples);

    if let Some(mem) = process_memory_usage() {
        println!("\n  Final: {}", mem.display());
    }
    println!();
}
