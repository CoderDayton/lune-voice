//! Per-stage pipeline latency benchmark.
//!
//! Measures each pipeline stage independently: ASR load, session create,
//! push_audio, poll_final, confidence calibration.
//! Supports cross-backend comparison when both models are present.
//!
//! Run:  cargo run --release --example pipeline_stages -p lune-app
//! GPU:  cargo run --release --features cuda --example pipeline_stages -p lune-app
//! Both: ASR=all cargo run --release --example pipeline_stages -p lune-app

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::time::Instant;

use bench_utils::{
    Fixture, benchmark_asr_entries, char_error_rate, init_ort, load_fixtures, percentile,
    process_memory_usage, word_error_rate, workspace_root,
};
use lune_asr::AsrRegistry;
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use lune_core::config::DevicePolicy;

struct StageTiming {
    fixture: &'static str,
    audio_secs: f64,
    session_create_ms: f64,
    push_audio_ms: f64,
    poll_final_ms: f64,
    calibration_ms: f64,
    total_ms: f64,
    wer: f64,
    cer: f64,
}

struct BackendReport {
    name: &'static str,
    load_ms: f64,
    timings: Vec<StageTiming>,
}

fn run_backend(
    entry: &bench_utils::BenchmarkAsrEntry,
    device: DevicePolicy,
    models_dir: &std::path::Path,
    fixtures: &[Fixture],
    rt: &tokio::runtime::Runtime,
) -> BackendReport {
    println!("\n  ── {} ──", entry.display_name);

    let load_t0 = Instant::now();
    let asr = AsrRegistry::from_config(&entry.config, device, models_dir).expect("load ASR");
    let load_ms = load_t0.elapsed().as_secs_f64() * 1000.0;
    println!("    Model load: {load_ms:.0}ms");
    if let Some(mem) = process_memory_usage() {
        println!("    Memory after load: {}", mem.display());
    }

    let fixture_filter = std::env::var("LUNE_BENCH_FIXTURE").ok();
    let mut timings = Vec::new();

    for fixture in fixtures {
        if let Some(ref needle) = fixture_filter
            && !fixture.name.contains(needle.as_str())
        {
            continue;
        }

        let total_t0 = Instant::now();

        // Stage 1: session create
        let t0 = Instant::now();
        let session_config = AsrSessionConfig {
            language: Some("en".into()),
            ..Default::default()
        };
        let session = rt
            .block_on(asr.start_session(session_config))
            .expect("start session");
        let session_create_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Stage 2: push audio
        let t0 = Instant::now();
        rt.block_on(asr.push_audio(&session, &fixture.pcm))
            .expect("push audio");
        let push_audio_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Stage 3: poll final (encode + decode)
        let t0 = Instant::now();
        let segments = rt.block_on(asr.poll_final(&session)).expect("poll final");
        let poll_final_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let raw_text: String = segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");

        let calibration_ms = 0.0_f64;

        let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;
        let wer = word_error_rate(&raw_text, &fixture.reference);
        let cer = char_error_rate(&raw_text, &fixture.reference);

        timings.push(StageTiming {
            fixture: fixture.name,
            audio_secs: fixture.audio_secs,
            session_create_ms,
            push_audio_ms,
            poll_final_ms,
            calibration_ms,
            total_ms,
            wer,
            cer,
        });
    }

    BackendReport {
        name: entry.display_name,
        load_ms,
        timings,
    }
}

fn print_report(report: &BackendReport) {
    println!("\n  ┌─ {} ─ Load: {:.0}ms ─┐", report.name, report.load_ms);
    println!(
        "  {:20} {:>7} {:>8} {:>8} {:>8} {:>7} {:>8} {:>6} {:>6}",
        "Fixture", "Audio", "SessCr", "Push", "Poll", "Calib", "Total", "WER%", "CER%"
    );
    println!("  {}", "─".repeat(105));

    for t in &report.timings {
        println!(
            "  {:20} {:6.1}s {:7.1} {:7.1} {:7.1} {:6.1} {:7.1} {:5.1} {:5.1}",
            t.fixture,
            t.audio_secs,
            t.session_create_ms,
            t.push_audio_ms,
            t.poll_final_ms,
            t.calibration_ms,
            t.total_ms,
            t.wer * 100.0,
            t.cer * 100.0,
        );
    }

    if report.timings.len() > 1 {
        let polls: Vec<f64> = report.timings.iter().map(|t| t.poll_final_ms).collect();
        let totals: Vec<f64> = report.timings.iter().map(|t| t.total_ms).collect();
        let wers: Vec<f64> = report.timings.iter().map(|t| t.wer * 100.0).collect();
        let rtfs: Vec<f64> = report
            .timings
            .iter()
            .map(|t| (t.total_ms / 1000.0) / t.audio_secs)
            .collect();

        println!("  {}", "─".repeat(105));
        println!(
            "  Poll P50: {:.1}ms  P95: {:.1}ms  |  Total P50: {:.1}ms  P95: {:.1}ms",
            percentile(&polls, 50.0),
            percentile(&polls, 95.0),
            percentile(&totals, 50.0),
            percentile(&totals, 95.0),
        );
        println!(
            "  WER  P50: {:.1}%   P95: {:.1}%   |  RTF P50: {:.4}  P95: {:.4}",
            percentile(&wers, 50.0),
            percentile(&wers, 95.0),
            percentile(&rtfs, 50.0),
            percentile(&rtfs, 95.0),
        );
    }

    // Per-fixture breakdown: where is time spent?
    if !report.timings.is_empty() {
        let total_poll: f64 = report.timings.iter().map(|t| t.poll_final_ms).sum();
        let total_push: f64 = report.timings.iter().map(|t| t.push_audio_ms).sum();
        let total_sess: f64 = report.timings.iter().map(|t| t.session_create_ms).sum();
        let total_cal: f64 = report.timings.iter().map(|t| t.calibration_ms).sum();
        let grand: f64 = total_poll + total_push + total_sess + total_cal;

        if grand > 0.0 {
            println!("\n  Time distribution:");
            println!(
                "    Poll(encode+decode): {:5.1}%  |  Push(buffer): {:5.1}%  |  Session: {:5.1}%",
                total_poll / grand * 100.0,
                total_push / grand * 100.0,
                total_sess / grand * 100.0,
            );
            println!(
                "    Calibration:         {:5.1}%",
                total_cal / grand * 100.0
            );
        }
    }
}

fn print_comparison(reports: &[BackendReport]) {
    if reports.len() < 2 {
        return;
    }
    println!("\n  ╔══════════════════════════════════════════════════════════════════╗");
    println!("  ║                   CROSS-BACKEND COMPARISON                     ║");
    println!("  ╚══════════════════════════════════════════════════════════════════╝");

    println!(
        "\n  {:20} {:>12} {:>12} {:>12} {:>12}",
        "", "Load(ms)", "Poll P50", "WER P50", "RTF P50"
    );
    println!("  {}", "─".repeat(70));

    for r in reports {
        let polls: Vec<f64> = r.timings.iter().map(|t| t.poll_final_ms).collect();
        let wers: Vec<f64> = r.timings.iter().map(|t| t.wer * 100.0).collect();
        let rtfs: Vec<f64> = r
            .timings
            .iter()
            .map(|t| (t.total_ms / 1000.0) / t.audio_secs)
            .collect();
        println!(
            "  {:20} {:11.0} {:11.1} {:10.1}% {:11.4}",
            r.name,
            r.load_ms,
            percentile(&polls, 50.0),
            percentile(&wers, 50.0),
            percentile(&rtfs, 50.0),
        );
    }

    // Find common fixtures for direct comparison
    let common: Vec<&str> = reports[0]
        .timings
        .iter()
        .filter_map(|t| {
            if reports[1].timings.iter().any(|t2| t2.fixture == t.fixture) {
                Some(t.fixture)
            } else {
                None
            }
        })
        .collect();

    if !common.is_empty() {
        println!(
            "\n  Per-fixture delta (positive = {} faster):",
            reports[0].name
        );
        for name in &common {
            let a = reports[0]
                .timings
                .iter()
                .find(|t| t.fixture == *name)
                .unwrap();
            let b = reports[1]
                .timings
                .iter()
                .find(|t| t.fixture == *name)
                .unwrap();
            let speed_delta = b.total_ms - a.total_ms;
            let wer_delta = (a.wer - b.wer) * 100.0;
            println!(
                "    {:20}  speed: {:+.1}ms  WER: {:+.1}pp",
                name, speed_delta, wer_delta,
            );
        }
    }
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                   PIPELINE STAGE LATENCY                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();
    let root = workspace_root();
    let device = if has_gpu {
        DevicePolicy::Cuda
    } else {
        DevicePolicy::Cpu
    };
    let models_dir = bench_utils::models_dir();

    let entries = benchmark_asr_entries();
    if entries.is_empty() {
        eprintln!("  [skip] No ASR model dirs found under models/");
        return;
    }

    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        eprintln!("  [skip] No fixtures found in test_fixtures/");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut reports = Vec::new();

    for entry in &entries {
        let report = run_backend(entry, device, &models_dir, &fixtures, &rt);
        print_report(&report);
        reports.push(report);
    }

    print_comparison(&reports);

    if let Some(mem) = process_memory_usage() {
        println!("\n  Final: {}", mem.display());
    }
    println!();
}
