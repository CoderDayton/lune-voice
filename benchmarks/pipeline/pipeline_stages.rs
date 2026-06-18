//! Per-stage pipeline latency + sub-stage profiling benchmark.
//!
//! Decomposes the full dictation pipeline into measured stages
//!
//!   session create → push audio → ASR poll_final → LLM correction
//!
//! and, for the Qwen3 backend, splits the ASR stage further into its recorded
//! sub-stages (`mel → encoder → prefill → decode`, plus decode tokens/sec)
//! read back from [`AsrRegistry::last_profile`]. This is the breakdown that
//! says *which stage dominates* before reaching for a sampling profiler.
//!
//! Each fixture runs `WARMUP` discarded passes then `ITERS` timed passes, so
//! the reported P50/P95 reflect steady state — CUDA arenas warm and the
//! `.onnx.opt` graph cached — rather than first-call cost.
//!
//! Env:
//!   ITERS=N         timed iterations per fixture           (default 3)
//!   WARMUP=N        discarded warmup iterations per fixture (default 1)
//!   LUNE_LLM=off    skip the LLM correction stage
//!   PROFILE=1       profiling mode: tight full-pipeline loop over every
//!                   fixture for PROFILE_SECS, minimal output — feed a
//!                   sampling profiler this build.
//!   PROFILE_SECS=N  duration of the profiling loop          (default 25)
//!
//! Run:   cargo run --release --example pipeline_stages -p lune-app
//! GPU:   cargo run --release --features cuda --example pipeline_stages -p lune-app
//! Flame: PROFILE=1 samply record -- \
//!          target/release/examples/pipeline_stages

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, Fixture, char_error_rate, percentile, process_memory_usage, word_error_rate,
};
use lune_asr::AsrRegistry;
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use lune_core::config::builtin_registry;
use lune_core::llm::{PostProcessInput, TextPostProcessor};

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

/// One timed pass over a single fixture: every stage, plus the ASR sub-stage
/// profile read back from the backend.
#[derive(Clone, Default)]
struct Sample {
    audio_secs: f64,
    sess_ms: f64,
    push_ms: f64,
    poll_ms: f64,
    // ASR sub-stages (Qwen3 only; zero when the backend records no profile).
    mel_ms: f64,
    encoder_ms: f64,
    prompt_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    decode_steps: usize,
    // Decode-loop per-op breakdown (IO-binding path).
    d_embed_ms: f64,
    d_bind_ms: f64,
    d_run_ms: f64,
    d_argmax_ms: f64,
    d_kv_ms: f64,
    // Post-ASR.
    llm_ms: f64,
    total_ms: f64,
    wer: f64,
    cer: f64,
}

/// Run the full pipeline once for `fixture`. When `score` is false the text
/// metrics are skipped (profiling loop doesn't need them).
fn run_once(
    harness: &BenchmarkHarness,
    asr: &AsrRegistry,
    llm: &dyn TextPostProcessor,
    style_id: &str,
    fixture: &Fixture,
    score: bool,
) -> Sample {
    let rt = &harness.rt;
    let total_t0 = Instant::now();

    // Stage 1 — session create.
    let t0 = Instant::now();
    let lang = BenchmarkHarness::language_for_fixture(fixture.name);
    let session_config = AsrSessionConfig {
        language: Some(lang.into()),
        ..Default::default()
    };
    let session = rt
        .block_on(asr.start_session(session_config))
        .expect("start session");
    let sess_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Stage 2 — push audio (buffer fill).
    let t0 = Instant::now();
    rt.block_on(asr.push_audio(&session, &fixture.pcm))
        .expect("push audio");
    let push_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Stage 3 — ASR poll_final (mel + encode + decode).
    let t0 = Instant::now();
    let segments = rt.block_on(asr.poll_final(&session)).expect("poll final");
    let poll_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let raw_text: String = segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");

    // ASR sub-stage profile recorded by the backend during poll_final.
    let prof = asr.last_profile().unwrap_or_default();

    // Stage 4 — LLM correction on the ASR output.
    let style_preset = builtin_registry()
        .get(style_id)
        .or_else(|| builtin_registry().get("minimal"))
        .expect("style preset");
    let input = PostProcessInput {
        raw_text: raw_text.clone(),
        style_preset,
        app_category: None,
        dictionary: Vec::new(),
        language: lang.into(),
        correction_needed: true,
    };
    let t0 = Instant::now();
    let llm_out = rt.block_on(llm.transform(&input));
    let llm_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    let _ = rt.block_on(asr.end_session(session));

    let (wer, cer) = if score {
        let final_text = llm_out
            .as_ref()
            .map(|o| o.cleaned_text.clone())
            .unwrap_or(raw_text);
        (
            word_error_rate(&final_text, &fixture.reference),
            char_error_rate(&final_text, &fixture.reference),
        )
    } else {
        (0.0, 0.0)
    };

    Sample {
        audio_secs: fixture.audio_secs,
        sess_ms,
        push_ms,
        poll_ms,
        mel_ms: prof.mel_ms,
        encoder_ms: prof.encoder_ms,
        prompt_ms: prof.prompt_ms,
        prefill_ms: prof.prefill_ms,
        decode_ms: prof.decode_ms,
        decode_steps: prof.decode_steps,
        d_embed_ms: prof.decode_embed_ms,
        d_bind_ms: prof.decode_bind_ms,
        d_run_ms: prof.decode_run_ms,
        d_argmax_ms: prof.decode_argmax_ms,
        d_kv_ms: prof.decode_kv_ms,
        llm_ms,
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
        "\n  {:18}{:>6}{:>7}{:>7}{:>7}{:>8}{:>8}{:>7}{:>7}{:>7}",
        "Fixture", "Audio", "Sess", "Push", "Mel", "Encoder", "Decode", "LLM", "Total", "RTF"
    );
    println!("  {}", "─".repeat(92));
    for (name, s) in samples {
        let rtf = (s.total_ms / 1000.0) / s.audio_secs.max(1e-9);
        println!(
            "  {:18}{:5.1}s{:7.1}{:7.1}{:7.1}{:8.1}{:8.1}{:7.1}{:7.1}{:7.3}",
            truncate(name, 18),
            s.audio_secs,
            s.sess_ms,
            s.push_ms,
            s.mel_ms,
            s.encoder_ms,
            s.decode_ms,
            s.llm_ms,
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
    let llms = pick(&|s| s.llm_ms);
    let decodes = pick(&|s| s.decode_ms);
    let encoders = pick(&|s| s.encoder_ms);
    let mels = pick(&|s| s.mel_ms);
    let wers = pick(&|s| s.wer * 100.0);
    let cers = pick(&|s| s.cer * 100.0);
    let rtfs = pick(&|s| (s.total_ms / 1000.0) / s.audio_secs.max(1e-9));
    // tokens/sec across the decode loop (steps / decode seconds).
    let toks: Vec<f64> = samples
        .iter()
        .filter(|s| s.decode_ms > 0.0 && s.decode_steps > 0)
        .map(|s| s.decode_steps as f64 / (s.decode_ms / 1000.0))
        .collect();

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
    row("mel ms", &mels);
    row("encoder ms", &encoders);
    row("decode ms", &decodes);
    row("ASR poll ms", &polls);
    row("LLM ms", &llms);
    row("total ms", &totals);
    println!(
        "  {:14}  P50 {:8.4}  P95 {:8.4}                 (wall/audio)",
        "RTF",
        col(&rtfs, 50.0),
        col(&rtfs, 95.0)
    );
    if !toks.is_empty() {
        println!(
            "  {:14}  P50 {:8.1}  P95 {:8.1}                 (decode steps/s)",
            "tokens/s",
            col(&toks, 50.0),
            col(&toks, 95.0)
        );
    }
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

    // Decode-loop op breakdown — where the dominant stage spends its time.
    let d_embed: f64 = samples.iter().map(|s| s.d_embed_ms).sum();
    let d_bind: f64 = samples.iter().map(|s| s.d_bind_ms).sum();
    let d_run: f64 = samples.iter().map(|s| s.d_run_ms).sum();
    let d_argmax: f64 = samples.iter().map(|s| s.d_argmax_ms).sum();
    let d_kv: f64 = samples.iter().map(|s| s.d_kv_ms).sum();
    let d_total = d_embed + d_bind + d_run + d_argmax + d_kv;
    if d_total > 0.0 {
        let total_steps: usize = samples.iter().map(|s| s.decode_steps).sum();
        let per_step = |x: f64| {
            if total_steps > 0 {
                x * 1000.0 / total_steps as f64
            } else {
                0.0
            }
        };
        println!("\n  Decode-loop op breakdown (share of measured decode ops · µs/token):");
        let pct = |x: f64| x / d_total * 100.0;
        println!(
            "    embed   {:5.1}% {:7.1}µs   decoder-run {:5.1}% {:7.1}µs   argmax {:5.1}% {:7.1}µs",
            pct(d_embed),
            per_step(d_embed),
            pct(d_run),
            per_step(d_run),
            pct(d_argmax),
            per_step(d_argmax),
        );
        println!(
            "    bind    {:5.1}% {:7.1}µs   kv-collect  {:5.1}% {:7.1}µs",
            pct(d_bind),
            per_step(d_bind),
            pct(d_kv),
            per_step(d_kv),
        );
    }

    // Time distribution — where the wall clock actually goes.
    let sum = |f: &dyn Fn(&Sample) -> f64| -> f64 { samples.iter().map(f).sum() };
    let s_mel = sum(&|s| s.mel_ms);
    let s_enc = sum(&|s| s.encoder_ms);
    let s_prompt = sum(&|s| s.prompt_ms);
    let s_prefill = sum(&|s| s.prefill_ms);
    let s_decode = sum(&|s| s.decode_ms);
    let s_llm = sum(&|s| s.llm_ms);
    let s_sess = sum(&|s| s.sess_ms);
    let s_push = sum(&|s| s.push_ms);
    let grand = sum(&|s| s.total_ms);

    if grand > 0.0 {
        println!("\n  Time distribution (share of total wall clock):");
        let pct = |x: f64| x / grand * 100.0;
        println!(
            "    mel {:5.1}%   encoder {:5.1}%   prompt {:5.1}%   prefill {:5.1}%   decode {:5.1}%",
            pct(s_mel),
            pct(s_enc),
            pct(s_prompt),
            pct(s_prefill),
            pct(s_decode),
        );
        println!(
            "    LLM {:5.1}%   session {:5.1}%   push {:5.1}%   other {:5.1}%",
            pct(s_llm),
            pct(s_sess),
            pct(s_push),
            pct(grand - s_mel - s_enc - s_prompt - s_prefill - s_decode - s_llm - s_sess - s_push)
                .max(0.0),
        );
    }
}

/// Tight full-pipeline loop for a sampling profiler. Runs every fixture in a
/// round-robin until the time budget is spent, then reports throughput.
fn profile_loop(
    harness: &BenchmarkHarness,
    asr: &AsrRegistry,
    llm: &dyn TextPostProcessor,
    style_id: &str,
    secs: f64,
) {
    println!("\n  ══ PROFILE mode — looping pipeline for {secs:.0}s ══");
    println!("  Attach a sampler to this process (samply record -- <bin>).");
    // One warm pass per fixture so we never profile cold-start.
    for fx in &harness.fixtures {
        let _ = run_once(harness, asr, llm, style_id, fx, false);
    }
    let start = Instant::now();
    let mut iters = 0usize;
    let mut audio_secs = 0.0f64;
    while start.elapsed().as_secs_f64() < secs {
        for fx in &harness.fixtures {
            let _ = run_once(harness, asr, llm, style_id, fx, false);
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
    println!("║          PIPELINE STAGE LATENCY  ·  sub-stage profile          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let harness = BenchmarkHarness::builder()
        .with_llm()
        .with_tracing()
        .build();

    if harness.fixtures.is_empty() {
        eprintln!("  [skip] No fixtures found in fixtures/audio/");
        return;
    }

    // The harness already loaded the ASR backend (no second 1 GB load here).
    let asr: &AsrRegistry = &harness.asr;
    let llm: &dyn TextPostProcessor = harness.llm.as_ref();
    let style_id = std::env::var("STYLE").unwrap_or_else(|_| "minimal".into());

    println!(
        "  Device: {}   Fixtures: {}   Style: {style_id}",
        if harness.has_gpu { "CUDA" } else { "CPU" },
        harness.fixtures.len(),
    );
    if let Some(mem) = process_memory_usage() {
        println!("  Memory after load: {}", mem.display());
    }

    if env_flag("PROFILE") {
        let secs = env_usize("PROFILE_SECS", 25) as f64;
        profile_loop(&harness, asr, llm, &style_id, secs);
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
            let _ = run_once(&harness, asr, llm, &style_id, fx, false);
        }
        let mut last = Sample::default();
        for _ in 0..iters {
            let s = run_once(&harness, asr, llm, &style_id, fx, true);
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
