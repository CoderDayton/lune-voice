//! Hybrid-VAD tuning benchmark — finds the `DEFAULT_VAD_CONFIG` knobs.
//!
//! Run: cargo run --release --example vad_tune -p lune-app
//! GPU: cargo run --release --features cuda --example vad_tune -p lune-app
//!
//! No labelled VAD ground truth exists, so the gate is tuned against the
//! objective that actually matters: end-to-end **ASR WER**. For each candidate
//! [`VadPipelineConfig`], every fixture's PCM is run through the real
//! [`HybridVad`] → the gated audio is fed to the real ASR (`poll_final`) → WER
//! is scored against the reference. The LLM is intentionally bypassed so the
//! signal reflects the gate's effect on raw transcription, not LLM repair.
//!
//! Baselines: **no-VAD** (push all audio) and **Earshot-only** (Silero off),
//! to size the hybrid delta. Then a coordinate-descent sweep over the knobs.
//! Env: `VAD_ROUNDS=N` refinement rounds (default 1); `LUNE_BENCH_FIXTURE=substr`
//! restricts the panel; `VAD_PCEN=0` disables PCEN on the Earshot stage.

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::time::Instant;

use bench_utils::{BenchmarkHarness, percentile, word_error_rate};
use lune_app::vad::{DEFAULT_VAD_CONFIG, GateAction, HybridVad, VadPipelineConfig};
use lune_asr::AsrRegistry;
use lune_audio::AudioChunk;
use lune_core::asr::{AsrBackend, AsrSessionConfig};

// ─── Scoring ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    mean_wer: f64,
    p90_wer: f64,
    /// Mean WER over just the noisy fixtures (`*_noise_snr*`) — where Silero's
    /// rejection should pay off, otherwise drowned out by the clean clips.
    noisy_wer: f64,
    /// Fraction of input audio that survived the gate (1.0 for no-VAD).
    gated_ratio: f64,
    mean_segments: f64,
    elapsed_s: f64,
}

/// How a fixture's PCM is turned into the audio the ASR sees.
#[derive(Clone, Copy)]
enum Gating {
    /// Push every sample (upper bound on recall, lower bound on noise rejection).
    NoVad,
    /// Run the real hybrid gate with this config (Silero off = Earshot-only).
    Hybrid(VadPipelineConfig),
}

const CHUNK: usize = 1280; // ~80 ms at 16 kHz, the live capture chunk size.

/// Run one fixture's PCM through the gate, concatenating the gated audio
/// (verified speech + any injected pauses) exactly as the capture loop would.
fn gate_pcm(config: VadPipelineConfig, pcm: &[f32], pcen: bool) -> (Vec<f32>, usize) {
    let mut hybrid = HybridVad::with_config(config, pcen).expect("build hybrid vad");
    let mut out: Vec<f32> = Vec::with_capacity(pcm.len());
    let mut segments = 0usize;
    let apply = |actions: Vec<GateAction>, out: &mut Vec<f32>, segments: &mut usize| {
        for a in actions {
            match a {
                GateAction::PushSamples(s) => out.extend_from_slice(&s),
                GateAction::InjectPause(n) => out.extend(std::iter::repeat_n(0.0, n)),
                GateAction::SegmentFinalized => *segments += 1,
            }
        }
    };
    for chunk in pcm.chunks(CHUNK) {
        let c = AudioChunk {
            samples: chunk.to_vec(),
            timestamp_ms: 0,
        };
        apply(hybrid.process_chunk(&c), &mut out, &mut segments);
    }
    apply(hybrid.finish(), &mut out, &mut segments);
    (out, segments)
}

/// Drive the real ASR over a finished audio buffer and return the raw transcript.
fn transcribe(asr: &AsrRegistry, rt: &tokio::runtime::Runtime, pcm: &[f32], lang: &str) -> String {
    if pcm.is_empty() {
        return String::new();
    }
    let cfg = AsrSessionConfig {
        language: Some(lang.to_string()),
        ..Default::default()
    };
    let session = rt.block_on(asr.start_session(cfg)).expect("start session");
    rt.block_on(asr.push_audio(&session, pcm))
        .expect("push audio");
    let segments = rt.block_on(asr.poll_final(&session)).expect("poll final");
    let text = segments
        .iter()
        .map(|s| s.text.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let _ = rt.block_on(asr.end_session(session));
    text
}

fn evaluate(harness: &BenchmarkHarness, gating: Gating, pcen: bool) -> Score {
    let t0 = Instant::now();
    let mut wers = Vec::with_capacity(harness.fixtures.len());
    let mut noisy = Vec::new();
    let mut gated_total = 0usize;
    let mut input_total = 0usize;
    let mut seg_total = 0usize;
    for fx in &harness.fixtures {
        let lang = BenchmarkHarness::language_for_fixture(fx.name);
        let (audio, segments): (Vec<f32>, usize) = match gating {
            Gating::NoVad => (fx.pcm.clone(), 0),
            Gating::Hybrid(cfg) => gate_pcm(cfg, &fx.pcm, pcen),
        };
        input_total += fx.pcm.len();
        gated_total += audio.len();
        seg_total += segments;
        let hyp = transcribe(&harness.asr, &harness.rt, &audio, lang);
        let wer = word_error_rate(&hyp, &fx.reference) * 100.0;
        wers.push(wer);
        if fx.name.contains("snr") {
            noisy.push(wer);
        }
    }
    let n = wers.len().max(1) as f64;
    Score {
        mean_wer: wers.iter().sum::<f64>() / n,
        p90_wer: percentile(&wers, 90.0),
        noisy_wer: if noisy.is_empty() {
            0.0
        } else {
            noisy.iter().sum::<f64>() / noisy.len() as f64
        },
        gated_ratio: if input_total == 0 {
            0.0
        } else {
            gated_total as f64 / input_total as f64
        },
        mean_segments: seg_total as f64 / n,
        elapsed_s: t0.elapsed().as_secs_f64(),
    }
}

fn print_row(label: &str, s: Score) {
    println!(
        "  {label:<34} WER {:>5.1}%  p90 {:>5.1}%  noisy {:>5.1}%  gated {:>5.1}%  segs {:>4.1}  {:>5.1}s",
        s.mean_wer,
        s.p90_wer,
        s.noisy_wer,
        s.gated_ratio * 100.0,
        s.mean_segments,
        s.elapsed_s
    );
}

// ─── Coordinate descent ──────────────────────────────────────────────────────

/// Evaluate each candidate, print it, and return the one with the lowest mean
/// WER (ties broken toward the more aggressive gate — lower `gated_ratio`).
fn sweep(
    harness: &BenchmarkHarness,
    pcen: bool,
    incumbent: VadPipelineConfig,
    knob: &str,
    candidates: Vec<(String, VadPipelineConfig)>,
) -> VadPipelineConfig {
    // Require a real improvement over the incumbent to move: coordinate descent
    // must never adopt a config worse than where it started, and the margin
    // stops it chasing sub-0.1% mean-WER noise.
    const MARGIN: f64 = 0.1;
    println!("\n── sweep: {knob} ──");
    let inc_score = evaluate(harness, Gating::Hybrid(incumbent), pcen);
    print_row("(keep incumbent)", inc_score);
    let mut best = (incumbent, inc_score);
    for (label, cfg) in candidates {
        let s = evaluate(harness, Gating::Hybrid(cfg), pcen);
        print_row(&label, s);
        if s.mean_wer + MARGIN < best.1.mean_wer {
            best = (cfg, s);
        }
    }
    let (cfg, s) = best;
    println!(
        "  → best {knob}: WER {:.1}%  (onset {:.2} offset {:.2} earshot {:.2} hang {}ms pre {}ms min {}ms pause {})",
        s.mean_wer,
        cfg.silero_onset,
        cfg.silero_offset,
        cfg.earshot_activity_threshold,
        cfg.hangover_ms,
        cfg.preroll_ms,
        cfg.min_chunk_ms,
        cfg.pause_injection
    );
    cfg
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  HYBRID VAD TUNING BENCHMARK                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let harness = BenchmarkHarness::builder().with_tracing().build();
    if harness.fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found");
        return;
    }
    let pcen = std::env::var("VAD_PCEN").map(|v| v != "0").unwrap_or(true);
    let rounds: usize = std::env::var("VAD_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    println!(
        "  Fixtures: {}   PCEN: {}   rounds: {}   ASR: {:?}",
        harness.fixtures.len(),
        pcen,
        rounds,
        harness.asr.kind()
    );

    // ── Baselines ──────────────────────────────────────────────────────────
    println!("\n── baselines ──");
    let no_vad = evaluate(&harness, Gating::NoVad, pcen);
    print_row("no-VAD (push all)", no_vad);
    let earshot_cfg = VadPipelineConfig {
        silero_enabled: false,
        ..DEFAULT_VAD_CONFIG
    };
    print_row(
        "earshot-only",
        evaluate(&harness, Gating::Hybrid(earshot_cfg), pcen),
    );
    print_row(
        "hybrid (current defaults)",
        evaluate(&harness, Gating::Hybrid(DEFAULT_VAD_CONFIG), pcen),
    );

    // ── Coordinate descent ─────────────────────────────────────────────────
    let mut best = DEFAULT_VAD_CONFIG;
    for round in 1..=rounds {
        println!("\n════════ round {round}/{rounds} ════════");

        best = sweep(
            &harness,
            pcen,
            best,
            "silero_onset",
            [0.3f32, 0.4, 0.5, 0.6, 0.7]
                .iter()
                .map(|&v| {
                    (
                        format!("onset={v:.2}"),
                        VadPipelineConfig {
                            silero_onset: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );

        best = sweep(
            &harness,
            pcen,
            best,
            "silero_offset",
            [0.2f32, 0.3, 0.35, 0.45]
                .iter()
                .filter(|&&v| v < best.silero_onset)
                .map(|&v| {
                    (
                        format!("offset={v:.2}"),
                        VadPipelineConfig {
                            silero_offset: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );

        best = sweep(
            &harness,
            pcen,
            best,
            "earshot_activity_threshold",
            [0.2f32, 0.3, 0.4, 0.5]
                .iter()
                .map(|&v| {
                    (
                        format!("earshot={v:.2}"),
                        VadPipelineConfig {
                            earshot_activity_threshold: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );

        best = sweep(
            &harness,
            pcen,
            best,
            "hangover_ms",
            [200u32, 350, 500, 800]
                .iter()
                .map(|&v| {
                    (
                        format!("hangover={v}ms"),
                        VadPipelineConfig {
                            hangover_ms: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );

        best = sweep(
            &harness,
            pcen,
            best,
            "preroll_ms",
            [150u32, 250, 350, 500]
                .iter()
                .map(|&v| {
                    (
                        format!("preroll={v}ms"),
                        VadPipelineConfig {
                            preroll_ms: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );

        best = sweep(
            &harness,
            pcen,
            best,
            "min_chunk_ms",
            [100u32, 200, 300, 400]
                .iter()
                .map(|&v| {
                    (
                        format!("min_chunk={v}ms"),
                        VadPipelineConfig {
                            min_chunk_ms: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );

        best = sweep(
            &harness,
            pcen,
            best,
            "pause_injection",
            [false, true]
                .iter()
                .map(|&v| {
                    (
                        format!("pause_injection={v}"),
                        VadPipelineConfig {
                            pause_injection: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );
    }

    // ── Winner ─────────────────────────────────────────────────────────────
    // Safety: a lowered onset can leave the incumbent offset above it; keep the
    // hysteresis gap so the shipped default is always valid.
    if best.silero_offset >= best.silero_onset {
        best.silero_offset = (best.silero_onset - 0.10).max(0.05);
    }
    let final_score = evaluate(&harness, Gating::Hybrid(best), pcen);
    println!("\n════════ WINNER ════════");
    print_row("no-VAD", no_vad);
    print_row("tuned hybrid", final_score);
    println!(
        "\n  Paste into DEFAULT_VAD_CONFIG (crates/lune-app/src/vad/config.rs):\n\
         \n    sample_rate: 16_000,\
         \n    silero_enabled: true,\
         \n    earshot_prefilter: {},\
         \n    earshot_activity_threshold: {:.2},\
         \n    silero_onset: {:.2},\
         \n    silero_offset: {:.2},\
         \n    preroll_ms: {},\
         \n    hangover_ms: {},\
         \n    min_chunk_ms: {},\
         \n    max_chunk_ms: {},\
         \n    pause_injection: {},",
        best.earshot_prefilter,
        best.earshot_activity_threshold,
        best.silero_onset,
        best.silero_offset,
        best.preroll_ms,
        best.hangover_ms,
        best.min_chunk_ms,
        best.max_chunk_ms,
        best.pause_injection,
    );
    println!();
}
