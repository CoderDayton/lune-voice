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
//! Tunes on the real dictation recordings in `fixtures/bench/` (commands,
//! long_form, proper_nouns, punctuation, quiet_dictation) plus the `*_noise_snr*`
//! clips from `fixtures/audio` (tagged `noisy`) for noise-rejection coverage.
//! Per-category WER is reported so recall-critical `quiet_dictation` and the
//! `noisy` precision case are both visible, not averaged away.
//!
//! Baselines: **no-VAD** (push all audio) and **Earshot-only** (Silero off),
//! to size the hybrid delta. Then a coordinate-descent sweep over the knobs.
//! Env: `VAD_ROUNDS=N` refinement rounds (default 1); `LUNE_BENCH_FIXTURE=substr`
//! restricts the panel (e.g. `=quiet`); `VAD_PCEN=0` disables Earshot PCEN.

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::path::Path;
use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, Fixture, cased_cer, concat_errors, load_wav_f32, percentile, spacing_delta,
    word_error_rate, workspace_root,
};
use lune_app::vad::{DEFAULT_VAD_CONFIG, GateAction, HybridVad, VadPipelineConfig};
use lune_asr::AsrRegistry;
use lune_audio::AudioChunk;
use lune_core::asr::{AsrBackend, AsrSessionConfig};

// ─── Scoring ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    mean_wer: f64,
    p90_wer: f64,
    /// Mean cased CER (%) — case, spacing, and punctuation all count. The
    /// signal WER erases; surfaces the `So Make` over-capitalization class.
    mean_cased_cer: f64,
    /// Mean fused-word boundaries per fixture (`MakeSure`). The splice
    /// signature; the Phase-1 acceptance gate wants this ≈ 0.
    mean_concat: f64,
    /// Mean signed word-count delta per fixture (hyp − ref). Negative = the
    /// gate is merging words.
    mean_spacing: f64,
    /// Mean WER over just the `quiet_dictation` clips — the recall-critical
    /// case where an over-eager gate clips low-volume speech.
    quiet_wer: f64,
    /// Mean WER over just the `noisy/*` SNR clips — the precision case where a
    /// leaky gate feeds background noise to the recognizer.
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
/// (verified speech with intra-utterance pauses intact) as the capture loop would.
fn gate_pcm(config: VadPipelineConfig, pcm: &[f32], pcen: bool) -> (Vec<f32>, usize) {
    let mut hybrid = HybridVad::with_config(config, pcen).expect("build hybrid vad");
    let mut out: Vec<f32> = Vec::with_capacity(pcm.len());
    let mut segments = 0usize;
    let apply = |actions: Vec<GateAction>, out: &mut Vec<f32>, segments: &mut usize| {
        for a in actions {
            match a {
                GateAction::PushSamples(s) => out.extend_from_slice(&s),
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

/// Load the real dictation recordings under `fixtures/bench/<category>/`,
/// tagging each fixture name `"<category>/<stem>"` so per-category WER can be
/// reported. Skips any `.wav` without a sibling `.txt` reference.
fn load_bench_fixtures(root: &Path) -> Vec<Fixture> {
    let bench_dir = root.join("fixtures/bench");
    let mut out = Vec::new();
    let Ok(cats) = std::fs::read_dir(&bench_dir) else {
        return out;
    };
    let mut cat_dirs: Vec<std::path::PathBuf> = cats
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    cat_dirs.sort();
    for cat_dir in cat_dirs {
        let category = cat_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let Ok(entries) = std::fs::read_dir(&cat_dir) else {
            continue;
        };
        let mut wavs: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
            .collect();
        wavs.sort();
        for wav in wavs {
            let txt = wav.with_extension("txt");
            if !txt.exists() {
                continue;
            }
            let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let pcm = load_wav_f32(&wav);
            let reference = std::fs::read_to_string(&txt)
                .expect("read reference")
                .trim()
                .to_string();
            let audio_secs = pcm.len() as f64 / 16_000.0;
            // Leaked intentionally: load_* runs once in main() and the Fixture
            // name lives for the whole run, so the bounded leak is simpler than
            // threading a string arena through the harness.
            let name: &'static str = Box::leak(format!("{category}/{stem}").into_boxed_str());
            out.push(Fixture {
                name,
                pcm,
                reference,
                audio_secs,
            });
        }
    }
    out
}

/// Load the noisy SNR variants from `fixtures/audio` (the `*_noise_snr*` clips),
/// tagged `"noisy/<stem>"`, so the gate is also tuned to reject background noise
/// — the real bench recordings are mostly quiet-environment.
fn load_noisy_audio_fixtures(root: &Path) -> Vec<Fixture> {
    let dir = root.join("fixtures/audio");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    let mut wavs: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.contains("noise"))
        })
        .collect();
    wavs.sort();
    for wav in wavs {
        let txt = wav.with_extension("txt");
        if !txt.exists() {
            continue;
        }
        let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let pcm = load_wav_f32(&wav);
        let reference = std::fs::read_to_string(&txt)
            .expect("read reference")
            .trim()
            .to_string();
        let audio_secs = pcm.len() as f64 / 16_000.0;
        // Bounded leak — see load_bench_fixtures; runs once, lives the whole run.
        let name: &'static str = Box::leak(format!("noisy/{stem}").into_boxed_str());
        out.push(Fixture {
            name,
            pcm,
            reference,
            audio_secs,
        });
    }
    out
}

/// Category for a fixture, from its `"<category>/<stem>"` name.
fn category_of(name: &str) -> &str {
    name.split('/').next().unwrap_or("?")
}

/// All scores for one gated fixture. WER and `cased_cer` are percentages.
#[derive(Clone, Copy)]
struct FixtureScore {
    wer: f64,
    cased_cer: f64,
    concat: usize,
    spacing: i64,
    gated: usize,
    segments: usize,
}

/// Gate one fixture and score its transcript across every metric.
fn fixture_score(
    harness: &BenchmarkHarness,
    fx: &Fixture,
    gating: Gating,
    pcen: bool,
) -> FixtureScore {
    let lang = BenchmarkHarness::language_for_fixture(fx.name);
    let (audio, segments) = match gating {
        Gating::NoVad => (fx.pcm.clone(), 0),
        Gating::Hybrid(cfg) => gate_pcm(cfg, &fx.pcm, pcen),
    };
    let hyp = transcribe(&harness.asr, &harness.rt, &audio, lang);
    // LUNE_BENCH_PRINT=1 dumps the raw transcript per fixture — the only way to
    // eyeball case/spacing artifacts (MakeSure) the scalar metrics summarize.
    if std::env::var("LUNE_BENCH_PRINT").is_ok_and(|v| v != "0") {
        println!("      [{}] {:?}", fx.name, hyp);
    }
    FixtureScore {
        wer: word_error_rate(&hyp, &fx.reference) * 100.0,
        cased_cer: cased_cer(&hyp, &fx.reference) * 100.0,
        concat: concat_errors(&hyp, &fx.reference),
        spacing: spacing_delta(&hyp, &fx.reference),
        gated: audio.len(),
        segments,
    }
}

fn evaluate(harness: &BenchmarkHarness, fixtures: &[Fixture], gating: Gating, pcen: bool) -> Score {
    let t0 = Instant::now();
    let mut wers = Vec::with_capacity(fixtures.len());
    let mut quiet = Vec::new();
    let mut noisy = Vec::new();
    let mut cers = Vec::new();
    let mut concats = Vec::new();
    let mut spacings = Vec::new();
    let mut gated_total = 0usize;
    let mut input_total = 0usize;
    let mut seg_total = 0usize;
    for fx in fixtures {
        let r = fixture_score(harness, fx, gating, pcen);
        input_total += fx.pcm.len();
        gated_total += r.gated;
        seg_total += r.segments;
        wers.push(r.wer);
        cers.push(r.cased_cer);
        concats.push(r.concat as f64);
        spacings.push(r.spacing as f64);
        match category_of(fx.name) {
            "quiet_dictation" => quiet.push(r.wer),
            "noisy" => noisy.push(r.wer),
            _ => {}
        }
    }
    let n = wers.len().max(1) as f64;
    Score {
        mean_wer: wers.iter().sum::<f64>() / n,
        p90_wer: percentile(&wers, 90.0),
        mean_cased_cer: cers.iter().sum::<f64>() / n,
        mean_concat: concats.iter().sum::<f64>() / n,
        mean_spacing: spacings.iter().sum::<f64>() / n,
        quiet_wer: if quiet.is_empty() {
            0.0
        } else {
            quiet.iter().sum::<f64>() / quiet.len() as f64
        },
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

/// Per-category mean WER for a gating config. Re-transcribes every fixture, so
/// only call it at the ends (baselines / winner), not inside the sweep.
fn report_by_category(
    harness: &BenchmarkHarness,
    fixtures: &[Fixture],
    gating: Gating,
    pcen: bool,
) {
    use std::collections::BTreeMap;
    let mut by_cat: BTreeMap<&str, Vec<FixtureScore>> = BTreeMap::new();
    for fx in fixtures {
        let r = fixture_score(harness, fx, gating, pcen);
        by_cat.entry(category_of(fx.name)).or_default().push(r);
    }
    for (cat, rs) in by_cat {
        let n = rs.len().max(1) as f64;
        let wer = rs.iter().map(|r| r.wer).sum::<f64>() / n;
        let cer = rs.iter().map(|r| r.cased_cer).sum::<f64>() / n;
        let concat = rs.iter().map(|r| r.concat as f64).sum::<f64>() / n;
        let spacing = rs.iter().map(|r| r.spacing as f64).sum::<f64>() / n;
        println!(
            "      {cat:<16} WER {wer:>5.1}%  cer {cer:>5.1}%  concat {concat:>4.1}  spc {spacing:>+5.1}  ({} clips)",
            rs.len()
        );
    }
}

fn print_row(label: &str, s: Score) {
    println!(
        "  {label:<30} WER {:>5.1}%  p90 {:>5.1}%  cer {:>5.1}%  cat {:>4.1}  spc {:>+5.1}  quiet {:>5.1}%  noisy {:>5.1}%  gated {:>5.1}%  segs {:>4.1}  {:>5.1}s",
        s.mean_wer,
        s.p90_wer,
        s.mean_cased_cer,
        s.mean_concat,
        s.mean_spacing,
        s.quiet_wer,
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
    fixtures: &[Fixture],
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
    let inc_score = evaluate(harness, fixtures, Gating::Hybrid(incumbent), pcen);
    print_row("(keep incumbent)", inc_score);
    let mut best = (incumbent, inc_score);
    for (label, cfg) in candidates {
        let s = evaluate(harness, fixtures, Gating::Hybrid(cfg), pcen);
        print_row(&label, s);
        if s.mean_wer + MARGIN < best.1.mean_wer {
            best = (cfg, s);
        }
    }
    let (cfg, s) = best;
    println!(
        "  → best {knob}: WER {:.1}%  (onset {:.2} offset {:.2} earshot {:.2} hang {}ms pre {}ms min {}ms gap {}ms)",
        s.mean_wer,
        cfg.silero_onset,
        cfg.silero_offset,
        cfg.earshot_activity_threshold,
        cfg.hangover_ms,
        cfg.preroll_ms,
        cfg.min_speech_ms,
        cfg.long_gap_ms
    );
    cfg
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  HYBRID VAD TUNING BENCHMARK                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .skip_fixtures()
        .build();
    let root = workspace_root();
    let mut fixtures = load_bench_fixtures(&root);
    fixtures.extend(load_noisy_audio_fixtures(&root));
    if let Ok(needle) = std::env::var("LUNE_BENCH_FIXTURE") {
        fixtures.retain(|f| f.name.contains(needle.as_str()));
    }
    if fixtures.is_empty() {
        println!("  [skip] No recordings found under fixtures/bench/");
        return;
    }
    let pcen = std::env::var("VAD_PCEN").map(|v| v != "0").unwrap_or(true);
    let rounds: usize = std::env::var("VAD_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    println!(
        "  Fixtures: {} (bench + noisy)   PCEN: {}   rounds: {}   ASR: {:?}",
        fixtures.len(),
        pcen,
        rounds,
        harness.asr.kind()
    );

    // ── Baselines ──────────────────────────────────────────────────────────
    println!("\n── baselines ──");
    let no_vad = evaluate(&harness, &fixtures, Gating::NoVad, pcen);
    print_row("no-VAD (push all)", no_vad);
    let earshot_cfg = VadPipelineConfig {
        silero_enabled: false,
        ..DEFAULT_VAD_CONFIG
    };
    print_row(
        "earshot-only",
        evaluate(&harness, &fixtures, Gating::Hybrid(earshot_cfg), pcen),
    );
    print_row(
        "hybrid (current defaults)",
        evaluate(
            &harness,
            &fixtures,
            Gating::Hybrid(DEFAULT_VAD_CONFIG),
            pcen,
        ),
    );

    // ── Coordinate descent ─────────────────────────────────────────────────
    let mut best = DEFAULT_VAD_CONFIG;
    for round in 1..=rounds {
        println!("\n════════ round {round}/{rounds} ════════");

        // Joint onset × offset grid. The two are coupled — a higher onset lets
        // the offset trim more aggressively while keeping a valid hysteresis gap
        // — so they are swept together; coordinate descent on each alone can't
        // see the coupling. Offset is expressed as onset − gap so every cell is
        // valid (offset < onset), floored at 0.05.
        let mut onset_offset = Vec::new();
        for onset in [0.30f32, 0.40, 0.50, 0.60] {
            for gap in [0.05f32, 0.10, 0.15, 0.20] {
                let offset = (onset - gap).max(0.05);
                onset_offset.push((
                    format!("onset={onset:.2} offset={offset:.2}"),
                    VadPipelineConfig {
                        silero_onset: onset,
                        silero_offset: offset,
                        ..best
                    },
                ));
            }
        }
        best = sweep(
            &harness,
            &fixtures,
            pcen,
            best,
            "silero onset×offset",
            onset_offset,
        );

        best = sweep(
            &harness,
            &fixtures,
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
            &fixtures,
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
            &fixtures,
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
            &fixtures,
            pcen,
            best,
            "min_speech_ms",
            [100u32, 200, 300, 400]
                .iter()
                .map(|&v| {
                    (
                        format!("min_speech={v}ms"),
                        VadPipelineConfig {
                            min_speech_ms: v,
                            ..best
                        },
                    )
                })
                .collect(),
        );

        // `long_gap_ms` is deliberately NOT swept: WER is blind to the word
        // fusion it controls (see DEFAULT_VAD_CONFIG in vad/config.rs), so a WER
        // sweep would shrink it back toward the old splice behaviour. It stays
        // pinned at the shipped default.
    }

    // ── Winner ─────────────────────────────────────────────────────────────
    // Safety: a lowered onset can leave the incumbent offset above it; keep the
    // hysteresis gap so the shipped default is always valid.
    if best.silero_offset >= best.silero_onset {
        best.silero_offset = (best.silero_onset - 0.10).max(0.05);
    }
    let final_score = evaluate(&harness, &fixtures, Gating::Hybrid(best), pcen);
    println!("\n════════ WINNER ════════");
    print_row("no-VAD", no_vad);
    print_row("tuned hybrid", final_score);
    println!();
    println!("  per-category WER (no-VAD):");
    report_by_category(&harness, &fixtures, Gating::NoVad, pcen);
    println!("  per-category WER (tuned hybrid):");
    report_by_category(&harness, &fixtures, Gating::Hybrid(best), pcen);
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
         \n    max_chunk_ms: {},\
         \n    min_speech_ms: {},\
         \n    long_gap_ms: {},",
        best.earshot_prefilter,
        best.earshot_activity_threshold,
        best.silero_onset,
        best.silero_offset,
        best.preroll_ms,
        best.hangover_ms,
        best.max_chunk_ms,
        best.min_speech_ms,
        best.long_gap_ms,
    );
    println!();
}
