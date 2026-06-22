//! VAD knob sweep — which `VadPipelineConfig` knob actually moves transcription
//! quality?
//!
//! Run: cargo run --release --features cuda --example vad_knob_ab -p lune-app
//!
//! Companion to the `long_gap_ms` finding (segmentation moved 3.8x, quality
//! flat). Here every candidate knob is swept one-at-a-time around the
//! `DEFAULT_VAD_CONFIG` baseline over the full dictation panel + noisy clips,
//! and the WER / cased-CER span across each sweep tells us whether that knob is
//! "live" (changes the transcript) or inert (changes only segmentation). WER is
//! punctuation-blind but catches the failures gate knobs cause — clipped onsets,
//! dropped speech, admitted noise, merged words.

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::io::Write;
use std::path::Path;

use bench_utils::{
    BenchmarkHarness, Fixture, cased_cer, concat_errors, load_wav_f32, word_error_rate,
    workspace_root,
};
use lune_app::vad::{DEFAULT_VAD_CONFIG, GateAction, HybridVad, VadPipelineConfig};
use lune_asr::AsrRegistry;
use lune_audio::AudioChunk;
use lune_core::asr::{AsrBackend, AsrSessionConfig};

const CHUNK: usize = 1280; // ~80 ms at 16 kHz, the live capture chunk size.

/// Run one fixture's PCM through the gate; return the gated audio and span count.
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

fn push_pair(out: &mut Vec<Fixture>, wav: &Path, tag: &str) {
    let txt = wav.with_extension("txt");
    if !txt.exists() {
        return;
    }
    let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
    let pcm = load_wav_f32(wav);
    let reference = std::fs::read_to_string(&txt)
        .expect("read reference")
        .trim()
        .to_string();
    let audio_secs = pcm.len() as f64 / 16_000.0;
    let name: &'static str = Box::leak(format!("{tag}/{stem}").into_boxed_str());
    out.push(Fixture {
        name,
        pcm,
        reference,
        audio_secs,
    });
}

/// The full dictation panel under `fixtures/bench/<category>/` plus the
/// `*noise*` precision clips from `fixtures/audio/`.
fn load_panel(root: &Path) -> Vec<Fixture> {
    let mut out = Vec::new();
    let bench = root.join("fixtures/bench");
    if let Ok(cats) = std::fs::read_dir(&bench) {
        let mut dirs: Vec<std::path::PathBuf> = cats
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for dir in dirs {
            let cat = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            if let Ok(entries) = std::fs::read_dir(&dir) {
                let mut wavs: Vec<std::path::PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
                    .collect();
                wavs.sort();
                for wav in wavs {
                    push_pair(&mut out, &wav, &cat);
                }
            }
        }
    }
    let audio = root.join("fixtures/audio");
    if let Ok(entries) = std::fs::read_dir(&audio) {
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
            push_pair(&mut out, &wav, "noisy");
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Score {
    wer: f64,
    cer: f64,
    concat: usize,
    segments: usize,
    gated: f64,
}

fn evaluate(
    harness: &BenchmarkHarness,
    fixtures: &[Fixture],
    config: VadPipelineConfig,
    pcen: bool,
) -> Score {
    let (mut wer, mut cer) = (0.0f64, 0.0f64);
    let (mut concat, mut segments) = (0usize, 0usize);
    let (mut gated_num, mut gated_den) = (0usize, 0usize);
    for fx in fixtures {
        let lang = BenchmarkHarness::language_for_fixture(fx.name);
        let (audio, s) = gate_pcm(config, &fx.pcm, pcen);
        segments += s;
        gated_num += audio.len();
        gated_den += fx.pcm.len().max(1);
        let hyp = transcribe(&harness.asr, &harness.rt, &audio, lang);
        wer += word_error_rate(&hyp, &fx.reference);
        cer += cased_cer(&hyp, &fx.reference);
        concat += concat_errors(&hyp, &fx.reference);
    }
    let n = fixtures.len() as f64;
    Score {
        wer: wer / n * 100.0,
        cer: cer / n * 100.0,
        concat,
        segments,
        gated: gated_num as f64 / gated_den as f64 * 100.0,
    }
}

struct Knob {
    name: &'static str,
    variants: Vec<(String, VadPipelineConfig)>,
}

fn main() {
    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .skip_fixtures()
        .build();
    let root = workspace_root();
    let pcen = std::env::var("VAD_PCEN").map(|v| v != "0").unwrap_or(true);
    let fixtures = load_panel(&root);
    if fixtures.is_empty() {
        println!("  [skip] no fixtures found");
        return;
    }

    // Default values for reference: preroll 500, hangover 350, onset 0.30,
    // offset 0.20, earshot 0.35, min_speech 100, max_intra_silence 240.
    let d = DEFAULT_VAD_CONFIG;
    let knobs: Vec<Knob> = vec![
        Knob {
            name: "preroll_ms",
            variants: [0u32, 125, 250, 500, 1000]
                .iter()
                .map(|&v| (format!("{v}"), VadPipelineConfig { preroll_ms: v, ..d }))
                .collect(),
        },
        Knob {
            name: "hangover_ms",
            variants: [50u32, 200, 350, 700]
                .iter()
                .map(|&v| {
                    (
                        format!("{v}"),
                        VadPipelineConfig {
                            hangover_ms: v,
                            ..d
                        },
                    )
                })
                .collect(),
        },
        Knob {
            name: "silero_onset",
            variants: [0.22f32, 0.30, 0.45, 0.60]
                .iter()
                .map(|&v| {
                    (
                        format!("{v:.2}"),
                        VadPipelineConfig {
                            silero_onset: v,
                            ..d
                        },
                    )
                })
                .collect(),
        },
        Knob {
            name: "silero_offset",
            variants: [0.05f32, 0.12, 0.20, 0.28]
                .iter()
                .map(|&v| {
                    (
                        format!("{v:.2}"),
                        VadPipelineConfig {
                            silero_offset: v,
                            ..d
                        },
                    )
                })
                .collect(),
        },
        Knob {
            name: "earshot_activity_threshold",
            variants: [0.20f32, 0.35, 0.50, 0.70]
                .iter()
                .map(|&v| {
                    (
                        format!("{v:.2}"),
                        VadPipelineConfig {
                            earshot_activity_threshold: v,
                            ..d
                        },
                    )
                })
                .collect(),
        },
        Knob {
            name: "min_speech_ms",
            variants: [0u32, 100, 300, 600]
                .iter()
                .map(|&v| {
                    (
                        format!("{v}"),
                        VadPipelineConfig {
                            min_speech_ms: v,
                            ..d
                        },
                    )
                })
                .collect(),
        },
        Knob {
            name: "max_intra_silence_ms",
            variants: [0u32, 120, 240, 500, 1000]
                .iter()
                .map(|&v| {
                    (
                        format!("{v}"),
                        VadPipelineConfig {
                            max_intra_silence_ms: v,
                            ..d
                        },
                    )
                })
                .collect(),
        },
    ];

    println!();
    println!(
        "  VAD knob sweep   ({} fixtures, PCEN {pcen}, ASR {:?})",
        fixtures.len(),
        harness.asr.kind()
    );
    let base = evaluate(&harness, &fixtures, d, pcen);
    println!(
        "  baseline (default): WER {:.2}%  casedCER {:.2}%  concat {}  seg {}  gated {:.1}%\n",
        base.wer, base.cer, base.concat, base.segments, base.gated
    );

    let mut summary: Vec<(&str, f64, f64)> = Vec::new(); // (knob, wer span, cer span)
    for knob in &knobs {
        println!("  ── {} ──", knob.name);
        println!(
            "  {:>8}   {:>7} {:>9} {:>7} {:>6} {:>7}",
            "value", "WER%", "casedCER%", "concat", "seg", "gated%"
        );
        let mut wers = Vec::new();
        let mut cers = Vec::new();
        for (label, cfg) in &knob.variants {
            let s = evaluate(&harness, &fixtures, *cfg, pcen);
            let marker = if *cfg == d { "  <- default" } else { "" };
            println!(
                "  {label:>8}   {:>7.2} {:>9.2} {:>7} {:>6} {:>6.1}{marker}",
                s.wer, s.cer, s.concat, s.segments, s.gated
            );
            wers.push(s.wer);
            cers.push(s.cer);
        }
        let wer_span = wers.iter().cloned().fold(f64::MIN, f64::max)
            - wers.iter().cloned().fold(f64::MAX, f64::min);
        let cer_span = cers.iter().cloned().fold(f64::MIN, f64::max)
            - cers.iter().cloned().fold(f64::MAX, f64::min);
        let verdict = if wer_span > 0.5 || cer_span > 0.5 {
            "LIVE — moves quality"
        } else {
            "inert — quality flat"
        };
        println!("  span: WER {wer_span:.2}  casedCER {cer_span:.2}  => {verdict}\n");
        summary.push((knob.name, wer_span, cer_span));
        let _ = std::io::stdout().flush();
    }

    summary.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  ══ knobs ranked by WER span (quality leverage) ══");
    for (name, wer_span, cer_span) in &summary {
        let tag = if *wer_span > 0.5 || *cer_span > 0.5 {
            "LIVE "
        } else {
            "inert"
        };
        println!("  [{tag}] {name:<28} WER span {wer_span:>5.2}  casedCER span {cer_span:>5.2}");
    }
}
