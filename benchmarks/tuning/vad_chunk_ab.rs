//! Focused VAD quality sweeps with PER-CATEGORY WER — chases the leads the
//! coarse `vad_knob_ab` surfaced without averaging wins away.
//!
//! Run: cargo run --release --features cuda --example vad_chunk_ab -p lune-app
//!
//! Two sweeps:
//!   1. `preroll_ms` (fine) — confirm whether the small WER edge over the 500 ms
//!      default is real or measurement noise, and see which category it helps.
//!   2. `max_chunk_ms` (the chunking cap) — long-form clips (54–70 s) hit the
//!      20 s force-split with no overlap, so a mid-sentence cut can clip a word.
//!      Does raising/lowering the cap move long-form WER?
//!
//! Per-category WER is printed because the panel is dominated by easy clips
//! (commands/pause ≈ 0 %); a knob that only helps long-form is invisible in the
//! aggregate.

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use bench_utils::{
    BenchmarkHarness, Fixture, cased_cer, load_wav_f32, word_error_rate, workspace_root,
};
use lune_app::vad::{DEFAULT_VAD_CONFIG, GateAction, HybridVad, VadPipelineConfig};
use lune_asr::AsrRegistry;
use lune_audio::AudioChunk;
use lune_core::asr::{AsrBackend, AsrSessionConfig};

const CHUNK: usize = 1280;

/// Categories shown in the per-category breakdown (the ones with error
/// headroom). Easy categories still count toward the overall mean.
const SHOWN: &[&str] = &[
    "long_form",
    "punctuation",
    "proper_nouns",
    "quiet_dictation",
    "noisy",
];

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

fn category_of(name: &str) -> &str {
    name.split('/').next().unwrap_or("?")
}

struct Eval {
    wer: f64,
    cer: f64,
    segments: usize,
    per_cat: BTreeMap<String, f64>, // category -> mean WER%
}

fn evaluate(
    harness: &BenchmarkHarness,
    fixtures: &[Fixture],
    config: VadPipelineConfig,
    pcen: bool,
) -> Eval {
    let (mut wer, mut cer) = (0.0f64, 0.0f64);
    let mut segments = 0usize;
    let mut cat: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for fx in fixtures {
        let lang = BenchmarkHarness::language_for_fixture(fx.name);
        let (audio, s) = gate_pcm(config, &fx.pcm, pcen);
        segments += s;
        let hyp = transcribe(&harness.asr, &harness.rt, &audio, lang);
        let w = word_error_rate(&hyp, &fx.reference);
        wer += w;
        cer += cased_cer(&hyp, &fx.reference);
        let e = cat
            .entry(category_of(fx.name).to_string())
            .or_insert((0.0, 0));
        e.0 += w;
        e.1 += 1;
    }
    let n = fixtures.len() as f64;
    let per_cat = cat
        .into_iter()
        .map(|(k, (s, c))| (k, s / c as f64 * 100.0))
        .collect();
    Eval {
        wer: wer / n * 100.0,
        cer: cer / n * 100.0,
        segments,
        per_cat,
    }
}

fn print_header() {
    print!(
        "  {:>8}   {:>6} {:>8} {:>4}",
        "value", "WER%", "casedCER%", "seg"
    );
    for c in SHOWN {
        print!("  {:>10}", &c[..c.len().min(10)]);
    }
    println!();
}

fn print_row(label: &str, e: &Eval, is_default: bool) {
    print!(
        "  {label:>8}   {:>6.2} {:>8.2} {:>4}",
        e.wer, e.cer, e.segments
    );
    for c in SHOWN {
        match e.per_cat.get(*c) {
            Some(v) => print!("  {v:>10.2}"),
            None => print!("  {:>10}", "-"),
        }
    }
    println!("{}", if is_default { "  <- default" } else { "" });
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
    let d = DEFAULT_VAD_CONFIG;

    println!();
    println!(
        "  VAD chunk/preroll sweep   ({} fixtures, PCEN {pcen}, ASR {:?})",
        fixtures.len(),
        harness.asr.kind()
    );
    println!("  (per-category columns are mean WER% within that category)\n");

    // ── 1. Fine preroll sweep ────────────────────────────────────────────────
    println!("  ══ preroll_ms (fine) ══");
    print_header();
    for ms in [200u32, 300, 400, 500, 600, 800, 1000, 1500] {
        let cfg = VadPipelineConfig {
            preroll_ms: ms,
            ..d
        };
        let e = evaluate(&harness, &fixtures, cfg, pcen);
        print_row(&ms.to_string(), &e, ms == d.preroll_ms);
        let _ = std::io::stdout().flush();
    }

    // ── 2. The chunking cap ──────────────────────────────────────────────────
    println!("\n  ══ max_chunk_ms (force-split cap) ══");
    print_header();
    for ms in [4_000u32, 8_000, 12_000, 20_000, 40_000, 80_000] {
        let cfg = VadPipelineConfig {
            max_chunk_ms: ms,
            ..d
        };
        let e = evaluate(&harness, &fixtures, cfg, pcen);
        print_row(&ms.to_string(), &e, ms == d.max_chunk_ms);
        let _ = std::io::stdout().flush();
    }

    println!("\n  done.");
}
