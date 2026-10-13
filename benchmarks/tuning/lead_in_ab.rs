// SPDX-License-Identifier: GPL-3.0-or-later

//! lead_in_ab — does audio in front of the first word fix a wrong opening
//! word on a short utterance?
//!
//! Each clip is cut to a short utterance and decoded under every
//! [`LeadIn`] policy, at three amounts of recorded lead-in:
//!
//!   0 ms    speech starts with the hotkey press
//!   150 ms  a short breath before speaking
//!   600 ms  more than the endpoint trim keeps
//!
//! Scored against the reference's opening words: `first` is the share of
//! clips whose first word is wrong, `head` the mean word errors in the first
//! [`HEAD_WORDS`] words, `empty` the clips that decoded to nothing.
//!
//! Corpus: the English clips in `fixtures/audio` and `fixtures/bench/*`, cut
//! to [`HEAD_MS`] after the onset. `LUNE_BENCH_DIR=<dir>` scores your own
//! `.wav` + `.txt` pairs instead, uncut. `LUNE_BENCH_AGC=1` runs each
//! utterance through the capture path's PCEN gain first (the `vad_pcen`
//! setting).
//!
//! Run: WHISPER_MODEL=large cargo run --release --example lead_in_ab \
//!        -p lune-voice --features whisper-cpp
//!      (LUNE_BENCH_PRINT=1 lists every wrong opening; LUNE_BENCH_FIXTURE=<sub> filters.)

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::path::Path;

use bench_utils::{
    BenchmarkHarness, Fixture, benchmark_asr_entry, levenshtein, load_fixtures, load_wav_f32,
    models_dir, normalize_text, workspace_root,
};
use lune_asr::{LeadIn, WhisperCppBackend, WhisperCppConfig};
use lune_audio::{PCEN_HOP_SIZE, PcenAgc, PcenAgcConfig};
use lune_core::asr::{AsrBackend, AsrSessionConfig};

const SAMPLES_PER_MS: usize = 16;

/// Length of the utterance cut from a fixture, measured from the onset.
const HEAD_MS: usize = 2_500;

/// Opening words scored against the reference.
const HEAD_WORDS: usize = 3;

/// Recorded lead-in kept before the onset, per condition.
const RECORDED_LEADS_MS: [usize; 3] = [0, 150, 600];

const ARMS: [(&str, LeadIn); 6] = [
    ("recorded", LeadIn::Recorded),
    ("noise>=400", LeadIn::AtLeast { ms: 400 }),
    ("noise>=1000", LeadIn::AtLeast { ms: 1_000 }),
    ("zeros+500", LeadIn::Silence { ms: 500 }),
    ("zeros+1000", LeadIn::Silence { ms: 1_000 }),
    ("zeros+3000", LeadIn::Silence { ms: 3_000 }),
];

/// First sample of speech: the first 10 ms frame, followed by another, whose
/// RMS passes 5% of the clip's loudest frame.
fn onset_sample(pcm: &[f32]) -> usize {
    const FRAME: usize = 10 * SAMPLES_PER_MS;
    let levels: Vec<f32> = pcm
        .chunks_exact(FRAME)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / FRAME as f32).sqrt())
        .collect();
    let threshold = levels.iter().copied().fold(0.0, f32::max) * 0.05;
    levels
        .windows(2)
        .position(|w| w[0] > threshold && w[1] > threshold)
        .map_or(0, |frame| frame * FRAME)
}

/// The clip from `lead_ms` before the onset, cut to `head_ms` after it when
/// given. A cut end is faded over 30 ms so it does not click.
fn utterance(pcm: &[f32], onset: usize, lead_ms: usize, head_ms: Option<usize>) -> Vec<f32> {
    let start = onset.saturating_sub(lead_ms * SAMPLES_PER_MS);
    let end = head_ms.map_or(pcm.len(), |ms| (onset + ms * SAMPLES_PER_MS).min(pcm.len()));
    let mut out = pcm[start..end].to_vec();
    if head_ms.is_some() {
        let fade = (30 * SAMPLES_PER_MS).min(out.len());
        let tail = out.len() - fade;
        for (i, s) in out[tail..].iter_mut().enumerate() {
            *s *= 1.0 - i as f32 / fade as f32;
        }
    }
    out
}

/// The utterance as the capture path hands it to the recognizer with
/// `vad_pcen` on: gain applied per hop, the last partial hop at the final gain.
fn with_agc(pcm: &[f32]) -> Vec<f32> {
    let mut agc = PcenAgc::new(PcenAgcConfig::default(), 16_000, PCEN_HOP_SIZE)
        .expect("default PCEN config is valid");
    let mut out = pcm.to_vec();
    let mut hops = out.chunks_exact_mut(PCEN_HOP_SIZE);
    for hop in &mut hops {
        agc.process_frame(hop);
    }
    let gain = agc.last_gain();
    for s in hops.into_remainder() {
        *s *= gain;
    }
    out
}

/// Every `.wav` in `dir` that has a `.txt` reference beside it.
fn load_pairs(dir: &Path, prefix: &str) -> Vec<Fixture> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut wavs: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .filter(|p| p.with_extension("txt").exists())
        .collect();
    wavs.sort();
    wavs.into_iter()
        .map(|wav| {
            let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let pcm = load_wav_f32(&wav);
            let reference = std::fs::read_to_string(wav.with_extension("txt"))
                .expect("read reference")
                .trim()
                .to_string();
            Fixture {
                name: Box::leak(format!("{prefix}{stem}").into_boxed_str()),
                audio_secs: pcm.len() as f64 / 16_000.0,
                pcm,
                reference,
            }
        })
        .collect()
}

/// The English read-speech set plus every `fixtures/bench/<category>`.
fn load_repo_fixtures() -> Vec<Fixture> {
    let root = workspace_root();
    let mut out: Vec<Fixture> = load_fixtures(&root)
        .into_iter()
        .filter(|f| BenchmarkHarness::language_for_fixture(f.name) == "en")
        .collect();
    let mut categories: Vec<_> = std::fs::read_dir(root.join("fixtures/bench"))
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    categories.sort();
    for dir in categories.iter().filter(|p| p.is_dir()) {
        let category = dir.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        out.extend(load_pairs(dir, &format!("{category}/")));
    }
    out
}

fn words(text: &str) -> Vec<String> {
    normalize_text(text)
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn transcribe(asr: &WhisperCppBackend, rt: &tokio::runtime::Runtime, pcm: &[f32]) -> String {
    let cfg = AsrSessionConfig {
        language: Some("en".to_string()),
        ..Default::default()
    };
    rt.block_on(async {
        let session = asr.start_session(cfg).await.expect("start session");
        asr.push_audio(&session, pcm).await.expect("push audio");
        let segments = asr.poll_final(&session).await.expect("poll final");
        let _ = asr.end_session(session).await;
        segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    })
}

/// One arm's totals at one recorded lead-in.
#[derive(Default, Clone, Copy)]
struct Tally {
    clips: usize,
    first_wrong: usize,
    head_errors: usize,
    empty: usize,
}

fn main() {
    let entry = benchmark_asr_entry().unwrap_or_else(|msg| panic!("lead_in_ab: {msg}"));
    let models = models_dir();
    let use_gpu = lune_persist::AppPaths::resolve()
        .map(|p| lune_asr::gpu_backend::cuda_available(p.data_dir()))
        .unwrap_or(false);

    let own_dir = std::env::var("LUNE_BENCH_DIR").ok();
    let mut fixtures = match &own_dir {
        Some(dir) => load_pairs(Path::new(dir), ""),
        None => load_repo_fixtures(),
    };
    if let Ok(needle) = std::env::var("LUNE_BENCH_FIXTURE") {
        fixtures.retain(|f| f.name.contains(needle.as_str()));
    }
    let head_ms = own_dir.is_none().then_some(HEAD_MS);
    let print = std::env::var("LUNE_BENCH_PRINT").is_ok_and(|v| v != "0");
    let agc = std::env::var("LUNE_BENCH_AGC").is_ok_and(|v| v != "0");
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    println!(
        "\nlead_in_ab: {} clips, {} ({}), agc {}, first {HEAD_WORDS} words scored\n",
        fixtures.len(),
        entry.display_name,
        if use_gpu { "gpu" } else { "cpu" },
        if agc { "on" } else { "off" },
    );

    let mut table = [[Tally::default(); RECORDED_LEADS_MS.len()]; ARMS.len()];
    for (arm, (label, lead_in)) in ARMS.iter().enumerate() {
        // One model in memory at a time: the previous arm's backend is
        // dropped before this one loads.
        let asr = WhisperCppBackend::new(WhisperCppConfig {
            model_path: models.join(entry.model.dir()).join(entry.model.ggml_file()),
            use_gpu,
            vad_model_path: Some(
                models
                    .join(lune_core::models::DIR_WHISPER_VAD)
                    .join(lune_core::models::WHISPER_VAD_GGML),
            ),
            lead_in: *lead_in,
        })
        .unwrap_or_else(|e| panic!("lead_in_ab: ASR load failed — {e}"));

        for fx in &fixtures {
            let reference = words(&fx.reference);
            let want = &reference[..reference.len().min(HEAD_WORDS)];
            let onset = onset_sample(&fx.pcm);
            for (col, lead_ms) in RECORDED_LEADS_MS.iter().enumerate() {
                let mut pcm = utterance(&fx.pcm, onset, *lead_ms, head_ms);
                if agc {
                    pcm = with_agc(&pcm);
                }
                let text = transcribe(&asr, &rt, &pcm);
                let hyp = words(&text);
                let got = &hyp[..hyp.len().min(HEAD_WORDS)];
                let first_wrong = got.first() != want.first();
                let tally = &mut table[arm][col];
                tally.clips += 1;
                tally.first_wrong += first_wrong as usize;
                tally.head_errors += levenshtein(got, want);
                tally.empty += hyp.is_empty() as usize;
                if print && first_wrong {
                    println!(
                        "  {label:12} lead {lead_ms:>3} ms  {:28} want {:?}  got {:?}",
                        fx.name,
                        want.join(" "),
                        got.join(" ")
                    );
                }
            }
        }
    }

    for (col, lead_ms) in RECORDED_LEADS_MS.iter().enumerate() {
        println!("\n  recorded lead-in {lead_ms} ms");
        println!(
            "  {:12}  {:>7}  {:>6}  {:>5}",
            "arm", "first", "head", "empty"
        );
        for (arm, (label, _)) in ARMS.iter().enumerate() {
            let t = table[arm][col];
            let n = t.clips.max(1) as f64;
            println!(
                "  {label:12}  {:>6.1}%  {:>6.2}  {:>5}",
                100.0 * t.first_wrong as f64 / n,
                t.head_errors as f64 / n,
                t.empty
            );
        }
    }
}
