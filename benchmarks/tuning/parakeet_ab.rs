// SPDX-License-Identifier: GPL-3.0-or-later

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::path::{Path, PathBuf};
use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, Fixture, benchmark_asr_entry, load_fixtures, load_wav_f32, models_dir,
    normalize_text, workspace_root,
};
use lune_asr::{
    LeadIn, ParakeetCppBackend, ParakeetCppConfig, WhisperCppBackend, WhisperCppConfig,
};
use lune_audio::{PCEN_HOP_SIZE, PcenAgc, PcenAgcConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::ParakeetModel;

const GARBAGE_DROP: f32 = 0.25;
const RISK_FLAGS: [f32; 2] = [0.40, 0.50];
const SWEEP: [f32; 9] = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50];

#[derive(Clone, Copy)]
enum Gain {
    Off,
    Auto,
    Always,
}

const GAINS: [(&str, Gain); 3] = [
    ("off", Gain::Off),
    ("auto", Gain::Auto),
    ("always", Gain::Always),
];

fn with_gain(pcm: &[f32], gain: Gain) -> Vec<f32> {
    let mut out = pcm.to_vec();
    match gain {
        Gain::Off => {}
        Gain::Auto => {
            lune_audio::auto_gain::apply_auto(&mut out, 16_000, &PcenAgcConfig::default())
                .expect("default PCEN config is valid");
        }
        Gain::Always => {
            let mut agc = PcenAgc::new(PcenAgcConfig::default(), 16_000, PCEN_HOP_SIZE)
                .expect("default PCEN config is valid");
            let mut hops = out.chunks_exact_mut(PCEN_HOP_SIZE);
            for hop in &mut hops {
                agc.process_frame(hop);
            }
            let last = agc.last_gain();
            for s in hops.into_remainder() {
                *s *= last;
            }
        }
    }
    out
}

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

fn align(hyp: &[String], reference: &[String]) -> (usize, Vec<bool>) {
    let (m, n) = (hyp.len(), reference.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if hyp[i - 1] == reference[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
            };
        }
    }
    let mut right = vec![false; m];
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        if hyp[i - 1] == reference[j - 1] && dp[i][j] == dp[i - 1][j - 1] {
            right[i - 1] = true;
            i -= 1;
            j -= 1;
        } else if dp[i][j] == dp[i - 1][j - 1] + 1 {
            i -= 1;
            j -= 1;
        } else if dp[i][j] == dp[i - 1][j] + 1 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    (dp[m][n], right)
}

fn transcribe<B: AsrBackend>(
    asr: &B,
    rt: &tokio::runtime::Runtime,
    pcm: &[f32],
) -> Vec<AsrSegment> {
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
    })
}

#[derive(Default)]
struct Tally {
    ref_words: usize,
    errors: usize,
    right: usize,
    wrong: usize,
    right_flagged: [usize; 2],
    wrong_flagged: [usize; 2],
    risks: Vec<(f32, bool)>,
    segments: usize,
    dropped: usize,
    dropped_words: usize,
    min_confidence: f32,
    decode_secs: f64,
}

fn score<B: AsrBackend>(
    asr: &B,
    rt: &tokio::runtime::Runtime,
    fixtures: &[Fixture],
    gain: Gain,
    print: Option<&str>,
) -> Tally {
    let mut tally = Tally {
        min_confidence: 1.0,
        ..Default::default()
    };
    for fx in fixtures {
        let pcm = with_gain(&fx.pcm, gain);
        let started = Instant::now();
        let segments = transcribe(asr, rt, &pcm);
        tally.decode_secs += started.elapsed().as_secs_f64();

        let mut hyp = Vec::new();
        let mut risk = Vec::new();
        for seg in &segments {
            tally.segments += 1;
            if let Some(c) = seg.confidence {
                tally.min_confidence = tally.min_confidence.min(c);
                if c < GARBAGE_DROP {
                    tally.dropped += 1;
                    tally.dropped_words += words(&seg.text).len();
                }
            }
            for wc in &seg.word_confidences {
                for w in words(&wc.word) {
                    hyp.push(w);
                    risk.push(wc.word_error_risk);
                }
            }
        }
        let reference = words(&fx.reference);
        let (errors, right) = align(&hyp, &reference);
        tally.ref_words += reference.len();
        tally.errors += errors;
        for (ok, r) in right.iter().zip(&risk) {
            let (count, flagged) = if *ok {
                (&mut tally.right, &mut tally.right_flagged)
            } else {
                (&mut tally.wrong, &mut tally.wrong_flagged)
            };
            *count += 1;
            tally.risks.push((*r, *ok));
            for (slot, threshold) in flagged.iter_mut().zip(RISK_FLAGS) {
                *slot += (*r >= threshold) as usize;
            }
        }
        if let Some(label) = print {
            println!(
                "    {label:34} {:30} {:>5.1}%  {errors}/{}",
                fx.name,
                100.0 * errors as f64 / reference.len().max(1) as f64,
                reference.len()
            );
        }
    }
    tally
}

fn report(label: &str, t: &Tally) {
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    println!(
        "  {label:34} {:>5.2}%  {:>4}/{:<5} {:>5.1}% {:>5.1}%  {:>5.1}% {:>5.1}%  {:>4.2} {:>2}/{:<3} {:>6.1}s",
        pct(t.errors, t.ref_words),
        t.errors,
        t.ref_words,
        pct(t.wrong_flagged[0], t.wrong),
        pct(t.right_flagged[0], t.right),
        pct(t.wrong_flagged[1], t.wrong),
        pct(t.right_flagged[1], t.right),
        t.min_confidence,
        t.dropped,
        t.segments,
        t.decode_secs,
    );
}

fn sweep(t: &Tally) {
    let share = |ok: bool, threshold: f32| {
        let of: Vec<_> = t.risks.iter().filter(|(_, o)| *o == ok).collect();
        100.0 * of.iter().filter(|(r, _)| *r >= threshold).count() as f64 / of.len().max(1) as f64
    };
    for (name, ok) in [("wrong flagged", false), ("right flagged", true)] {
        print!("      {name:14}");
        for threshold in SWEEP {
            print!("  >={threshold:.2} {:>5.1}%", share(ok, threshold));
        }
        println!();
    }
}

fn main() {
    let models = models_dir();
    let use_gpu = lune_persist::AppPaths::resolve()
        .map(|p| lune_asr::gpu_backend::gpu_available(p.data_dir()))
        .unwrap_or(false);
    let vad = models
        .join(lune_core::models::DIR_WHISPER_VAD)
        .join(lune_core::models::WHISPER_VAD_GGML);

    let mut fixtures = match std::env::var("LUNE_BENCH_DIR") {
        Ok(dir) => load_pairs(Path::new(&dir), ""),
        Err(_) => load_repo_fixtures(),
    };
    if let Ok(needle) = std::env::var("LUNE_BENCH_FIXTURE") {
        fixtures.retain(|f| f.name.contains(needle.as_str()));
    }
    let print = std::env::var("LUNE_BENCH_PRINT").is_ok_and(|v| v != "0");
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let installed = ParakeetModel::default();
    let mut parakeet: Vec<PathBuf> = vec![models.join(installed.dir()).join(installed.ggml_file())];
    if let Ok(extra) = std::env::var("PARAKEET_MODELS") {
        parakeet.extend(
            extra
                .split(',')
                .filter(|p| !p.is_empty())
                .map(PathBuf::from),
        );
    }
    parakeet.retain(|p| p.is_file());

    let audio_secs: f64 = fixtures.iter().map(|f| f.audio_secs).sum();
    println!(
        "\nparakeet_ab: {} clips, {:.0} s, {}\n",
        fixtures.len(),
        audio_secs,
        if use_gpu { "gpu" } else { "cpu" }
    );
    println!(
        "  {:34} {:>6}  {:>10} {:>13}  {:>13}  {:>4} {:>6} {:>7}",
        "model / gain", "WER", "errors", "flag>=.40 w/r", "flag>=.50 w/r", "minc", "drop", "decode"
    );

    for path in &parakeet {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .trim_start_matches("ggml-")
            .to_owned();
        let asr = ParakeetCppBackend::new(ParakeetCppConfig {
            model: installed,
            model_path: path.clone(),
            use_gpu,
            vad_model_path: Some(vad.clone()),
        })
        .unwrap_or_else(|e| panic!("parakeet_ab: {name} failed to load — {e}"));
        for (gain_name, gain) in GAINS {
            let label = format!("{name} / {gain_name}");
            let tally = score(&asr, &rt, &fixtures, gain, print.then_some(label.as_str()));
            report(&label, &tally);
            sweep(&tally);
        }
    }

    match benchmark_asr_entry() {
        Ok(entry) => {
            let asr = WhisperCppBackend::new(WhisperCppConfig {
                model_path: models.join(entry.model.dir()).join(entry.model.ggml_file()),
                use_gpu,
                vad_model_path: Some(vad),
                lead_in: LeadIn::default(),
            })
            .unwrap_or_else(|e| panic!("parakeet_ab: whisper failed to load — {e}"));
            for (gain_name, gain) in GAINS {
                let label = format!("{} / {gain_name}", entry.display_name);
                let tally = score(&asr, &rt, &fixtures, gain, print.then_some(label.as_str()));
                report(&label, &tally);
                sweep(&tally);
            }
        }
        Err(msg) => println!("  whisper skipped: {msg}"),
    }
}
