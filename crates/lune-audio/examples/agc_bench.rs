//! AGC frontend benchmark: PCEN off vs on.
//!
//! Compares Earshot VAD behavior with and without the PCEN-style adaptive
//! AGC frontend across a panel of clean and noisy fixtures.
//!
//! Per fixture, runs both branches frame-by-frame (256-sample hops at 16 kHz)
//! and reports per-frame VAD score distribution, speech-frame counts, mean
//! AGC gain, and per-frame CPU cost.
//!
//! Usage:
//!   cargo run --release --example agc_bench -p lune-audio
//!   FIXTURES=path1.wav,path2.wav cargo run --release --example agc_bench -p lune-audio
//!   PANEL=all cargo run --release --example agc_bench -p lune-audio

use std::path::{Path, PathBuf};
use std::time::Instant;

use earshot::Detector;
use lune_audio::{PcenAgc, PcenAgcConfig};

const HOP_SIZE: usize = lune_audio::PCEN_HOP_SIZE;
const SAMPLE_RATE: u32 = 16_000;
const SPEECH_START: f32 = 0.5;

#[derive(Clone, Copy, Default)]
struct FrameStat {
    score: f32,
    is_speech_decision: bool,
}

#[derive(Default)]
struct BranchResult {
    frames: Vec<FrameStat>,
    speech_decisions: usize,
    avg_score: f32,
    p50_score: f32,
    p95_score: f32,
    elapsed_us: u128,
}

impl BranchResult {
    fn finalize(&mut self) {
        if self.frames.is_empty() {
            return;
        }
        let n = self.frames.len();
        let mut sorted: Vec<f32> = self.frames.iter().map(|f| f.score).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = sorted.iter().sum::<f32>() / n as f32;
        let p50 = sorted[n / 2];
        let p95 = sorted[(n * 95 / 100).min(n - 1)];
        self.avg_score = mean;
        self.p50_score = p50;
        self.p95_score = p95;
        self.speech_decisions = self.frames.iter().filter(|f| f.is_speech_decision).count();
    }
}

fn run_branch(samples: &[f32], pcen: Option<PcenAgcConfig>) -> BranchResult {
    let mut agc = pcen.map(|cfg| PcenAgc::new(cfg, SAMPLE_RATE, HOP_SIZE).expect("build agc"));
    let mut detector = Detector::default();
    let mut work = vec![0.0f32; HOP_SIZE];
    let mut out = BranchResult::default();
    out.frames.reserve(samples.len() / HOP_SIZE + 1);

    let t0 = Instant::now();
    for chunk in samples.chunks(HOP_SIZE) {
        if chunk.len() < HOP_SIZE {
            break;
        }
        work.clear();
        work.extend_from_slice(chunk);
        if let Some(a) = agc.as_mut() {
            a.process_frame(&mut work);
        }
        let score = detector.predict_f32(&work);
        out.frames.push(FrameStat {
            score,
            is_speech_decision: score >= SPEECH_START,
        });
    }
    out.elapsed_us = t0.elapsed().as_micros();
    out.finalize();
    out
}

fn load_wav_f32(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, SAMPLE_RATE, "WAV must be 16kHz");
    assert_eq!(spec.channels, 1, "WAV must be mono");
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.expect("read sample") as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("read sample"))
            .collect(),
    }
}

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/audio")
}

fn default_panel() -> Vec<&'static str> {
    vec![
        "librispeech_clean.wav",
        "librispeech_clean_noise_snr10.wav",
        "librispeech_clean_noise_snr5.wav",
        "librispeech_clean_noise_snr0.wav",
        "librispeech_40744.wav",
    ]
}

fn full_panel() -> Vec<String> {
    let dir = fixtures_root();
    let mut out: Vec<String> = std::fs::read_dir(&dir)
        .expect("fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".wav"))
        .collect();
    out.sort();
    out
}

fn resolve_fixtures() -> Vec<PathBuf> {
    if let Ok(env) = std::env::var("FIXTURES") {
        return env.split(',').map(|s| PathBuf::from(s.trim())).collect();
    }
    let panel = std::env::var("PANEL").unwrap_or_default();
    let root = fixtures_root();
    let names: Vec<String> = if panel == "all" {
        full_panel()
    } else {
        default_panel().into_iter().map(String::from).collect()
    };
    names.into_iter().map(|n| root.join(n)).collect()
}

fn fmt_pct(num: usize, den: usize) -> String {
    if den == 0 {
        "  -".into()
    } else {
        format!("{:>4.1}%", 100.0 * num as f32 / den as f32)
    }
}

fn main() {
    let fixtures = resolve_fixtures();
    println!("AGC bench: PCEN off vs on  ·  hop=256  ·  threshold={SPEECH_START}");
    println!("Fixtures: {}\n", fixtures.len());

    println!(
        "{:<38}  {:>6}  {:>9}  {:>9}  {:>9}  {:>7}  {:>7}",
        "Fixture", "frames", "spch_off", "spch_on", "Δspch", "us/fr off", "us/fr on"
    );
    println!("{}", "─".repeat(105));

    let mut tot_frames = 0usize;
    let mut tot_off = 0usize;
    let mut tot_on = 0usize;
    let mut tot_us_off: u128 = 0;
    let mut tot_us_on: u128 = 0;
    let mut score_p50_off: Vec<f32> = vec![];
    let mut score_p50_on: Vec<f32> = vec![];

    for path in &fixtures {
        if !path.exists() {
            eprintln!("missing: {}", path.display());
            continue;
        }
        let samples = load_wav_f32(path);
        let off = run_branch(&samples, None);
        let on = run_branch(&samples, Some(PcenAgcConfig::default()));
        let frames = off.frames.len();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let display = if name.len() > 36 { &name[..36] } else { &name };

        let us_off = off.elapsed_us as f64 / frames.max(1) as f64;
        let us_on = on.elapsed_us as f64 / frames.max(1) as f64;
        let delta = on.speech_decisions as i64 - off.speech_decisions as i64;
        println!(
            "{:<38}  {:>6}  {:>9}  {:>9}  {:>+9}  {:>7.2}  {:>7.2}",
            display, frames, off.speech_decisions, on.speech_decisions, delta, us_off, us_on
        );

        tot_frames += frames;
        tot_off += off.speech_decisions;
        tot_on += on.speech_decisions;
        tot_us_off += off.elapsed_us;
        tot_us_on += on.elapsed_us;
        score_p50_off.push(off.p50_score);
        score_p50_on.push(on.p50_score);
    }

    println!("{}", "─".repeat(105));
    let us_off = tot_us_off as f64 / tot_frames.max(1) as f64;
    let us_on = tot_us_on as f64 / tot_frames.max(1) as f64;
    println!(
        "{:<38}  {:>6}  {:>9}  {:>9}  {:>+9}  {:>7.2}  {:>7.2}",
        "TOTAL",
        tot_frames,
        tot_off,
        tot_on,
        tot_on as i64 - tot_off as i64,
        us_off,
        us_on
    );

    println!();
    println!("Speech-frame ratio:");
    println!(
        "  off: {} / {}  ({})",
        tot_off,
        tot_frames,
        fmt_pct(tot_off, tot_frames)
    );
    println!(
        "  on : {} / {}  ({})",
        tot_on,
        tot_frames,
        fmt_pct(tot_on, tot_frames)
    );

    let overhead_pct = if us_off > 0.0 {
        100.0 * (us_on - us_off) / us_off
    } else {
        0.0
    };
    println!();
    println!("CPU per frame:");
    println!("  off: {:>6.2} us", us_off);
    println!("  on : {:>6.2} us  ({:+.1}%)", us_on, overhead_pct);

    if !score_p50_off.is_empty() {
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        println!();
        println!("Mean per-fixture median VAD score:");
        println!("  off: {:.3}", mean(&score_p50_off));
        println!("  on : {:.3}", mean(&score_p50_on));
    }
}
