//! Generate noisy variants of the clean fixtures for LLM-correction
//! stress-testing.
//!
//! Reads `fixtures/audio/librispeech_clean.wav` and
//! `fixtures/audio/custom_recording.wav`, adds reproducible Gaussian noise
//! at multiple SNR levels, and writes `<name>_noise_snr<db>.wav` + a
//! matching `.txt` reference (identical to the source transcript, since
//! noise doesn't change the ground truth).
//!
//! The existing clean and 40744 fixtures already cover the "high-confidence
//! ASR" path. These noisy variants drag raw per-word confidences into the
//! 0.5–0.8 range so the marking threshold, calibrator, runner-up alternates,
//! and LLM corrector actually fire during benchmarks.
//!
//! Run: `cargo run --release --example noisify_fixtures -p lune-voice`
//!
//! The PRNG is seeded deterministically so re-running produces bit-identical
//! output — reproducible WAVs mean reproducible WER ablations.

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::path::{Path, PathBuf};

use bench_utils::workspace_root;

/// SNR levels (dB) to generate per base fixture. 10 dB is mildly noisy,
/// 5 dB is heavily noisy but still intelligible to a human, 0 dB means
/// signal and noise have equal RMS power.
const SNR_LEVELS_DB: &[i32] = &[10, 5, 0];

/// Base fixtures to noisify. Each must exist in `fixtures/audio/` as both
/// `<name>.wav` and `<name>.txt`.
const BASE_FIXTURES: &[&str] = &["librispeech_clean", "custom_recording"];

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              NOISY FIXTURE GENERATOR                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let root = workspace_root();
    let fixtures_dir = root.join("fixtures/audio");

    for name in BASE_FIXTURES {
        let wav_path = fixtures_dir.join(format!("{name}.wav"));
        let txt_path = fixtures_dir.join(format!("{name}.txt"));

        if !wav_path.exists() {
            println!("  [skip] {wav_path:?} not found");
            continue;
        }

        println!("▸ {name}");
        let (spec, samples) = read_wav(&wav_path);
        let signal_rms = rms(&samples);
        println!(
            "    loaded {} samples, {:.1}s, signal RMS = {:.5}",
            samples.len(),
            samples.len() as f64 / spec.sample_rate as f64,
            signal_rms
        );

        let reference = std::fs::read_to_string(&txt_path).unwrap_or_else(|e| {
            panic!("read {}: {e}", txt_path.display());
        });

        for &snr_db in SNR_LEVELS_DB {
            let out_name = format!("{name}_noise_snr{snr_db}");
            let out_wav = fixtures_dir.join(format!("{out_name}.wav"));
            let out_txt = fixtures_dir.join(format!("{out_name}.txt"));

            let noisy = add_gaussian_noise(&samples, signal_rms, snr_db);
            write_wav(&out_wav, spec, &noisy);
            std::fs::write(&out_txt, &reference).expect("write reference");

            let noise_rms = rms_diff(&samples, &noisy);
            let measured_snr = 20.0 * (signal_rms / noise_rms).log10();
            println!(
                "    {:>5} dB target → {:.1} dB measured → {}",
                snr_db,
                measured_snr,
                out_wav.file_name().unwrap().to_string_lossy()
            );
        }
        println!();
    }

    println!("  Done. Re-run `calibrate_confidence` to pick up the new fixtures.");
    println!();
}

// ── WAV I/O ─────────────────────────────────────────────────────────────────

fn read_wav(path: &Path) -> (hound::WavSpec, Vec<f32>) {
    let reader =
        hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono");
    assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz");

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(Result::ok)
            .collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    (spec, samples)
}

fn write_wav(path: &Path, spec: hound::WavSpec, samples: &[f32]) {
    // Write as 16-bit PCM regardless of input format so the output is
    // compact and uniformly loadable by the rest of the benchmark suite.
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: spec.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, out_spec)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    for &s in samples {
        // Hard-clip to i16 range to avoid wrap-around from added noise.
        let clipped = s.clamp(-1.0, 1.0);
        let int_sample = (clipped * 32_767.0) as i16;
        writer.write_sample(int_sample).expect("write sample");
    }
    writer.finalize().expect("finalize");
}

// ── Noise generation ────────────────────────────────────────────────────────

/// Add reproducible Gaussian noise to `samples` such that the resulting
/// noise RMS hits the target SNR relative to `signal_rms`.
///
/// `snr_db = 20 * log10(signal_rms / noise_rms)`, so
/// `noise_rms = signal_rms * 10^(-snr_db / 20)`.
fn add_gaussian_noise(samples: &[f32], signal_rms: f32, snr_db: i32) -> Vec<f32> {
    let noise_rms = signal_rms * 10.0_f32.powf(-snr_db as f32 / 20.0);
    // Seed includes the SNR so different levels use different noise streams.
    let mut rng = XorShift64::new(0xD1CE_D1CE_D1CE_D1CE ^ (snr_db as u64).wrapping_mul(0x9E37));
    samples
        .iter()
        .map(|&s| s + rng.next_gaussian() * noise_rms)
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = a.iter().zip(b).map(|(&x, &y)| (y - x).powi(2)).sum();
    (sum_sq / a.len() as f32).sqrt()
}

// ── Deterministic PRNG (xorshift64 + Box-Muller) ───────────────────────────

struct XorShift64 {
    state: u64,
    // Cache the second Box-Muller sample so every other call is free.
    cached: Option<f32>,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
            cached: None,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_uniform(&mut self) -> f32 {
        // 24 significand bits → [0, 1) in f32 without bias.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal sample via Box-Muller. Returns both samples from
    /// each polar pair, caching the second for the next call.
    fn next_gaussian(&mut self) -> f32 {
        if let Some(cached) = self.cached.take() {
            return cached;
        }
        // Reject u1 == 0 to avoid log(0). Loop is bounded — probability of
        // a zero draw is 2^-24 per attempt.
        let u1 = loop {
            let v = self.next_uniform();
            if v > 0.0 {
                break v;
            }
        };
        let u2 = self.next_uniform();
        let mag = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * u2;
        let z0 = mag * angle.cos();
        let z1 = mag * angle.sin();
        self.cached = Some(z1);
        z0
    }
}

#[allow(dead_code)]
fn _ignored_workaround() -> PathBuf {
    // Kept so `PathBuf` stays imported when the rest of the file uses only
    // `Path` — dead-code warnings on an example target are promoted to
    // errors in the lefthook `clippy -D warnings` gate.
    PathBuf::new()
}
