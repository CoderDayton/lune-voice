//! Mel-spectrogram feature extraction throughput benchmark.
//!
//! Profiles `extract_features()` across realistic audio lengths used by the
//! Parakeet TDT pipeline — from single inference-trigger chunks (300 ms) up
//! to long dictation sessions (30 s). Validates real-time factor (RTF) and
//! measures how much of the inference budget mel preprocessing consumes.
//!
//! Run with:
//!   cargo run --example mel_throughput -p lune-asr --release

use std::time::Instant;

use lune_asr::extract_features;

const SR: usize = 16_000;
const N_FFT: usize = 512;
const HOP: usize = 160;
const N_MELS: usize = 128;

fn banner(s: &str) {
    let width = 66;
    println!("\n╔{:═<width$}╗", "");
    println!("║  {:<width$}║", s, width = width - 2);
    println!("╚{:═<width$}╝\n", "");
}

fn section(s: &str) {
    println!(
        "── {} {}",
        s,
        "─".repeat(67usize.saturating_sub(s.len() + 4))
    );
}

/// Expected output frame count for a given number of input samples.
fn expected_frames(n_samples: usize) -> usize {
    let padded = N_FFT / 2 + n_samples + N_FFT / 2;
    (padded - N_FFT) / HOP + 1
}

/// Time `f` for `reps` repetitions; return per-call `Duration`.
fn bench<F: Fn()>(reps: u32, f: F) -> std::time::Duration {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed() / reps
}

fn main() {
    banner("MEL-SPECTROGRAM THROUGHPUT BENCHMARK");

    // ── Stage 1: OnceLock cache initialization ────────────────────────────────
    section("Stage 1: Static Cache Initialization (OnceLock)");
    println!("  MEL_FILTERBANK and HANN_WINDOW are computed once per process.");
    println!("  First call pays the init cost; subsequent calls are free.\n");

    // Use a 300ms chunk — the current inference trigger threshold.
    let chunk_300 = vec![0.1f32; SR * 300 / 1000];

    let t0 = Instant::now();
    let _ = extract_features(&chunk_300);
    let cold = t0.elapsed();

    let t1 = Instant::now();
    let _ = extract_features(&chunk_300);
    let warm = t1.elapsed();

    println!(
        "  Cold call (filterbank + window init): {:>8.3} ms",
        cold.as_secs_f64() * 1000.0
    );
    println!(
        "  Warm call (OnceLock cache hit):       {:>8.3} ms",
        warm.as_secs_f64() * 1000.0
    );
    println!(
        "  One-time init overhead:               {:>8.3} ms\n",
        cold.saturating_sub(warm).as_secs_f64() * 1000.0
    );

    // ── Stage 2: Latency scaling by audio length ──────────────────────────────
    section("Stage 2: Extraction Latency by Audio Length");
    println!(
        "  {:<10}  {:>9}  {:>9}  {:>11}  {:>14}  {:>8}",
        "Duration", "Samples", "Frames", "Latency", "Throughput", "RTF"
    );
    println!("  {}", "─".repeat(72));

    let lengths_ms: &[usize] = &[30, 100, 300, 500, 1_000, 2_000, 5_000, 10_000, 30_000];

    for &ms in lengths_ms {
        let n_samples = SR * ms / 1_000;
        let audio = vec![0.1f32; n_samples];

        // One warmup pass (cache already hot after Stage 1).
        let _ = extract_features(&audio);

        // Fewer reps for long audio to keep runtime reasonable.
        let reps = if ms >= 10_000 { 5 } else { 50 };
        let per_call = bench(reps, || {
            let _ = extract_features(&audio);
        });

        let latency_ms = per_call.as_secs_f64() * 1000.0;
        let audio_ms = ms as f64;
        let samples_per_sec = n_samples as f64 / per_call.as_secs_f64();
        // RTF < 1.0 = faster than real-time (lower is better).
        let rtf = latency_ms / audio_ms;
        let n_frames = expected_frames(n_samples);

        println!(
            "  {:<10}  {:>9}  {:>9}  {:>9.3} ms  {:>11.0} S/s  {:>7.4}x",
            format!("{} ms", ms),
            n_samples,
            n_frames,
            latency_ms,
            samples_per_sec,
            rtf
        );
    }

    // ── Stage 3: Sustained streaming (back-to-back 300ms chunks) ─────────────
    section("Stage 3: Sustained Streaming Throughput");
    println!("  Simulates continuous inference: 300ms chunks arriving back-to-back.");
    println!("  Matches the real pipeline: poll_results fires every ~300ms.\n");

    let n_iters = 100u32;
    let t = Instant::now();
    for _ in 0..n_iters {
        let _ = extract_features(&chunk_300);
    }
    let total = t.elapsed();
    let per_chunk = total / n_iters;
    let audio_total_ms = 300.0 * n_iters as f64;
    let wall_ms = total.as_secs_f64() * 1000.0;
    let sustained_rtf = wall_ms / audio_total_ms;

    println!("  Chunks processed:    {}", n_iters);
    println!(
        "  Audio simulated:     {:.0} ms ({:.1} s)",
        audio_total_ms,
        audio_total_ms / 1000.0
    );
    println!("  Total wall time:     {:.2} ms", wall_ms);
    println!(
        "  Per-chunk (mean):    {:.3} ms  ({} frames out)",
        per_chunk.as_secs_f64() * 1000.0,
        expected_frames(chunk_300.len())
    );
    println!(
        "  Sustained RTF:       {:.4}x  → {:.1}% of 300ms budget consumed",
        sustained_rtf,
        sustained_rtf * 100.0
    );

    // ── Stage 4: Percentile distribution (300ms chunk, 100 samples) ──────────
    section("Stage 4: Latency Percentiles (300ms chunk, 100 samples)");

    let mut timings: Vec<f64> = (0..100)
        .map(|_| {
            let t = Instant::now();
            let _ = extract_features(&chunk_300);
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pct = |p: usize| timings[timings.len() * p / 100];
    println!(
        "  min={:.3}  p50={:.3}  p90={:.3}  p99={:.3}  max={:.3}  (ms)",
        timings[0],
        pct(50),
        pct(90),
        pct(99),
        timings.last().unwrap()
    );

    // ── Stage 5: Output shape and numerical stability ─────────────────────────
    section("Stage 5: Output Shape and Numerical Stability");
    println!(
        "  {:<10}  {:<20}  {:<18}  All finite",
        "Duration", "Shape (frames × mels)", "Expected frames"
    );
    println!("  {}", "─".repeat(66));

    for &ms in &[0usize, 30, 300, 1_000, 5_000] {
        let n_samples = SR * ms / 1_000;
        let audio = vec![0.1f32; n_samples];
        let features = extract_features(&audio);
        let shape = features.shape();
        let expected = expected_frames(n_samples);
        let all_finite = features.iter().all(|v| v.is_finite());
        let shape_str = format!("({}, {})", shape[0], shape[1]);
        println!(
            "  {:<10}  {:<20}  {:<18}  {}",
            format!("{} ms", ms),
            shape_str,
            expected,
            all_finite
        );
        // Sanity assertions — fail loudly if the pipeline would produce garbage.
        assert_eq!(shape[1], N_MELS, "mel bins must be {N_MELS}");
        if ms > 0 {
            assert!(all_finite, "NaN/Inf in features for {ms}ms audio");
        }
    }

    // ── Budget analysis ───────────────────────────────────────────────────────
    banner("PREPROCESSING BUDGET ANALYSIS");

    let p50_ms = pct(50);
    // Measured from pipeline_latency benchmark (Parakeet TDT 0.6B, CPU, INT8).
    let inference_est_ms = 100.0f64;
    let budget_ms = 300.0f64;
    let headroom_ms = budget_ms - p50_ms - inference_est_ms;

    println!("  Mel extraction consumes a fraction of the 300ms inference budget.");
    println!("  The bottleneck is TDT ONNX inference, not preprocessing.\n");

    println!("  300ms inference cycle budget:");
    println!(
        "    Mel extraction (p50):   {:>7.3} ms  ({:.1}%)",
        p50_ms,
        p50_ms / budget_ms * 100.0
    );
    println!(
        "    TDT inference (est):    {:>7.1} ms  ({:.1}%)",
        inference_est_ms,
        inference_est_ms / budget_ms * 100.0
    );
    println!(
        "    Segment tracker ops:      {:>5.3} ms  (<0.1%)",
        0.077 * expected_frames(chunk_300.len()) as f64 / 1000.0
    );
    println!("    {}", "─".repeat(48));
    println!(
        "    Remaining headroom:     {:>7.1} ms  ({:.1}%)",
        headroom_ms,
        headroom_ms / budget_ms * 100.0
    );
    println!();
    println!("  Throughput upper bound (mel only, no inference):");
    println!(
        "    {:.0} chunks/s  →  {:.0}x real-time",
        1000.0 / p50_ms,
        1000.0 / p50_ms / (1000.0 / 300.0)
    );
}
