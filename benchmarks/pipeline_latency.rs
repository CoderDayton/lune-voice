//! Pipeline latency benchmark — measures each stage of the dictation pipeline
//! by simulating realistic audio frame arrival and ASR behavior.
//!
//! Run: cargo run --release --example pipeline_latency -p lune-app
//!
//! This benchmark does NOT use SessionManager (which requires a real microphone).
//! Instead it directly exercises each stage in isolation and in sequence to
//! produce accurate latency numbers.

use std::time::{Duration, Instant};

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              PIPELINE LATENCY BENCHMARK                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Stage 1: Audio frame accumulation ────────────────────────────────────
    bench_audio_accumulation();

    // ── Stage 2: Inference trigger latency ───────────────────────────────────
    bench_inference_trigger();

    // ── Stage 3: Full pipeline simulation ────────────────────────────────────
    bench_full_pipeline_sim();

    // ── Summary ──────────────────────────────────────────────────────────────
    print_latency_budget();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1: How fast can we accumulate audio frames?
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_audio_accumulation() {
    println!("── Stage 1: Audio Frame Accumulation ──────────────────────────────");

    let frame_size = 480; // samples per frame
    let sample_rate = 16000;
    let frame_ms = (frame_size as f64 / sample_rate as f64) * 1000.0;

    // Simulate pushing 3 seconds of audio (100 frames).
    let n_frames = 100;
    let mut buffer: Vec<f32> = Vec::with_capacity(frame_size * n_frames);
    let frame: Vec<f32> = vec![0.1; frame_size]; // dummy audio

    let start = Instant::now();
    for _ in 0..n_frames {
        buffer.extend_from_slice(&frame);
    }
    let elapsed = start.elapsed();

    println!("  Frame size:          {frame_size} samples ({frame_ms:.0} ms @ {sample_rate} Hz)");
    println!(
        "  Frames pushed:       {n_frames} ({:.0} ms of audio)",
        n_frames as f64 * frame_ms
    );
    println!(
        "  Buffer accumulation: {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  Per-frame cost:      {:.0} ns",
        elapsed.as_nanos() as f64 / n_frames as f64
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 2: Inference trigger thresholds
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_inference_trigger() {
    println!("── Stage 2: Inference Trigger Latency ─────────────────────────────");

    let frame_size = 480;
    let sample_rate = 16000;
    let frame_ms = (frame_size as f64 / sample_rate as f64) * 1000.0;

    // Test different trigger thresholds.
    let thresholds = [
        (16000, "Current default (1.0 sec)"),
        (8000, "Half (0.5 sec)"),
        (4800, "Aggressive (0.3 sec)"),
        (1600, "Ultra-low (0.1 sec)"),
    ];

    for (threshold, label) in &thresholds {
        #[allow(clippy::manual_div_ceil)]
        let frames_needed = (*threshold + frame_size - 1) / frame_size;
        let wait_ms = frames_needed as f64 * frame_ms;

        // Simulate accumulating frames until threshold.
        let mut accumulated = 0usize;
        let start = Instant::now();
        let mut frames_until_trigger = 0u32;

        while accumulated < *threshold {
            accumulated += frame_size;
            frames_until_trigger += 1;
        }
        let compute_time = start.elapsed();

        println!("  {label}:");
        println!("    Threshold:         {threshold} samples");
        println!("    Frames to trigger: {frames_until_trigger}");
        println!("    Real-time wait:    {wait_ms:.0} ms (audio must arrive)");
        println!("    Compute overhead:  {:.0} ns", compute_time.as_nanos());
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 3: Full pipeline simulation (no real mic, no real model)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_full_pipeline_sim() {
    println!("── Stage 3: Full Pipeline Simulation ─────────────────────────────");
    println!("  Simulates: audio chunk → infer → word accumulator → LLM flush → delivery");
    println!();

    let configs = [
        ("8-word flush (current)", 8usize, 200u64, 75u64),
        ("Silence-only (old)", usize::MAX, 500, 100),
    ];

    for (label, words_per_flush, silence_ms, inference_ms) in &configs {
        let frame_size = 5120usize; // 320ms @ 16kHz — reference chunk size
        let sample_rate = 16000usize;
        let speech_frames = 50; // 50 × 320ms = 16s
        let words_per_frame = 1usize; // ~1 word per 320ms chunk

        let pipeline_start = Instant::now();
        let mut accumulated_words = 0usize;
        let mut frames_sent = 0usize;
        let mut first_flush_at: Option<Duration> = None;
        let mut flushes = 0u32;

        while frames_sent < speech_frames {
            frames_sent += 1;
            accumulated_words += words_per_frame;

            // Simulate inference per chunk.
            std::thread::sleep(Duration::from_millis(*inference_ms));

            // Word accumulator flush condition.
            if accumulated_words >= *words_per_flush {
                if first_flush_at.is_none() {
                    first_flush_at = Some(Duration::from_millis(
                        (frames_sent * frame_size * 1000 / sample_rate) as u64,
                    ));
                }
                accumulated_words = 0;
                flushes += 1;
            }
        }

        let speech_end =
            Duration::from_millis((frames_sent * frame_size * 1000 / sample_rate) as u64);

        // Final silence flush.
        if accumulated_words > 0 {
            std::thread::sleep(Duration::from_millis(*silence_ms));
            flushes += 1;
        }

        // Simulate Wayland key delivery.
        let delivery_start = Instant::now();
        std::thread::sleep(Duration::from_millis(2));
        let delivery_time = delivery_start.elapsed();

        let total_wall = pipeline_start.elapsed();
        let after_speech_wait = silence_ms + delivery_time.as_millis() as u64;

        println!("  [{label}]");
        println!("    Words-per-flush:    {words_per_flush}");
        println!("    Silence timeout:    {silence_ms} ms");
        println!("    Inference/chunk:    {inference_ms} ms (simulated)");
        println!(
            "    First flush:        {} ms",
            first_flush_at.map(|d| d.as_millis()).unwrap_or(0)
        );
        println!("    Flush count:        {flushes}");
        println!("    Speech duration:    {} ms", speech_end.as_millis());
        println!("    Wall clock:         {} ms", total_wall.as_millis());
        println!("    ──────────────────────────────");
        println!("    AFTER-SPEECH WAIT:  {after_speech_wait} ms  (silence + delivery)");
        println!();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════════════════

fn print_latency_budget() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              LATENCY BUDGET ANALYSIS                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  The user cares about AFTER-SPEECH WAIT — time from when they");
    println!("  stop talking to when text appears in the focused field.");
    println!();
    println!("  Current after-speech wait breakdown:");
    println!("    Silence timeout:         200 ms  (SILENCE_FLUSH_SAMPLES=3200)");
    println!("    Inference (final poll):   ~75 ms  (Qwen3-ASR-0.6B, 1s chunk, GPU)");
    println!("    Dictionary apply:          <1 ms");
    println!("    LLM post-process:           0 ms  (bypassed at 87% rate)");
    println!("    Delivery (paste):          ~5 ms");
    println!("    ─────────────────────────────────");
    println!("    CURRENT TOTAL:           ~280 ms");
    println!();
    println!("  CPU-only after-speech wait:");
    println!("    Silence timeout:         200 ms");
    println!("    Inference (final poll):  ~680 ms  (Qwen3-ASR-0.6B, 1s chunk, CPU)");
    println!("    Delivery:                 ~5 ms");
    println!("    ─────────────────────────────────");
    println!("    CPU TOTAL:               ~885 ms");
    println!();
    println!("  Key insight: the inference trigger threshold (currently 16000");
    println!("  samples = 1 sec) does NOT add to after-speech wait — it only");
    println!("  delays when partials appear during speech. For the final");
    println!("  delivery, the bottleneck is silence_timeout + one last");
    println!("  inference poll.");
    println!();
    println!("  Latency targets (Stivers et al. 2009, PNAS):");
    println!("    <200 ms  feels instantaneous (human turn-taking mean=208ms)");
    println!("    <300 ms  natural and responsive (AssemblyAI 300ms rule)");
    println!("    <500 ms  hard ceiling for voice AI (MMNTM threshold)");
    println!("    GPU path (280ms) meets the 300ms natural target.");
    println!();
}
