//! Pipeline latency benchmark — measures each stage of the dictation pipeline
//! by simulating realistic audio frame arrival and ASR behavior.
//!
//! Run: cargo run --release --example pipeline_latency -p lune-app
//!
//! This benchmark does NOT use SessionManager (which requires a real microphone).
//! Instead it directly exercises each stage in isolation and in sequence to
//! produce accurate latency numbers.

use std::time::{Duration, Instant};

use lune_asr::segment::SegmentTracker;

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

    // ── Stage 3: Segment tracker ─────────────────────────────────────────────
    bench_segment_tracker();

    // ── Stage 4: Silence timeout simulation ──────────────────────────────────
    bench_silence_timeout();

    // ── Stage 5: Full pipeline simulation ────────────────────────────────────
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
// Stage 3: Segment tracker micro-benchmark
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_segment_tracker() {
    println!("── Stage 3: Segment Tracker ──────────────────────────────────────");

    let iterations = 100_000u64;

    // push_partial throughput (simulates ASR returning updated text).
    let start = Instant::now();
    let mut tracker = SegmentTracker::new(500);
    for i in 0..iterations {
        tracker.push_partial(format!("partial text update {i}"), Some(i * 30), None);
    }
    let push_elapsed = start.elapsed();

    // check_silence_finalization throughput.
    let start = Instant::now();
    for _ in 0..iterations {
        tracker.check_silence_finalization();
    }
    let check_elapsed = start.elapsed();

    // drain.
    let start = Instant::now();
    let drained = tracker.drain();
    let drain_elapsed = start.elapsed();

    println!(
        "  push_partial × {iterations}:               {:>6.2} ms ({:.0} ns/op)",
        push_elapsed.as_secs_f64() * 1000.0,
        push_elapsed.as_nanos() as f64 / iterations as f64
    );
    println!(
        "  check_silence × {iterations}:              {:>6.2} ms ({:.0} ns/op)",
        check_elapsed.as_secs_f64() * 1000.0,
        check_elapsed.as_nanos() as f64 / iterations as f64
    );
    println!(
        "  drain ({} segments):                    {:>6.3} ms",
        drained.len(),
        drain_elapsed.as_secs_f64() * 1000.0
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 4: Silence timeout measurement
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_silence_timeout() {
    println!("── Stage 4: Silence Timeout ─────────────────────────────────────");

    let timeouts = [500, 300, 200, 150, 100];

    for timeout_ms in &timeouts {
        let mut tracker = SegmentTracker::new(*timeout_ms);
        tracker.push_partial("hello world".into(), Some(0), None);

        // Wait for the timeout, then check.
        std::thread::sleep(Duration::from_millis(*timeout_ms + 10));

        let start = Instant::now();
        tracker.check_silence_finalization();
        let check_time = start.elapsed();

        let segments = tracker.drain();
        let finalized = segments.first().map(|s| s.is_final).unwrap_or(false);

        println!(
            "  Timeout {timeout_ms:>3} ms: finalized={finalized}  check_cost={:.0} ns",
            check_time.as_nanos()
        );
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 5: Full pipeline simulation (no real mic, no real model)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_full_pipeline_sim() {
    println!("── Stage 5: Full Pipeline Simulation ─────────────────────────────");
    println!("  Simulates: audio frames → ASR accumulation → inference → segment");
    println!("             → silence timeout → finalize → delivery");
    println!();

    let configs = [
        ("Previous defaults", 16000, 500u64, 100u64),
        ("Current (applied)", 4800, 200, 100),
        ("Ultra-low", 1600, 150, 100),
    ];

    for (label, trigger_threshold, silence_ms, inference_ms) in &configs {
        let frame_size = 480usize;
        let sample_rate = 16000usize;

        // Simulate 1.5 seconds of continuous speech.
        let speech_frames = 50; // 50 × 30ms = 1500ms
        let speech_text = "So I was thinking we should go to the store.";

        let pipeline_start = Instant::now();

        // Phase 1: Accumulate audio until inference triggers.
        let mut accumulated_samples = 0usize;
        let mut frames_sent = 0usize;
        let mut first_inference_at: Option<Duration> = None;

        let mut tracker = SegmentTracker::new(*silence_ms);

        while frames_sent < speech_frames {
            // Simulate frame arrival at real-time pace.
            // In the real pipeline, frames arrive every 30ms from cpal.
            // Here we just track the simulated time.
            accumulated_samples += frame_size;
            frames_sent += 1;

            // Check if inference should trigger.
            if accumulated_samples >= *trigger_threshold {
                if first_inference_at.is_none() {
                    let simulated_audio_time = Duration::from_millis(
                        (frames_sent * frame_size * 1000 / sample_rate) as u64,
                    );
                    first_inference_at = Some(simulated_audio_time);
                }

                // Simulate inference latency.
                std::thread::sleep(Duration::from_millis(*inference_ms));
                accumulated_samples = 0;

                // ASR returns a partial.
                tracker.push_partial(speech_text.to_string(), Some(frames_sent as u64 * 30), None);
            }
        }

        let speech_end =
            Duration::from_millis((frames_sent * frame_size * 1000 / sample_rate) as u64);

        // Phase 2: Silence timeout.
        std::thread::sleep(Duration::from_millis(*silence_ms + 10));
        tracker.check_silence_finalization();

        // Phase 3: Drain finalized segments.
        let segments = tracker.drain();
        let has_final = segments.iter().any(|s| s.is_final);

        // Phase 4: Simulate delivery (clipboard + paste).
        let delivery_start = Instant::now();
        // arboard::Clipboard::new + set_text + enigo Ctrl+V ≈ 5ms
        std::thread::sleep(Duration::from_millis(5));
        let delivery_time = delivery_start.elapsed();

        let total_wall = pipeline_start.elapsed();

        // Calculate simulated end-to-end latency.
        // The user perceives: speech_end + silence_timeout + delivery.
        let perceived_latency =
            speech_end.as_millis() as u64 + *silence_ms + delivery_time.as_millis() as u64;

        println!("  [{label}]");
        println!(
            "    Trigger threshold: {} samples ({:.0} ms)",
            trigger_threshold,
            *trigger_threshold as f64 / sample_rate as f64 * 1000.0
        );
        println!("    Silence timeout:   {silence_ms} ms");
        println!("    Inference latency: {inference_ms} ms (simulated)");
        println!(
            "    First inference:   {} ms after speech start",
            first_inference_at.map(|d| d.as_millis()).unwrap_or(0)
        );
        println!("    Speech duration:   {} ms", speech_end.as_millis());
        println!("    Finalized:         {has_final}");
        println!("    Wall clock:        {} ms", total_wall.as_millis());
        println!("    ──────────────────────────────");
        println!(
            "    PERCEIVED LATENCY: {perceived_latency} ms  (speech_end + silence + delivery)"
        );
        println!(
            "    After-speech wait: {} ms  (silence_timeout + delivery)",
            *silence_ms + delivery_time.as_millis() as u64
        );
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
    println!("    Silence timeout:         500 ms  ← biggest knob");
    println!("    Inference (final poll):  ~100 ms  (Parakeet TDT 0.6B CPU)");
    println!("    Dictionary apply:          <1 ms");
    println!("    LLM post-process:           0 ms  (bypassed)");
    println!("    Delivery (paste):          ~5 ms");
    println!("    ─────────────────────────────────");
    println!("    CURRENT TOTAL:           ~605 ms");
    println!();
    println!("  With optimized settings:");
    println!("    Silence timeout:         200 ms");
    println!("    Inference:              ~100 ms");
    println!("    Delivery:                 ~5 ms");
    println!("    ─────────────────────────────────");
    println!("    OPTIMIZED TOTAL:         ~305 ms");
    println!();
    println!("  Key insight: the inference trigger threshold (currently 16000");
    println!("  samples = 1 sec) does NOT add to after-speech wait — it only");
    println!("  delays when partials appear during speech. For the final");
    println!("  delivery, the bottleneck is silence_timeout + one last");
    println!("  inference poll.");
    println!();
    println!("  RECOMMENDATIONS:");
    println!("    1. Lower silence_timeout: 500 → 200 ms  (save 300 ms)");
    println!("    2. Lower inference trigger: 16000 → 4800  (faster partials)");
    println!("    3. Both: total after-speech ≈ 305 ms (feels instant)");
    println!();
}
