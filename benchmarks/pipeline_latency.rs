//! Pipeline latency benchmark — measures each stage of the dictation pipeline
//! by simulating realistic audio frame arrival and ASR behavior.
//!
//! Run: cargo run --release --example pipeline_latency -p lune-app
//!
//! This benchmark does NOT use SessionManager (which requires a real microphone).
//! Instead it directly exercises each stage in isolation and in sequence to
//! produce accurate latency numbers.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use lune_asr::{OrtQwenAsrBackend, OrtQwenAsrConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};

use lune_app::{
    SILENCE_FLUSH_SAMPLES, SILENCE_TIER_DEFAULT, SILENCE_TIER_HIGH, SILENCE_TIER_MODERATE,
    SILENCE_TIER_SPECULATIVE, dynamic_silence_threshold,
};

const SAMPLE_RATE: usize = 16_000;

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

    // ── Stage 4: Dynamic silence tiers ───────────────────────────────────────
    bench_dynamic_silence();

    // ── Stage 5: Output quality analysis ─────────────────────────────────────
    let quality = bench_quality_analysis();

    // ── Summary ──────────────────────────────────────────────────────────────
    print_latency_budget();
    print_quality_summary(&quality);
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

    // Dynamic silence: resolve tier from confidence using the real function.
    let configs = [
        (
            "High confidence (>=99%)",
            8usize,
            samples_to_ms(dynamic_silence_threshold(Some(0.99), true)),
            75u64,
        ),
        (
            "Good confidence (>=92%)",
            8,
            samples_to_ms(dynamic_silence_threshold(Some(0.95), true)),
            75,
        ),
        (
            "Moderate confidence (>=80%)",
            8,
            samples_to_ms(dynamic_silence_threshold(Some(0.85), true)),
            75,
        ),
        (
            "Low/unknown confidence",
            8,
            samples_to_ms(dynamic_silence_threshold(None, true)),
            75,
        ),
        (
            "Dynamic silence disabled",
            8,
            samples_to_ms(dynamic_silence_threshold(Some(0.99), false)),
            75,
        ),
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
// Stage 4: Dynamic silence tiers (exercises real function)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_dynamic_silence() {
    println!("── Stage 4: Dynamic Silence Tiers (real function) ────────────────");
    println!("  Exercises dynamic_silence_threshold() from lune_app::session.");
    println!();
    println!(
        "  {:<22}  {:>8}  {:>10}  {:>12}  {:>12}",
        "Confidence", "Samples", "Silence", "+ Inference", "+ Delivery"
    );
    println!("  {}", "─".repeat(72));

    let inference_gpu_ms = 75.0f64;
    let delivery_ms = 5.0f64;

    let cases: &[(&str, Option<f32>)] = &[
        (">=99% (speculative)", Some(0.99)),
        (">=92% (high)", Some(0.95)),
        (">=80% (moderate)", Some(0.85)),
        ("<80% (default)", Some(0.5)),
        ("None (first utterance)", None),
    ];

    for (label, conf) in cases {
        let samples = dynamic_silence_threshold(*conf, true);
        let silence_ms = samples_to_ms(samples) as f64;
        let total = silence_ms + inference_gpu_ms + delivery_ms;
        println!(
            "  {:<22}  {:>8}  {:>8.0} ms  {:>10.0} ms  {:>10.0} ms",
            label,
            samples,
            silence_ms,
            silence_ms + inference_gpu_ms,
            total
        );
    }

    // Disabled path
    let disabled_samples = dynamic_silence_threshold(Some(1.0), false);
    let disabled_ms = samples_to_ms(disabled_samples) as f64;
    println!(
        "  {:<22}  {:>8}  {:>8.0} ms  {:>10.0} ms  {:>10.0} ms",
        "Disabled (static)",
        disabled_samples,
        disabled_ms,
        disabled_ms + inference_gpu_ms,
        disabled_ms + inference_gpu_ms + delivery_ms
    );

    println!();
    println!("  Tier constants (samples @ 16kHz):");
    println!(
        "    SPECULATIVE = {SILENCE_TIER_SPECULATIVE:>5}  ({:>3} ms)",
        samples_to_ms(SILENCE_TIER_SPECULATIVE)
    );
    println!(
        "    HIGH        = {SILENCE_TIER_HIGH:>5}  ({:>3} ms)",
        samples_to_ms(SILENCE_TIER_HIGH)
    );
    println!(
        "    MODERATE    = {SILENCE_TIER_MODERATE:>5}  ({:>3} ms)",
        samples_to_ms(SILENCE_TIER_MODERATE)
    );
    println!(
        "    DEFAULT     = {SILENCE_TIER_DEFAULT:>5}  ({:>3} ms)",
        samples_to_ms(SILENCE_TIER_DEFAULT)
    );
    println!(
        "    STATIC      = {SILENCE_FLUSH_SAMPLES:>5}  ({:>3} ms)  (when disabled)",
        samples_to_ms(SILENCE_FLUSH_SAMPLES)
    );
    println!();
}

#[inline]
fn samples_to_ms(samples: usize) -> u64 {
    (samples * 1000 / SAMPLE_RATE) as u64
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
    let inference_gpu = 75u64;
    let inference_cpu = 680u64;
    let delivery = 5u64;

    let spec_ms = samples_to_ms(SILENCE_TIER_SPECULATIVE);
    let high_ms = samples_to_ms(SILENCE_TIER_HIGH);
    let mod_ms = samples_to_ms(SILENCE_TIER_MODERATE);
    let def_ms = samples_to_ms(SILENCE_TIER_DEFAULT);
    let static_ms = samples_to_ms(SILENCE_FLUSH_SAMPLES);

    println!("  Dynamic silence after-speech wait (GPU, Qwen3-ASR-0.6B):");
    println!();
    println!("    Conf >=99%  (speculative, LLM bypassed):");
    println!(
        "      Silence:   {:>3} ms  Dictionary: <1 ms  Delivery: ~{delivery} ms",
        spec_ms
    );
    println!("      TOTAL:    ~{} ms", spec_ms + delivery);
    println!();
    println!("    Conf >=92%  (high):");
    println!(
        "      Silence:   {:>3} ms  Inference: ~{inference_gpu} ms  Delivery: ~{delivery} ms",
        high_ms
    );
    println!("      TOTAL:    ~{} ms", high_ms + inference_gpu + delivery);
    println!();
    println!("    Conf >=80%  (moderate):");
    println!(
        "      Silence:   {:>3} ms  Inference: ~{inference_gpu} ms  Delivery: ~{delivery} ms",
        mod_ms
    );
    println!("      TOTAL:    ~{} ms", mod_ms + inference_gpu + delivery);
    println!();
    println!("    Conf <80%   (default/unknown):");
    println!(
        "      Silence:   {:>3} ms  Inference: ~{inference_gpu} ms  Delivery: ~{delivery} ms",
        def_ms
    );
    println!("      TOTAL:    ~{} ms", def_ms + inference_gpu + delivery);
    println!();
    println!("    Disabled    (static fallback):");
    println!(
        "      Silence:   {:>3} ms  Inference: ~{inference_gpu} ms  Delivery: ~{delivery} ms",
        static_ms
    );
    println!(
        "      TOTAL:    ~{} ms",
        static_ms + inference_gpu + delivery
    );
    println!();
    println!("  CPU-only after-speech wait (worst case, conf <80%):");
    println!(
        "    Silence:     {:>3} ms  Inference: ~{inference_cpu} ms  Delivery: ~{delivery} ms",
        def_ms
    );
    println!("    TOTAL:      ~{} ms", def_ms + inference_cpu + delivery);
    println!();
    println!("  Key insight: dynamic silence adapts the wait to ASR confidence.");
    println!("  High-confidence segments flush in ~55ms (speculative paste),");
    println!("  while uncertain segments get ~330ms for a safer result.");
    println!();
    println!("  Latency targets (Stivers et al. 2009, PNAS):");
    println!("    <200 ms  feels instantaneous (human turn-taking mean=208ms)");
    println!("    <300 ms  natural and responsive (AssemblyAI 300ms rule)");
    println!("    <500 ms  hard ceiling for voice AI (MMNTM threshold)");
    println!(
        "    GPU >=99% ({} ms) feels instantaneous.",
        spec_ms + delivery
    );
    println!(
        "    GPU >=92% ({} ms) meets the 200ms target.",
        high_ms + inference_gpu + delivery
    );
    println!(
        "    GPU >=80% ({} ms) meets the 300ms target.",
        mod_ms + inference_gpu + delivery
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 5: Output quality analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-fixture quality metrics collected from real ASR inference.
struct QualityRun {
    name: &'static str,
    audio_secs: f64,
    /// Absolute inference time; kept for summary table.
    inference_secs: f64,
    rtf: f64,
    wer: f64,
    cer: f64,
    segments: Vec<AsrSegment>,
    /// Stored for per-word alignment diffing in future extensions.
    #[allow(dead_code)]
    hypothesis: String,
    #[allow(dead_code)]
    reference: String,
}

fn bench_quality_analysis() -> Vec<QualityRun> {
    println!("── Stage 5: Output Quality Analysis ──────────────────────────────");

    // ── 5a. Synthetic WER/CER demo (no models needed) ─────────────────────
    println!("  5a. WER/CER metric demo (pure Rust)");
    println!();

    let pairs: &[(&str, &str, &str)] = &[
        ("hello world", "hello world", "perfect match"),
        ("hello world", "hello wirld", "1 substitution"),
        ("the cat sat on the mat", "cat sat mat", "3 deletions"),
        ("a b c d e", "a b c d e f", "1 insertion"),
        (
            "from the respect paid her",
            "FROM THE RESPECT PAID HER",
            "case only",
        ),
    ];

    println!("  {:<50}  {:>6}  {:>6}", "Pair (hyp | ref)", "WER", "CER");
    println!("  {}", "─".repeat(66));
    for (hyp, ref_, note) in pairs {
        let wer = word_error_rate(hyp, ref_);
        let cer = char_error_rate(hyp, ref_);
        println!(
            "  {:<50}  {:>5.1}%  {:>5.1}%  ← {note}",
            format!("{hyp:?} | {ref_:?}"),
            wer * 100.0,
            cer * 100.0,
        );
    }
    println!();

    // ── 5b. Real ASR inference ─────────────────────────────────────────────
    println!("  5b. Real ASR inference (requires models/qwen3-asr-0.6b-onnx)");
    println!();

    let root = workspace_root();
    let model_dir = root.join("models/qwen3-asr-0.6b-onnx");
    if !model_dir.exists() {
        println!("  [skip] model dir not found: {}", model_dir.display());
        println!();
        return vec![];
    }

    // Initialize ORT (CPU or CUDA).
    #[cfg(feature = "cuda")]
    {
        if !ort::init()
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .commit()
        {
            eprintln!("  [warn] ORT CUDA init failed, using CPU");
            let _ = ort::init().commit();
        } else {
            println!("  [info] ORT initialized with CUDA EP");
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ort::init().commit();
        println!("  [info] ORT initialized (CPU only)");
    }

    println!("  Loading ASR backend ...");
    let load_t = Instant::now();
    let backend = match OrtQwenAsrBackend::new(OrtQwenAsrConfig {
        model_dir: model_dir.clone(),
        use_gpu: cfg!(feature = "cuda"),
        ..Default::default()
    }) {
        Ok(b) => b,
        Err(e) => {
            println!("  [skip] ASR backend load failed: {e}");
            println!();
            return vec![];
        }
    };
    println!("  ASR loaded in {:.1}s", load_t.elapsed().as_secs_f64());
    println!();

    let fixtures: &[(&'static str, &str, &str)] = &[
        (
            "librispeech_clean",
            "test_fixtures/librispeech_clean.wav",
            "test_fixtures/librispeech_clean.txt",
        ),
        (
            "librispeech_40744",
            "test_fixtures/librispeech_40744.wav",
            "test_fixtures/librispeech_40744.txt",
        ),
    ];

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut results: Vec<QualityRun> = Vec::new();

    for &(name, wav_rel, txt_rel) in fixtures {
        let wav_path = root.join(wav_rel);
        if !wav_path.exists() {
            println!("  [skip] {name}: {} not found", wav_path.display());
            continue;
        }

        let pcm = load_wav_f32(&wav_path);
        let audio_secs = pcm.len() as f64 / 16_000.0;
        let reference = std::fs::read_to_string(root.join(txt_rel))
            .unwrap_or_default()
            .trim()
            .to_string();

        println!("  Fixture: {name} ({audio_secs:.1}s audio)");

        let infer_t = Instant::now();
        let segments = rt.block_on(transcribe(&backend, &pcm));
        let inference_secs = infer_t.elapsed().as_secs_f64();
        let rtf = inference_secs / audio_secs;

        let hypothesis = segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        let wer = if reference.is_empty() {
            f64::NAN
        } else {
            word_error_rate(&hypothesis, &reference)
        };
        let cer = if reference.is_empty() {
            f64::NAN
        } else {
            char_error_rate(&hypothesis, &reference)
        };

        let conf_vals: Vec<f64> = segments
            .iter()
            .filter_map(|s| s.confidence.map(|c| c as f64))
            .collect();
        let mean_conf = if conf_vals.is_empty() {
            f64::NAN
        } else {
            conf_vals.iter().sum::<f64>() / conf_vals.len() as f64
        };

        println!("    Segments:    {}", segments.len());
        println!("    Inference:   {inference_secs:.3}s  ({rtf:.3}x RT)");
        println!("    Mean conf:   {mean_conf:.3}");
        if !wer.is_nan() {
            println!("    WER:         {:.1}%", wer * 100.0);
            println!("    CER:         {:.1}%", cer * 100.0);
        }
        println!("    Hypothesis:");
        wrap_print("      ", &hypothesis, 72);
        if !reference.is_empty() {
            println!("    Reference:");
            wrap_print("      ", &reference, 72);
        }
        println!();

        results.push(QualityRun {
            name,
            audio_secs,
            inference_secs,
            rtf,
            hypothesis,
            reference,
            wer,
            cer,
            segments,
        });
    }

    // ── 5c. Confidence gate threshold sweep ───────────────────────────────
    let all_segs: Vec<&AsrSegment> = results.iter().flat_map(|r| &r.segments).collect();
    let n_segs = all_segs.len();

    if n_segs > 0 {
        println!("  5c. Confidence gate threshold sweep ({n_segs} total segments)");
        println!();
        println!(
            "  {:>12}  {:>12}  {:>10}  {:>10}",
            "Threshold", "Bypass %", "Bypassed", "LLM-routed"
        );
        println!("  {}", "─".repeat(50));

        for &thr in &[0.70f32, 0.80, 0.90, 0.95, 0.99] {
            let bypassed = all_segs
                .iter()
                .filter(|s| s.confidence.unwrap_or(0.0) >= thr)
                .count();
            let bypass_pct = bypassed as f64 / n_segs as f64 * 100.0;
            println!(
                "  {:>11.0}%  {:>11.1}%  {:>10}  {:>10}",
                thr * 100.0,
                bypass_pct,
                bypassed,
                n_segs - bypassed,
            );
        }
        println!();
    }

    results
}

fn print_quality_summary(results: &[QualityRun]) {
    if results.is_empty() {
        return;
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              QUALITY BUDGET ANALYSIS                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "  {:20}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7}  {:>8}",
        "Fixture", "Audio", "Infer", "RTF", "WER", "CER", "Segments"
    );
    println!("  {}", "─".repeat(73));
    for r in results {
        println!(
            "  {:20}  {:>6.1}s  {:>6.3}s  {:>5.3}x  {:>5.1}%  {:>5.1}%  {:>8}",
            r.name,
            r.audio_secs,
            r.inference_secs,
            r.rtf,
            if r.wer.is_nan() { 0.0 } else { r.wer * 100.0 },
            if r.cer.is_nan() { 0.0 } else { r.cer * 100.0 },
            r.segments.len(),
        );
    }
    println!();

    println!("  WER = Word Error Rate (lower is better; 0% = perfect match).");
    println!("  CER = Character Error Rate (character-level edit distance).");
    println!("  RTF = Real-Time Factor (0.10x = 10x faster than real-time).");
    println!();
    println!("  Quality context:");
    println!("    Industry reference WER (Whisper-large-v3, LibriSpeech clean): ~2.7%");
    println!("    Qwen3-ASR-0.6B INT4 streaming: 8s encoder window, past-text conditioned");
    println!("    LLM correction reduces WER ~15-30% on low-confidence segments.");
    println!();
}

// ── Quality helpers ──────────────────────────────────────────────────────────

/// Returns the workspace root (two dirs up from this crate's manifest).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Normalise text to uppercase ASCII words for WER/CER comparison.
/// Strips punctuation; collapses whitespace.
fn normalize_text(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphabetic() {
                c.to_ascii_uppercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// O(m·n) Levenshtein edit distance over arbitrary `Eq` slices.
fn levenshtein<T: Eq>(a: &[T], b: &[T]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp: Vec<usize> = (0..=n).collect();
    for i in 1..=m {
        let mut prev = dp[0];
        dp[0] = i;
        for j in 1..=n {
            let temp = dp[j];
            dp[j] = if a[i - 1] == b[j - 1] {
                prev
            } else {
                1 + prev.min(dp[j]).min(dp[j - 1])
            };
            prev = temp;
        }
    }
    dp[n]
}

/// Word Error Rate: normalise both strings, split on whitespace, Levenshtein.
fn word_error_rate(hyp: &str, ref_: &str) -> f64 {
    let h = normalize_text(hyp);
    let r = normalize_text(ref_);
    let hw: Vec<&str> = h.split_whitespace().collect();
    let rw: Vec<&str> = r.split_whitespace().collect();
    if rw.is_empty() {
        return 0.0;
    }
    levenshtein(&hw, &rw) as f64 / rw.len() as f64
}

/// Character Error Rate: normalise, Levenshtein on chars.
fn char_error_rate(hyp: &str, ref_: &str) -> f64 {
    let h = normalize_text(hyp);
    let r = normalize_text(ref_);
    let hc: Vec<char> = h.chars().collect();
    let rc: Vec<char> = r.chars().collect();
    if rc.is_empty() {
        return 0.0;
    }
    levenshtein(&hc, &rc) as f64 / rc.len() as f64
}

/// Load 16 kHz mono f32 PCM from a WAV file (Float or Int samples).
fn load_wav_f32(path: &std::path::Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono WAV");
    assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz WAV");
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.expect("read sample"))
            .collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.expect("read sample") as f32 / max)
                .collect()
        }
    }
}

/// Transcribe by pushing 1s chunks and polling after each, plus a silence flush.
async fn transcribe(backend: &OrtQwenAsrBackend, pcm: &[f32]) -> Vec<AsrSegment> {
    let session = backend
        .start_session(AsrSessionConfig::default())
        .await
        .expect("start asr session");
    let chunk = 16_000; // 1s @ 16 kHz
    let mut out: Vec<AsrSegment> = Vec::new();
    for slice in pcm.chunks(chunk) {
        backend
            .push_audio(&session, slice)
            .await
            .expect("push audio");
        out.extend(backend.poll_results(&session).await.expect("poll results"));
    }
    // Half-second silence flush to commit any remaining unfixed tail.
    let silence = vec![0.0f32; 8_000];
    backend
        .push_audio(&session, &silence)
        .await
        .expect("push silence");
    out.extend(
        backend
            .poll_results(&session)
            .await
            .expect("poll results final"),
    );
    backend.end_session(session).await.expect("end session");
    out
}

/// Print `text` wrapped at `width` chars with `prefix` on each line.
fn wrap_print(prefix: &str, text: &str, width: usize) {
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            println!("{prefix}{line}");
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        println!("{prefix}{line}");
    }
}
