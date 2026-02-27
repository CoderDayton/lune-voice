//! Pipeline latency benchmark — real ONNX inference on real WAV fixtures.
//!
//! Run: cargo run --release --example pipeline_latency -p lune-app
//! GPU: cargo run --release --features cuda --example pipeline_latency -p lune-app
//!
//! All 5 stages use real model inference — no simulated data, no thread::sleep.
//! Gracefully skips if model directories are missing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use lune_asr::{OrtQwenAsrBackend, OrtQwenAsrConfig, compute_qwen_mel};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::StylePreset;
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_llm::{OrtLlmBackend, OrtLlmConfig};

use lune_app::{
    SILENCE_FLUSH_SAMPLES, SILENCE_TIER_DEFAULT, SILENCE_TIER_HIGH, SILENCE_TIER_MODERATE,
    SILENCE_TIER_SPECULATIVE, dynamic_silence_threshold,
};

const SAMPLE_RATE: usize = 16_000;

// ─── Data structures ────────────────────────────────────────────────────────

struct Fixture {
    name: &'static str,
    pcm: Vec<f32>,
    reference: String,
    audio_secs: f64,
}

struct AsrResult {
    segments: Vec<AsrSegment>,
    hypothesis: String,
    total_secs: f64,
    chunk_latencies_ms: Vec<f64>,
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              PIPELINE LATENCY BENCHMARK (Real Inference)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── ORT init: try CUDA, auto-fall-back to CPU ───────────────────────────
    let has_gpu = ort::init()
        .with_execution_providers([ort::ep::CUDA::default().build()])
        .commit();
    if has_gpu {
        println!("  [info] ORT: CUDA EP active");
    } else {
        let _ = ort::init().commit();
        println!("  [info] ORT: CPU only (CUDA not available)");
    }

    let root = workspace_root();

    // ── Load ASR backend ────────────────────────────────────────────────────
    let asr_model_dir = root.join("models/qwen3-asr-0.6b-onnx");
    if !asr_model_dir.exists() {
        println!(
            "  [skip] ASR model dir not found: {}",
            asr_model_dir.display()
        );
        println!("  Download models first. Exiting.");
        return;
    }

    println!("  Loading ASR backend ...");
    let t = Instant::now();
    let asr = match OrtQwenAsrBackend::new(OrtQwenAsrConfig {
        model_dir: asr_model_dir,
        use_gpu: has_gpu,
        ..Default::default()
    }) {
        Ok(b) => b,
        Err(e) => {
            println!("  [skip] ASR backend load failed: {e}");
            return;
        }
    };
    println!("  ASR loaded in {:.1}s", t.elapsed().as_secs_f64());

    // ── Load LLM backend (optional — stages 1-3 still run without it) ─────
    let llm_model_dir = root.join("models/qwen3-0.6b-onnx");
    let llm: Option<OrtLlmBackend> = if llm_model_dir.exists() {
        println!("  Loading LLM backend ...");
        let t = Instant::now();
        match OrtLlmBackend::new(OrtLlmConfig {
            model_dir: llm_model_dir,
            use_gpu: has_gpu,
            ..Default::default()
        }) {
            Ok(b) => {
                println!("  LLM loaded in {:.1}s", t.elapsed().as_secs_f64());
                Some(b)
            }
            Err(e) => {
                println!("  [warn] LLM backend load failed: {e}");
                println!("  Stages 4-5 LLM portions will be skipped.");
                None
            }
        }
    } else {
        println!("  [info] LLM model dir not found — stages 4-5 LLM portions skipped");
        None
    };

    // ── Load fixtures ───────────────────────────────────────────────────────
    let fixture_defs: &[(&'static str, &str, &str)] = &[
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

    let mut fixtures: Vec<Fixture> = Vec::new();
    for &(name, wav_rel, txt_rel) in fixture_defs {
        let wav_path = root.join(wav_rel);
        if !wav_path.exists() {
            println!("  [skip] {name}: {} not found", wav_path.display());
            continue;
        }
        let pcm = load_wav_f32(&wav_path);
        let audio_secs = pcm.len() as f64 / SAMPLE_RATE as f64;
        let reference = std::fs::read_to_string(root.join(txt_rel))
            .unwrap_or_default()
            .trim()
            .to_string();
        println!("  Loaded {name}: {audio_secs:.1}s ({} samples)", pcm.len());
        fixtures.push(Fixture {
            name,
            pcm,
            reference,
            audio_secs,
        });
    }

    if fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found. Exiting.");
        return;
    }
    println!();

    // ── Run stages ──────────────────────────────────────────────────────────
    bench_mel_extraction(&fixtures);
    let asr_results = bench_asr_streaming(&asr, &fixtures);
    bench_confidence_gate(&asr_results);
    let llm_results = bench_llm_postprocess(llm.as_ref(), &asr_results, &fixtures);
    bench_full_summary(&fixtures, &asr_results, &llm_results);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1: Mel Extraction (real PCM → real compute_qwen_mel)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_mel_extraction(fixtures: &[Fixture]) {
    println!("── Stage 1: Mel Extraction ────────────────────────────────────────");
    println!();

    let warmup_runs = 2;
    let timed_runs = 5;

    for fix in fixtures {
        // Warmup
        for _ in 0..warmup_runs {
            let _ = compute_qwen_mel(&fix.pcm);
        }

        // Timed runs
        let mut times_ms: Vec<f64> = Vec::with_capacity(timed_runs);
        for _ in 0..timed_runs {
            let t = Instant::now();
            let _ = compute_qwen_mel(&fix.pcm);
            times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rtf = (mean_ms / 1000.0) / fix.audio_secs;
        let throughput = fix.pcm.len() as f64 / (mean_ms / 1000.0);

        println!(
            "  {} ({:.1}s audio, {} runs):",
            fix.name, fix.audio_secs, timed_runs
        );
        println!("    Latency:     {mean_ms:.3}ms mean  [{min_ms:.3} .. {max_ms:.3}]");
        println!(
            "    RTF:         {rtf:.6}x  ({:.0}x faster than real-time)",
            1.0 / rtf
        );
        println!("    Throughput:  {throughput:.0} samples/s");
        println!();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 2: ASR Streaming Inference (real model, per-chunk timing)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_asr_streaming(asr: &OrtQwenAsrBackend, fixtures: &[Fixture]) -> Vec<AsrResult> {
    println!("── Stage 2: ASR Streaming Inference ──────────────────────────────");
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut results: Vec<AsrResult> = Vec::new();

    for fix in fixtures {
        let (segments, total_secs, chunk_latencies_ms) =
            rt.block_on(transcribe_timed(asr, &fix.pcm));

        let hypothesis = segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        let rtf = total_secs / fix.audio_secs;

        // Per-chunk stats
        let n = chunk_latencies_ms.len();
        let mean_chunk = if n > 0 {
            chunk_latencies_ms.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let min_chunk = chunk_latencies_ms
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_chunk = chunk_latencies_ms
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let p99_chunk = percentile(&chunk_latencies_ms, 99.0);

        let conf_vals: Vec<f64> = segments
            .iter()
            .filter_map(|s| s.confidence.map(|c| c as f64))
            .collect();
        let mean_conf = if conf_vals.is_empty() {
            f64::NAN
        } else {
            conf_vals.iter().sum::<f64>() / conf_vals.len() as f64
        };

        println!("  {} ({:.1}s audio):", fix.name, fix.audio_secs);
        println!("    Total:       {total_secs:.3}s  ({rtf:.3}x RT)");
        println!("    Segments:    {}", segments.len());
        println!("    Mean conf:   {mean_conf:.3}");
        println!("    Chunk latency ({n} chunks):");
        println!(
            "      mean={mean_chunk:.1}ms  min={min_chunk:.1}ms  max={max_chunk:.1}ms  p99={p99_chunk:.1}ms"
        );
        println!("    Hypothesis:");
        wrap_print("      ", &hypothesis, 72);
        println!();

        results.push(AsrResult {
            segments,
            hypothesis,
            total_secs,
            chunk_latencies_ms,
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 3: Confidence Gate Sweep (real segment confidences)
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_confidence_gate(asr_results: &[AsrResult]) {
    println!("── Stage 3: Confidence Gate Sweep ──────────────────────────────────");
    println!();

    let all_segs: Vec<&AsrSegment> = asr_results.iter().flat_map(|r| &r.segments).collect();
    let n_segs = all_segs.len();

    if n_segs == 0 {
        println!("  No segments to analyze.");
        println!();
        return;
    }

    println!("  Total segments: {n_segs}");
    println!();

    // Threshold sweep (mean threshold, with min_token_threshold = 0.70)
    println!(
        "  {:>12}  {:>10}  {:>10}  {:>12}  {:>14}",
        "Threshold", "Bypass %", "Bypassed", "LLM-routed", "Silence tier"
    );
    println!("  {}", "─".repeat(64));

    for &thr in &[0.70f32, 0.80, 0.90, 0.95, 0.99] {
        let bypassed = all_segs
            .iter()
            .filter(|s| s.is_confident_gated(thr, 0.70))
            .count();
        let bypass_pct = bypassed as f64 / n_segs as f64 * 100.0;
        let silence_samples = dynamic_silence_threshold(Some(thr), true);
        let silence_ms = samples_to_ms(silence_samples);

        println!(
            "  {:>11.0}%  {:>9.1}%  {:>10}  {:>12}  {:>12} ms",
            thr * 100.0,
            bypass_pct,
            bypassed,
            n_segs - bypassed,
            silence_ms,
        );
    }

    println!();
    println!("  Silence tier constants (samples @ 16kHz):");
    println!(
        "    SPECULATIVE = {:>5}  ({:>3} ms)",
        SILENCE_TIER_SPECULATIVE,
        samples_to_ms(SILENCE_TIER_SPECULATIVE)
    );
    println!(
        "    HIGH        = {:>5}  ({:>3} ms)",
        SILENCE_TIER_HIGH,
        samples_to_ms(SILENCE_TIER_HIGH)
    );
    println!(
        "    MODERATE    = {:>5}  ({:>3} ms)",
        SILENCE_TIER_MODERATE,
        samples_to_ms(SILENCE_TIER_MODERATE)
    );
    println!(
        "    DEFAULT     = {:>5}  ({:>3} ms)",
        SILENCE_TIER_DEFAULT,
        samples_to_ms(SILENCE_TIER_DEFAULT)
    );
    println!(
        "    STATIC      = {:>5}  ({:>3} ms)  (when disabled)",
        SILENCE_FLUSH_SAMPLES,
        samples_to_ms(SILENCE_FLUSH_SAMPLES)
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 4: LLM Post-Processing (real model on low-confidence segments)
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-fixture LLM results for summary.
struct LlmResult {
    /// Number of segments that failed the confidence gate.
    #[allow(dead_code)]
    segments_processed: usize,
    /// Total LLM inference time for this fixture.
    total_secs: f64,
    /// LLM-corrected full hypothesis (gate-bypassed segments keep original text).
    corrected_hypothesis: String,
}

fn bench_llm_postprocess(
    llm: Option<&OrtLlmBackend>,
    asr_results: &[AsrResult],
    fixtures: &[Fixture],
) -> Vec<Option<LlmResult>> {
    println!("── Stage 4: LLM Post-Processing ──────────────────────────────────");
    println!();

    let llm = match llm {
        Some(l) => l,
        None => {
            println!("  [skip] LLM backend not available.");
            println!();
            return asr_results.iter().map(|_| None).collect();
        }
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let style = Arc::new(StylePreset::general());
    let mut results: Vec<Option<LlmResult>> = Vec::new();

    for (asr_res, fix) in asr_results.iter().zip(fixtures.iter()) {
        let low_conf_segs: Vec<&AsrSegment> = asr_res
            .segments
            .iter()
            .filter(|s| !s.is_confident_gated(0.95, 0.70))
            .collect();

        if low_conf_segs.is_empty() {
            println!("  {} — all segments bypassed gate, no LLM needed", fix.name);
            println!();
            results.push(Some(LlmResult {
                segments_processed: 0,
                total_secs: 0.0,
                corrected_hypothesis: asr_res.hypothesis.clone(),
            }));
            continue;
        }

        println!(
            "  {} — {} / {} segments below gate (mean<0.95 or min_tok<0.70):",
            fix.name,
            low_conf_segs.len(),
            asr_res.segments.len()
        );

        let mut total_llm_secs = 0.0;
        let mut corrected_parts: Vec<String> = Vec::new();

        for seg in &asr_res.segments {
            if seg.is_confident_gated(0.95, 0.70) {
                corrected_parts.push(seg.text.trim().to_string());
                continue;
            }

            let input = PostProcessInput {
                raw_text: seg.text.trim().to_string(),
                style_preset: Arc::clone(&style),
                app_category: None,
                dictionary: vec![],
                context: None,
            };

            let t = Instant::now();
            match rt.block_on(llm.transform(&input)) {
                Ok(output) => {
                    let llm_ms = t.elapsed().as_secs_f64() * 1000.0;
                    total_llm_secs += t.elapsed().as_secs_f64();
                    let conf_str = seg
                        .confidence
                        .map(|c| format!("{c:.3}"))
                        .unwrap_or_else(|| "none".into());
                    println!(
                        "    [{llm_ms:.0}ms] conf={conf_str}  \"{}\"\n             → \"{}\"",
                        seg.text.trim(),
                        output.cleaned_text.trim()
                    );
                    corrected_parts.push(output.cleaned_text.trim().to_string());
                }
                Err(e) => {
                    println!("    [error] LLM transform failed: {e}");
                    corrected_parts.push(seg.text.trim().to_string());
                }
            }
        }

        let corrected_hypothesis = corrected_parts
            .into_iter()
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        println!(
            "    Total LLM time: {:.3}s ({} segments)",
            total_llm_secs,
            low_conf_segs.len()
        );
        println!();

        results.push(Some(LlmResult {
            segments_processed: low_conf_segs.len(),
            total_secs: total_llm_secs,
            corrected_hypothesis,
        }));
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 5: Full Pipeline Summary
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_full_summary(
    fixtures: &[Fixture],
    asr_results: &[AsrResult],
    llm_results: &[Option<LlmResult>],
) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              END-TO-END PIPELINE SUMMARY                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Summary table
    println!(
        "  {:20}  {:>8}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7}  {:>8}",
        "Fixture", "Audio", "ASR", "LLM", "Total", "RTF", "WER", "CER"
    );
    println!("  {}", "─".repeat(82));

    for (i, fix) in fixtures.iter().enumerate() {
        let asr_res = &asr_results[i];
        let llm_res = llm_results.get(i).and_then(|r| r.as_ref());

        let llm_secs = llm_res.map(|r| r.total_secs).unwrap_or(0.0);
        let total_secs = asr_res.total_secs + llm_secs;
        let rtf = total_secs / fix.audio_secs;

        // Use LLM-corrected hypothesis if available, else raw ASR
        let final_hyp = llm_res
            .map(|r| r.corrected_hypothesis.as_str())
            .unwrap_or(&asr_res.hypothesis);

        let (wer, cer) = if fix.reference.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            (
                word_error_rate(final_hyp, &fix.reference),
                char_error_rate(final_hyp, &fix.reference),
            )
        };

        println!(
            "  {:20}  {:>6.1}s  {:>6.3}s  {:>6.3}s  {:>5.3}s  {:>5.3}x  {:>5.1}%  {:>5.1}%",
            fix.name,
            fix.audio_secs,
            asr_res.total_secs,
            llm_secs,
            total_secs,
            rtf,
            if wer.is_nan() { 0.0 } else { wer * 100.0 },
            if cer.is_nan() { 0.0 } else { cer * 100.0 },
        );
    }
    println!();

    // Per-fixture detail
    for (i, fix) in fixtures.iter().enumerate() {
        let asr_res = &asr_results[i];
        let llm_res = llm_results.get(i).and_then(|r| r.as_ref());

        let final_hyp = llm_res
            .map(|r| r.corrected_hypothesis.as_str())
            .unwrap_or(&asr_res.hypothesis);

        // After-speech latency estimate
        let mean_conf: Option<f32> = {
            let vals: Vec<f32> = asr_res
                .segments
                .iter()
                .filter_map(|s| s.confidence)
                .collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f32>() / vals.len() as f32)
            }
        };
        let silence_samples = dynamic_silence_threshold(mean_conf, true);
        let silence_ms = samples_to_ms(silence_samples);

        // Last-chunk inference time as representative "after speech" inference cost
        let last_chunk_ms = asr_res.chunk_latencies_ms.last().copied().unwrap_or(0.0);
        let delivery_ms: u64 = 5; // typical uinput delivery
        let after_speech_ms = silence_ms + last_chunk_ms as u64 + delivery_ms;

        println!("  {} detail:", fix.name);
        println!("    After-speech latency estimate:");
        println!(
            "      Silence: {} ms + last-chunk infer: {:.0} ms + delivery: ~{} ms = ~{} ms",
            silence_ms, last_chunk_ms, delivery_ms, after_speech_ms
        );

        if !fix.reference.is_empty() {
            println!("    Hypothesis:");
            wrap_print("      ", final_hyp, 72);
            println!("    Reference:");
            wrap_print("      ", &fix.reference, 72);
        }
        println!();
    }

    // Quality context
    println!("  WER = Word Error Rate (lower is better; 0% = perfect match).");
    println!("  CER = Character Error Rate (character-level edit distance).");
    println!("  RTF = Real-Time Factor (0.10x = 10x faster than real-time).");
    println!();
    println!("  Quality context:");
    println!("    Industry reference WER (Whisper-large-v3, LibriSpeech clean): ~2.7%");
    println!("    Qwen3-ASR-0.6B INT4 streaming: 8s encoder window, past-text conditioned");
    println!("    LLM correction reduces WER ~15-30% on low-confidence segments.");
    println!();
    println!("  Latency targets (Stivers et al. 2009, PNAS):");
    println!("    <200 ms  feels instantaneous (human turn-taking mean=208ms)");
    println!("    <300 ms  natural and responsive (AssemblyAI 300ms rule)");
    println!("    <500 ms  hard ceiling for voice AI (MMNTM threshold)");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers (shared, canonical implementations — not duplicated)
// ═══════════════════════════════════════════════════════════════════════════════

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Transcribe with per-chunk timing. Returns (segments, total_secs, chunk_latencies_ms).
async fn transcribe_timed(
    backend: &OrtQwenAsrBackend,
    pcm: &[f32],
) -> (Vec<AsrSegment>, f64, Vec<f64>) {
    let session = backend
        .start_session(AsrSessionConfig::default())
        .await
        .expect("start asr session");

    let chunk_size = 16_000; // 1s @ 16 kHz
    let mut out: Vec<AsrSegment> = Vec::new();
    let mut chunk_latencies: Vec<f64> = Vec::new();
    let total_start = Instant::now();

    for slice in pcm.chunks(chunk_size) {
        let chunk_start = Instant::now();
        backend
            .push_audio(&session, slice)
            .await
            .expect("push audio");
        out.extend(backend.poll_results(&session).await.expect("poll results"));
        chunk_latencies.push(chunk_start.elapsed().as_secs_f64() * 1000.0);
    }

    // Silence flush
    let silence = vec![0.0f32; 8_000];
    let chunk_start = Instant::now();
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
    chunk_latencies.push(chunk_start.elapsed().as_secs_f64() * 1000.0);

    let total_secs = total_start.elapsed().as_secs_f64();
    backend.end_session(session).await.expect("end session");

    (out, total_secs, chunk_latencies)
}

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

#[inline]
fn samples_to_ms(samples: usize) -> u64 {
    (samples * 1000 / SAMPLE_RATE) as u64
}

/// Compute the p-th percentile of a slice (linear interpolation).
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}
