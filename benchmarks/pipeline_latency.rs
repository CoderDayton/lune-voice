//! Pipeline latency benchmark — real ONNX inference on real WAV fixtures.
//!
//! Run: cargo run --release --example pipeline_latency -p lune-app
//! GPU: cargo run --release --features cuda --example pipeline_latency -p lune-app
//!
//! All 5 stages use real model inference — no simulated data, no thread::sleep.
//! Gracefully skips if model directories are missing.

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    Fixture, brent_minimize, char_error_rate, init_ort, load_fixtures, percentile, word_error_rate,
    workspace_root, wrap_print,
};
use lune_asr::{AsrRegistry, compute_qwen_mel};
use lune_audio::{VadSegmenter, VadSegmenterConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::{AsrBackendConfig, DevicePolicy, StylePreset};
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_llm::{LlamaCppBackend, LlamaCppConfig};

struct AsrResult {
    segments: Vec<AsrSegment>,
    hypothesis: String,
    #[allow(dead_code)]
    total_secs: f64,
    #[allow(dead_code)]
    chunk_latencies_ms: Vec<f64>,
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              PIPELINE LATENCY BENCHMARK (Real Inference)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();

    let root = workspace_root();

    // ── Load ASR backends (all available) ──────────────────────────────────
    let device = if has_gpu {
        DevicePolicy::Cuda
    } else {
        DevicePolicy::Cpu
    };
    let parakeet_dir = root.join("models/parakeet-tdt-0.6b");
    let qwen_dir = root.join("models/qwen3-asr-0.6b-onnx");

    let mut backends: Vec<(&str, AsrRegistry)> = Vec::new();

    if parakeet_dir.exists() {
        print!("  Loading Parakeet TDT ... ");
        let t = Instant::now();
        let cfg = AsrBackendConfig::ParakeetTdt {
            model_dir: parakeet_dir,
        };
        match AsrRegistry::from_config(&cfg, device) {
            Ok(b) => {
                println!("{:.1}s", t.elapsed().as_secs_f64());
                backends.push(("Parakeet-TDT-int8", b));
            }
            Err(e) => println!("[fail] {e}"),
        }
    }

    if qwen_dir.exists() {
        print!("  Loading OrtQwen ASR (batch) ... ");
        let t = Instant::now();
        let cfg = AsrBackendConfig::OrtQwenAsr {
            model_dir: qwen_dir,
        };
        match AsrRegistry::from_config(&cfg, device) {
            Ok(b) => {
                println!("{:.1}s", t.elapsed().as_secs_f64());
                backends.push(("OrtQwen-INT4-batch", b));
            }
            Err(e) => println!("[fail] {e}"),
        }
    }

    if backends.is_empty() {
        println!("  [skip] No ASR model dirs found. Download models first. Exiting.");
        return;
    }

    // Primary backend is the first loaded (used for stages 3-6)
    let primary_name = backends[0].0;

    // ── Load LLM backend (optional — stages 1-3 still run without it) ─────
    let llm_model_path = root.join("models/llm-corrector-gguf/qwen2.5-1.5b-instruct.Q4_K_M.gguf");
    let rt_llm = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime for LLM init");
    let llm: Option<LlamaCppBackend> = if llm_model_path.exists() {
        println!("  Loading LLM backend ...");
        let t = Instant::now();
        match rt_llm.block_on(LlamaCppBackend::new(LlamaCppConfig {
            model_path: llm_model_path,
            n_gpu_layers: if has_gpu { 999 } else { 0 },
            ..Default::default()
        })) {
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
        println!("  [info] LLM model not found — stages 4-5 LLM portions skipped");
        None
    };

    // ── Load fixtures ───────────────────────────────────────────────────────
    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found. Exiting.");
        return;
    }
    println!();

    // ── Load FP16 decoder ASR backend (optional — Stage 7, OrtQwen only) ──
    let fp16_model_dir = root.join("models/qwen3-asr-0.6b-onnx-fp16dec");
    let asr_fp16: Option<AsrRegistry> = if fp16_model_dir.exists() {
        println!("  Loading FP16-decoder ASR backend ...");
        let t = Instant::now();
        let fp16_config = AsrBackendConfig::OrtQwenAsr {
            model_dir: fp16_model_dir,
        };
        match AsrRegistry::from_config(&fp16_config, device) {
            Ok(b) => {
                println!(
                    "  FP16-decoder ASR loaded in {:.1}s",
                    t.elapsed().as_secs_f64()
                );
                Some(b)
            }
            Err(e) => {
                println!("  [warn] FP16-decoder ASR load failed: {e}");
                None
            }
        }
    } else {
        println!("  [info] FP16-decoder model not found — Stage 7 will be skipped");
        None
    };

    // ── Run stages ──────────────────────────────────────────────────────────
    // Stage 1 (mel extraction) — Qwen-specific
    if backends.iter().any(|(_, b)| b.as_ort_qwen().is_some()) {
        bench_mel_extraction(&fixtures);
    } else {
        println!("── Stage 1: Mel Extraction ── [skipped — no OrtQwen backend]");
        println!();
    }

    // Stage 2: Run ASR on ALL backends for comparison
    let mut primary_results: Option<Vec<AsrResult>> = None;
    for (name, backend) in &backends {
        println!("══ Backend: {} ══", name);
        let is_batch = name.contains("batch");
        let pipeline = backend.pipeline_config();
        let vad_config = VadSegmenterConfig {
            speech_threshold: pipeline.speech_threshold,
            min_speech_ms: pipeline.min_speech_ms,
            sentence_close_ms: pipeline.sentence_close_ms,
            hard_close_ms: pipeline.hard_close_ms,
            max_segment_ms: pipeline.max_segment_ms,
            adaptive: pipeline.adaptive_vad,
        };
        if !is_batch {
            println!("  VAD max_segment_ms: {}", vad_config.max_segment_ms);
        } else {
            println!("  Mode: batch + overlap stitching (no VAD segmenter)");
        }
        println!();
        let results = bench_asr_streaming(backend, &fixtures, vad_config, is_batch);
        if primary_results.is_none() {
            primary_results = Some(results);
        }
    }
    let asr_results = primary_results.unwrap();
    println!("  (Stages 3-6 use primary backend: {})", primary_name);
    println!();

    // Stages 3-6 use primary backend results
    bench_confidence_gate(&asr_results);
    bench_threshold_sweep(llm.as_ref(), &asr_results, &fixtures);
    bench_vad_timing_sweep(&backends[0].1, &fixtures);
    bench_adaptive_vad(&backends[0].1, &fixtures);
    if let Some(ref fp16) = asr_fp16 {
        bench_fp16_decoder(fp16, &asr_results, &fixtures);
    }
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

fn bench_asr_streaming(
    asr: &(impl AsrBackend + Sync),
    fixtures: &[Fixture],
    vad_config: VadSegmenterConfig,
    use_batch: bool,
) -> Vec<AsrResult> {
    println!("── Stage 2: ASR Inference ────────────────────────────────────────");
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut results: Vec<AsrResult> = Vec::new();

    for fix in fixtures {
        let (segments, total_secs, chunk_latencies_ms) = if use_batch {
            rt.block_on(transcribe_batch(asr, &fix.pcm))
        } else {
            rt.block_on(transcribe_timed_with_config(
                asr,
                &fix.pcm,
                vad_config.clone(),
            ))
        };

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
// Stage 3: Per-Word Confidence Analysis
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_confidence_gate(asr_results: &[AsrResult]) {
    println!("── Stage 3: Per-Word Confidence Analysis ──────────────────────────");
    println!();

    let all_segs: Vec<&AsrSegment> = asr_results.iter().flat_map(|r| &r.segments).collect();
    let n_segs = all_segs.len();

    if n_segs == 0 {
        println!("  No segments to analyze.");
        println!();
        return;
    }

    // Collect all word confidences across segments.
    let all_word_confs: Vec<f32> = all_segs
        .iter()
        .flat_map(|s| s.word_confidences.iter().map(|wc| wc.confidence))
        .collect();
    let n_words = all_word_confs.len();

    println!("  Total segments: {n_segs}");
    println!("  Total words (with confidence): {n_words}");

    if n_words > 0 {
        let mut sorted = all_word_confs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = sorted.iter().sum::<f32>() / n_words as f32;
        let min = sorted[0];
        let max = sorted[n_words - 1];
        let p10 = sorted[(n_words as f64 * 0.10) as usize];
        let p25 = sorted[(n_words as f64 * 0.25) as usize];
        let p50 = sorted[n_words / 2];

        println!();
        println!("  Word confidence distribution:");
        println!("    mean={mean:.3}  min={min:.3}  max={max:.3}");
        println!("    p10={p10:.3}  p25={p25:.3}  p50={p50:.3}");
    }
    println!();

    // Threshold sweep: at each level, show % words marked and % segments triggering LLM.
    println!(
        "  {:>12}  {:>12}  {:>12}  {:>14}",
        "Threshold", "Words marked", "% marked", "Segs w/ marks"
    );
    println!("  {}", "─".repeat(56));

    for &thr in &[0.70f32, 0.80, 0.85, 0.90, 0.95] {
        let marked_words = all_word_confs.iter().filter(|&&c| c < thr).count();
        let marked_pct = if n_words > 0 {
            marked_words as f64 / n_words as f64 * 100.0
        } else {
            0.0
        };
        let segs_with_marks = all_segs
            .iter()
            .filter(|s| s.has_uncertain_words(thr))
            .count();

        println!(
            "  {:>11.0}%  {:>12}  {:>10.1}%  {:>8} / {}",
            thr * 100.0,
            marked_words,
            marked_pct,
            segs_with_marks,
            n_segs,
        );
    }
    println!();

    // Print example prepared text for each result set.
    for asr_res in asr_results {
        let (prepared, has_uncertain) = lune_llm::prepare_input(&asr_res.segments, 0.85, "en");
        if !prepared.is_empty() {
            println!(
                "  Prepared text (threshold=0.85, uncertain={}):",
                has_uncertain
            );
            wrap_print("    ", &prepared, 72);
            println!();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 4: Threshold Sweep — ASR once, LLM at each threshold, compare WER
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-fixture result at a given threshold.
struct ThresholdResult {
    threshold: f32,
    words_marked: usize,
    segs_with_marks: usize,
    llm_ms: f64,
    hypothesis: String,
    wer: f64,
    cer: f64,
    bypassed: bool,
}

fn bench_threshold_sweep(
    llm: Option<&LlamaCppBackend>,
    asr_results: &[AsrResult],
    fixtures: &[Fixture],
) {
    println!("── Stage 4: Threshold Sweep (ASR once → LLM at each threshold) ──");
    println!();

    // Also compute raw ASR WER (no LLM) as baseline.
    for (i, fix) in fixtures.iter().enumerate() {
        let asr_res = &asr_results[i];
        if !fix.reference.is_empty() {
            let raw_wer = word_error_rate(&asr_res.hypothesis, &fix.reference) * 100.0;
            let raw_cer = char_error_rate(&asr_res.hypothesis, &fix.reference) * 100.0;
            println!(
                "  {} — raw ASR (no LLM): WER={:.1}%  CER={:.1}%",
                fix.name, raw_wer, raw_cer
            );
        }
    }
    println!();

    // Diagnostic: per-word confidence + marked text at 0.75 threshold.
    println!("── Pipeline Trace (per-word confidence @ 75%) ──");
    println!();
    for (i, fix) in fixtures.iter().enumerate() {
        let asr_res = &asr_results[i];
        println!("  {} — word confidences:", fix.name);
        for seg in &asr_res.segments {
            for wc in &seg.word_confidences {
                let marker = if wc.confidence < 0.75 { " ◄" } else { "" };
                print!(
                    "    {:>6.1}%  {}{}\n",
                    wc.confidence * 100.0,
                    wc.word,
                    marker
                );
            }
        }
        let (prepared, has_uncertain) = lune_llm::prepare_input(&asr_res.segments, 0.75, "en");
        println!("  Marked input (uncertain={}):", has_uncertain);
        wrap_print("    ", &prepared, 90);
        println!("  Raw ASR:");
        wrap_print("    ", &asr_res.hypothesis, 90);
        if !fix.reference.is_empty() {
            println!("  Reference:");
            wrap_print("    ", &fix.reference, 90);
        }
        println!();
    }
    println!();

    let llm = match llm {
        Some(l) => l,
        None => {
            println!("  [skip] LLM backend not available — cannot sweep thresholds.");
            println!();
            return;
        }
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let style = Arc::new(StylePreset::casual());
    let thresholds: &[f32] = &[0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

    // Collect all word confidences for distribution context.
    let all_word_confs: Vec<f32> = asr_results
        .iter()
        .flat_map(|r| r.segments.iter())
        .flat_map(|s| s.word_confidences.iter().map(|wc| wc.confidence))
        .collect();
    let n_words = all_word_confs.len();

    // Cache LLM results by prepared text — same text at different thresholds reuses result.
    let mut llm_cache: std::collections::HashMap<String, (String, f64)> =
        std::collections::HashMap::new();

    let mut all_results: Vec<Vec<ThresholdResult>> = Vec::new();

    for (i, (asr_res, fix)) in asr_results.iter().zip(fixtures.iter()).enumerate() {
        println!("  {} — sweeping {} thresholds:", fix.name, thresholds.len());

        let mut fixture_results: Vec<ThresholdResult> = Vec::new();

        for &thr in thresholds {
            let (prepared, has_uncertain) = lune_llm::prepare_input(&asr_res.segments, thr, "en");

            let words_marked = asr_res
                .segments
                .iter()
                .flat_map(|s| &s.word_confidences)
                .filter(|wc| wc.confidence < thr)
                .count();

            let segs_with_marks = asr_res
                .segments
                .iter()
                .filter(|s| s.has_uncertain_words(thr))
                .count();

            // Run LLM per-threshold with the actual prepared text (different markings per threshold).
            // Cache by prepared text to avoid re-running LLM when two thresholds produce identical text.
            let (hyp, llm_ms, bypassed) = if !has_uncertain {
                (prepared, 0.0, true)
            } else if let Some((corrected, ms)) = llm_cache.get(&prepared) {
                (corrected.clone(), *ms, false)
            } else {
                let t = Instant::now();
                let input = PostProcessInput {
                    raw_text: prepared.clone(),
                    style_preset: Arc::clone(&style),
                    app_category: None,
                    dictionary: vec![],
                    context: None,
                    language: "en".into(),
                };
                let corrected = match rt.block_on(llm.transform(&input)) {
                    Ok(output) => output.cleaned_text.trim().to_string(),
                    Err(e) => {
                        println!("    [warn] LLM failed for fixture {i} thr={thr}: {e}");
                        prepared.clone()
                    }
                };
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "    LLM [{:.0}%]: {ms:.0}ms → {:?}",
                    thr * 100.0,
                    &corrected[..corrected.len().min(80)]
                );
                llm_cache.insert(prepared, (corrected.clone(), ms));
                (corrected, ms, false)
            };

            let (wer, cer) = if fix.reference.is_empty() {
                (0.0, 0.0)
            } else {
                (
                    word_error_rate(&hyp, &fix.reference) * 100.0,
                    char_error_rate(&hyp, &fix.reference) * 100.0,
                )
            };

            fixture_results.push(ThresholdResult {
                threshold: thr,
                words_marked,
                segs_with_marks,
                llm_ms,
                hypothesis: hyp,
                wer,
                cer,
                bypassed,
            });
        }

        // Print table for this fixture.
        let fix_words: usize = asr_res
            .segments
            .iter()
            .flat_map(|s| &s.word_confidences)
            .count();

        println!();
        println!(
            "    {:>5}  {:>7}  {:>8}  {:>7}  {:>7}  {:>7}  {}",
            "Thr", "Marked", "Segs", "LLM ms", "WER %", "CER %", "Note"
        );
        println!("    {}", "─".repeat(62));

        for r in &fixture_results {
            let note = if r.bypassed { "BYPASS" } else { "" };
            println!(
                "    {:>4.0}%  {:>3}/{:<3}  {:>4}/{:<3}  {:>7.0}  {:>6.1}  {:>6.1}  {}",
                r.threshold * 100.0,
                r.words_marked,
                fix_words,
                r.segs_with_marks,
                asr_res.segments.len(),
                r.llm_ms,
                r.wer,
                r.cer,
                note,
            );
        }
        println!();

        all_results.push(fixture_results);
    }

    // ── Best threshold selection ────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              THRESHOLD SWEEP SUMMARY                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // For each threshold, compute average WER across fixtures.
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}",
        "Thr", "Avg WER %", "Avg CER %", "Avg LLM ms"
    );
    println!("  {}", "─".repeat(42));

    let mut best_thr = 0.0f32;
    let mut best_wer = f64::MAX;

    for (ti, &thr) in thresholds.iter().enumerate() {
        let mut sum_wer = 0.0;
        let mut sum_cer = 0.0;
        let mut sum_llm = 0.0;
        let mut count = 0;

        for fixture_results in &all_results {
            if let Some(r) = fixture_results.get(ti) {
                sum_wer += r.wer;
                sum_cer += r.cer;
                sum_llm += r.llm_ms;
                count += 1;
            }
        }

        if count > 0 {
            let avg_wer = sum_wer / count as f64;
            let avg_cer = sum_cer / count as f64;
            let avg_llm = sum_llm / count as f64;

            let marker = if avg_wer < best_wer {
                best_wer = avg_wer;
                best_thr = thr;
                " ◄ best"
            } else {
                ""
            };

            println!(
                "  {:>4.0}%  {:>9.1}  {:>9.1}  {:>9.0}{}",
                thr * 100.0,
                avg_wer,
                avg_cer,
                avg_llm,
                marker,
            );
        }
    }
    println!();

    println!(
        "  Recommended threshold: {:.0}% (lowest avg WER)",
        best_thr * 100.0
    );
    println!("  Total words with confidence data: {n_words}");
    println!();

    // Show best hypothesis for each fixture at the recommended threshold.
    for (i, fix) in fixtures.iter().enumerate() {
        if let Some(results) = all_results.get(i) {
            if let Some(best) = results
                .iter()
                .find(|r| (r.threshold - best_thr).abs() < 0.001)
            {
                println!(
                    "  {} at {:.0}% (WER={:.1}%, CER={:.1}%):",
                    fix.name,
                    best_thr * 100.0,
                    best.wer,
                    best.cer
                );
                println!("    Hypothesis:");
                wrap_print("      ", &best.hypothesis, 72);
                if !fix.reference.is_empty() {
                    println!("    Reference:");
                    wrap_print("      ", &fix.reference, 72);
                }
                println!();
            }
        }
    }

    // Quality context.
    println!("  WER = Word Error Rate (lower is better; 0% = perfect match).");
    println!("  CER = Character Error Rate (character-level edit distance).");
    println!();
    println!("  Quality context:");
    println!("    Industry reference WER (Whisper-large-v3, LibriSpeech clean): ~2.7%");
    println!("    Qwen3-ASR-0.6B INT4 streaming: 8s encoder window, past-text conditioned");
    println!("    Per-word [?word?] marking → surgical LLM span correction on uncertain words.");
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 5: VAD Timing Sweep — grid search over sentence_close × hard_close
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_vad_timing_sweep(asr: &(impl AsrBackend + Sync), fixtures: &[Fixture]) {
    use std::cell::RefCell;

    println!("── Stage 5: VAD Timing Sweep (Brent's method) ─────────────────────");
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // Shared eval log — records every point Brent probes.
    struct Probe {
        sentence_ms: u32,
        hard_ms: u32,
        avg_wer: f64,
        per_fixture: Vec<(f64, usize)>, // (wer%, n_segs) per fixture
    }
    let log: RefCell<Vec<Probe>> = RefCell::new(Vec::new());

    // Evaluate a (sentence_close, hard_close) config → avg WER%.
    let evaluate = |sentence_ms: u32, hard_ms: u32| -> f64 {
        let config = VadSegmenterConfig {
            sentence_close_ms: sentence_ms,
            hard_close_ms: hard_ms,
            ..Default::default()
        };
        let mut per_fixture = Vec::new();
        let mut wer_sum = 0.0;

        for fix in fixtures {
            let (segments, _, _) =
                rt.block_on(transcribe_timed_with_config(asr, &fix.pcm, config.clone()));
            let hypothesis = segments
                .iter()
                .map(|s| s.text.trim())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let n_segs = segments.len();
            let wer = word_error_rate(&hypothesis, &fix.reference) * 100.0;
            per_fixture.push((wer, n_segs));
            wer_sum += wer;
        }

        let avg = wer_sum / fixtures.len().max(1) as f64;

        // Live trace
        print!("    {:>6} {:>6}", sentence_ms, hard_ms);
        for (wer, segs) in &per_fixture {
            print!("  {:>5.1} {:>4}", wer, segs);
        }
        println!("  {:>6.1}%", avg);

        log.borrow_mut().push(Probe {
            sentence_ms,
            hard_ms,
            avg_wer: avg,
            per_fixture,
        });
        avg
    };

    // Table header.
    let print_header = || {
        print!("    {:>6} {:>6}", "sent", "hard");
        for fix in fixtures {
            let short: String = fix.name.chars().take(8).collect();
            print!("  {:>5} {:>4}", short, "segs");
        }
        println!("  avg_wer");
        print!("    {:>6} {:>6}", "──────", "──────");
        for _ in fixtures {
            print!("  {:>5} {:>4}", "─────", "────");
        }
        println!("  ───────");
    };

    // Run Brent for each hard_close ratio.
    // Reduces the 2D problem to 1D: optimize sentence_close, hard_close = ratio × sentence_close.
    let ratios = [2.0, 3.0];

    let mut winners: Vec<(u32, u32, f64)> = Vec::new(); // (sentence_ms, hard_ms, avg_wer)

    for &ratio in &ratios {
        println!(
            "  Brent search: sentence_close ∈ [50, 1000]ms, hard_close = {ratio:.0}× sentence"
        );
        print_header();

        let optimal = brent_minimize(
            |sent_f| {
                let sent = (sent_f.round() as u32).clamp(50, 3000);
                let hard = (sent as f64 * ratio) as u32;
                evaluate(sent, hard)
            },
            50.0,
            1000.0,
            0.02, // ~2% relative tolerance → converges to ~20-50ms precision
        );

        let sent = (optimal.round() as u32).clamp(50, 3000);
        let hard = (sent as f64 * ratio) as u32;

        // Find the actual WER for this point (may already be in the log).
        let wer = log
            .borrow()
            .iter()
            .find(|p| p.sentence_ms == sent && p.hard_ms == hard)
            .map(|p| p.avg_wer)
            .unwrap_or_else(|| evaluate(sent, hard));

        winners.push((sent, hard, wer));
        println!();
    }

    // ── Results ─────────────────────────────────────────────────────────
    // Sort all probed points by WER.
    let mut all: Vec<_> = log.into_inner();
    all.sort_by(|a, b| a.avg_wer.partial_cmp(&b.avg_wer).unwrap());

    println!("  Top 5 across all probes ({} total evals):", all.len());
    for (i, p) in all.iter().take(5).enumerate() {
        print!(
            "    {}. sent={}ms hard={}ms  avg_wer={:.1}%  |",
            i + 1,
            p.sentence_ms,
            p.hard_ms,
            p.avg_wer,
        );
        for (j, (wer, segs)) in p.per_fixture.iter().enumerate() {
            if j > 0 {
                print!(",");
            }
            print!(" {:.1}%/{segs}s", wer);
        }
        println!();
    }

    let best = &all[0];
    println!();
    println!(
        "  ★ Optimal: sentence_close={}ms  hard_close={}ms  avg_wer={:.1}%",
        best.sentence_ms, best.hard_ms, best.avg_wer,
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark-specific helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Transcribe with per-chunk timing. Returns (segments, total_secs, chunk_latencies_ms).
async fn transcribe_timed(
    backend: &(impl AsrBackend + Sync),
    pcm: &[f32],
) -> (Vec<AsrSegment>, f64, Vec<f64>) {
    transcribe_timed_with_config(backend, pcm, VadSegmenterConfig::default()).await
}

/// Batch transcription: push all audio, call poll_final once.
/// Used for overlap-stitched batch mode (no VadSegmenter).
async fn transcribe_batch(
    backend: &(impl AsrBackend + Sync),
    pcm: &[f32],
) -> (Vec<AsrSegment>, f64, Vec<f64>) {
    let session = backend
        .start_session(AsrSessionConfig::default())
        .await
        .expect("start asr session");

    let total_start = Instant::now();

    // Push all audio at once
    backend.push_audio(&session, pcm).await.expect("push audio");

    // Single poll_final triggers overlap-stitched decode
    let chunk_start = Instant::now();
    let out = backend
        .poll_final(&session)
        .await
        .expect("poll_final batch");
    let chunk_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;

    let total_secs = total_start.elapsed().as_secs_f64();
    backend.end_session(session).await.expect("end session");

    (out, total_secs, vec![chunk_ms])
}

/// Like `transcribe_timed` but with a custom VAD config.
async fn transcribe_timed_with_config(
    backend: &(impl AsrBackend + Sync),
    pcm: &[f32],
    vad_config: VadSegmenterConfig,
) -> (Vec<AsrSegment>, f64, Vec<f64>) {
    let session = backend
        .start_session(AsrSessionConfig::default())
        .await
        .expect("start asr session");

    let mut segmenter = VadSegmenter::new(vad_config);
    let chunk_size = 16_000; // 1s @ 16 kHz
    let mut out: Vec<AsrSegment> = Vec::new();
    let mut chunk_latencies: Vec<f64> = Vec::new();
    let total_start = Instant::now();

    for slice in pcm.chunks(chunk_size) {
        let (boundaries, in_speech) = segmenter.push(slice);
        if in_speech {
            let chunk_start = Instant::now();
            backend
                .push_audio(&session, slice)
                .await
                .expect("push audio");
            out.extend(backend.poll_results(&session).await.expect("poll results"));
            chunk_latencies.push(chunk_start.elapsed().as_secs_f64() * 1000.0);
        }
        // Flush ASR + reset context at segment boundaries
        for _boundary in &boundaries {
            let chunk_start = Instant::now();
            out.extend(
                backend
                    .poll_final(&session)
                    .await
                    .expect("poll_final at boundary"),
            );
            chunk_latencies.push(chunk_start.elapsed().as_secs_f64() * 1000.0);
            backend
                .reset_context(&session)
                .await
                .expect("reset at boundary");
        }
    }

    // Flush remaining speech from segmenter + ASR
    segmenter.finish();
    let silence = vec![0.0f32; 16_000];
    let chunk_start = Instant::now();
    backend
        .push_audio(&session, &silence)
        .await
        .expect("push silence");
    out.extend(
        backend
            .poll_final(&session)
            .await
            .expect("poll results final"),
    );
    chunk_latencies.push(chunk_start.elapsed().as_secs_f64() * 1000.0);

    let total_secs = total_start.elapsed().as_secs_f64();
    backend.end_session(session).await.expect("end session");

    (out, total_secs, chunk_latencies)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 6: Adaptive VAD Validation
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_adaptive_vad(asr: &(impl AsrBackend + Sync), fixtures: &[Fixture]) {
    println!("── Stage 6: Adaptive VAD Validation ──────────────────────────────");
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // Run each fixture with adaptive VAD and compare to static defaults
    for fix in fixtures {
        // Static baseline
        let (static_segs, static_secs, _) = rt.block_on(transcribe_timed(asr, &fix.pcm));
        let static_text = segments_to_text(&static_segs);
        let static_wer = word_error_rate(&static_text, &fix.reference) * 100.0;

        // Adaptive mode
        let adaptive_config = VadSegmenterConfig::adaptive();
        let (adaptive_segs, adaptive_secs, _) =
            rt.block_on(transcribe_timed_with_config(asr, &fix.pcm, adaptive_config));
        let adaptive_text = segments_to_text(&adaptive_segs);
        let adaptive_wer = word_error_rate(&adaptive_text, &fix.reference) * 100.0;

        // Get adaptive state from a fresh run to show convergence
        let mut segmenter = VadSegmenter::new(VadSegmenterConfig::adaptive());
        for chunk in fix.pcm.chunks(16_000) {
            segmenter.push(chunk);
        }
        segmenter.finish();
        let adaptive_state = segmenter
            .debug_adaptive_state()
            .unwrap_or_else(|| "N/A".into());

        let delta = adaptive_wer - static_wer;
        let arrow = if delta < -0.5 {
            "▼ improved"
        } else if delta > 0.5 {
            "▲ regressed"
        } else {
            "≈ neutral"
        };

        println!("  {}:", fix.name);
        println!(
            "    Static:   WER {:.1}%  ({} segs, {:.2}s)",
            static_wer,
            static_segs.len(),
            static_secs
        );
        println!(
            "    Adaptive: WER {:.1}%  ({} segs, {:.2}s)  [{} Δ{:+.1}%]",
            adaptive_wer,
            adaptive_segs.len(),
            adaptive_secs,
            arrow,
            delta
        );
        println!("    Learned:  {}", adaptive_state);
        println!();
    }
}

fn segments_to_text(segs: &[AsrSegment]) -> String {
    segs.iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 7: FP16 Decoder vs INT4 Decoder Comparison
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_fp16_decoder(
    fp16_asr: &(impl AsrBackend + Sync),
    int4_results: &[AsrResult],
    fixtures: &[Fixture],
) {
    println!("── Stage 7: FP16 Decoder vs INT4 Decoder ─────────────────────────");
    println!();
    println!("  INT4 decoder: quantized weights (MatMulNBits), ~516 MB");
    println!("  FP16 decoder: full precision, ~1137 MB");
    println!("  Hypothesis: INT4 autoregressive error compounding degrades WER.");
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    for (i, fix) in fixtures.iter().enumerate() {
        let int4_res = &int4_results[i];
        let int4_wer = word_error_rate(&int4_res.hypothesis, &fix.reference) * 100.0;
        let int4_cer = char_error_rate(&int4_res.hypothesis, &fix.reference) * 100.0;

        // Run FP16 decoder on same audio
        let (fp16_segs, fp16_secs, _) = rt.block_on(transcribe_timed(fp16_asr, &fix.pcm));
        let fp16_text = fp16_segs
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let fp16_wer = word_error_rate(&fp16_text, &fix.reference) * 100.0;
        let fp16_cer = char_error_rate(&fp16_text, &fix.reference) * 100.0;

        let delta_wer = fp16_wer - int4_wer;
        let arrow = if delta_wer < -0.5 {
            "▼ improved"
        } else if delta_wer > 0.5 {
            "▲ regressed"
        } else {
            "≈ neutral"
        };

        println!("  {} ({:.1}s audio):", fix.name, fix.audio_secs);
        println!(
            "    INT4:  WER {:.1}%  CER {:.1}%  ({} segs, {:.2}s)",
            int4_wer,
            int4_cer,
            int4_res.segments.len(),
            int4_res.total_secs
        );
        println!(
            "    FP16:  WER {:.1}%  CER {:.1}%  ({} segs, {:.2}s)  [{} Δ{:+.1}%]",
            fp16_wer,
            fp16_cer,
            fp16_segs.len(),
            fp16_secs,
            arrow,
            delta_wer
        );
        println!("    INT4 hypothesis:");
        wrap_print("      ", &int4_res.hypothesis, 72);
        println!("    FP16 hypothesis:");
        wrap_print("      ", &fp16_text, 72);
        if !fix.reference.is_empty() {
            println!("    Reference:");
            wrap_print("      ", &fix.reference, 72);
        }
        println!();
    }
}
