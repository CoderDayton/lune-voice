//! Reranking ablation — compare WER across correction paths:
//!
//!   1. Raw ASR (baseline)
//!   2. + Nemotron reranker only
//!   3. + LLM correction only
//!   4. + Reranker → LLM (full pipeline)
//!
//! Shows whether the reranker, the LLM, or both contribute to quality,
//! and whether the LLM alone can subsume the reranker.
//!
//! Run:  cargo run --release --example rerank_ablation -p lune-app
//! GPU:  cargo run --release --features cuda --example rerank_ablation -p lune-app

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    Fixture, benchmark_asr_entry, init_ort, load_fixtures, word_error_rate, workspace_root,
};
use lune_asr::{AsrRegistry, IsotonicCalibrator, NemotronReranker, NemotronRerankerConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::DevicePolicy;
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_llm::backend::ort_style::{OrtStyleLlm, OrtStyleLlmConfig};

fn collect_nbest(segments: &[AsrSegment]) -> Vec<String> {
    let any_multi = segments.iter().any(|s| s.hypotheses.len() >= 2);
    if !any_multi {
        return Vec::new();
    }
    let argmaxes: Vec<&str> = segments
        .iter()
        .map(|s| {
            s.hypotheses
                .first()
                .map(String::as_str)
                .unwrap_or(s.text.as_str())
        })
        .collect();
    let argmax_sentence = argmaxes.join(" ");
    let mut result: Vec<String> = vec![argmax_sentence.clone()];
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(argmax_sentence.to_ascii_lowercase());

    for (seg_idx, seg) in segments.iter().enumerate() {
        for alt in seg.hypotheses.iter().skip(1) {
            if result.len() >= 5 {
                break;
            }
            let mut pieces: Vec<&str> = argmaxes.clone();
            pieces[seg_idx] = alt.as_str();
            let aggregate = pieces.join(" ");
            let key = aggregate.to_ascii_lowercase();
            if seen.insert(key) {
                result.push(aggregate);
            }
        }
        if result.len() >= 5 {
            break;
        }
    }
    result
}

fn rerank_pick(reranker: &NemotronReranker, hypotheses: &[String], margin: f32) -> Option<String> {
    if hypotheses.len() < 2 {
        return None;
    }
    let query = "most fluent and contextually plausible transcription";
    let scores = reranker.score(query, hypotheses).ok()?;
    let argmax_score = scores.first().copied().unwrap_or(0.0);
    let (best_idx, best_score) = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, &s)| (i, s))
        .unwrap_or((0, argmax_score));
    let chosen_idx = if best_idx == 0 || (best_score - argmax_score) >= margin {
        best_idx
    } else {
        0
    };
    Some(hypotheses[chosen_idx].clone())
}

#[allow(dead_code)]
struct PathResult {
    name: &'static str,
    text: String,
    wer: f64,
    ms: f64,
}

struct FixtureResult {
    fixture: &'static str,
    audio_secs: f64,
    nbest_count: usize,
    paths: Vec<PathResult>,
}

fn run_fixture(
    fixture: &Fixture,
    asr: &AsrRegistry,
    reranker: Option<&NemotronReranker>,
    llm: Option<&dyn TextPostProcessor>,
    calibrator: &IsotonicCalibrator,
    margin: f32,
    mark_threshold: f32,
    rt: &tokio::runtime::Runtime,
) -> FixtureResult {
    // Decode
    let session_config = AsrSessionConfig {
        language: Some("en".into()),
        ..Default::default()
    };
    let session = rt
        .block_on(asr.start_session(session_config))
        .expect("start");
    rt.block_on(asr.push_audio(&session, &fixture.pcm))
        .expect("push");
    let mut segments = rt.block_on(asr.poll_final(&session)).expect("poll");

    // Calibrate + build hypotheses
    for seg in &mut segments {
        for wc in &mut seg.word_confidences {
            wc.confidence = calibrator.calibrate(wc.confidence);
        }
        seg.build_hypotheses(mark_threshold, 3);
    }

    let raw_text: String = segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let hypotheses = collect_nbest(&segments);
    let nbest_count = hypotheses.len();

    let ref_text = &fixture.reference;
    let mut paths = Vec::new();

    // Path 1: Raw ASR
    paths.push(PathResult {
        name: "Raw ASR",
        wer: word_error_rate(&raw_text, ref_text),
        text: raw_text.clone(),
        ms: 0.0,
    });

    // Path 2: Reranker only
    if let Some(rr) = reranker {
        let t0 = Instant::now();
        let chosen = rerank_pick(rr, &hypotheses, margin).unwrap_or_else(|| raw_text.clone());
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        paths.push(PathResult {
            name: "Reranker",
            wer: word_error_rate(&chosen, ref_text),
            text: chosen,
            ms,
        });
    }

    // Path 3: LLM correction only (no reranker, passes hypotheses to LLM)
    if let Some(llm) = llm {
        let t0 = Instant::now();
        let style = lune_core::config::StylePreset::casual();
        let input = PostProcessInput {
            raw_text: raw_text.clone(),
            style_preset: Arc::new(style),
            app_category: None,
            dictionary: Vec::new(),
            context: None,
            language: "en".into(),
            hypotheses: hypotheses.clone(),
        };
        let result = rt.block_on(llm.transform(&input));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(output) => {
                paths.push(PathResult {
                    name: "LLM only",
                    wer: word_error_rate(&output.cleaned_text, ref_text),
                    text: output.cleaned_text,
                    ms,
                });
            }
            Err(e) => {
                eprintln!("    [warn] LLM failed for {}: {e}", fixture.name);
            }
        }
    }

    // Path 4: Reranker → LLM (full pipeline)
    if let (Some(rr), Some(llm)) = (reranker, llm) {
        let t0 = Instant::now();
        let after_rerank = rerank_pick(rr, &hypotheses, margin).unwrap_or_else(|| raw_text.clone());
        let style = lune_core::config::StylePreset::casual();
        let input = PostProcessInput {
            raw_text: after_rerank,
            style_preset: Arc::new(style),
            app_category: None,
            dictionary: Vec::new(),
            context: None,
            language: "en".into(),
            hypotheses: hypotheses.clone(),
        };
        let result = rt.block_on(llm.transform(&input));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(output) => {
                paths.push(PathResult {
                    name: "Rerank+LLM",
                    wer: word_error_rate(&output.cleaned_text, ref_text),
                    text: output.cleaned_text,
                    ms,
                });
            }
            Err(e) => {
                eprintln!("    [warn] Rerank+LLM failed for {}: {e}", fixture.name);
            }
        }
    }

    FixtureResult {
        fixture: fixture.name,
        audio_secs: fixture.audio_secs,
        nbest_count,
        paths,
    }
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                   RERANKING ABLATION                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();
    let root = workspace_root();
    let device = if has_gpu {
        DevicePolicy::Cuda
    } else {
        DevicePolicy::Cpu
    };
    let models_dir = root.join("models");

    let asr_entry = match benchmark_asr_entry(&root) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("  [skip] {msg}");
            return;
        }
    };
    println!("  ASR: {}", asr_entry.display_name);

    let t0 = Instant::now();
    let asr = AsrRegistry::from_config(&asr_entry.config, device, &models_dir).expect("load ASR");
    println!(
        "  ASR loaded in {:.0}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Load reranker
    let reranker_dir = models_dir.join("nemotron-rerank-1b");
    let reranker = if reranker_dir.join("tokenizer.json").is_file() {
        print!("  Loading reranker ... ");
        let t = Instant::now();
        match NemotronReranker::new(NemotronRerankerConfig {
            model_dir: reranker_dir,
            use_gpu: has_gpu,
        }) {
            Ok(rr) => {
                println!("{:.0}ms", t.elapsed().as_secs_f64() * 1000.0);
                Some(rr)
            }
            Err(e) => {
                println!("failed: {e}");
                None
            }
        }
    } else {
        println!("  Reranker: not found");
        None
    };

    // Load LLM
    let style_dir = models_dir.join("qwen3-0.6b");
    let llm: Option<Box<dyn TextPostProcessor>> = if style_dir.join("tokenizer.json").is_file() {
        print!("  Loading LLM ... ");
        let t = Instant::now();
        match OrtStyleLlm::new(OrtStyleLlmConfig {
            model_dir: style_dir,
            use_gpu: has_gpu,
            max_new_tokens: 128,
        }) {
            Ok(l) => {
                println!("{:.0}ms", t.elapsed().as_secs_f64() * 1000.0);
                Some(Box::new(l) as Box<dyn TextPostProcessor>)
            }
            Err(e) => {
                println!("failed: {e}");
                None
            }
        }
    } else {
        println!("  LLM: not found");
        None
    };

    if reranker.is_none() && llm.is_none() {
        eprintln!("\n  [skip] Neither reranker nor LLM available — nothing to compare");
        return;
    }

    // Load calibrator
    let slug = asr_entry.backend_kind.slug();
    let data_dir = std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".local/share/lune-voice"))
        .unwrap_or_default();
    let cal_path = data_dir.join(format!("calibrator_{slug}.json"));
    let calibrator = std::fs::read_to_string(&cal_path)
        .ok()
        .and_then(|j| serde_json::from_str::<IsotonicCalibrator>(&j).ok())
        .unwrap_or_else(IsotonicCalibrator::identity);

    let mark_threshold = std::env::var("LUNE_MARK_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.65);
    let margin = std::env::var("LUNE_RERANKER_MARGIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(5.0);

    println!("  Mark threshold: {mark_threshold}  Reranker margin: {margin}");

    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        eprintln!("  [skip] No fixtures");
        return;
    }

    let fixture_filter = std::env::var("LUNE_BENCH_FIXTURE").ok();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut results = Vec::new();

    for fixture in &fixtures {
        if let Some(ref needle) = fixture_filter {
            if !fixture.name.contains(needle.as_str()) {
                continue;
            }
        }
        let result = run_fixture(
            fixture,
            &asr,
            reranker.as_ref(),
            llm.as_deref(),
            &calibrator,
            margin,
            mark_threshold,
            &rt,
        );
        results.push(result);
    }

    // Print per-fixture table
    println!();
    let path_names: Vec<&str> = if let Some(first) = results.first() {
        first.paths.iter().map(|p| p.name).collect()
    } else {
        return;
    };

    print!("  {:24} {:>5} {:>6}", "Fixture", "Audio", "Nbest");
    for name in &path_names {
        print!("  {:>12}", format!("{name} WER%"));
    }
    println!();
    println!("  {}", "─".repeat(24 + 5 + 6 + path_names.len() * 14 + 4));

    for r in &results {
        print!(
            "  {:24} {:4.1}s {:>5}",
            r.fixture, r.audio_secs, r.nbest_count
        );
        for p in &r.paths {
            print!("  {:>11.1}%", p.wer * 100.0);
        }
        println!();
    }

    // Aggregate
    if results.len() > 1 {
        println!("  {}", "─".repeat(24 + 5 + 6 + path_names.len() * 14 + 4));
        print!("  {:24} {:>5} {:>6}", "AVERAGE", "", "");
        for (i, _name) in path_names.iter().enumerate() {
            let avg: f64 = results
                .iter()
                .filter_map(|r| r.paths.get(i).map(|p| p.wer * 100.0))
                .sum::<f64>()
                / results.len() as f64;
            print!("  {:>11.1}%", avg);
        }
        println!();
    }

    // Latency summary
    println!("\n  Latency (ms):");
    for (i, name) in path_names.iter().enumerate() {
        if i == 0 {
            continue;
        } // raw has no latency
        let times: Vec<f64> = results
            .iter()
            .filter_map(|r| r.paths.get(i).map(|p| p.ms))
            .collect();
        if times.is_empty() {
            continue;
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let max = times.iter().cloned().fold(f64::MIN, f64::max);
        println!("    {name:12}  avg: {avg:7.1}  max: {max:7.1}");
    }

    // Delta summary
    println!("\n  WER delta vs raw (negative = improvement):");
    for (i, name) in path_names.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let deltas: Vec<f64> = results
            .iter()
            .filter_map(|r| {
                let raw_wer = r.paths.first()?.wer;
                let this_wer = r.paths.get(i)?.wer;
                Some((this_wer - raw_wer) * 100.0)
            })
            .collect();
        if deltas.is_empty() {
            continue;
        }
        let avg = deltas.iter().sum::<f64>() / deltas.len() as f64;
        let improved = deltas.iter().filter(|d| **d < -0.001).count();
        let regressed = deltas.iter().filter(|d| **d > 0.001).count();
        println!(
            "    {name:12}  avg: {:+5.1}pp  improved: {improved}/{} regressed: {regressed}/{}",
            avg,
            deltas.len(),
            deltas.len(),
        );
    }

    println!();
}
