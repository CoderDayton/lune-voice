//! LLM KV cache prefix benchmark.
//!
//! Compares cold (no cache) vs warm (cached prefix) latency over repeated
//! apply_style calls with the same system prompt.
//!
//! GPU: cargo run --release --features cuda --example llm_kv_cache_bench -p lune-llm
//! CPU: cargo run --release --example llm_kv_cache_bench -p lune-llm

use std::path::PathBuf;
use std::time::Instant;

use lune_core::config::StylePreset;
use lune_llm::backend::ort_llm::{OrtStyleLlm, OrtStyleLlmConfig};

const INPUTS: &[&str] = &[
    "hello wrld how you doing today",
    "i went store and brought some groceries",
    "tell me what time it",
    "the wether is really nice today isnt it",
    "can you help me with this computor problem",
    "i dont no why it isnt working properly",
    "lets meet at the hotle and git some sleep",
    "please send me the reporte when your done",
    "i need to bye some cloth from the mall",
    "she said she wood come to the party",
];

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    let dir = PathBuf::from(home).join(".local/share/lune-voice/models/qwen3-0.6b");
    assert!(dir.exists(), "model not found at {}", dir.display());
    dir
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    let use_gpu = cfg!(feature = "cuda");
    println!("╔══════════════════════════════════════════════╗");
    println!("║     LLM KV Cache Prefix Benchmark           ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  device: {}", if use_gpu { "CUDA" } else { "CPU" });
    println!("  inputs: {}", INPUTS.len());
    println!();

    let llm = OrtStyleLlm::new(OrtStyleLlmConfig {
        model_dir: model_dir(),
        use_gpu,
        max_new_tokens: 128,
    })
    .expect("failed to load model");

    let style = StylePreset::minimal();

    // ── Run all inputs, collect per-call timings ────────────────────────
    let mut timings_ms: Vec<f64> = Vec::with_capacity(INPUTS.len());
    let mut outputs: Vec<String> = Vec::with_capacity(INPUTS.len());

    for (i, input) in INPUTS.iter().enumerate() {
        let t0 = Instant::now();
        let result = llm
            .apply_style(input, "en", &style, &[])
            .expect("apply_style failed");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        timings_ms.push(ms);
        outputs.push(result);
        let tag = if i == 0 { "COLD" } else { "WARM" };
        println!("  [{tag:4}] {ms:7.1}ms  {input}",);
        println!("  {:>14}  {}", "→", &outputs[i]);
    }

    println!();

    // ── Summary stats ──────────────────────────────────────────────────
    let cold_ms = timings_ms[0];
    let warm_timings: &[f64] = &timings_ms[1..];
    let warm_avg: f64 = warm_timings.iter().sum::<f64>() / warm_timings.len() as f64;
    let warm_min: f64 = warm_timings.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let warm_max: f64 = warm_timings.iter().copied().reduce(f64::max).unwrap_or(0.0);
    let warm_p50 = percentile(warm_timings, 0.5);
    let warm_p95 = percentile(warm_timings, 0.95);

    println!("  ┌─────────────────────────────────────┐");
    println!("  │ Cold (1st call):     {cold_ms:7.1} ms      │");
    println!("  │ Warm avg:            {warm_avg:7.1} ms      │");
    println!("  │ Warm min:            {warm_min:7.1} ms      │");
    println!("  │ Warm max:            {warm_max:7.1} ms      │");
    println!("  │ Warm p50:            {warm_p50:7.1} ms      │");
    println!("  │ Warm p95:            {warm_p95:7.1} ms      │");
    println!(
        "  │ Speedup (cold/avg):  {:7.2}x       │",
        cold_ms / warm_avg
    );
    println!("  └─────────────────────────────────────┘");
}

fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
