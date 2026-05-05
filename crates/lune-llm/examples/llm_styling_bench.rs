//! Styling pass benchmark.
//!
//! Runs one fixture through correction, then applies every style preset
//! and prints the output + timing for each.  Use this to tweak prompts.
//!
//! GPU: cargo run --release --features cuda --example llm_styling_bench -p lune-llm
//! CPU: cargo run --release --example llm_styling_bench -p lune-llm

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use lune_core::config::StylePreset;
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_llm::backend::ort_llm::{OrtStyleLlm, OrtStyleLlmConfig};

const DEFAULT_FIXTURE: &str = "so i went to the store and brought some groceries and then i came home and i cooked dinner for my family and it was really good they said they liked it a lot";

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    let dir = PathBuf::from(home).join(".local/share/lune-voice/models/qwen3-0.6b");
    assert!(dir.exists(), "model not found at {}", dir.display());
    dir
}

fn load_fixture() -> String {
    if let Some(path) = std::env::args().nth(1) {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"))
            .trim()
            .to_string()
    } else {
        DEFAULT_FIXTURE.to_string()
    }
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    let use_gpu = cfg!(feature = "cuda");
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║          LLM Styling Pass Benchmark                 ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  device: {}", if use_gpu { "CUDA" } else { "CPU" });
    println!();

    let llm = OrtStyleLlm::new(OrtStyleLlmConfig {
        model_dir: model_dir(),
        use_gpu,
        max_new_tokens: 256,
    })
    .expect("failed to load model");

    let fixture = load_fixture();
    println!("  INPUT:  {fixture}");
    println!();

    // ── Correction pass (warms the prefix cache) ─────────────────────
    // Use the minimal preset so transform() runs correction only (no styling).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let minimal_input = PostProcessInput {
        raw_text: fixture.clone(),
        style_preset: Arc::new(StylePreset::minimal()),
        app_category: None,
        dictionary: vec![],
        context: None,
        language: "en".into(),
        uncertain_words: vec![],
    };

    let t0 = Instant::now();
    let corrected_output = rt
        .block_on(llm.transform(&minimal_input))
        .expect("correction failed");
    let correction_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let corrected = &corrected_output.cleaned_text;

    println!("  CORRECTED ({correction_ms:.1}ms):");
    println!("    {corrected}");
    println!();

    // ── Styling passes ───────────────────────────────────────────────
    let presets = StylePreset::all_builtin();
    let mut style_timings: Vec<(&str, f64)> = Vec::with_capacity(presets.len());

    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  Style Results                                         │");
    println!("  └─────────────────────────────────────────────────────────┘");

    for preset in presets {
        if !preset.needs_styling() {
            println!("  [{:12}] SKIPPED (no styling needed)", preset.id);
            println!();
            continue;
        }

        let t0 = Instant::now();
        let styled = match llm.apply_styling(corrected, preset) {
            Ok(s) => s,
            Err(e) => {
                println!("  [{:12}] ERROR: {e}", preset.id);
                println!();
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        style_timings.push((&preset.id, ms));

        let char_delta = styled.len() as isize - corrected.len() as isize;
        let word_count = styled.split_whitespace().count();

        println!(
            "  [{:12}] {ms:7.1}ms  ({word_count} words, {char_delta:+} chars)",
            preset.id
        );
        println!("    {styled}");
        println!();
    }

    // ── Summary ──────────────────────────────────────────────────────
    if style_timings.is_empty() {
        return;
    }

    let total_style_ms: f64 = style_timings.iter().map(|(_, ms)| ms).sum();
    let avg_style_ms = total_style_ms / style_timings.len() as f64;
    let min_style = style_timings
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let max_style = style_timings
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  Summary                                               │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │  Correction:      {correction_ms:7.1} ms                          │");
    println!("  │  Styling avg:     {avg_style_ms:7.1} ms                          │");
    println!(
        "  │  Styling min:     {:7.1} ms  ({:12})            │",
        min_style.1, min_style.0
    );
    println!(
        "  │  Styling max:     {:7.1} ms  ({:12})            │",
        max_style.1, max_style.0
    );
    println!(
        "  │  Total (corr+avg): {:7.1} ms                          │",
        correction_ms + avg_style_ms
    );
    println!("  └─────────────────────────────────────────────────────────┘");
}
