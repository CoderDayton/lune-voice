//! Integration tests for the KV cache prefix caching.
//! Requires the Qwen3-0.6B ONNX model on disk.
//! Run: cargo test -p lune-llm --test kv_cache_integration -- --ignored

use std::path::PathBuf;
use std::time::Instant;

use lune_llm::backend::ort_llm::{OrtStyleLlm, OrtStyleLlmConfig};

fn model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = PathBuf::from(home).join(".local/share/lune-voice/models/qwen3-0.6b");
    dir.exists().then_some(dir)
}

fn make_llm() -> OrtStyleLlm {
    let dir = model_dir().expect("model not found — skip with cargo test (no --ignored)");
    let config = OrtStyleLlmConfig {
        model_dir: dir,
        use_gpu: true,
        cuda_required: false,
        max_new_tokens: 128,
    };
    OrtStyleLlm::new(config).expect("failed to load model")
}

fn run(llm: &OrtStyleLlm, input: &str) -> (String, std::time::Duration) {
    let t0 = Instant::now();
    let result = llm
        .apply_correction(input, "en")
        .expect("apply_correction failed");
    (result, t0.elapsed())
}

/// Repeated calls with the same system prompt should reuse the prefix cache.
/// The second call should be faster than the first (first prefills + freezes).
#[test]
#[ignore]
fn cached_prefix_speedup() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    let llm = make_llm();

    // Warmup / first call — builds and freezes the prefix cache.
    let (text1, dur1) = run(&llm, "hello wrld how you doing");
    println!(
        "Call 1 (cold): {:.1}ms — {text1}",
        dur1.as_secs_f64() * 1000.0
    );

    // Second call — should reuse the cached prefix.
    let (text2, dur2) = run(&llm, "i went store and brought some groceries");
    println!(
        "Call 2 (warm): {:.1}ms — {text2}",
        dur2.as_secs_f64() * 1000.0
    );

    // Third call — also cached.
    let (text3, dur3) = run(&llm, "tell me what time it");
    println!(
        "Call 3 (warm): {:.1}ms — {text3}",
        dur3.as_secs_f64() * 1000.0
    );

    assert!(!text1.is_empty(), "empty output on call 1");
    assert!(!text2.is_empty(), "empty output on call 2");
    assert!(!text3.is_empty(), "empty output on call 3");

    // The warm calls should be at least somewhat faster.
    // We don't enforce a hard ratio because model load jitter is high,
    // but we print the speedup for manual inspection.
    let speedup = dur1.as_secs_f64() / dur3.as_secs_f64();
    println!("Speedup (call1/call3): {speedup:.2}x");
}

/// Changing language (which changes the system prompt) should invalidate the cache
/// and produce valid output with the new prompt.
#[test]
#[ignore]
fn cache_invalidation_on_language_change() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    let llm = make_llm();

    // English — builds cache.
    let r1 = llm.apply_correction("hello wrld", "en").expect("en failed");
    println!("EN: {r1}");

    // French — different system prompt, cache should be invalidated.
    let r2 = llm
        .apply_correction("bonjur le monde", "fr")
        .expect("fr failed");
    println!("FR: {r2}");

    // Back to English — should rebuild cache.
    let r3 = llm
        .apply_correction("how are you doing today", "en")
        .expect("en2 failed");
    println!("EN2: {r3}");

    assert!(!r1.is_empty());
    assert!(!r2.is_empty());
    assert!(!r3.is_empty());
}

/// Output quality with cached prefix should not be catastrophically worse
/// than what the model produces. Basic sanity: clean input stays clean.
#[test]
#[ignore]
fn cached_output_quality() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    let llm = make_llm();

    // Warm up cache.
    let _ = run(&llm, "warmup text for cache");

    // Clean text should pass through mostly unchanged.
    let (clean, _) = run(&llm, "The quick brown fox jumps over the lazy dog.");
    println!("Clean: {clean}");
    assert!(
        clean.contains("quick brown fox"),
        "mangled clean text with cached prefix: {clean}"
    );

    // Obvious typo should be corrected.
    let (fixed, _) = run(&llm, "i dont no why computor isnt working");
    println!("Fixed: {fixed}");
    let lower = fixed.to_lowercase();
    assert!(
        lower.contains("know") || lower.contains("computer"),
        "cached prefix broke correction: {fixed}"
    );
}

/// Multiple rapid-fire calls to stress the freeze/decompress cycle.
#[test]
#[ignore]
fn rapid_fire_stability() {
    let llm = make_llm();

    let inputs = [
        "hello wrld",
        "i went store",
        "tell me what time it",
        "the wether is nice today",
        "can you help me with this",
        "i need to bye some groceries",
        "lets meet at the hotle",
        "please send me the reporte",
    ];

    for (i, input) in inputs.iter().enumerate() {
        let (result, dur) = run(&llm, input);
        println!("[{i}] {:.1}ms — {result}", dur.as_secs_f64() * 1000.0);
        assert!(!result.is_empty(), "empty on call {i}");
    }
}
