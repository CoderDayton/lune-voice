//! Smoke test: load Qwen3-0.6B ONNX model and run single correction.
//! Skip with `cargo test` (ignored by default).

use std::path::PathBuf;
use std::sync::Arc;

use lune_core::config::StylePreset;
use lune_core::dict::DictionaryEntry;
use lune_core::llm::{LlmService, PostProcessInput, TextPostProcessor};
use lune_llm::CorrectionPipeline;
use lune_llm::backend::ort_llm::{OrtStyleLlm, OrtStyleLlmConfig};

fn model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = PathBuf::from(home).join(".local/share/lune-voice/models/qwen3-0.6b");
    dir.exists().then_some(dir)
}

fn make_llm() -> OrtStyleLlm {
    let dir = model_dir().expect("model not found");
    let config = OrtStyleLlmConfig {
        model_dir: dir,
        use_gpu: true,
        cuda_required: false,
        max_new_tokens: 512,
    };
    OrtStyleLlm::new(config).expect("failed to load model")
}

fn run_case(llm: &OrtStyleLlm, input: &str, uncertain: &[(String, usize)]) -> String {
    let result = llm
        .apply_correction(input, "en", uncertain)
        .expect("apply_correction failed");
    println!("INPUT:  {input}");
    if !uncertain.is_empty() {
        let words: Vec<&str> = uncertain.iter().map(|(w, _)| w.as_str()).collect();
        println!("HINTS:  {}", words.join(", "));
    }
    println!("OUTPUT: {result}");
    println!("---");
    result
}

#[test]
#[ignore]
fn smoke_correction() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    let llm = make_llm();

    // Basic typo + capitalization
    let r1 = run_case(&llm, "hello wrld how you doing today.", &[]);

    // Missing words + misheard word with hint
    let r2 = run_case(
        &llm,
        "I went store and brought some groceries",
        &[("brought".into(), 4)],
    );

    // Missing small words
    let _r3 = run_case(&llm, "can you tell me what time it", &[]);

    // Already correct - should pass through
    let r4 = run_case(&llm, "The quick brown fox jumps over the lazy dog.", &[]);

    // Multiple issues
    let r5 = run_case(
        &llm,
        "i dont no why the computor isnt working",
        &[("no".into(), 2), ("computor".into(), 4)],
    );

    let _r6 = run_case(&llm, "let go to the mall and bye some cloth", &[]);

    let _r7 = run_case(&llm, "let meet at my hotle and git some sleep!", &[]);

    // No echo of system prompt or hints
    assert!(!r1.contains("Fix"), "leaked system prompt: {r1}");
    assert!(
        !r5.contains("Correct the following transcript:"),
        "leaked user prompt: {r5}"
    );
    assert!(
        !r5.contains("Possibly misheard"),
        "leaked user prompt: {r5}"
    );
    assert!(!r1.contains("Transcript"), "leaked prompt structure: {r1}");
    assert!(!r1.is_empty(), "output empty");
    assert!(
        !r2.contains("Possibly misheard"),
        "leaked hint prefix: {r2}"
    );

    // Clean text should be mostly unchanged
    assert!(r4.contains("quick brown fox"), "mangled clean text: {r4}");
}

#[test]
#[ignore]
fn dictionary_example_correction_smoke() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    let llm = Arc::new(make_llm());
    let pipeline = CorrectionPipeline::new(Arc::clone(&llm) as Arc<dyn LlmService>);
    let input = PostProcessInput {
        raw_text: "I gave a talk in Barolo about transcription.".into(),
        style_preset: Arc::new(StylePreset::minimal()),
        app_category: None,
        dictionary: vec![DictionaryEntry {
            phrase: "Barolo".into(),
            replacement: "Boorloo".into(),
            priority: 10,
        }],
        context: None,
        language: "en".into(),
        uncertain_words: vec![],
        correction_needed: true,
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let output = rt.block_on(pipeline.transform(&input)).unwrap();
    println!("OUTPUT: {}", output.cleaned_text);
    assert!(
        output.cleaned_text.contains("Boorloo"),
        "dictionary example did not produce replacement: {}",
        output.cleaned_text
    );
}
