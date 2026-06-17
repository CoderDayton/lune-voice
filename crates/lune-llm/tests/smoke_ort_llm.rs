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

fn run_case(llm: &OrtStyleLlm, input: &str) -> String {
    let result = llm
        .apply_correction(input, "en")
        .expect("apply_correction failed");
    println!("INPUT:  {input}");
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
    let r1 = run_case(&llm, "hello wrld how you doing today.");

    // Missing words + misheard word
    let r2 = run_case(&llm, "I went store and brought some groceries");

    // Missing small words
    let _r3 = run_case(&llm, "can you tell me what time it");

    // Already correct - should pass through
    let r4 = run_case(&llm, "The quick brown fox jumps over the lazy dog.");

    // Multiple issues
    let r5 = run_case(&llm, "i dont no why the computor isnt working");

    let _r6 = run_case(&llm, "let go to the mall and bye some cloth");

    let _r7 = run_case(&llm, "let meet at my hotle and git some sleep!");

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
        language: "en".into(),
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

/// Decisive check for the open-vocab edit gate: a real Qwen3 decode must hand
/// back NON-EMPTY, per-word-aligned decode margins. If `complete_scored`
/// returned empty margins on real output, the gate's `admits(None)` is always
/// false and the whole channel is a silent no-op — indistinguishable from "the
/// gate didn't help." This test rules that out by exercising the production
/// scored path and asserting one margin per output word.
#[test]
fn complete_scored_emits_aligned_word_margins() {
    // Runs whenever the real model is present, so default `cargo test` exercises
    // the production margin path; skips cleanly when models aren't on disk.
    if model_dir().is_none() {
        eprintln!("[skip] qwen3-0.6b model not present");
        return;
    }
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();

    let llm = make_llm();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // Static system + the public production user-prompt builder. The system
    // text does not affect margin attribution — only the decoded answer's
    // token→word structure does — so a representative prompt is faithful.
    let system = "You correct speech-to-text transcripts for language en. \
                  Fix spelling, spacing, and punctuation. Output only the \
                  corrected transcript, nothing else."
        .to_string();

    let cases = [
        "i dont no why the computor isnt working",
        "let go to the mall and bye some cloth",
        "The quick brown fox jumps over the lazy dog.",
    ];

    for input in cases {
        let user = OrtStyleLlm::user_prompt_with_examples(input, "en", &[]);
        let (out, margins) = rt
            .block_on(llm.complete_scored(&system, &user))
            .expect("complete_scored failed");
        let out_words = out.split_whitespace().count();
        println!("INPUT:   {input}");
        println!("OUTPUT:  {out}");
        println!(
            "MARGINS: {} words; {:?}",
            margins.len(),
            margins
                .iter()
                .map(|m| (m.word.as_str(), m.margin))
                .collect::<Vec<_>>()
        );
        println!("---");

        assert!(!out.is_empty(), "empty output for {input:?}");
        assert!(
            !margins.is_empty(),
            "EMPTY word_margins (silent no-op) for {input:?} -> {out:?}"
        );
        // word_margins_from_tokens returns exactly one margin per output word on
        // success, so equality here also proves the token→word alignment held.
        assert_eq!(
            margins.len(),
            out_words,
            "margin count {} != output word count {} for {out:?}",
            margins.len(),
            out_words
        );
    }
}
