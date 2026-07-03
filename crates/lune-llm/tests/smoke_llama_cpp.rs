//! Real-model smoke test for the llama.cpp styling backend.
//!
//! Ignored by default (needs a GGUF on disk + the C++ build). Run with:
//!   LUNE_TEST_LLM_GGUF=/path/to/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
//!   LIBCLANG_PATH=/lib/x86_64-linux-gnu \
//!   cargo test -p lune-llm --features llama-cpp --test smoke_llama_cpp -- --ignored --nocapture

#![cfg(feature = "llama-cpp")]

use lune_core::llm::LlmService;
use lune_llm::backend::llama_cpp::{LlamaCppBackend, LlamaCppConfig};

fn backend() -> Option<LlamaCppBackend> {
    backend_with_tokens(256)
}

fn backend_with_tokens(max_new_tokens: usize) -> Option<LlamaCppBackend> {
    let path = std::env::var("LUNE_TEST_LLM_GGUF").ok()?;
    LlamaCppBackend::new(LlamaCppConfig {
        model_path: path.into(),
        use_gpu: false,
        max_new_tokens,
    })
    .ok()
}

const SYSTEM: &str = "You rewrite the user's dictated text to be clear and well punctuated. Output only the rewritten text.";
const USER: &str =
    "so um i was thinking we could like meet tomorrow maybe around noon if that works";

#[tokio::test]
#[ignore = "needs a real GGUF model on disk (set LUNE_TEST_LLM_GGUF)"]
async fn completes_clean_styled_text() {
    let Some(llm) = backend() else {
        eprintln!("LUNE_TEST_LLM_GGUF not set / load failed; skipping");
        return;
    };
    let out = llm.complete(SYSTEM, USER).await.expect("complete");
    eprintln!("OUTPUT: {out:?}");
    assert!(!out.trim().is_empty(), "empty completion");
    // Thinking must be disabled: no reasoning block leaks into the output.
    assert!(!out.contains("<think>"), "think block leaked: {out}");
    assert!(!out.contains("</think>"), "think close leaked: {out}");
    // It should look like prose, not meta-commentary.
    assert!(
        !lune_llm::prompt::contains_forbidden_output(&out),
        "meta-commentary in output: {out}"
    );
}

#[tokio::test]
#[ignore = "needs a real GGUF model on disk (set LUNE_TEST_LLM_GGUF)"]
async fn generates_few_shot_examples_for_abstract_style() {
    // Regression: an abstract/meta style description ("improve the prompt...")
    // previously made the model narrate the rule instead of emitting concrete
    // pairs, producing malformed JSON (wrong arity, unbalanced arrays).
    let Some(llm) = backend_with_tokens(512) else {
        eprintln!("LUNE_TEST_LLM_GGUF not set / load failed; skipping");
        return;
    };
    let pairs = lune_llm::few_shot::generate_few_shot_examples(
        &llm,
        "Prompt Enhancer",
        "Improve the prompt by making it more specific and actionable, while keeping the \
         original length and meaning intact.",
    )
    .await
    .expect("few-shot generation");
    eprintln!("PAIRS: {pairs:?}");
    assert!(!pairs.is_empty(), "no examples generated");
    for (input, output) in &pairs {
        assert!(!input.trim().is_empty(), "empty example input");
        assert!(!output.trim().is_empty(), "empty example output");
    }
}

#[tokio::test]
#[ignore = "needs a real GGUF model on disk (set LUNE_TEST_LLM_GGUF)"]
async fn generates_few_shot_examples() {
    let Some(llm) = backend_with_tokens(512) else {
        eprintln!("LUNE_TEST_LLM_GGUF not set / load failed; skipping");
        return;
    };
    // The production styles-page path: same function, same 512-token budget
    // (LazyLocalLlm::new(..., 512) in runtime.rs).
    let pairs = lune_llm::few_shot::generate_few_shot_examples(
        &llm,
        "Casual",
        "Replace every colon with a comma. Remove all quotation marks. \
         Lowercase everything except proper nouns. Use contractions.",
    )
    .await
    .expect("few-shot generation");
    eprintln!("PAIRS: {pairs:?}");
    assert!(!pairs.is_empty(), "no examples generated");
    for (input, output) in &pairs {
        assert!(!input.trim().is_empty(), "empty example input");
        assert!(!output.trim().is_empty(), "empty example output");
        // The `<think>` token is hard-banned in the sampler (llama_cpp.rs), so
        // reasoning can never reopen and swallow the real answer.
        assert!(
            !input.contains("<think"),
            "think leaked into input: {input}"
        );
        assert!(
            !output.contains("<think"),
            "think leaked into output: {output}"
        );
    }
}

#[tokio::test]
#[ignore = "needs a real GGUF model on disk (set LUNE_TEST_LLM_GGUF)"]
async fn deterministic_across_runs() {
    let Some(llm) = backend() else {
        return;
    };
    let a = llm.complete(SYSTEM, USER).await.expect("complete a");
    let b = llm.complete(SYSTEM, USER).await.expect("complete b");
    assert_eq!(a, b, "greedy decode should be deterministic");
}

#[tokio::test]
#[ignore = "needs a real GGUF model on disk (set LUNE_TEST_LLM_GGUF)"]
async fn handles_empty_input() {
    let Some(llm) = backend() else {
        return;
    };
    // Must not panic or hang on empty user text.
    let _ = llm.complete(SYSTEM, "").await.expect("complete empty");
}
