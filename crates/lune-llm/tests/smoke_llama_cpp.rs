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
    let path = std::env::var("LUNE_TEST_LLM_GGUF").ok()?;
    LlamaCppBackend::new(LlamaCppConfig {
        model_path: path.into(),
        use_gpu: false,
        max_new_tokens: 256,
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
