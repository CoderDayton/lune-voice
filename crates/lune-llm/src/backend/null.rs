//! No-op text post-processor.
//!
//! Returns the raw input text unchanged. Used as the default
//! `TextPostProcessor` when no LLM correction backend is configured.

use lune_core::Result;
use lune_core::llm::{LlmService, PostProcessInput, PostProcessOutput, TextPostProcessor};

/// Returns the input transcript verbatim.
#[derive(Debug, Default)]
pub struct NullPostProcessor;

impl NullPostProcessor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl TextPostProcessor for NullPostProcessor {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        Ok(PostProcessOutput {
            cleaned_text: input.raw_text.clone(),
            metadata: None,
            edit_observations: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl LlmService for NullPostProcessor {
    async fn complete(&self, _system: &str, _user: &str) -> Result<String> {
        Err(lune_core::Error::llm("LLM backend is disabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::config::StylePreset;
    use std::sync::Arc;

    #[tokio::test]
    async fn passthrough_returns_input_unchanged() {
        let processor = NullPostProcessor::new();
        let input = PostProcessInput {
            raw_text: "hello world".into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: None,
            dictionary: vec![],
            language: "en".into(),
            correction_needed: true,
        };
        let out = processor.transform(&input).await.unwrap();
        assert_eq!(out.cleaned_text, "hello world");
    }
}
