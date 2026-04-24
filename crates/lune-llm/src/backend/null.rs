//! No-op text post-processor.
//!
//! Returns the raw input text unchanged. Used as the default
//! `TextPostProcessor` when no remote LLM correction is configured —
//! all local correction now goes through the Nemotron cross-encoder
//! reranker, which is wired separately on `SessionManager` and bypasses
//! this trait entirely. The trait stays for the OpenAI remote path.

use lune_core::Result;
use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

/// Returns the input transcript verbatim, with `[?word?]` markers
/// stripped if any leak through. Strictly additive: never modifies
/// content, only removes residual annotation syntax.
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
        // Strip any residual annotation markers the upstream marking
        // pass may have emitted, then return the cleaned text. The
        // reranker handles all real correction at the session layer;
        // this trait is only invoked as a passthrough now.
        let cleaned = crate::prompt::strip_markers(&input.raw_text);
        Ok(PostProcessOutput {
            cleaned_text: cleaned,
            metadata: None,
        })
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
            context: None,
            language: "en".into(),
            hypotheses: Vec::new(),
        };
        let out = processor.transform(&input).await.unwrap();
        assert_eq!(out.cleaned_text, "hello world");
    }

    #[tokio::test]
    async fn passthrough_strips_residual_markers() {
        let processor = NullPostProcessor::new();
        let input = PostProcessInput {
            raw_text: "the [?weather?] was nice".into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: None,
            dictionary: vec![],
            context: None,
            language: "en".into(),
            hypotheses: Vec::new(),
        };
        let out = processor.transform(&input).await.unwrap();
        assert_eq!(out.cleaned_text, "the weather was nice");
    }
}
