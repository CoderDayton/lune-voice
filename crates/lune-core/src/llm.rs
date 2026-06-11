use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::config::StylePreset;
use crate::dict::DictionaryEntry;

/// Input to text post-processor.
#[derive(Debug, Clone)]
pub struct PostProcessInput {
    pub raw_text: String,
    pub style_preset: Arc<StylePreset>,
    /// Target application category hint.
    pub app_category: Option<AppCategory>,
    pub dictionary: Vec<DictionaryEntry>,
    /// Recent session text for cross-utterance disambiguation.
    /// Tail of `session_text` truncated to 500 chars (125 words, 5 utterances).
    pub context: Option<String>,
    /// BCP-47 language code from ASR detection (e.g. "en", "eng", "fr").
    /// Defaults to "en" if not set by pipeline.
    pub language: String,
    /// Words flagged as likely incorrect, with their word-index position in text.
    pub uncertain_words: Vec<(String, usize)>,
    /// Whether the correction pass should run. The session gate sets this to
    /// false when it predicts a correction no-op (nothing unresolved, no risk
    /// hints, unambiguous punctuation); the styling pass still runs whenever
    /// the preset asks for it.
    pub correction_needed: bool,
}

/// Output from the text post-processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessOutput {
    pub cleaned_text: String,
    pub metadata: Option<OutputMetadata>,
}

/// Hints about output formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub has_bullets: bool,
    pub has_headings: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AppCategory {
    Email,
    Ide,
    Chat,
    Notes,
    Browser,
    Terminal,
}

/// Simple completion service: system prompt + user message → response text.
///
/// Intended for application-level tasks (e.g. generating few-shot examples)
/// that need a raw LLM call without the full post-processing pipeline.
#[async_trait]
pub trait LlmService: Send + Sync {
    async fn complete(&self, system: &str, user: &str) -> Result<String>;
}

/// Trait for LLM-based text post-processing.
#[async_trait]
pub trait TextPostProcessor: Send + Sync {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput>;

    /// Stream corrected text in chunks as they become available.
    ///
    /// Default implementation calls [`transform`] and sends the full result as
    /// a single chunk.  Backends with token-level access (e.g. local LLM)
    /// override this to emit word-boundary chunks during generation.
    async fn transform_streaming(
        &self,
        input: &PostProcessInput,
        chunk_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<PostProcessOutput> {
        let output = self.transform(input).await?;
        let _ = chunk_tx.send(output.cleaned_text.clone()).await;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_process_input_fields() {
        let input = PostProcessInput {
            raw_text: "hello um world".into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: Some(AppCategory::Email),
            dictionary: vec![],
            context: None,
            language: "en".into(),
            uncertain_words: Vec::new(),
            correction_needed: true,
        };
        assert_eq!(input.raw_text, "hello um world");
        assert_eq!(input.app_category, Some(AppCategory::Email));
        assert_eq!(input.style_preset.id, "casual");
    }

    #[test]
    fn post_process_output_round_trip() {
        let output = PostProcessOutput {
            cleaned_text: "Hello, world.".into(),
            metadata: Some(OutputMetadata {
                has_bullets: false,
                has_headings: false,
            }),
        };
        let json = serde_json::to_string(&output).unwrap();
        let parsed: PostProcessOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cleaned_text, "Hello, world.");
    }
}
