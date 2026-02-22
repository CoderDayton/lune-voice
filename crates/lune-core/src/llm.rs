use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::Result;
use crate::config::StylePreset;
use crate::dict::DictionaryEntry;

/// Input to the text post-processor.
#[derive(Debug, Clone)]
pub struct PostProcessInput {
    pub raw_text: String,
    /// Preceding text for context.
    pub prev_context: Option<String>,
    pub style_preset: Arc<StylePreset>,
    /// Target application category hint.
    pub app_category: Option<AppCategory>,
    pub dictionary: Vec<DictionaryEntry>,
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

/// Trait for LLM-based text post-processing.
#[async_trait::async_trait]
pub trait TextPostProcessor: Send + Sync {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_process_input_fields() {
        let input = PostProcessInput {
            raw_text: "hello um world".into(),
            prev_context: None,
            style_preset: Arc::new(StylePreset::general()),
            app_category: Some(AppCategory::Email),
            dictionary: vec![],
        };
        assert_eq!(input.raw_text, "hello um world");
        assert_eq!(input.app_category, Some(AppCategory::Email));
        assert_eq!(input.style_preset.id, "general");
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
