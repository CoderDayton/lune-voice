//! Unified LLM-based text post-processing pipeline.
//!
//! Wraps any `LlmService` backend and implements the full `TextPostProcessor`
//! interface with automatic correction + optional styling passes.

use async_trait::async_trait;
use lune_core::Result;
use lune_core::config::BASE_SYSTEM_PROMPT;
use lune_core::llm::{LlmService, PostProcessInput, PostProcessOutput, TextPostProcessor};
use std::sync::Arc;

/// Unified correction and styling pipeline.
///
/// Delegates to a shared `LlmService` for all LLM calls, ensuring:
/// - One system prompt construction (BASE_SYSTEM_PROMPT + dictionary terms)
/// - One user prompt format (via OrtStyleLlm::user_prompt)
/// - Two-pass workflow: correction → styling (if needed)
/// - Graceful degradation: raw text on any LLM failure
pub struct CorrectionPipeline {
    llm: Arc<dyn LlmService>,
}

impl CorrectionPipeline {
    /// Create a new pipeline wrapping an LLM service.
    pub fn new(llm: Arc<dyn LlmService>) -> Self {
        Self { llm }
    }

    /// Build the base system prompt with language and dictionary substitution.
    fn build_system_prompt(&self, input: &PostProcessInput) -> String {
        let mut system_content = BASE_SYSTEM_PROMPT.replace("{lang}", &input.language);

        if !input.dictionary.is_empty() {
            system_content.push_str("\nDictionary terms (use these exact spellings):");
            for entry in &input.dictionary {
                system_content.push_str("\n- ");
                system_content.push_str(&entry.phrase);
                system_content.push_str(" → ");
                system_content.push_str(&entry.replacement);
            }
        }

        system_content
    }

    /// Build the user prompt for the correction pass.
    fn build_correction_prompt(&self, input: &PostProcessInput) -> String {
        super::backend::ort_llm::OrtStyleLlm::user_prompt(
            &input.raw_text,
            &input.language,
            &input.uncertain_words,
        )
    }
}

#[async_trait]
impl TextPostProcessor for CorrectionPipeline {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let system = self.build_system_prompt(input);
        let user = self.build_correction_prompt(input);

        // Pass 1: Correction
        let mut text = match self.llm.complete(&system, &user).await {
            Ok(corrected) => corrected,
            Err(e) => {
                tracing::warn!(error = %e, "correction pass failed, returning raw text");
                return Ok(PostProcessOutput {
                    cleaned_text: input.raw_text.clone(),
                    metadata: None,
                });
            }
        };

        tracing::debug!(corrected_len = text.len(), "correction pass complete");

        // Pass 2: Styling (skip for minimal preset)
        if input.style_preset.needs_styling() {
            let style_prompt = input.style_preset.build_styling_prompt(&text);
            match self.llm.complete("", &style_prompt).await {
                Ok(styled) => {
                    tracing::debug!(styled_len = styled.len(), "styling pass complete");
                    text = styled;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "styling pass failed, keeping corrected text");
                }
            }
        }

        Ok(PostProcessOutput {
            cleaned_text: text,
            metadata: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::config::StylePreset;
    use std::sync::Arc;

    /// Mock LLM that returns predictable outputs.
    struct MockLlm {
        correction_output: String,
        styling_output: String,
    }

    #[async_trait]
    impl LlmService for MockLlm {
        async fn complete(&self, system: &str, _user: &str) -> Result<String> {
            // Distinguish based on system prompt content.
            if system.is_empty() {
                Ok(self.styling_output.clone())
            } else {
                Ok(self.correction_output.clone())
            }
        }
    }

    #[tokio::test]
    async fn correction_pass_only_for_minimal_style() {
        let mock = Arc::new(MockLlm {
            correction_output: "Corrected text".to_string(),
            styling_output: "Should not be used".to_string(),
        });
        let pipeline = CorrectionPipeline::new(mock);
        let input = PostProcessInput {
            raw_text: "raw text".into(),
            style_preset: Arc::new(StylePreset::minimal()),
            app_category: None,
            dictionary: vec![],
            context: None,
            language: "en".into(),
            uncertain_words: vec![],
        };

        let output = pipeline.transform(&input).await.unwrap();
        assert_eq!(output.cleaned_text, "Corrected text");
    }

    #[tokio::test]
    async fn correction_and_styling_for_non_minimal_style() {
        let mock = Arc::new(MockLlm {
            correction_output: "Corrected text".to_string(),
            styling_output: "Styled text".to_string(),
        });
        let pipeline = CorrectionPipeline::new(mock);
        let input = PostProcessInput {
            raw_text: "raw text".into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: None,
            dictionary: vec![],
            context: None,
            language: "en".into(),
            uncertain_words: vec![],
        };

        let output = pipeline.transform(&input).await.unwrap();
        assert_eq!(output.cleaned_text, "Styled text");
    }

    #[tokio::test]
    async fn graceful_degradation_on_correction_failure() {
        struct FailingLlm;

        #[async_trait]
        impl LlmService for FailingLlm {
            async fn complete(&self, _system: &str, _user: &str) -> Result<String> {
                Err(lune_core::Error::Llm("test failure".into()))
            }
        }

        let mock = Arc::new(FailingLlm);
        let pipeline = CorrectionPipeline::new(mock);
        let input = PostProcessInput {
            raw_text: "raw text".into(),
            style_preset: Arc::new(StylePreset::minimal()),
            app_category: None,
            dictionary: vec![],
            context: None,
            language: "en".into(),
            uncertain_words: vec![],
        };

        let output = pipeline.transform(&input).await.unwrap();
        assert_eq!(output.cleaned_text, "raw text");
    }
}
