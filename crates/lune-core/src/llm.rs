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
    /// BCP-47 language code from ASR detection (e.g. "en", "eng", "fr").
    /// Defaults to "en" if not set by pipeline.
    pub language: String,
    /// Whether the correction pass should run. The session gate sets this to
    /// false when it predicts a correction no-op (nothing unresolved,
    /// unambiguous punctuation); the styling pass still runs whenever
    /// the preset asks for it.
    pub correction_needed: bool,
}

/// One open-vocab edit-gate calibration observation, labelled against the
/// rule-merged decision. The session persists these to feed the next
/// calibration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EditObservation {
    /// The raw word already matched the delivered (truth) word — editing it
    /// would risk over-correction.
    pub observed_correct: bool,
    /// Decode margin of the LLM's proposed word; `None` when no edit was
    /// proposed at this slot.
    pub margin: Option<f32>,
    /// The LLM's proposed word equals the delivered word (a beneficial flip).
    pub flip_fixes: bool,
}

/// Output from the text post-processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessOutput {
    pub cleaned_text: String,
    pub metadata: Option<OutputMetadata>,
    /// Per-edit calibration observations from the open-vocab edit gate,
    /// labelled against the rule-merged decision. Empty unless the backend
    /// produced decode margins. The session persists these (it owns the
    /// `session_id`) to feed the next calibration.
    #[serde(default)]
    pub edit_observations: Vec<EditObservation>,
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

/// One output word paired with the decode decision margin that produced it.
///
/// `margin` is `logit(emitted) − max logit(other)` in logit space (≥ 0 under
/// greedy decode), min-aggregated over the word's sub-tokens — the weakest
/// token bounds confidence in the whole word. Consumed by the conformal
/// LLM-edit gate to decide whether a proposed edit clears the calibrated `tau`.
#[derive(Debug, Clone)]
pub struct WordMargin {
    pub word: String,
    pub margin: f32,
}

/// Simple completion service: system prompt + user message → response text.
///
/// Intended for application-level tasks (e.g. generating few-shot examples)
/// that need a raw LLM call without the full post-processing pipeline.
#[async_trait]
pub trait LlmService: Send + Sync {
    async fn complete(&self, system: &str, user: &str) -> Result<String>;

    /// Like [`complete`](Self::complete) but also returns the per-output-word
    /// decode margins. The default returns the completion with no margins, so
    /// backends without token-level access (remote APIs, the null backend)
    /// degrade safely: an empty margin vector yields no decisive observations,
    /// leaving the conformal edit gate disabled. Only the local ORT backend
    /// overrides this with real margins.
    async fn complete_scored(&self, system: &str, user: &str) -> Result<(String, Vec<WordMargin>)> {
        Ok((self.complete(system, user).await?, Vec::new()))
    }
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
            language: "en".into(),
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
            edit_observations: vec![
                EditObservation {
                    observed_correct: true,
                    margin: None,
                    flip_fixes: false,
                },
                EditObservation {
                    observed_correct: false,
                    margin: Some(2.0),
                    flip_fixes: true,
                },
            ],
        };
        let json = serde_json::to_string(&output).unwrap();
        let parsed: PostProcessOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cleaned_text, "Hello, world.");
        assert_eq!(parsed.edit_observations.len(), 2);
    }
}
