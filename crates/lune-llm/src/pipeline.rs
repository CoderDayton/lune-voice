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
/// - One user prompt format (via OrtStyleLlm::user_prompt_with_examples)
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
    fn build_correction_prompt(&self, input: &PostProcessInput, raw_text: &str) -> String {
        let examples = dictionary_examples(input);
        crate::backend::ort_llm::OrtStyleLlm::user_prompt_with_examples(
            raw_text,
            &input.language,
            &examples,
        )
    }
}

fn dictionary_examples(input: &PostProcessInput) -> Vec<(String, String)> {
    input
        .dictionary
        .iter()
        .take(6)
        .filter_map(|entry| {
            let phrase = clean_example_fragment(&entry.phrase);
            let replacement = clean_example_fragment(&entry.replacement);
            if phrase.is_empty() || replacement.is_empty() {
                return None;
            }
            Some((
                format!("I said {phrase} in the meeting."),
                format!("I said {replacement} in the meeting."),
            ))
        })
        .collect()
}

fn clean_example_fragment(s: &str) -> String {
    const MAX_CHARS: usize = 80;
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(MAX_CHARS)
        .collect()
}

#[async_trait]
impl TextPostProcessor for CorrectionPipeline {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let seeded_raw_text = apply_dictionary_seed(&input.raw_text, &input.dictionary);
        let system = self.build_system_prompt(input);
        let user = self.build_correction_prompt(input, &seeded_raw_text);

        // Pass 1: Correction
        let mut text = match self.llm.complete(&system, &user).await {
            Ok(corrected) => corrected,
            Err(e) => {
                tracing::warn!(error = %e, "correction pass failed, returning raw text");
                return Ok(PostProcessOutput {
                    cleaned_text: seeded_raw_text,
                    metadata: None,
                });
            }
        };

        tracing::debug!(corrected_len = text.len(), "correction pass complete");

        // Hardening: number-space repair + forbidden-meta-output guard +
        // alignment-based merge against the raw input. Previously duplicated
        // in OrtStyleLlm::transform; lifted here so every backend benefits.
        text = crate::prompt::fix_all_number_spaces(&text);
        if crate::backend::ort_llm::contains_forbidden_output(&text) {
            tracing::warn!(output = %text, "LLM produced meta-text, falling back to raw");
            return Ok(PostProcessOutput {
                cleaned_text: seeded_raw_text,
                metadata: None,
            });
        }
        text =
            crate::backend::ort_llm::validate_and_merge(&seeded_raw_text, &text, &input.language);

        // Pass 2: Styling (gated by preset.skip_styling)
        if input.style_preset.needs_styling() {
            let style_prompt = input.style_preset.build_styling_prompt(&text);
            match self.llm.complete("", &style_prompt).await {
                Ok(styled) => {
                    if !styled.is_empty()
                        && !crate::backend::ort_llm::contains_forbidden_output(&styled)
                    {
                        tracing::debug!(styled_len = styled.len(), "styling pass complete");
                        text = styled;
                    } else {
                        tracing::warn!(
                            "styling pass produced empty or forbidden output, keeping corrected text"
                        );
                    }
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

/// Pre-replace single-token dictionary phrases in the raw transcript before
/// the LLM ever sees it.
///
/// Policy: the user's dictionary wins unconditionally — every case-insensitive
/// whole-word hit is rewritten, and the post-pass validator then accepts the
/// replacement as an `Op::Match`. The trade-off is that homonyms ("lit" →
/// "lit-up") cannot be selectively kept; users opt in by adding the entry.
///
/// Multi-word phrases are skipped here — seeding them safely needs phrase
/// boundaries we don't have at this layer. The LLM still sees them via the
/// system prompt + correction examples.
fn apply_dictionary_seed(text: &str, dictionary: &[lune_core::dict::DictionaryEntry]) -> String {
    if dictionary.is_empty() {
        return text.to_string();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::with_capacity(words.len());
    for word in words {
        let (prefix, core, suffix) = split_edge_punctuation(word);
        let replacement = dictionary
            .iter()
            .filter(|entry| !entry.phrase.contains(char::is_whitespace))
            .find(|entry| core.eq_ignore_ascii_case(&entry.phrase))
            .map(|entry| entry.replacement.as_str());
        match replacement {
            Some(replacement) => {
                let mut s = String::with_capacity(prefix.len() + replacement.len() + suffix.len());
                s.push_str(prefix);
                s.push_str(replacement);
                s.push_str(suffix);
                out.push(s);
            }
            None => out.push(word.to_string()),
        }
    }
    out.join(" ")
}

// Peel off any leading/trailing non-alphanumeric characters so dictionary
// matches survive ASCII *and* unicode punctuation (smart quotes, em-dashes,
// guillemets). Using `is_alphanumeric` also keeps accented letters in the
// core, which `is_ascii_punctuation` would have left dangling.
fn split_edge_punctuation(word: &str) -> (&str, &str, &str) {
    let core_start = word
        .find(|c: char| c.is_alphanumeric())
        .unwrap_or(word.len());
    let core_end = word
        .rfind(|c: char| c.is_alphanumeric())
        .map(|i| i + word[i..].chars().next().map(char::len_utf8).unwrap_or(0))
        .unwrap_or(core_start);
    (
        &word[..core_start],
        &word[core_start..core_end],
        &word[core_end..],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::config::StylePreset;
    use lune_core::dict::DictionaryEntry;
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
        // The post-correction validator drops outputs that share no words with
        // the input — see `backend::ort_llm::validate_and_merge`. Mocks must
        // therefore preserve enough vocabulary to clear the 40% LCS gate.
        let mock = Arc::new(MockLlm {
            correction_output: "raw text corrected".to_string(),
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
        // skip_styling=true on minimal → only the correction pass runs.
        assert_eq!(output.cleaned_text, "raw text");
    }

    #[tokio::test]
    async fn correction_and_styling_for_non_minimal_style() {
        let mock = Arc::new(MockLlm {
            correction_output: "raw text corrected".to_string(),
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

    #[test]
    fn dictionary_entries_become_correction_examples() {
        let input = PostProcessInput {
            raw_text: "I visited Barolo yesterday".into(),
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
        };

        let examples = dictionary_examples(&input);
        assert_eq!(
            examples,
            vec![(
                "I said Barolo in the meeting.".to_string(),
                "I said Boorloo in the meeting.".to_string()
            )]
        );
    }

    #[test]
    fn dictionary_seed_applies_single_word_replacements() {
        let dictionary = vec![DictionaryEntry {
            phrase: "Barolo".into(),
            replacement: "Boorloo".into(),
            priority: 10,
        }];
        assert_eq!(
            apply_dictionary_seed("I visited Barolo.", &dictionary),
            "I visited Boorloo."
        );
    }

    #[test]
    fn dictionary_seed_strips_unicode_edge_punctuation() {
        let dictionary = vec![DictionaryEntry {
            phrase: "Barolo".into(),
            replacement: "Boorloo".into(),
            priority: 10,
        }];
        // Smart quotes and em-dash are not ASCII punctuation, but we still
        // want the dictionary match to fire when they surround the core word.
        assert_eq!(
            apply_dictionary_seed("\u{201C}Barolo\u{201D}\u{2014}", &dictionary),
            "\u{201C}Boorloo\u{201D}\u{2014}"
        );
    }
}
