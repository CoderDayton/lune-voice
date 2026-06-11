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
            input.context.as_deref(),
            &input.uncertain_words,
        )
    }
}

fn dictionary_examples(input: &PostProcessInput) -> Vec<(String, String)> {
    let use_english_carrier = matches!(
        lune_core::lang::display_name(&input.language),
        Some("English")
    );
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
            if use_english_carrier {
                Some((
                    format!("I said {phrase} in the meeting."),
                    format!("I said {replacement} in the meeting."),
                ))
            } else {
                // For any other language we hand the LLM bare phrase pairs
                // rather than ship an English carrier sentence into a
                // non-English correction prompt.
                Some((phrase, replacement))
            }
        })
        .collect()
}

fn clean_example_fragment(s: &str) -> String {
    const MAX_CHARS: usize = 80;
    let mut out = String::with_capacity(s.len().min(MAX_CHARS * 4));
    let mut chars_taken = 0usize;
    let mut first = true;
    for tok in s.split_whitespace() {
        if !first {
            if chars_taken >= MAX_CHARS {
                break;
            }
            out.push(' ');
            chars_taken += 1;
        }
        for c in tok.chars() {
            if chars_taken >= MAX_CHARS {
                return out;
            }
            out.push(c);
            chars_taken += 1;
        }
        first = false;
    }
    out
}

#[async_trait]
impl TextPostProcessor for CorrectionPipeline {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let seeded_raw_text = apply_dictionary_seed(&input.raw_text, &input.dictionary);

        // Pass 1: Correction — skipped when the session gate predicted a
        // correction no-op (nothing unresolved, no risk hints, unambiguous
        // punctuation). Styling below runs either way, and a FAILED
        // correction no longer aborts the pipeline: styling still gets its
        // chance on the raw text.
        let mut text = if !input.correction_needed {
            tracing::debug!("correction pass skipped by gate; styling only");
            seeded_raw_text.clone()
        } else {
            let system = self.build_system_prompt(input);
            let user = self.build_correction_prompt(input, &seeded_raw_text);
            match self.llm.complete(&system, &user).await {
                Ok(corrected) => {
                    tracing::debug!(corrected_len = corrected.len(), "correction pass complete");
                    // Hardening: number-space repair + forbidden-meta-output
                    // guard + alignment-based merge against the raw input.
                    // Previously duplicated in OrtStyleLlm::transform; lifted
                    // here so every backend benefits.
                    let corrected = crate::prompt::fix_all_number_spaces(&corrected);
                    if crate::backend::ort_llm::contains_forbidden_output(&corrected) {
                        tracing::warn!(output = %corrected, "LLM produced meta-text, falling back to raw");
                        seeded_raw_text.clone()
                    } else {
                        let merged = crate::backend::ort_llm::validate_and_merge(
                            &seeded_raw_text,
                            &corrected,
                            &input.language,
                        );
                        // Make validator rejections observable: a downstream
                        // "LLM made no changes" can mean the model proposed
                        // nothing OR everything it proposed was reverted here
                        // — opposite tuning problems.
                        if merged != corrected {
                            tracing::info!(
                                fully_reverted = merged == seeded_raw_text,
                                proposed_len = corrected.len(),
                                merged_len = merged.len(),
                                "validator rejected some LLM edits"
                            );
                        }
                        merged
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "correction pass failed, using raw text");
                    seeded_raw_text.clone()
                }
            }
        };

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

/// Pre-replace dictionary phrases in the raw transcript before the LLM ever
/// sees it. Single-word and multi-word phrases are both handled; the
/// post-pass validator then accepts each replacement as an `Op::Match`.
///
/// Matching is Unicode case-insensitive on the alphanumeric core of each
/// whitespace-delimited token, so leading/trailing punctuation (ASCII or
/// smart quotes / em-dashes / guillemets) does not block the match. When
/// several phrases could match at the same position, the longest one wins.
///
/// Whitespace between non-matching tokens is preserved byte-for-byte from
/// the input — tabs, newlines, and runs of spaces all survive.
fn apply_dictionary_seed(text: &str, dictionary: &[lune_core::dict::DictionaryEntry]) -> String {
    if dictionary.is_empty() {
        return text.to_string();
    }

    // Pre-tokenize each dictionary phrase into lowercase tokens. Longest
    // phrase first so multi-word entries win over their single-word
    // substrings at the same position.
    let mut phrases: Vec<(Vec<String>, &str)> = dictionary
        .iter()
        .filter_map(|entry| {
            let toks: Vec<String> = entry.phrase.split_whitespace().map(unicode_lower).collect();
            if toks.iter().any(|t| t.is_empty()) || toks.is_empty() {
                None
            } else {
                Some((toks, entry.replacement.as_str()))
            }
        })
        .collect();
    phrases.sort_by_key(|(toks, _)| std::cmp::Reverse(toks.len()));

    let tokens = tokenize_word_spans(text);
    // Lowercase each token's alphanumeric core once up front. The matcher
    // is O(tokens * phrases * phrase_len) and used to allocate a fresh
    // String per comparison; precomputing cuts those allocations entirely.
    let lowered_cores: Vec<String> = tokens
        .iter()
        .map(|&(s, e)| unicode_lower(split_edge_punctuation(&text[s..e]).1))
        .collect();
    let mut out = String::with_capacity(text.len());
    let mut last_end = 0usize;
    let mut t = 0usize;
    while t < tokens.len() {
        let matched = phrases.iter().find_map(|(phrase_toks, replacement)| {
            let n = phrase_toks.len();
            if t + n > tokens.len() {
                return None;
            }
            let all_match = (0..n).all(|k| lowered_cores[t + k] == phrase_toks[k]);
            if all_match {
                Some((n, *replacement))
            } else {
                None
            }
        });

        if let Some((n, replacement)) = matched {
            let (first_s, first_e) = tokens[t];
            let (last_s, last_e) = tokens[t + n - 1];
            // Preserve everything before the matched span verbatim.
            out.push_str(&text[last_end..first_s]);
            // Keep the first token's leading punctuation and the last
            // token's trailing punctuation; drop anything in between
            // (internal whitespace and punctuation of the matched span)
            // since the replacement is a single unit.
            let (prefix, _, _) = split_edge_punctuation(&text[first_s..first_e]);
            let (_, _, suffix) = split_edge_punctuation(&text[last_s..last_e]);
            out.push_str(prefix);
            out.push_str(replacement);
            out.push_str(suffix);
            last_end = last_e;
            t += n;
        } else {
            t += 1;
        }
    }
    out.push_str(&text[last_end..]);
    out
}

fn unicode_lower(s: &str) -> String {
    s.chars().flat_map(char::to_lowercase).collect()
}

/// Byte-range spans of every whitespace-delimited token in `text`. Used
/// to rebuild output with the input's whitespace intact.
fn tokenize_word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
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
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, c)| i + c.len_utf8())
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
            correction_needed: true,
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
            correction_needed: true,
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
                Err(lune_core::Error::llm("test failure"))
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
            correction_needed: true,
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
            correction_needed: true,
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
    fn dictionary_examples_drop_english_carrier_for_non_english() {
        let input = PostProcessInput {
            raw_text: "...".into(),
            style_preset: Arc::new(StylePreset::minimal()),
            app_category: None,
            dictionary: vec![DictionaryEntry {
                phrase: "Bordeaux".into(),
                replacement: "Bordo".into(),
                priority: 10,
            }],
            context: None,
            language: "fr".into(),
            uncertain_words: vec![],
            correction_needed: true,
        };
        let examples = dictionary_examples(&input);
        assert_eq!(
            examples,
            vec![("Bordeaux".to_string(), "Bordo".to_string())]
        );
    }

    #[test]
    fn dictionary_seed_matches_unicode_case_variants() {
        let dictionary = vec![DictionaryEntry {
            phrase: "métro".into(),
            replacement: "subway".into(),
            priority: 10,
        }];
        // Uppercase non-ASCII letter must fold to its lowercase form.
        assert_eq!(
            apply_dictionary_seed("J'aime le MÉTRO.", &dictionary),
            "J'aime le subway."
        );
    }

    #[test]
    fn dictionary_seed_preserves_internal_whitespace() {
        let dictionary = vec![DictionaryEntry {
            phrase: "Barolo".into(),
            replacement: "Boorloo".into(),
            priority: 10,
        }];
        // Tabs and runs of spaces between non-matching tokens must survive
        // — apply_dictionary_seed used to collapse them via split_whitespace.
        assert_eq!(
            apply_dictionary_seed("I  visited\tBarolo today", &dictionary),
            "I  visited\tBoorloo today"
        );
    }

    #[test]
    fn dictionary_seed_applies_multi_word_phrase() {
        let dictionary = vec![DictionaryEntry {
            phrase: "machine learning".into(),
            replacement: "ML".into(),
            priority: 10,
        }];
        assert_eq!(
            apply_dictionary_seed("I work on machine learning daily.", &dictionary),
            "I work on ML daily."
        );
    }

    #[test]
    fn dictionary_seed_multi_word_drops_internal_punctuation() {
        let dictionary = vec![DictionaryEntry {
            phrase: "machine learning".into(),
            replacement: "ML".into(),
            priority: 10,
        }];
        // Punctuation between matched phrase tokens is intentionally
        // collapsed — the replacement is a single unit, so the comma
        // between "machine" and "learning" is dropped along with the
        // internal whitespace. Pin this so a future refactor doesn't
        // silently start emitting "ML,".
        assert_eq!(
            apply_dictionary_seed("I prefer machine, learning techniques.", &dictionary),
            "I prefer ML techniques."
        );
    }

    #[test]
    fn dictionary_seed_multi_word_is_case_insensitive_and_punctuation_tolerant() {
        let dictionary = vec![DictionaryEntry {
            phrase: "machine learning".into(),
            replacement: "ML".into(),
            priority: 10,
        }];
        // Punctuation between phrase tokens and surrounding the phrase must
        // not block the match; surface case of the user input should not matter.
        assert_eq!(
            apply_dictionary_seed("Hello, Machine Learning! Yes.", &dictionary),
            "Hello, ML! Yes."
        );
    }

    #[test]
    fn dictionary_seed_prefers_longest_match() {
        let dictionary = vec![
            DictionaryEntry {
                phrase: "machine".into(),
                replacement: "MCH".into(),
                priority: 10,
            },
            DictionaryEntry {
                phrase: "machine learning".into(),
                replacement: "ML".into(),
                priority: 10,
            },
        ];
        // The longer phrase wins even when a single-word entry could fire first.
        assert_eq!(
            apply_dictionary_seed("I love machine learning.", &dictionary),
            "I love ML."
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
