//! LLM-based text styling pipeline.
//!
//! Wraps any `LlmService` backend and implements the `TextPostProcessor`
//! interface as a styling-only pass: seed dictionary phrases, then (when the
//! preset asks for it) rewrite the text in the requested style. Grammar /
//! punctuation correction is no longer part of the pipeline.

use async_trait::async_trait;
use lune_core::Result;
use lune_core::llm::{LlmService, PostProcessInput, PostProcessOutput, TextPostProcessor};
use std::collections::HashMap;
use std::sync::Arc;

/// Styling pipeline.
///
/// Delegates to a shared `LlmService` for the styling call, with graceful
/// degradation: the seeded raw text is kept on any LLM failure.
pub struct StylePipeline {
    llm: Arc<dyn LlmService>,
}

impl StylePipeline {
    /// Create a new pipeline wrapping an LLM service.
    pub fn new(llm: Arc<dyn LlmService>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl TextPostProcessor for StylePipeline {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let mut text = apply_dictionary_seed(&input.raw_text, &input.dictionary);
        // The caller invokes this pass only when a style is selected (styling is
        // gated by the presence of a selection, not a per-preset flag), so the
        // styling LLM always runs here.
        let (system, user) = input.style_preset.build_styling_prompt(&text);
        match self.llm.complete(&system, &user).await {
            Ok(styled)
                if !styled.is_empty() && !crate::prompt::contains_forbidden_output(&styled) =>
            {
                text = crate::prompt::fix_all_number_spaces(&styled);
            }
            Ok(_) => {
                tracing::warn!(
                    "styling pass produced empty or forbidden output, keeping seeded text"
                )
            }
            Err(e) => tracing::warn!(error = %e, "styling pass failed, keeping seeded text"),
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
    // Index phrases by their first token: a phrase can only match at a position
    // whose token equals `phrase_toks[0]`, so the matcher consults just that
    // bucket instead of every phrase — O(tokens × phrases) → O(tokens ×
    // phrases-sharing-first-token). Insertion order is preserved, so each
    // bucket stays longest-first (a longer phrase still wins on overlap).
    let mut by_first: HashMap<&str, Vec<(&[String], &str)>> = HashMap::new();
    for (toks, repl) in &phrases {
        by_first
            .entry(toks[0].as_str())
            .or_default()
            .push((toks.as_slice(), repl));
    }

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
        let matched = by_first.get(lowered_cores[t].as_str()).and_then(|bucket| {
            bucket.iter().find_map(|(phrase_toks, replacement)| {
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
            })
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

    /// Mock LLM that returns a fixed styled string for the styling call.
    struct StylingLlm {
        styling_output: String,
    }

    #[async_trait]
    impl LlmService for StylingLlm {
        async fn complete(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(self.styling_output.clone())
        }
    }

    #[tokio::test]
    async fn non_minimal_preset_applies_styling() {
        let mock = Arc::new(StylingLlm {
            styling_output: "Styled text".to_string(),
        });
        let pipeline = StylePipeline::new(mock);
        let input = PostProcessInput {
            raw_text: "raw text".into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: None,
            dictionary: vec![],
            language: "en".into(),
        };

        let output = pipeline.transform(&input).await.unwrap();
        assert_eq!(output.cleaned_text, "Styled text");
    }

    #[tokio::test]
    async fn forbidden_styling_output_keeps_seeded_text() {
        let mock = Arc::new(StylingLlm {
            styling_output: "Here is the styled text".to_string(),
        });
        let pipeline = StylePipeline::new(mock);
        let input = PostProcessInput {
            raw_text: "raw text".into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: None,
            dictionary: vec![],
            language: "en".into(),
        };

        let output = pipeline.transform(&input).await.unwrap();
        assert_eq!(output.cleaned_text, "raw text");
    }

    #[tokio::test]
    async fn graceful_degradation_on_styling_failure() {
        struct FailingLlm;

        #[async_trait]
        impl LlmService for FailingLlm {
            async fn complete(&self, _system: &str, _user: &str) -> Result<String> {
                Err(lune_core::Error::llm("test failure"))
            }
        }

        let mock = Arc::new(FailingLlm);
        let pipeline = StylePipeline::new(mock);
        let input = PostProcessInput {
            raw_text: "raw text".into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: None,
            dictionary: vec![],
            language: "en".into(),
        };

        let output = pipeline.transform(&input).await.unwrap();
        assert_eq!(output.cleaned_text, "raw text");
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
