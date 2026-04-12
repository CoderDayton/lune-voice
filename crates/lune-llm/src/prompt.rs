//! Prompt construction for ASR transcript correction.
//!
//! Provides a shared system prompt and ChatML formatting used by both
//! the inference backend (`llamacpp_backend.rs`) and the training script
//! (`unsloth/train.py`). The system prompt in `train.py` must be kept
//! byte-identical to [`SYSTEM`].

use crate::fillers::{self, FillerError};

/// System prompt for ASR transcript correction.
///
/// Kept minimal — no few-shot examples. The fine-tuned model learns the
/// task from training data; the generic model handles it from instructions.
pub const SYSTEM: &str = "You are a transcript correction tool. Your only job is to fix words annotated as uncertain.\n\
Annotation formats:\n\
- {guess|alt1|alt2}: ASR's top pick is `guess`, followed by 1-2 runner-up candidates. Pick whichever fits the sentence best, or substitute a different word if none fit.\n\
- [?word?]: the ASR is uncertain but has no specific alternative. Correct it from context.\n\
Rules:\n\
- Output ONLY the corrected transcript.\n\
- Copy every unannotated word EXACTLY as-is, including profanity, slang, and repeated words.\n\
- Prefer preserving unannotated words unless the sentence is clearly broken.\n\
- Delete [?um?] [?uh?] {um|...} {uh|...} and filler variants entirely.\n\
- NEVER censor, rephrase, reorder, summarize, add punctuation, or remove words.\n\
Examples:\n\
Input:  the {fox|box|ox} jumped over the lazy dog\n\
Output: the fox jumped over the lazy dog\n\
Input:  it was a {whether|weather} report\n\
Output: it was a weather report\n\
Input:  I need to [?recieve?] the package\n\
Output: I need to receive the package";

/// Build a ChatML inference prompt (assistant turn left open).
///
/// ```text
/// <|im_start|>system
/// {system}<|im_end|>
/// <|im_start|>user
/// {input}<|im_end|>
/// <|im_start|>assistant
/// ```
pub fn build_prompt(input: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n\
         <|im_start|>user\n{input}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Build a ChatML training row (assistant turn closed with target).
///
/// Used by `train.py` and for dataset verification.
pub fn format_training_row(input: &str, target: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n\
         <|im_start|>user\n{input}<|im_end|>\n\
         <|im_start|>assistant\n{target}<|im_end|>"
    )
}

/// Full pre-inference pipeline: strip fillers → build prompt.
pub fn prepare_prompt(raw: &str, lang: &str) -> Result<String, FillerError> {
    let cleaned = fillers::remove_fillers(raw, lang)?;
    Ok(build_prompt(&cleaned))
}

/// Post-process model output: strip leaked special tokens + residual markers.
///
/// mistral.rs may or may not strip stop tokens depending on sampler config.
/// Safer to handle explicitly.
pub fn post_process(output: &str) -> String {
    let s = output.trim();
    // Strip any leaked ChatML tokens
    let s = s.replace("<|im_end|>", "").replace("<|im_start|>", "");
    let s = s.trim();
    // Strip residual [?word?] / [word] markers the model failed to resolve
    strip_markers(s)
}

/// Prompt builder with configurable system prompt and language.
///
/// Holds `lang` and `system` as fields for cleaner API when adding
/// ES/FR variants with different system prompts.
pub struct PromptBuilder {
    pub system: String,
    pub lang: String,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self {
            system: SYSTEM.to_string(),
            lang: "en".to_string(),
        }
    }
}

impl PromptBuilder {
    pub fn new(lang: impl Into<String>) -> Self {
        Self {
            system: SYSTEM.to_string(),
            lang: lang.into(),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = system.into();
        self
    }

    /// Build inference prompt (assistant turn open).
    pub fn build_prompt(&self, input: &str) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{input}<|im_end|>\n\
             <|im_start|>assistant\n",
            self.system
        )
    }

    /// Build training row (assistant turn closed).
    pub fn format_training_row(&self, input: &str, target: &str) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{input}<|im_end|>\n\
             <|im_start|>assistant\n{target}<|im_end|>",
            self.system
        )
    }

    /// Strip fillers for this language, then build inference prompt.
    pub fn prepare_prompt(&self, raw: &str) -> Result<String, FillerError> {
        let cleaned = fillers::remove_fillers(raw, &self.lang)?;
        Ok(self.build_prompt(&cleaned))
    }
}

/// Extract unmarked words from a marked input string.
///
/// Words inside annotation markers (`[?..?]` or `{guess|alternate}`) are
/// skippable by the LLM — they can be replaced, deleted, or corrected.
/// Unmarked words MUST be preserved.
fn extract_unmarked_words(input: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars();

    while let Some(c) = chars.next() {
        if c == '[' {
            // `[?word?]` legacy marker. Anything inside counts as annotated.
            let rest = chars.as_str();
            if rest.starts_with('?') {
                flush_words(&mut current, &mut result);
                chars.next(); // consume the '?'
                loop {
                    match chars.next() {
                        Some('?') if chars.as_str().starts_with(']') => {
                            chars.next(); // consume ']'
                            break;
                        }
                        Some(_) => {}
                        None => break, // unclosed marker — EOF
                    }
                }
            } else {
                current.push(c);
            }
        } else if c == '{' {
            // `{guess|alternate}` runner-up marker. Everything inside is
            // skippable — the LLM may return any of the listed words or
            // substitute a different one entirely.
            flush_words(&mut current, &mut result);
            for ic in chars.by_ref() {
                if ic == '}' {
                    break;
                }
            }
        } else {
            current.push(c);
        }
    }
    flush_words(&mut current, &mut result);
    result
}

#[inline]
fn flush_words(buf: &mut String, out: &mut Vec<String>) {
    for w in buf.split_whitespace() {
        out.push(w.to_lowercase());
    }
    buf.clear();
}

/// Strip leading/trailing ASCII punctuation from a word, preserving
/// apostrophes (for contractions like "don't", "I'm").
#[inline]
fn strip_word_punctuation(word: &str) -> &str {
    word.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
}

/// Validate that all unmarked words from the input appear in the LLM output
/// in order (case-insensitive, punctuation-normalized).
///
/// This catches the most dangerous LLM failure mode: silently changing or
/// dropping words that were NOT marked as uncertain. If the LLM was only
/// supposed to fix `[?word?]` spans, every other word must survive.
///
/// Punctuation normalization prevents false negatives when the LLM adds
/// trailing punctuation (e.g. "works" vs "works,") despite being told not to.
///
/// Returns `true` if all unmarked input words appear in the output in order.
pub fn validate_preservation(input: &str, output: &str) -> bool {
    let unmarked = extract_unmarked_words(input);
    if unmarked.is_empty() {
        return true; // all words were marked — LLM can change anything
    }
    if output.is_empty() {
        return false;
    }

    let output_words: Vec<String> = output
        .split_whitespace()
        .map(|w| strip_word_punctuation(&w.to_lowercase()).to_string())
        .filter(|w| !w.is_empty())
        .collect();

    // Greedy order-preserving subsequence match
    let mut out_idx = 0;
    for word in &unmarked {
        let found = output_words[out_idx..].iter().position(|w| w == word);
        match found {
            Some(pos) => out_idx += pos + 1,
            None => return false,
        }
    }
    true
}

/// Strip residual annotation markers from LLM output.
///
/// Handles three formats the correction prompt may leak through:
/// - `[?word?]` — legacy uncertainty marker, strips to `word`
/// - `[word]`   — partial strip by a small model, recovers `word`
/// - `{guess|alternate}` — runner-up marker, strips to `guess` (first option)
///
/// Unclosed brackets/braces are preserved verbatim so stripping is
/// lossless on partial LLM output.
pub fn strip_markers(text: &str) -> String {
    if !text.contains('[') && !text.contains('{') {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            // Collect everything up to the closing ']'.
            let mut inner = String::new();
            let mut found_close = false;
            for ic in chars.by_ref() {
                if ic == ']' {
                    found_close = true;
                    break;
                }
                inner.push(ic);
            }
            if found_close {
                // Strip [?word?] → word, [word] → word.
                let stripped = inner.strip_prefix('?').unwrap_or(&inner);
                let stripped = stripped.strip_suffix('?').unwrap_or(stripped);
                result.push_str(stripped);
            } else {
                // Unclosed bracket — keep as-is.
                result.push('[');
                result.push_str(&inner);
            }
        } else if c == '{' {
            // `{guess|alternate}` → `guess`. Collect everything up to the
            // closing brace, then split on the first `|` and keep the first
            // option. Unclosed braces preserve the original source verbatim.
            let mut inner = String::new();
            let mut found_close = false;
            for ic in chars.by_ref() {
                if ic == '}' {
                    found_close = true;
                    break;
                }
                inner.push(ic);
            }
            if found_close {
                let guess = inner.split('|').next().unwrap_or(&inner);
                result.push_str(guess.trim());
            } else {
                result.push('{');
                result.push_str(&inner);
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_markers_question_marks() {
        assert_eq!(strip_markers("hello [?world?] there"), "hello world there");
    }

    #[test]
    fn strip_markers_bare_brackets() {
        assert_eq!(strip_markers("the [most] important"), "the most important");
    }

    #[test]
    fn strip_markers_no_brackets() {
        assert_eq!(strip_markers("hello world"), "hello world");
    }

    #[test]
    fn strip_markers_unclosed_bracket() {
        assert_eq!(strip_markers("hello [world"), "hello [world");
    }

    #[test]
    fn strip_markers_mixed() {
        assert_eq!(
            strip_markers("the [?oth?] service [provided] for [?them?]"),
            "the oth service provided for them"
        );
    }

    #[test]
    fn build_prompt_format() {
        let p = build_prompt("the [?whether?] was nice");
        assert!(p.starts_with("<|im_start|>system\n"));
        assert!(p.contains("<|im_start|>user\nthe [?whether?] was nice<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn training_row_format() {
        let r = format_training_row("the [?whether?] was nice", "the weather was nice");
        assert!(r.contains("<|im_start|>user\nthe [?whether?] was nice<|im_end|>"));
        assert!(r.contains("<|im_start|>assistant\nthe weather was nice<|im_end|>"));
    }

    #[test]
    fn post_process_strips_leaked_tokens() {
        assert_eq!(
            post_process("the weather was nice<|im_end|>"),
            "the weather was nice"
        );
        assert_eq!(
            post_process("<|im_start|>the weather was nice"),
            "the weather was nice"
        );
    }

    #[test]
    fn post_process_strips_residual_markers() {
        assert_eq!(
            post_process("the [?weather?] was nice"),
            "the weather was nice"
        );
    }

    #[test]
    fn prompt_builder_custom_lang() {
        let pb = PromptBuilder::new("fr");
        assert_eq!(pb.lang, "fr");
        let p = pb.build_prompt("test");
        assert!(p.contains(SYSTEM));
    }

    #[test]
    fn prompt_builder_custom_system() {
        let pb = PromptBuilder::new("en").with_system("Custom prompt.");
        let p = pb.build_prompt("test");
        assert!(p.contains("Custom prompt."));
        assert!(!p.contains(SYSTEM));
    }

    // ── Preservation validation tests ───────────────────────────────────

    #[test]
    fn validate_preserves_all_unmarked_words() {
        // LLM correctly replaced [?recieve?] with "receive" — valid
        let input = "the [?recieve?] button works";
        let output = "the receive button works";
        assert!(validate_preservation(input, output));
    }

    #[test]
    fn validate_detects_dropped_unmarked_word() {
        // LLM silently dropped "works" — invalid
        let input = "the [?recieve?] button works";
        let output = "the receive button";
        assert!(!validate_preservation(input, output));
    }

    #[test]
    fn validate_detects_changed_unmarked_word() {
        // LLM changed "button" to "feature" — invalid
        let input = "the [?recieve?] button works";
        let output = "the receive feature works";
        assert!(!validate_preservation(input, output));
    }

    #[test]
    fn validate_allows_case_changes() {
        // Case changes on unmarked words are acceptable
        let input = "the [?recieve?] Button works";
        let output = "the receive button works";
        assert!(validate_preservation(input, output));
    }

    #[test]
    fn validate_allows_trailing_punctuation() {
        // LLM added a comma and period — should still pass
        let input = "the [?recieve?] button works";
        let output = "the receive button, works.";
        assert!(validate_preservation(input, output));
    }

    #[test]
    fn validate_contraction_with_punctuation() {
        // Apostrophes in contractions preserved, trailing period stripped
        let input = "I [?dont?] know";
        let output = "I don't know.";
        assert!(validate_preservation(input, output));
    }

    #[test]
    fn validate_no_markers_all_preserved() {
        // No markers at all — every word must be preserved
        let input = "hello world";
        let output = "hello world";
        assert!(validate_preservation(input, output));
    }

    #[test]
    fn validate_no_markers_word_changed() {
        // No markers — LLM changed a word it shouldn't have
        let input = "hello world";
        let output = "hello earth";
        assert!(!validate_preservation(input, output));
    }

    #[test]
    fn validate_all_markers() {
        // All words are marked — LLM can change everything
        let input = "[?helo?] [?wrld?]";
        let output = "hello world";
        assert!(validate_preservation(input, output));
    }

    #[test]
    fn validate_empty_output_fails() {
        let input = "the [?recieve?] button";
        let output = "";
        assert!(!validate_preservation(input, output));
    }

    #[test]
    fn validate_filler_markers_removed() {
        // Fillers wrapped in [?..?] removed by LLM — the unmarked words must remain
        let input = "the [?um?] meeting [?recieve?] starts";
        let output = "the meeting receive starts";
        assert!(validate_preservation(input, output));
    }

    // ── Runner-up `{guess|alternate}` format tests ──────────────────────

    #[test]
    fn strip_markers_runner_up_picks_first_word() {
        assert_eq!(strip_markers("the {fox|box} jumped"), "the fox jumped");
    }

    #[test]
    fn strip_markers_runner_up_mixed_with_legacy() {
        // A single segment may carry both formats when some words had no
        // meaningful runner-up alternate on Parakeet.
        assert_eq!(
            strip_markers("the {fox|box} [?and?] the [?dog?]"),
            "the fox and the dog"
        );
    }

    #[test]
    fn strip_markers_unclosed_brace_preserved() {
        assert_eq!(strip_markers("the {fox|box"), "the {fox|box");
    }

    #[test]
    fn strip_markers_brace_without_pipe() {
        // Defensive: `{word}` with no pipe — still strips to `word`.
        assert_eq!(strip_markers("the {fox} jumped"), "the fox jumped");
    }

    #[test]
    fn strip_markers_no_brackets_or_braces() {
        assert_eq!(strip_markers("hello world"), "hello world");
    }

    #[test]
    fn validate_preservation_runner_up_marker_skippable() {
        // The word inside `{..|..}` is annotated, so the LLM may replace it.
        let input = "the {fox|box} jumped";
        let output = "the cat jumped";
        assert!(validate_preservation(input, output));
    }

    #[test]
    fn validate_preservation_runner_up_unmarked_word_dropped() {
        // Unmarked "jumped" must survive even when a runner-up marker
        // sits next to it.
        let input = "the {fox|box} jumped";
        let output = "the cat";
        assert!(!validate_preservation(input, output));
    }

    #[test]
    fn post_process_strips_runner_up_markers() {
        assert_eq!(
            post_process("the {weather|whether} was nice"),
            "the weather was nice"
        );
    }
}
