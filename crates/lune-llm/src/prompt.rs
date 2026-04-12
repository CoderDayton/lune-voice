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

/// System prompt for **N-best hypothesis rescoring** — the ProGRes-style
/// path (arXiv:2409.00217) used when the ASR backend exposes multiple
/// full-sentence candidates.
///
/// The LLM's task is constrained to "pick the most fluent candidate from
/// the numbered list and wrap it in angle brackets". This is strictly
/// easier for a small instruction-tuned corrector than free-form marker
/// correction because:
///
/// 1. Each candidate is a complete, grammatical sentence the model
///    already knows how to reason about. No novel marker syntax.
/// 2. The task reduces to multi-choice selection, which stock Qwen-1.5B
///    handles reliably without fine-tuning.
/// 3. Output validation is trivial: parse for `<...>`, match against the
///    candidate list, fall back to candidate 1 (the argmax) on any
///    parse or match failure. The corrector is strictly additive —
///    output can never regress below the argmax.
pub const N_BEST_SYSTEM: &str = "You are an ASR hypothesis rescorer. You will be given a numbered list of transcription candidates from a speech recognizer, ordered from most to least confident (candidate 1 is the most likely). Your job:\n\
1. Default to candidate 1. Only pick a different candidate if candidate 1 contains a word that is clearly wrong in context (e.g. a homophone mistake, a nonsense word, or a grammar break). Minor phrasing differences do NOT justify switching.\n\
2. Copy your chosen candidate verbatim — do not add, remove, or reorder words.\n\
3. Wrap your choice in angle brackets like <your choice here>.\n\
4. Do not add explanation, commentary, or justification.";

/// Build a ChatML inference prompt for the N-best rescoring path.
///
/// Renders the candidate list in the ProGRes format: a leading header
/// line, one candidate per line prefixed with its 1-based rank, and an
/// open `<` to bias the sampler toward the delimiter convention. Returns
/// the prompt plus the bracketed-answer prefix so `post_process_n_best`
/// can extract the completion.
pub fn build_n_best_prompt(hypotheses: &[String]) -> String {
    let mut body = String::with_capacity(256 + hypotheses.len() * 64);
    body.push_str("Transcription candidates in order of ASR confidence:\n");
    for (i, hyp) in hypotheses.iter().enumerate() {
        body.push_str(&format!("{}. {}\n", i + 1, hyp));
    }
    body.push_str("\nMost fluent candidate:");
    format!(
        "<|im_start|>system\n{N_BEST_SYSTEM}<|im_end|>\n\
         <|im_start|>user\n{body}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Post-process LLM output from the N-best rescoring path.
///
/// Algorithm (strictly fallback-safe — can never regress below argmax):
/// 1. Parse for `<...>` content; if found, trim and compare against each
///    hypothesis case-insensitively. Match → return that hypothesis.
/// 2. No brackets: scan each line of the output, trim, and compare.
///    Match → return that hypothesis.
/// 3. Trim leaked ChatML tokens and compare the whole cleaned output.
/// 4. No match → return `hypotheses[0]` (the argmax).
///
/// This guarantees the corrector is **strictly additive**: the output
/// is always a verbatim hypothesis from the list, so the best-case is
/// beam-quality and the worst-case is argmax.
pub fn post_process_n_best(raw: &str, hypotheses: &[String]) -> String {
    debug_assert!(!hypotheses.is_empty(), "n-best list must be non-empty");
    let cleaned = raw
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .trim()
        .to_string();

    // 1. Bracketed answer.
    if let Some(start) = cleaned.find('<')
        && let Some(end) = cleaned[start + 1..].find('>')
    {
        let inside = cleaned[start + 1..start + 1 + end].trim();
        if let Some(matched) = match_hypothesis(inside, hypotheses) {
            return matched;
        }
    }

    // 2. Line-by-line scan (handles models that forget the brackets).
    for line in cleaned.lines() {
        // Strip leading "N.", "N)", or bullet characters the model may
        // have copied from the prompt format.
        let trimmed = line
            .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == ' ')
            .trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(matched) = match_hypothesis(trimmed, hypotheses) {
            return matched;
        }
    }

    // 3. Whole-output match.
    if let Some(matched) = match_hypothesis(cleaned.trim(), hypotheses) {
        return matched;
    }

    // 4. Fallback: argmax.
    hypotheses[0].clone()
}

/// Case-insensitive punctuation-tolerant hypothesis lookup. Normalises
/// both sides by lowercasing and stripping leading/trailing ASCII
/// punctuation so `<fox jumped.>` still matches `fox jumped`.
fn match_hypothesis(candidate: &str, hypotheses: &[String]) -> Option<String> {
    let normalized = normalize_for_match(candidate);
    if normalized.is_empty() {
        return None;
    }
    hypotheses
        .iter()
        .find(|h| normalize_for_match(h) == normalized)
        .cloned()
}

#[inline]
fn normalize_for_match(s: &str) -> String {
    s.trim()
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

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

    // ── N-best rescoring path ───────────────────────────────────────────

    fn sample_hypotheses() -> Vec<String> {
        vec![
            "the quick brown fox jumped".into(),
            "the quick brown box jumped".into(),
            "the quick brown ox jumped".into(),
        ]
    }

    #[test]
    fn build_n_best_prompt_renders_numbered_list() {
        let p = build_n_best_prompt(&sample_hypotheses());
        assert!(p.contains("1. the quick brown fox jumped"));
        assert!(p.contains("2. the quick brown box jumped"));
        assert!(p.contains("3. the quick brown ox jumped"));
        assert!(p.contains("Most fluent candidate:"));
        assert!(p.starts_with("<|im_start|>system\n"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn post_process_n_best_bracketed_answer() {
        let hyps = sample_hypotheses();
        let raw = "<the quick brown box jumped>";
        assert_eq!(
            post_process_n_best(raw, &hyps),
            "the quick brown box jumped"
        );
    }

    #[test]
    fn post_process_n_best_bracketed_with_whitespace() {
        let hyps = sample_hypotheses();
        let raw = "  <  the quick brown ox jumped  >  ";
        assert_eq!(post_process_n_best(raw, &hyps), "the quick brown ox jumped");
    }

    #[test]
    fn post_process_n_best_trailing_punctuation_tolerated() {
        let hyps = sample_hypotheses();
        let raw = "<the quick brown fox jumped.>";
        assert_eq!(
            post_process_n_best(raw, &hyps),
            "the quick brown fox jumped"
        );
    }

    #[test]
    fn post_process_n_best_unbracketed_line_match() {
        // LLM forgot the brackets — line-scan fallback picks it up.
        let hyps = sample_hypotheses();
        let raw = "the quick brown box jumped";
        assert_eq!(
            post_process_n_best(raw, &hyps),
            "the quick brown box jumped"
        );
    }

    #[test]
    fn post_process_n_best_numbered_prefix_stripped() {
        // LLM echoed "2. ..." from the prompt — still parse it.
        let hyps = sample_hypotheses();
        let raw = "2. the quick brown box jumped";
        assert_eq!(
            post_process_n_best(raw, &hyps),
            "the quick brown box jumped"
        );
    }

    #[test]
    fn post_process_n_best_case_insensitive_match() {
        let hyps = sample_hypotheses();
        let raw = "<THE QUICK BROWN OX JUMPED>";
        assert_eq!(post_process_n_best(raw, &hyps), "the quick brown ox jumped");
    }

    #[test]
    fn post_process_n_best_unknown_output_falls_back_to_argmax() {
        // LLM produced a sentence not in the list — fall back to hypothesis[0].
        let hyps = sample_hypotheses();
        let raw = "<the slow brown fox ambled>";
        assert_eq!(post_process_n_best(raw, &hyps), hyps[0]);
    }

    #[test]
    fn post_process_n_best_empty_output_falls_back_to_argmax() {
        let hyps = sample_hypotheses();
        assert_eq!(post_process_n_best("", &hyps), hyps[0]);
    }

    #[test]
    fn post_process_n_best_strips_leaked_chatml_tokens() {
        let hyps = sample_hypotheses();
        let raw = "<|im_start|><the quick brown box jumped><|im_end|>";
        assert_eq!(
            post_process_n_best(raw, &hyps),
            "the quick brown box jumped"
        );
    }

    #[test]
    fn post_process_n_best_garbage_falls_back() {
        let hyps = sample_hypotheses();
        let raw = "I'm not sure, could be any of them.";
        assert_eq!(post_process_n_best(raw, &hyps), hyps[0]);
    }
}
