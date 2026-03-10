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
pub const SYSTEM: &str = "You are a transcript correction tool. Your only job is to fix words marked with [?word?].\n\
Rules:\n\
- Output ONLY the corrected transcript.\n\
- Copy every word NOT inside [?...?] exactly as-is.\n\
- Replace each [?word?] with the contextually correct word, no markers.\n\
- Delete [?um?] [?uh?] and filler variants entirely.\n\
- Do not rephrase, summarize, or add punctuation.";

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

/// Strip residual `[?word?]` and `[word]` markers from LLM output.
///
/// The span-correction prompt asks the LLM to replace `[?word?]` with corrected
/// text, but small models sometimes emit partially stripped markers like `[word]`
/// or fail to strip `[?word?]` entirely.
pub fn strip_markers(text: &str) -> String {
    if !text.contains('[') {
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
}
