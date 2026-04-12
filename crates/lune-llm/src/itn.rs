//! Inverse Text Normalization (ITN) — convert spoken-form ASR output to
//! written form.
//!
//! Wraps `text-processing-rs` (a Rust port of NVIDIA NeMo ITN grammars)
//! to normalize numbers, dates, currency, and other entities in the
//! transcript. Applied as a post-ASR step before the text reaches the
//! user.
//!
//! Examples:
//! - "two hundred thirty two" → "232"
//! - "january fifth twenty twenty six" → "January 5, 2026"
//! - "cinco por ciento" → "5%"

/// Apply inverse text normalization to ASR output.
///
/// Processes the full sentence, replacing spoken-form spans (numbers,
/// dates, currency, ordinals) with their written equivalents while
/// preserving all other words verbatim.
///
/// Returns the input unchanged if no normalizable spans are found.
pub fn normalize(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    text_processing_rs::normalize_sentence(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_cardinal_number() {
        let result = normalize("I have two hundred apples");
        assert!(result.contains("200"), "expected '200' in '{result}'");
    }

    #[test]
    fn normalize_leaves_plain_text_alone() {
        let input = "the quick brown fox jumped over the lazy dog";
        assert_eq!(normalize(input), input);
    }

    #[test]
    fn normalize_empty_string() {
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn normalize_ordinal() {
        let result = normalize("the twenty first of january");
        // Should contain "21" or "21st"
        assert!(result.contains("21"), "expected '21' in '{result}'");
    }

    #[test]
    fn normalize_preserves_surrounding_text() {
        let result = normalize("she bought three hundred books yesterday");
        assert!(result.starts_with("she bought"));
        assert!(result.ends_with("books yesterday"));
        assert!(result.contains("300"));
    }
}
