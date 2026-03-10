//! Input preparation for contextual token resolution.
//!
//! Combines per-word confidence marking with filler removal to produce
//! clean text with `[?word?]` markers on uncertain tokens — ready for
//! LLM-based contextual resolution.

use lune_core::asr::AsrSegment;

use crate::fillers;

/// Prepare ASR segments for contextual token resolution.
///
/// Pipeline:
/// 1. Mark uncertain words → `[?word?]` via per-word confidence
/// 2. Join all segments into a single utterance
/// 3. Remove filler words (bare and `[?filler?]`-wrapped)
///
/// Fillers that are also low-confidence (e.g. `[?um?]`) are removed by
/// `remove_fillers` — correct behavior since they carry no semantic content.
///
/// Returns `(prepared_text, has_uncertain)`:
/// - `prepared_text`: cleaned text with `[?word?]` markers on uncertain non-filler tokens
/// - `has_uncertain`: whether any `[?..?]` markers remain after filler removal
pub fn prepare_input(segments: &[AsrSegment], threshold: f32, lang: &str) -> (String, bool) {
    // 1. Mark uncertain words per segment, join
    let marked: String = segments
        .iter()
        .map(|s| s.mark_uncertain(threshold))
        .filter(|t| {
            let trimmed = t.trim();
            !trimmed.is_empty()
        })
        .collect::<Vec<_>>()
        .join(" ");

    if marked.trim().is_empty() {
        return (String::new(), false);
    }

    // 2. Remove fillers (handles both bare "um" and marked "[?um?]")
    let cleaned = match fillers::remove_fillers(&marked, lang) {
        Ok(c) => c,
        Err(_) => marked, // unsupported language — skip filler removal
    };

    // 3. Check if any uncertain markers remain (fillers may have consumed some)
    let has_uncertain = cleaned.contains("[?");

    (cleaned, has_uncertain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::asr::{AsrSegment, WordConfidence};

    fn seg(text: &str, wcs: Vec<(&str, f32)>) -> AsrSegment {
        AsrSegment {
            id: 0,
            text: text.into(),
            confidence: Some(0.9),
            min_token_confidence: None,
            is_final: true,
            start_ms: None,
            end_ms: None,
            word_confidences: wcs
                .into_iter()
                .map(|(w, c)| WordConfidence {
                    word: w.into(),
                    confidence: c,
                })
                .collect(),
        }
    }

    #[test]
    fn basic_marking_and_filler_removal() {
        let segments = vec![seg(
            "um the recieve button",
            vec![
                ("um", 0.3),
                ("the", 0.95),
                ("recieve", 0.4),
                ("button", 0.92),
            ],
        )];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        // "um" is filler AND low-conf → removed entirely
        // "recieve" is low-conf → marked [?recieve?]
        assert_eq!(text, "the [?recieve?] button");
        assert!(has_uncertain);
    }

    #[test]
    fn all_confident_no_fillers() {
        let segments = vec![seg("hello world", vec![("hello", 0.95), ("world", 0.90)])];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        assert_eq!(text, "hello world");
        assert!(!has_uncertain);
    }

    #[test]
    fn only_fillers_uncertain() {
        // All uncertain words are fillers → removed, no markers remain
        let segments = vec![seg(
            "um uh the result",
            vec![("um", 0.3), ("uh", 0.2), ("the", 0.95), ("result", 0.90)],
        )];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        assert_eq!(text, "the result");
        assert!(!has_uncertain);
    }

    #[test]
    fn multi_segment() {
        let segments = vec![
            seg("hello world", vec![("hello", 0.95), ("world", 0.90)]),
            seg(
                "um recieve this",
                vec![("um", 0.3), ("recieve", 0.4), ("this", 0.95)],
            ),
        ];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        assert_eq!(text, "hello world [?recieve?] this");
        assert!(has_uncertain);
    }

    #[test]
    fn empty_segments() {
        let (text, has_uncertain) = prepare_input(&[], 0.65, "en");
        assert_eq!(text, "");
        assert!(!has_uncertain);
    }

    #[test]
    fn unsupported_language_skips_filler_removal() {
        let segments = vec![seg(
            "um hallo welt",
            vec![("um", 0.3), ("hallo", 0.95), ("welt", 0.90)],
        )];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "de");
        // German unsupported → filler removal skipped, "um" stays as [?um?]
        assert_eq!(text, "[?um?] hallo welt");
        assert!(has_uncertain);
    }

    #[test]
    fn no_word_confidences_uses_raw_text() {
        let segments = vec![AsrSegment {
            id: 0,
            text: "hello um world".into(),
            confidence: Some(0.9),
            min_token_confidence: None,
            is_final: true,
            start_ms: None,
            end_ms: None,
            word_confidences: vec![],
        }];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        // No word confidences → mark_uncertain returns raw text
        // Bare "um" still removed by filler removal
        assert_eq!(text, "hello world");
        assert!(!has_uncertain);
    }
}
