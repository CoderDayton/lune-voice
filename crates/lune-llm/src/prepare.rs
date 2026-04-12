//! Input preparation for contextual token resolution.
//!
//! Combines per-word confidence marking with filler removal to produce
//! clean text with uncertainty annotations (`[?word?]` or `{guess|alternate}`)
//! — ready for LLM-based contextual resolution.

use lune_core::asr::AsrSegment;

use crate::fillers;

/// Returns `true` when `text` contains any uncertainty annotation
/// emitted by `mark_uncertain`: either a legacy `[?word?]` marker or a
/// runner-up `{guess|alternate}` marker. The `{...|...}` check is strict
/// (requires both braces and a pipe inside) to avoid false positives on
/// unrelated punctuation.
#[inline]
pub fn has_uncertainty_markers(text: &str) -> bool {
    if text.contains("[?") {
        return true;
    }
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '{' {
            continue;
        }
        let mut has_pipe = false;
        for ic in chars.by_ref() {
            match ic {
                '}' => {
                    if has_pipe {
                        return true;
                    }
                    break;
                }
                '|' => has_pipe = true,
                _ => {}
            }
        }
    }
    false
}

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
    let has_word_confidences = segments.iter().any(|s| !s.word_confidences.is_empty());

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

    // 2. Filler removal strategy depends on whether we have confidence data:
    //    - WITH confidence: only strip [?filler?]-marked fillers. Bare fillers
    //      that passed the confidence check are preserved (may be real words).
    //    - WITHOUT confidence: strip all fillers (bare + marked) since we can't
    //      distinguish real words from fillers without confidence signals.
    let cleaned = if has_word_confidences {
        match fillers::remove_fillers_marked_only(&marked, lang) {
            Ok(c) => c,
            Err(_) => marked,
        }
    } else {
        match fillers::remove_fillers(&marked, lang) {
            Ok(c) => c,
            Err(_) => marked,
        }
    };

    // 3. Check if any uncertain markers remain (fillers may have consumed some)
    let has_uncertain = has_uncertainty_markers(&cleaned);

    (cleaned, has_uncertain)
}

/// Contextual variant of [`prepare_input`] that uses neighbor-aware
/// confidence thresholds.
///
/// Instead of a flat threshold, each word's effective threshold is adjusted
/// based on its position (first-word leniency) and neighbor confidence
/// (strong neighbors protect borderline words; weak neighbors flag them).
///
/// This reduces false positives (unnecessary LLM calls on correct text)
/// and false negatives (missed errors in noisy regions).
pub fn prepare_input_contextual(
    segments: &[AsrSegment],
    threshold: f32,
    lang: &str,
) -> (String, bool) {
    let marked: String = segments
        .iter()
        .map(|s| s.mark_uncertain_contextual(threshold))
        .filter(|t| !t.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    if marked.trim().is_empty() {
        return (String::new(), false);
    }

    let cleaned = match fillers::remove_fillers_marked_only(&marked, lang) {
        Ok(c) => c,
        Err(_) => marked,
    };

    let has_uncertain = has_uncertainty_markers(&cleaned);
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
                .map(|(w, c)| WordConfidence::new(w.to_string(), c))
                .collect(),
            hypotheses: Vec::new(),
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
    fn no_word_confidences_strips_bare_fillers() {
        // No word confidences → mark_uncertain returns raw text unchanged.
        // Without confidence data, bare fillers ARE stripped (full removal)
        // because we have no signal to distinguish fillers from real words,
        // and removing common fillers is the safer default.
        let segments = vec![AsrSegment {
            id: 0,
            text: "hello um world".into(),
            confidence: Some(0.9),
            min_token_confidence: None,
            is_final: true,
            start_ms: None,
            end_ms: None,
            word_confidences: vec![],
            hypotheses: Vec::new(),
        }];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        assert_eq!(text, "hello world");
        assert!(!has_uncertain);
    }

    #[test]
    fn with_word_confidences_preserves_confident_fillers() {
        // With confidence data, a confident filler (e.g. "um" at 0.95) is
        // preserved — the speaker may have said a real word misheard as filler.
        let segments = vec![seg(
            "hello um world",
            vec![("hello", 0.95), ("um", 0.85), ("world", 0.90)],
        )];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        // "um" is above threshold, so not marked. Bare filler stripping is
        // disabled when we have confidence data. "um" survives.
        assert_eq!(text, "hello um world");
        assert!(!has_uncertain);
    }

    #[test]
    fn with_word_confidences_strips_uncertain_fillers() {
        // With confidence data, a low-confidence filler gets marked [?um?],
        // then remove_fillers_marked_only strips it.
        let segments = vec![seg(
            "hello um world",
            vec![("hello", 0.95), ("um", 0.30), ("world", 0.90)],
        )];
        let (text, has_uncertain) = prepare_input(&segments, 0.65, "en");
        assert_eq!(text, "hello world");
        assert!(!has_uncertain);
    }

    // ── Contextual marking integration tests ────────────────────────────

    #[test]
    fn contextual_marking_protects_borderline_with_strong_neighbors() {
        // "weather" at 0.62 — below flat threshold but above contextual
        // threshold when neighbors are confident.
        let segments = vec![seg(
            "the weather today",
            vec![("the", 0.95), ("weather", 0.62), ("today", 0.94)],
        )];
        let (text, has_uncertain) = prepare_input_contextual(&segments, 0.65, "en");
        assert_eq!(text, "the weather today");
        assert!(!has_uncertain);
    }

    #[test]
    fn contextual_marking_flags_error_cluster() {
        // All low-confidence — error cluster, all should be marked
        let segments = vec![seg(
            "tha wether tody",
            vec![("tha", 0.35), ("wether", 0.55), ("tody", 0.30)],
        )];
        let (text, has_uncertain) = prepare_input_contextual(&segments, 0.65, "en");
        assert!(text.contains("[?tha?]"));
        assert!(text.contains("[?wether?]"));
        assert!(text.contains("[?tody?]"));
        assert!(has_uncertain);
    }

    #[test]
    fn contextual_filler_in_error_cluster_removed() {
        // "um" is a filler AND low-conf with low-conf neighbors — removed
        let segments = vec![seg(
            "um tha recieve",
            vec![("um", 0.30), ("tha", 0.35), ("recieve", 0.40)],
        )];
        let (text, has_uncertain) = prepare_input_contextual(&segments, 0.65, "en");
        assert!(!text.contains("um"));
        assert!(text.contains("[?tha?]"));
        assert!(text.contains("[?recieve?]"));
        assert!(has_uncertain);
    }

    // ── has_uncertainty_markers helper ──────────────────────────────────

    #[test]
    fn detects_legacy_bracket_marker() {
        assert!(has_uncertainty_markers("the [?fox?] jumped"));
    }

    #[test]
    fn detects_runner_up_brace_marker() {
        assert!(has_uncertainty_markers("the {fox|box} jumped"));
    }

    #[test]
    fn detects_both_formats_in_same_string() {
        assert!(has_uncertainty_markers("the {fox|box} and the [?cat?]"));
    }

    #[test]
    fn ignores_lone_brace_without_pipe() {
        // A bare `{word}` without a pipe is not a runner-up marker —
        // `strip_markers` still strips it, but it is not "uncertain" for
        // the purpose of deciding whether to call the LLM.
        assert!(!has_uncertainty_markers("the {only} option"));
    }

    #[test]
    fn ignores_lone_brace_with_no_close() {
        assert!(!has_uncertainty_markers("set x = {foo |"));
    }

    #[test]
    fn ignores_plain_text() {
        assert!(!has_uncertainty_markers("hello world"));
    }

    #[test]
    fn ignores_pipe_outside_braces() {
        assert!(!has_uncertainty_markers("echo a | b"));
    }
}
