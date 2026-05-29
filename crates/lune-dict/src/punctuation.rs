//! Deterministic punctuation normalization + LLM-gate detection.
//!
//! ASR output (Qwen3-ASR in particular) often arrives lowercase and without
//! terminal punctuation even when every *word* is high-confidence. The word
//! heuristic in [`crate::corrector`] resolves spelling, not punctuation, so a
//! fully-resolved segment would otherwise be emitted verbatim — "turn on the
//! light then close the door" — and bypass the LLM entirely.
//!
//! This module mirrors the "resolve deterministically, escalate the residual"
//! philosophy of the corrector, applied to punctuation:
//!
//! * **Tier 1 — deterministic, no LLM:** capitalize the first letter; append a
//!   terminal period when the segment is a single resolved declarative. These
//!   fixes are unambiguous, so they never need the model.
//! * **Tier 2 — ambiguous, trips the LLM gate:** comma-less run-ons and
//!   interrogative-leading segments with no end mark. The right fix here
//!   (internal commas, a sentence split, `?` vs `.` vs `!`) cannot be guessed
//!   safely, so [`PunctuationOutcome::needs_llm`] is set and the caller routes
//!   the text through the LLM punctuation pass.
//!
//! Non-Latin scripts (CJK, Hangul, etc.) use different punctuation and casing
//! conventions, so they pass through untouched and never trip the gate.

/// Word count at or above which a separator-free segment is treated as a
/// run-on candidate the deterministic pass cannot safely punctuate. Tuned for
/// the "balanced" sensitivity: high enough to ignore ordinary short sentences,
/// low enough to catch genuinely jarring run-ons.
const RUN_ON_WORD_THRESHOLD: usize = 12;

/// Leading tokens that mark a segment as interrogative. When one of these opens
/// a segment that lacks an end mark, the statement-vs-question punctuation is
/// ambiguous, so it is deferred to the LLM rather than guessed.
const INTERROGATIVE_LEADS: &[&str] = &[
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom", "is", "are", "am",
    "was", "were", "do", "does", "did", "can", "could", "would", "should", "will", "shall", "has",
    "have", "had", "may", "might", "must",
];

/// Result of [`normalize`].
pub struct PunctuationOutcome {
    /// Text with deterministic, unambiguous fixes applied: a leading capital
    /// and, for resolved declaratives, a terminal period. When `needs_llm` is
    /// true the terminal mark is left to the LLM, so only capitalization is
    /// applied here.
    pub text: String,
    /// True when residual punctuation is ambiguous enough to warrant the LLM
    /// (run-on candidate, or interrogative-leading with no end mark).
    pub needs_llm: bool,
}

/// Normalize punctuation on a single dictation segment.
///
/// Returns the deterministically-fixed text plus whether the residual
/// punctuation should be handed to the LLM. The returned text is always safe to
/// emit as-is *and* safe to feed to the LLM as input.
pub fn normalize(text: &str) -> PunctuationOutcome {
    let trimmed = text.trim();
    if trimmed.is_empty() || !is_latin(trimmed) {
        return PunctuationOutcome {
            text: text.to_owned(),
            needs_llm: false,
        };
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let ends_terminal = ends_with_terminal(trimmed);
    let interrogative = words
        .first()
        .map(|w| is_interrogative_lead(w))
        .unwrap_or(false);

    // Run-on: long stretch with no internal separator or sentence break. Such a
    // segment likely needs internal commas or a split — the deterministic pass
    // can only touch the ends, so escalate it.
    let run_on = words.len() >= RUN_ON_WORD_THRESHOLD && !has_internal_separator(trimmed);

    // Interrogative lead without an end mark is ambiguous (`?` vs `.` vs `!`),
    // so let the LLM decide rather than guessing a question mark.
    let needs_llm = run_on || (interrogative && !ends_terminal);

    // Tier 1: always capitalize the first letter.
    let mut out = String::with_capacity(trimmed.len() + 1);
    let mut capitalized = false;
    for c in trimmed.chars() {
        if !capitalized && c.is_alphabetic() {
            out.extend(c.to_uppercase());
            capitalized = true;
        } else {
            out.push(c);
        }
    }

    // Tier 1: append a terminal period only for resolved declaratives. When
    // `needs_llm` is set the interrogative/run-on branch owns the end mark.
    if !ends_terminal && !needs_llm {
        out.push('.');
    }

    PunctuationOutcome {
        text: out,
        needs_llm,
    }
}

/// True when every alphabetic character is in a Latin range. Accented European
/// text counts as Latin; CJK, Hangul, Cyrillic, Greek, Arabic, etc. do not.
fn is_latin(text: &str) -> bool {
    text.chars().filter(|c| c.is_alphabetic()).all(|c| {
        c.is_ascii_alphabetic() || matches!(c, '\u{00C0}'..='\u{024F}') // Latin-1 Supplement + Latin Extended-A/B
    })
}

/// Closing quotes/brackets that may trail a sentence terminator.
const CLOSING: &[char] = &['"', '\'', ')', ']', '}', '»', '”', '’'];
/// Sentence-terminating marks.
const TERMINATORS: &[char] = &['.', '!', '?', '…'];

/// Whether the segment already ends in a sentence terminator, ignoring trailing
/// closing quotes/brackets (e.g. `he said "go".` or `(done!)`).
fn ends_with_terminal(text: &str) -> bool {
    text.trim_end_matches(CLOSING).ends_with(TERMINATORS)
}

/// Whether the body carries any internal separator or sentence break (a comma,
/// semicolon, colon, dash, or a non-final terminator). A trailing terminator
/// does not count.
fn has_internal_separator(text: &str) -> bool {
    // Drop a trailing terminator (and any closing quote/bracket) so it is not
    // mistaken for internal structure.
    let body = text.trim_end_matches(CLOSING).trim_end_matches(TERMINATORS);
    body.chars()
        .any(|c| matches!(c, ',' | ';' | ':' | '—' | '–' | '.' | '!' | '?'))
}

/// Whether `word` (with leading punctuation stripped, case-folded) opens an
/// interrogative.
fn is_interrogative_lead(word: &str) -> bool {
    let w = word.trim_start_matches(|c: char| !c.is_alphanumeric());
    INTERROGATIVE_LEADS
        .iter()
        .any(|lead| w.eq_ignore_ascii_case(lead))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_and_periods_plain_declarative() {
        let out = normalize("turn on the light");
        assert_eq!(out.text, "Turn on the light.");
        assert!(!out.needs_llm);
    }

    #[test]
    fn leaves_already_punctuated_text_alone() {
        let out = normalize("Turn on the light.");
        assert_eq!(out.text, "Turn on the light.");
        assert!(!out.needs_llm);
    }

    #[test]
    fn respects_existing_terminal_question_mark() {
        let out = normalize("what time is it?");
        assert_eq!(out.text, "What time is it?");
        assert!(!out.needs_llm);
    }

    #[test]
    fn interrogative_lead_without_mark_defers_to_llm() {
        let out = normalize("what time is it");
        // Capitalized but no terminal forced — the LLM owns `?` vs `.`.
        assert_eq!(out.text, "What time is it");
        assert!(out.needs_llm);
    }

    #[test]
    fn long_comma_less_run_defers_to_llm() {
        let out = normalize("turn on the light then close the door it is getting dark in here now");
        assert!(out.needs_llm, "12+ word separator-free run should escalate");
        assert!(out.text.starts_with("Turn"));
    }

    #[test]
    fn long_run_with_comma_stays_deterministic() {
        let out = normalize("turn on the light, then close the door it is getting dark");
        assert!(
            !out.needs_llm,
            "a comma signals existing structure, no escalation"
        );
        assert!(out.text.ends_with('.'));
    }

    #[test]
    fn terminal_detection_ignores_trailing_quote() {
        let out = normalize("he said \"go.\"");
        assert!(!out.needs_llm);
        assert!(out.text.ends_with("\"go.\""));
    }

    #[test]
    fn non_latin_text_passes_through_untouched() {
        let out = normalize("你好世界");
        assert_eq!(out.text, "你好世界");
        assert!(!out.needs_llm);
    }

    #[test]
    fn empty_text_passes_through() {
        let out = normalize("   ");
        assert!(!out.needs_llm);
    }

    #[test]
    fn short_interrogative_with_period_not_flagged() {
        // Declarative-ending despite an interrogative lead — not ambiguous.
        let out = normalize("how about that.");
        assert_eq!(out.text, "How about that.");
        assert!(!out.needs_llm);
    }
}
