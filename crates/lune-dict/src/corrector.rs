use std::sync::Arc;

use lune_ngram::NGramModel;

use crate::lexicon::{Lexicon, WordSource};

pub fn has_dict(language: &str) -> bool {
    crate::freq::has_dict(language)
}

/// Risk score (Heuristic Word Confidence Estimate) at which a word is treated
/// as "uncertain" — surfaced to the LLM via `unresolved_words` and used as the
/// gate for invoking the LLM at all (see [`LLM_GATE_RISK`]).
pub const HWCE_FLAG_THRESHOLD: f32 = 0.40;

/// Risk score that triggers the LLM correction pass. Aliased to
/// [`HWCE_FLAG_THRESHOLD`] so a session can never reach the LLM without at
/// least one word also qualifying as a hint — keeps `needs_llm=true` and
/// `unresolved_words` consistent.
pub const LLM_GATE_RISK: f32 = HWCE_FLAG_THRESHOLD;

/// Maximum risk at which a known high-frequency word is allowed to pass
/// through unchanged. Above this, the word is sent to the LLM as uncertain
/// even if the lexicon recognizes it.
pub const CANONICAL_OVERRIDE_RISK_CEIL: f32 = 0.75;

/// Maximum risk at which an unknown but proper-shaped token (capitalized
/// word, acronym, CamelCase) is preserved verbatim. Above this, the word is
/// flagged regardless of shape.
pub const PROPER_NOUN_RISK_CEIL: f32 = 0.80;

#[derive(Debug)]
pub struct WordScore {
    pub word: String,
    pub position: usize,
    pub candidate_score: f32,
}

/// Result of processing a single word during correction.
/// Contains the word to include in output and optional unresolved entry.
struct ProcessedWord {
    corrected_word: String,
    unresolved: Option<WordScore>,
}

pub struct CorrectionResult {
    pub corrected_text: String,
    pub word_risk_positions: Vec<(usize, f32)>,
    pub unresolved_words: Vec<WordScore>,
    pub all_resolved: bool,
    /// Risk-score entries that did not align to any text token. Non-zero
    /// indicates ASR word boundaries diverged from `split_whitespace`
    /// (punctuation splits, contractions, duplicate-without-match).
    pub dropped_risk_entries: usize,
}

/// Process a single word during correction.
///
/// Returns the word to include in the output and optionally a `WordScore`
/// for unresolved words that will be passed to the LLM.
fn process_single_word(
    word: &str,
    i: usize,
    stripped: &str,
    lexicon_info: &Option<crate::lexicon::WordInfo>,
    risk_score: f32,
    dict_replacements: &[(String, String)],
) -> ProcessedWord {
    // Empty or digit-containing words pass through unchanged
    if stripped.is_empty() || stripped.chars().any(|c| c.is_ascii_digit()) {
        return ProcessedWord {
            corrected_word: word.to_owned(),
            unresolved: None,
        };
    }

    let is_unknown = !lexicon_info.as_ref().is_some_and(|info| {
        info.freq >= Lexicon::CONFIDENT_FREQ || info.source == WordSource::Vocab
    });
    let flagged = risk_score >= HWCE_FLAG_THRESHOLD;

    // Dictionary replacements take priority
    if let Some((_, repl)) = dict_replacements
        .iter()
        .find(|(phrase, _)| phrase.eq_ignore_ascii_case(stripped))
    {
        return ProcessedWord {
            corrected_word: restore_punctuation(word, repl),
            unresolved: None,
        };
    }

    // Known, unflagged words pass through
    if !is_unknown && !flagged {
        return ProcessedWord {
            corrected_word: word.to_owned(),
            unresolved: None,
        };
    }

    // Apply canonical casing from the vocab (e.g. "github" → "GitHub").
    // Only override when the canonical itself carries uppercase information —
    // otherwise we'd flatten correct sentence-initial capitalization (e.g.
    // "All" → "all") because freq-dict canonicals fall back to lowercase.
    if let Some(info) = lexicon_info.as_ref() {
        let canonical_has_upper = info.canonical.chars().any(|c| c.is_ascii_uppercase());
        if canonical_has_upper && info.canonical.as_ref() != stripped {
            return ProcessedWord {
                corrected_word: restore_punctuation(word, &info.canonical),
                unresolved: None,
            };
        }

        let is_high_freq = info.freq >= Lexicon::CONFIDENT_FREQ;
        if info.source == WordSource::Vocab
            || (is_high_freq && risk_score < CANONICAL_OVERRIDE_RISK_CEIL)
        {
            return ProcessedWord {
                corrected_word: word.to_owned(),
                unresolved: None,
            };
        }
    }

    // Proper nouns and acronyms pass through unless very high risk
    let looks_proper = stripped.starts_with(|c: char| c.is_ascii_uppercase())
        || (stripped.len() >= 2 && stripped.chars().all(|c| c.is_ascii_uppercase()));
    // Mixed-case identifier: any uppercase + any lowercase, length >= 3.
    // Looser than "≥2 uppercase" so leading-lowercase forms like "iPhone",
    // "eBay", "macOS" are recognized as proper-shaped tokens.
    let is_camel_case = stripped.len() >= 3
        && stripped.chars().any(|c| c.is_ascii_uppercase())
        && stripped.chars().any(|c| c.is_ascii_lowercase());

    // Proper-shaped tokens are preserved verbatim under the risk ceiling
    // regardless of dictionary membership: a known word that *also* looks
    // like a proper noun (e.g. brand names or sentence-initial capitals)
    // shouldn't be rewritten by the heuristic at moderate risk.
    if (looks_proper || is_camel_case) && risk_score < PROPER_NOUN_RISK_CEIL {
        return ProcessedWord {
            corrected_word: word.to_owned(),
            unresolved: None,
        };
    }

    // Flagged words become uncertain — passed as hints to the LLM
    let unresolved = if flagged {
        Some(WordScore {
            word: stripped.to_owned(),
            position: i,
            candidate_score: 0.0,
        })
    } else {
        None
    };

    ProcessedWord {
        corrected_word: word.to_owned(),
        unresolved,
    }
}

pub struct HeuristicCorrector {
    #[allow(dead_code)]
    ngram: Arc<NGramModel>,
}

impl Default for HeuristicCorrector {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicCorrector {
    pub fn from_ngram(ngram: Arc<NGramModel>) -> Self {
        Self { ngram }
    }

    pub fn new() -> Self {
        Self::from_ngram(Arc::new(
            NGramModel::load_bundled().expect("bundled ngram model"),
        ))
    }

    pub fn correct(
        &self,
        text: &str,
        word_risk_scores: &[(String, f32)],
        dict_replacements: &[(String, String)],
        lang: &str,
    ) -> CorrectionResult {
        let words: Vec<&str> = text.split_whitespace().collect();
        let aligned = align_word_risk_positions(&words, word_risk_scores);
        let word_risk_positions = aligned.positions;
        let dropped_risk_entries = aligned.dropped;

        if !has_dict(lang) {
            return CorrectionResult {
                corrected_text: text.to_owned(),
                word_risk_positions,
                unresolved_words: Vec::new(),
                all_resolved: true,
                dropped_risk_entries,
            };
        }

        let lexicon = Lexicon::for_lang(lang);
        let mut result_words: Vec<String> = Vec::with_capacity(words.len());
        let mut unresolved = Vec::new();

        for (i, &word) in words.iter().enumerate() {
            let stripped = strip_punctuation(word);
            let lexicon_info = lexicon.lookup(stripped);
            let risk_score = word_risk_positions
                .iter()
                .find(|(position, _)| *position == i)
                .map(|(_, risk)| *risk)
                .unwrap_or(0.0);

            let processed = process_single_word(
                word,
                i,
                stripped,
                &lexicon_info,
                risk_score,
                dict_replacements,
            );

            result_words.push(processed.corrected_word);
            if let Some(word_score) = processed.unresolved {
                unresolved.push(word_score);
            }
        }

        let all_resolved = unresolved.is_empty();
        CorrectionResult {
            corrected_text: result_words.join(" "),
            word_risk_positions,
            unresolved_words: unresolved,
            all_resolved,
            dropped_risk_entries,
        }
    }
}

fn strip_punctuation(word: &str) -> &str {
    let s = word.trim_start_matches(|c: char| c.is_ascii_punctuation());
    s.trim_end_matches(|c: char| c.is_ascii_punctuation())
}

/// Aligned positions plus a count of risk entries that could not be matched
/// to any remaining text token.
struct AlignedRisks {
    positions: Vec<(usize, f32)>,
    dropped: usize,
}

fn align_word_risk_positions(words: &[&str], word_risk_scores: &[(String, f32)]) -> AlignedRisks {
    let mut out = Vec::new();
    let mut dropped = 0usize;
    let mut cursor = 0usize;

    for (risk_word, risk) in word_risk_scores {
        let risk_word = strip_punctuation(risk_word);
        if risk_word.is_empty() {
            continue;
        }

        if let Some(offset) = words[cursor..]
            .iter()
            .position(|word| strip_punctuation(word).eq_ignore_ascii_case(risk_word))
        {
            let idx = cursor + offset;
            out.push((idx, *risk));
            cursor = (idx + 1).min(words.len());
        } else {
            dropped += 1;
        }
    }

    AlignedRisks {
        positions: out,
        dropped,
    }
}

fn restore_punctuation(original: &str, corrected: &str) -> String {
    let prefix = &original[..original.len()
        - original
            .trim_start_matches(|c: char| c.is_ascii_punctuation())
            .len()];
    let suffix_start = original
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .len();
    let suffix = &original[suffix_start..];
    format!("{prefix}{corrected}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_word_not_flagged_passes_through() {
        let c = HeuristicCorrector::new();
        let result = c.correct("the quick brown fox", &[], &[], "en");
        assert_eq!(result.corrected_text, "the quick brown fox");
    }

    #[test]
    fn word_risk_positions_survive_heuristic() {
        let c = HeuristicCorrector::new();
        let scores = vec![("teh".into(), 0.80)];
        let result = c.correct("teh quick", &scores, &[], "en");
        assert_eq!(result.word_risk_positions, vec![(0, 0.80)]);
    }

    #[test]
    fn word_risk_positions_handle_duplicate_words_by_order() {
        let c = HeuristicCorrector::new();
        let scores = vec![("the".into(), 0.10), ("the".into(), 0.85)];
        let result = c.correct("the build fixed the deployment", &scores, &[], "en");
        assert_eq!(result.word_risk_positions, vec![(0, 0.10), (3, 0.85)]);
    }

    #[test]
    fn proper_noun_skipped_when_not_flagged() {
        let c = HeuristicCorrector::new();
        let result = c.correct("London is great", &[], &[], "en");
        assert!(result.corrected_text.contains("London"));
    }

    #[test]
    fn acronym_skipped_when_not_flagged() {
        let c = HeuristicCorrector::new();
        let result = c.correct("the NASA program", &[], &[], "en");
        assert!(result.corrected_text.contains("NASA"));
    }

    #[test]
    fn dict_replacement_takes_priority() {
        let c = HeuristicCorrector::new();
        let dict = vec![("helo".into(), "hello".into())];
        let result = c.correct("helo world", &[], &dict, "en");
        assert_eq!(result.corrected_text, "hello world");
    }

    #[test]
    fn unsupported_lang_bypasses_corrector() {
        let c = HeuristicCorrector::new();
        let result = c.correct("jambo dunia", &[], &[], "swahili");
        assert_eq!(result.corrected_text, "jambo dunia");
    }

    #[test]
    fn french_known_words_pass_through() {
        let c = HeuristicCorrector::new();
        let result = c.correct("bonjour le monde", &[], &[], "fr");
        assert_eq!(result.corrected_text, "bonjour le monde");
    }

    #[test]
    fn unresolved_words_reported() {
        let c = HeuristicCorrector::new();
        let risks = vec![("xyzqqq".to_string(), 0.9)];
        let result = c.correct("the xyzqqq fox", &risks, &[], "en");
        assert!(!result.unresolved_words.is_empty());
        assert!(result.unresolved_words.iter().any(|ws| ws.word == "xyzqqq"));
    }

    #[test]
    fn empty_input() {
        let c = HeuristicCorrector::new();
        let result = c.correct("", &[], &[], "en");
        assert_eq!(result.corrected_text, "");
    }

    #[test]
    fn single_char_word_not_corrected() {
        let c = HeuristicCorrector::new();
        let result = c.correct("I a", &[], &[], "en");
        assert_eq!(result.corrected_text, "I a");
    }

    #[test]
    fn flagged_curated_vocab_passes_through_without_correction() {
        let c = HeuristicCorrector::new();
        let risks = vec![("WebSocket".to_string(), 0.95)];
        let result = c.correct("WebSocket failed", &risks, &[], "en");
        assert_eq!(result.corrected_text, "WebSocket failed");
    }

    #[test]
    fn llm_gate_risk_matches_flag_threshold() {
        // Invariant: if the session-level gate fires, at least one word must
        // also qualify as uncertain — otherwise the LLM gets no hint.
        assert!((LLM_GATE_RISK - HWCE_FLAG_THRESHOLD).abs() < f32::EPSILON);
    }

    #[test]
    fn sub_gate_risk_does_not_fire_llm() {
        // A risk just under the gate must neither flag the word nor trigger
        // a needs_llm decision (the session uses LLM_GATE_RISK).
        let c = HeuristicCorrector::new();
        let just_under = LLM_GATE_RISK - 0.05;
        let risks = vec![("the".to_string(), just_under)];
        let result = c.correct("the quick brown fox", &risks, &[], "en");
        assert!(result.unresolved_words.is_empty());
        assert!(!risks.iter().any(|(_, r)| *r >= LLM_GATE_RISK));
    }

    #[test]
    fn camel_case_leading_lowercase_preserved() {
        // "iPhone" has only one uppercase char — under the old (≥2 upper)
        // predicate it would not have been recognized as a proper-shaped
        // token and could be marked unresolved.
        let c = HeuristicCorrector::new();
        let risks = vec![("iPhone".to_string(), 0.7)];
        let result = c.correct("the iPhone is new", &risks, &[], "en");
        assert!(result.corrected_text.contains("iPhone"));
        assert!(
            !result.unresolved_words.iter().any(|w| w.word == "iPhone"),
            "iPhone should not be flagged as unresolved at moderate risk"
        );
    }

    #[test]
    fn dropped_risks_counted() {
        // Risk word that doesn't appear in the text should be reported.
        let c = HeuristicCorrector::new();
        let risks = vec![("the".to_string(), 0.1), ("nonexistent".to_string(), 0.5)];
        let result = c.correct("the quick fox", &risks, &[], "en");
        assert_eq!(result.dropped_risk_entries, 1);
    }

    #[test]
    fn no_dropped_risks_on_full_match() {
        let c = HeuristicCorrector::new();
        let risks = vec![("the".to_string(), 0.1), ("fox".to_string(), 0.2)];
        let result = c.correct("the quick fox", &risks, &[], "en");
        assert_eq!(result.dropped_risk_entries, 0);
    }

    #[test]
    fn flagged_sentence_initial_capital_preserved() {
        let c = HeuristicCorrector::new();
        // High risk on "All" used to force-lowercase it via the canonical
        // override because freq-dict canonicals fall back to lowercase.
        let risks = vec![("All".to_string(), 0.85)];
        let result = c.correct("All I'm doing is trash work.", &risks, &[], "en");
        assert!(
            result.corrected_text.starts_with("All "),
            "expected leading 'All' preserved, got {:?}",
            result.corrected_text
        );
    }

    #[test]
    fn vocab_word_gets_canonical_casing() {
        let c = HeuristicCorrector::new();
        let risks = vec![("github".to_string(), 0.5)];
        let result = c.correct("push to github", &risks, &[], "en");
        assert_eq!(result.corrected_text, "push to GitHub");
    }

    #[test]
    fn has_dict_checks() {
        assert!(has_dict("en"));
        assert!(has_dict("eng"));
        assert!(has_dict("English"));
        assert!(has_dict(""));
        assert!(has_dict("fr"));
        assert!(has_dict("de"));
        assert!(has_dict("Japanese"));
        assert!(!has_dict("swahili"));
    }

    #[test]
    fn restore_punctuation_edges() {
        assert_eq!(restore_punctuation("\"hello\"", "hi"), "\"hi\"");
        assert_eq!(restore_punctuation("hello.", "hi"), "hi.");
        assert_eq!(restore_punctuation("hello", "hi"), "hi");
    }

    #[test]
    fn punctuation_preserved_with_dict() {
        let c = HeuristicCorrector::new();
        let dict = vec![("teh".into(), "the".into())];
        let result = c.correct("teh, quick", &[], &dict, "en");
        assert!(result.corrected_text.contains("the,"));
    }

    #[test]
    fn from_ngram_constructor_works() {
        use lune_ngram::{NGramModel, ScoringConfig};
        use std::sync::Arc;

        let model = NGramModel::new(ScoringConfig::trigram_default()).unwrap();
        let c = HeuristicCorrector::from_ngram(Arc::new(model));
        let result = c.correct("the quick", &[], &[], "en");
        assert_eq!(result.corrected_text, "the quick");
    }

    #[test]
    fn qd_002_no_false_positives() {
        let c = HeuristicCorrector::new();
        let text = "The memory leak was traced back to the event listener on the WebSocket connection. It wasn't being cleaned up when the component unmounted. I pushed a fix to the dev branch and it looks stable now.";
        let result = c.correct(text, &[], &[], "en");
        assert!(
            result.unresolved_words.is_empty(),
            "unexpected unresolved: {:?}",
            result.unresolved_words
        );
    }
}
