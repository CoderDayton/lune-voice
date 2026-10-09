// SPDX-License-Identifier: GPL-3.0-or-later

pub mod caprate;
pub mod commands;
pub mod corrector;
pub mod formatting;
pub mod freq;
pub mod lexicon;
pub mod snippets;
pub mod vocab;

use lune_core::dict::DictionaryEntry;

/// The user dictionary as the correction pipeline reads it.
///
/// Finds matches in ASR text (phrases — several words, or a word carrying
/// punctuation such as `node.js` — exactly as written; plain single words
/// exactly, else Levenshtein distance ≤ 1 for words of [`MIN_FUZZY_CHARS`] or
/// more) and selects relevant entries for LLM prompts.
/// Holds no database handle: the caller loads the entries and builds a new
/// value when they change.
#[derive(Debug, Clone, Default)]
pub struct UserDictionary {
    /// Highest priority first; `top_relevant` truncates in this order.
    entries: Vec<DictionaryEntry>,
}

impl UserDictionary {
    pub fn new(mut entries: Vec<DictionaryEntry>) -> Self {
        entries.sort_by_key(|e| std::cmp::Reverse(e.priority));
        Self { entries }
    }

    pub fn scan(&self, text: &str) -> Vec<DictMatch> {
        let (single, phrases): (Vec<&DictionaryEntry>, Vec<&DictionaryEntry>) = self
            .entries
            .iter()
            .partition(|e| corrector::is_plain_word(&e.phrase));

        let mut replacements = Vec::new();
        // `text` is not mutated across the loop, so split it once instead of
        // re-splitting per multi-word entry (was O(entries × text length)).
        let text_words: Vec<&str> = text.split_whitespace().collect();

        for entry in &phrases {
            let phrase_words: Vec<&str> = entry.phrase.split_whitespace().collect();
            let n = phrase_words.len();
            // An empty or whitespace-only phrase is not a plain word, so it
            // lands here and tokenizes to nothing. A zero-width window would
            // advance `i` by zero and the loop below would never terminate.
            if n == 0 {
                continue;
            }
            let mut i = 0;
            while i + n <= text_words.len() {
                let window = &text_words[i..i + n];
                if let Some((prefix, suffix)) =
                    corrector::phrase_window_matches(window, &phrase_words)
                {
                    // Without the surrounding punctuation, so the corrector
                    // can place the phrase again wherever it sits.
                    let joined = window.join(" ");
                    let matched = joined[prefix..joined.len() - suffix].to_owned();
                    // Phrase matches are word-boundary-exact, never fuzzy.
                    replacements.push(DictMatch {
                        matched,
                        replacement: entry.replacement.clone(),
                        fuzzy: false,
                    });
                    i += n;
                } else {
                    i += 1;
                }
            }
        }

        // Single-word matching is span-aware (`split_internal_punctuation`),
        // not just whitespace-split, so a word ASR glues to a neighbor with
        // no whitespace (e.g. an ellipsis: "is...Tayton.") still gets its
        // own lookup — the same tokenization `Lexicon::vocab_fuzzy_matches`
        // uses, so the user dictionary and the bundled vocab agree on what
        // counts as a word.
        for word in text.split_whitespace() {
            let (_, spans) = corrector::split_internal_punctuation(word);
            for (span, _) in &spans {
                if span.is_empty() {
                    continue;
                }
                // Exact (case-insensitive) match is trusted; an off-by-one
                // match is fuzzy and will be gated by the corrector.
                if let Some((entry, fuzzy)) = match_single_word(&single, span) {
                    replacements.push(DictMatch {
                        matched: (*span).to_owned(),
                        replacement: entry.replacement.clone(),
                        fuzzy,
                    });
                }
            }
        }

        replacements
    }

    /// Return up to `limit` entries most relevant to `text` for LLM injection.
    ///
    /// Relevance: entries [`Self::scan`] could match — a phrase on whole
    /// words, a plain single word exactly or as a near-match. Sorted by
    /// priority DESC; truncated to `limit`.
    pub fn top_relevant<'a>(&'a self, text: &str, limit: usize) -> Vec<&'a DictionaryEntry> {
        // Pre-split once into spans (not just whitespace tokens), same
        // tokenization as `scan`, so a glued-without-whitespace word is still
        // considered.
        let text_words: Vec<&str> = text.split_whitespace().collect();
        let spans: Vec<&str> = text
            .split_whitespace()
            .flat_map(|w| corrector::split_internal_punctuation(w).1)
            .map(|(span, _)| span)
            .collect();

        let mut relevant: Vec<&DictionaryEntry> = self
            .entries
            .iter()
            .filter(|e| {
                if corrector::is_plain_word(&e.phrase) {
                    return spans
                        .iter()
                        .any(|span| match_single_word(&[*e], span).is_some());
                }
                let phrase_words: Vec<&str> = e.phrase.split_whitespace().collect();
                !phrase_words.is_empty()
                    && text_words.windows(phrase_words.len()).any(|window| {
                        corrector::phrase_window_matches(window, &phrase_words).is_some()
                    })
            })
            .collect();

        // Already sorted priority DESC by the `entries` invariant; just truncate.
        relevant.truncate(limit);
        relevant
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// One dictionary match found by [`UserDictionary::scan`]. `fuzzy` is true
/// for single-word edit-distance-1 matches (not exact and not multi-word
/// phrase matches), which the corrector gates on low word confidence so a
/// garbled ASR onset isn't rewritten into a coincidentally-near dictionary term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictMatch {
    /// The text span that matched (as it appeared in the transcript).
    pub matched: String,
    /// The dictionary replacement to substitute.
    pub replacement: String,
    /// True only for fuzzy (edit-distance-1) single-word matches.
    pub fuzzy: bool,
}

/// Shortest word, in characters, on either side of a fuzzy match. One edit is
/// a third of a three-letter word: below this, "at" and "is" are both near
/// matches of an "ai" entry.
const MIN_FUZZY_CHARS: usize = 4;

/// The entry a single transcript word matches, and whether the match is fuzzy.
/// An exact (case-insensitive) entry always wins over a near-match, whatever
/// the priorities; among near-matches the first — highest-priority — wins.
fn match_single_word<'a>(
    entries: &[&'a DictionaryEntry],
    span: &str,
) -> Option<(&'a DictionaryEntry, bool)> {
    if let Some(entry) = entries.iter().find(|e| eq_ci(span, &e.phrase)) {
        return Some((entry, false));
    }
    let long_enough = |s: &str| s.chars().count() >= MIN_FUZZY_CHARS;
    if !long_enough(span) {
        return None;
    }
    entries
        .iter()
        .find(|e| long_enough(&e.phrase) && lev_le1_ci(span, &e.phrase))
        .map(|entry| (*entry, true))
}

/// Case-fold a word for use as a lookup key.
///
/// ASCII takes the allocation-cheap path; anything else folds with full
/// Unicode `to_lowercase`, so a dictionary entry stored as `École` still keys
/// the same as a transcript's `école`. `to_ascii_lowercase` alone leaves every
/// non-ASCII byte untouched, which silently defeats the case-insensitive
/// contract for accented entries.
#[inline]
pub(crate) fn fold_ci(s: &str) -> String {
    if s.is_ascii() {
        s.to_ascii_lowercase()
    } else {
        s.to_lowercase()
    }
}

/// Case-insensitive equality with the same folding rule as [`fold_ci`].
#[inline]
pub(crate) fn eq_ci(a: &str, b: &str) -> bool {
    if a.is_ascii() && b.is_ascii() {
        a.eq_ignore_ascii_case(b)
    } else {
        a.to_lowercase() == b.to_lowercase()
    }
}

/// Whether two dictionary phrases would be triggered by the same spoken words.
///
/// Correction matches case-insensitively on whitespace-delimited words, so
/// `"GitHub"` and `"github"` are one phrase wearing two spellings. Storing both
/// leaves which replacement wins down to row order — invisible to the user —
/// so the save path uses this to reject the second.
pub fn phrases_collide(a: &str, b: &str) -> bool {
    let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let a = normalize(a);
    !a.is_empty() && eq_ci(&a, &normalize(b))
}

/// O(n) Levenshtein distance ≤ 1 check, case-insensitive.
///
/// Replaces the `levenshtein(a.to_lowercase(), b.to_lowercase()) <= 1` pattern
/// in hot paths: no full DP table, early exit.
///
/// Pure-ASCII inputs — the overwhelming majority — take an allocation-free
/// byte path. Anything else is compared by *character*, since a byte-level
/// distance is wrong across multi-byte UTF-8: `café` is one character but two
/// bytes away from `cafe`, and accented names are exactly what a personal
/// dictionary is for.
#[inline]
pub(crate) fn lev_le1_ci(a: &str, b: &str) -> bool {
    if a.is_ascii() && b.is_ascii() {
        return lev_le1_by(a.as_bytes(), b.as_bytes(), |x, y| {
            x.eq_ignore_ascii_case(&y)
        });
    }
    let a: Vec<char> = a.chars().flat_map(char::to_lowercase).collect();
    let b: Vec<char> = b.chars().flat_map(char::to_lowercase).collect();
    lev_le1_by(&a, &b, |x, y| x == y)
}

/// Distance-≤-1 check over any comparable unit (bytes or chars).
fn lev_le1_by<T: Copy>(a: &[T], b: &[T], eq: impl Fn(T, T) -> bool) -> bool {
    // Orient so `short` ≤ `long_s`.
    let (short, long_s) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    match long_s.len() - short.len() {
        0 => {
            // Same length: at most 1 substitution allowed.
            let mut diffs = 0u32;
            for (&x, &y) in short.iter().zip(long_s.iter()) {
                if !eq(x, y) {
                    diffs += 1;
                    if diffs > 1 {
                        return false;
                    }
                }
            }
            true
        }
        1 => {
            // One insertion into shorter: find single divergence point.
            let mut si = 0;
            let mut li = 0;
            let mut skipped = false;
            while si < short.len() && li < long_s.len() {
                if eq(short[si], long_s[li]) {
                    si += 1;
                    li += 1;
                } else if skipped {
                    return false;
                } else {
                    skipped = true;
                    li += 1;
                }
            }
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(phrase: &str, replacement: &str, priority: u8) -> DictionaryEntry {
        DictionaryEntry {
            phrase: phrase.into(),
            replacement: replacement.into(),
            priority,
        }
    }

    /// `top_relevant` truncates in cache order, so the dictionary must own the
    /// priority ordering rather than trust whoever loaded the entries.
    #[test]
    fn a_user_dictionary_orders_entries_by_priority() {
        let dict = UserDictionary::new(vec![
            entry("rust", "Rust", 1),
            entry("github", "GitHub", 9),
            entry("tokio", "Tokio", 5),
        ]);
        let top = dict.top_relevant("rust github tokio", 2);
        let phrases: Vec<&str> = top.iter().map(|e| e.phrase.as_str()).collect();
        assert_eq!(phrases, ["github", "tokio"]);
    }

    /// A phrase that is whitespace but not empty passes the multi-word
    /// partition and tokenizes to zero words. A zero-width window matches
    /// vacuously and advances `i` by zero, so the scan never terminates —
    /// on *any* dictation, including an empty transcript.
    #[test]
    fn a_whitespace_only_phrase_does_not_loop_forever() {
        let m = UserDictionary::new(vec![entry("\t", "boom", 100)]);
        assert!(m.scan("hello there").is_empty());
        assert!(m.scan("").is_empty());
    }

    // ── lev_le1_ci ───────────────────────────────────────────────────────

    #[test]
    fn lev_le1_ci_identical() {
        assert!(lev_le1_ci("rust", "rust"));
    }

    #[test]
    fn lev_le1_ci_case_insensitive() {
        assert!(lev_le1_ci("Rust", "rust"));
        assert!(lev_le1_ci("GITHUB", "github"));
    }

    #[test]
    fn lev_le1_ci_one_sub() {
        assert!(lev_le1_ci("rast", "rust"));
    }

    #[test]
    fn lev_le1_ci_one_insert() {
        assert!(lev_le1_ci("rustt", "rust"));
    }

    #[test]
    fn lev_le1_ci_two_subs_rejected() {
        assert!(!lev_le1_ci("raxt", "rust"));
    }

    #[test]
    fn lev_le1_ci_length_diff_gt1_rejected() {
        assert!(!lev_le1_ci("abc", "abcde"));
    }

    #[test]
    fn lev_le1_ci_counts_characters_not_bytes() {
        // One character apart, two bytes apart: the byte-level check rejected
        // this, so an accented dictionary entry never fuzzy-matched.
        assert!(lev_le1_ci("cafe", "café"));
        assert!(lev_le1_ci("café", "cafés"));
        // Case folding is Unicode, not ASCII-only.
        assert!(lev_le1_ci("ÉCOLE", "école"));
        // Two characters apart is still rejected.
        assert!(!lev_le1_ci("café", "cofé!"));
    }

    // ── scan / top_relevant: internal-punctuation tokenization ─────────────

    #[test]
    fn scan_finds_fuzzy_match_glued_by_internal_punctuation() {
        let mgr = UserDictionary::new(vec![entry("dayton", "Dayton", 10)]);

        // Whisper sometimes glues a trailing-thought ellipsis directly onto
        // the next word with no whitespace: "is...Tayton." is ONE
        // split_whitespace token. The user dictionary must still find
        // "Tayton" as a near-match of the "dayton" entry, same as the
        // bundled vocab's fuzzy matching does.
        let matches = mgr.scan("hello my name is...Tayton.");
        assert!(
            matches
                .iter()
                .any(|m| m.matched.eq_ignore_ascii_case("tayton")
                    && m.replacement == "Dayton"
                    && m.fuzzy),
            "expected a fuzzy match for the glued word, got {matches:?}"
        );
    }

    #[test]
    fn top_relevant_finds_match_glued_by_internal_punctuation() {
        let mgr = UserDictionary::new(vec![entry("dayton", "Dayton", 10)]);

        let relevant = mgr.top_relevant("hello my name is...Tayton.", 10);
        assert!(relevant.iter().any(|e| e.phrase == "dayton"));
    }

    // ── scan: match selection ─────────────────────────────────────────────

    #[test]
    fn scan_prefers_an_exact_entry_over_a_higher_priority_near_match() {
        let mgr = UserDictionary::new(vec![entry("card", "Card9", 9), entry("cars", "CARS", 1)]);
        let matches = mgr.scan("cars");
        assert_eq!(
            matches,
            [DictMatch {
                matched: "cars".into(),
                replacement: "CARS".into(),
                fuzzy: false,
            }]
        );
    }

    #[test]
    fn scan_does_not_fuzzy_match_short_words() {
        let mgr = UserDictionary::new(vec![entry("ai", "AI", 5), entry("", "boom", 5)]);
        // "a", "at" and "is" are each one edit from "ai".
        assert!(mgr.scan("a cat is at home").is_empty());
        // The exact word still matches.
        let matches = mgr.scan("the ai works");
        assert_eq!(matches.len(), 1);
        assert!(!matches[0].fuzzy);
    }

    /// The scan and the corrector must agree on where a phrase sits, or a
    /// match the scan reports is never applied.
    #[test]
    fn multi_word_entry_applies_next_to_punctuation() {
        let mgr = UserDictionary::new(vec![entry("new york", "New York", 5)]);
        let gate = crate::corrector::LexicalGate::new();
        for (text, expected) in [
            ("i love new york.", "i love New York."),
            ("i love \"new york\" pizza", "i love \"New York\" pizza"),
        ] {
            let matches = mgr.scan(text);
            let none: &[(&str, f32, f32)] = &[];
            let result = gate.correct(text, none, &matches, "en");
            assert_eq!(result.corrected_text, expected);
        }
    }

    /// A phrase that carries punctuation is matched as written, against whole
    /// transcript words, and never against the bare word inside it.
    #[test]
    fn entries_with_punctuation_inside_apply() {
        let mgr = UserDictionary::new(vec![
            entry("node.js", "Node.js", 5),
            entry("c++", "C++", 5),
            entry(".net", ".NET", 5),
            entry("dr. smith", "Dr. Smith", 5),
        ]);
        let gate = crate::corrector::LexicalGate::new();
        let none: &[(&str, f32, f32)] = &[];
        for (text, expected) in [
            (
                "we write node.js, c++ and .net.",
                "we write Node.js, C++ and .NET.",
            ),
            ("go ask (dr. smith) first", "go ask (Dr. Smith) first"),
            // The bare words are different words.
            ("plan c of the net node", "plan c of the net node"),
        ] {
            let result = gate.correct(text, none, &mgr.scan(text), "en");
            assert_eq!(result.corrected_text, expected);
        }

        let relevant = mgr.top_relevant("we write node.js, c++ and .net.", 10);
        assert_eq!(relevant.len(), 3);
        assert!(mgr.top_relevant("plan c of the net node", 10).is_empty());
    }

    // ── top_relevant ──────────────────────────────────────────────────────

    #[test]
    fn top_relevant_matches_whole_words_only() {
        let mgr = UserDictionary::new(vec![entry("ai", "AI", 5), entry("new york", "New York", 5)]);
        // "ai" is inside "said" and "plain"; "new york" is inside "renew yorker".
        assert!(
            mgr.top_relevant("she said plain things to renew yorker", 10)
                .is_empty()
        );
        let relevant = mgr.top_relevant("the AI moved to New York.", 10);
        assert_eq!(relevant.len(), 2);
    }

    #[test]
    fn top_relevant_finds_match() {
        let mgr = UserDictionary::new(vec![
            entry("github", "GitHub", 5),
            entry("rust", "Rust", 10),
        ]);

        let relevant = mgr.top_relevant("I use github", 10);
        assert!(relevant.iter().any(|e| e.phrase == "github"));
    }

    #[test]
    fn top_relevant_respects_limit() {
        let mgr = UserDictionary::new(
            (0..15u8)
                .map(|i| entry(&format!("word{i}"), &format!("Word{i}"), i))
                .collect(),
        );
        let text: String = (0..15)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let relevant = mgr.top_relevant(&text, 10);
        assert_eq!(relevant.len(), 10);
    }
}
