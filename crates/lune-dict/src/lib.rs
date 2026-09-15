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
/// Finds matches in ASR text (multi-word exact match, single-word fuzzy with
/// Levenshtein distance ≤ 1) and selects relevant entries for LLM prompts.
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
        let (multi, single): (Vec<&DictionaryEntry>, Vec<&DictionaryEntry>) = self
            .entries
            .iter()
            .partition(|e| e.phrase.contains(char::is_whitespace));

        let mut replacements = Vec::new();
        // `text` is not mutated across the loop, so split it once instead of
        // re-splitting per multi-word entry (was O(entries × text length)).
        let text_words: Vec<&str> = text.split_whitespace().collect();

        for entry in &multi {
            let phrase_words: Vec<&str> = entry.phrase.split_whitespace().collect();
            let n = phrase_words.len();
            // A phrase that is whitespace but not empty lands in `multi` (it
            // contains whitespace) yet tokenizes to nothing. A zero-width
            // window matches vacuously and `i += n` advances by zero, so the
            // loop below would never terminate.
            if n == 0 {
                continue;
            }
            let mut i = 0;
            while i + n <= text_words.len() {
                let window_match = text_words[i..i + n]
                    .iter()
                    .zip(&phrase_words)
                    .all(|(tw, pw)| eq_ci(strip_trailing_punct(tw), pw));
                if window_match {
                    let matched: String = text_words[i..i + n].join(" ");
                    // Multi-word phrase matches are word-boundary-exact, never fuzzy.
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
            let (_, spans) = crate::corrector::split_internal_punctuation(word);
            for (span, _) in &spans {
                if span.is_empty() {
                    continue;
                }
                if let Some(entry) = single.iter().find(|e| lev_le1_ci(span, &e.phrase)) {
                    // Exact (case-insensitive) match is trusted; an off-by-one
                    // match is fuzzy and will be gated on low confidence by
                    // the corrector.
                    let fuzzy = !eq_ci(span, entry.phrase.as_str());
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
    /// Relevance: entries whose phrase matches (exact or fuzzy Levenshtein ≤ 1)
    /// a word in `text`. Sorted by priority DESC; truncated to `limit`.
    pub fn top_relevant<'a>(&'a self, text: &str, limit: usize) -> Vec<&'a DictionaryEntry> {
        // Pre-split once into spans (not just whitespace tokens), same
        // tokenization as `scan`, so a glued-without-whitespace word is still
        // considered. Keep original case; lev_le1_ci handles CI below.
        let spans: Vec<&str> = text
            .split_whitespace()
            .flat_map(|w| crate::corrector::split_internal_punctuation(w).1)
            .map(|(span, _)| span)
            .collect();

        let mut relevant: Vec<&DictionaryEntry> = self
            .entries
            .iter()
            .filter(|e| {
                // No-alloc ASCII CI substring match.
                if contains_ascii_ci(text, &e.phrase) {
                    return true;
                }
                // Fuzzy single-word match (no to_lowercase allocation).
                if !e.phrase.contains(char::is_whitespace) {
                    return spans.iter().any(|w| lev_le1_ci(w, &e.phrase));
                }
                false
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

/// Strip trailing ASCII punctuation for comparison purposes.
#[inline]
fn strip_trailing_punct(s: &str) -> &str {
    s.trim_end_matches(['.', ',', '!', '?', ';', ':'])
}

/// ASCII case-insensitive substring search with no heap allocation.
#[inline]
fn contains_ascii_ci(haystack: &str, needle: &str) -> bool {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.len() > h.len() {
        return false;
    }
    if n.is_empty() {
        return true;
    }
    h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n))
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

    // ── top_relevant ──────────────────────────────────────────────────────

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
