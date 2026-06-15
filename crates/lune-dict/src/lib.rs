pub mod caprate;
pub mod corrector;
pub mod freq;
pub mod homophone;
pub mod homophone_calib;
pub mod lexicon;
pub mod ngram;
pub mod punctuation;
pub mod vocab;

#[cfg(feature = "dictionary")]
use lune_core::dict::DictionaryEntry;
#[cfg(feature = "dictionary")]
use lune_persist::{Database, PersistResult};

/// Manages the user dictionary: in-memory cache + DB persistence.
///
/// Applies corrections to ASR text (multi-word exact match, single-word fuzzy
/// with Levenshtein distance ≤ 1) and selects relevant entries for LLM prompts.
#[cfg(feature = "dictionary")]
pub struct DictionaryManager {
    db: Database,
    cache: Vec<DictionaryEntry>,
}

#[cfg(feature = "dictionary")]
impl DictionaryManager {
    /// Create a manager and load the initial cache from the database.
    pub fn new(mut db: Database) -> PersistResult<Self> {
        let cache = db.list_dictionary_entries()?;
        Ok(Self { db, cache })
    }

    /// Reload the in-memory cache from the database.
    pub fn reload(&mut self) -> PersistResult<()> {
        self.cache = self.db.list_dictionary_entries()?;
        Ok(())
    }

    /// Current cached entries (sorted by priority DESC, as returned by DB).
    pub fn entries(&self) -> &[DictionaryEntry] {
        &self.cache
    }

    // ── CRUD ─────────────────────────────────────────────────────────────

    pub fn insert(&mut self, phrase: &str, replacement: &str, priority: u8) -> PersistResult<i32> {
        let id = self
            .db
            .insert_dictionary_entry(phrase, replacement, priority)?;
        self.cache.push(DictionaryEntry {
            phrase: phrase.into(),
            replacement: replacement.into(),
            priority,
        });
        // Re-sort by priority DESC to maintain invariant.
        self.cache
            .sort_unstable_by(|a, b| b.priority.cmp(&a.priority));
        Ok(id)
    }

    pub fn delete(&mut self, db_id: i32) -> PersistResult<bool> {
        // We don't store DB IDs in the cache, so reload after deletion.
        let deleted = self.db.delete_dictionary_entry(db_id)?;
        if deleted {
            self.reload()?;
        }
        Ok(deleted)
    }

    // ── Text correction ──────────────────────────────────────────────────

    /// Apply dictionary corrections to `text`.
    ///
    /// Multi-word entries are matched word-boundary-exact (case-insensitive)
    /// and replaced first. Single-word entries use Levenshtein distance ≤ 1
    /// fuzzy matching per word.
    pub fn apply(&self, text: &str) -> String {
        // Single-pass partition; cheaper than count()-based split_whitespace.
        let (multi, single): (Vec<&DictionaryEntry>, Vec<&DictionaryEntry>) = self
            .cache
            .iter()
            .partition(|e| e.phrase.contains(char::is_whitespace));

        let mut current = text.to_owned();
        for entry in &multi {
            current = apply_multi_word(&current, entry);
        }
        apply_single_words(&current, &single)
    }

    pub fn scan(&self, text: &str) -> Vec<(String, String)> {
        let (multi, single): (Vec<&DictionaryEntry>, Vec<&DictionaryEntry>) = self
            .cache
            .iter()
            .partition(|e| e.phrase.contains(char::is_whitespace));

        let mut replacements = Vec::new();

        for entry in &multi {
            let phrase_words: Vec<&str> = entry.phrase.split_whitespace().collect();
            let n = phrase_words.len();
            let text_words: Vec<&str> = text.split_whitespace().collect();
            let mut i = 0;
            while i + n <= text_words.len() {
                let window_match = text_words[i..i + n]
                    .iter()
                    .zip(&phrase_words)
                    .all(|(tw, pw)| strip_trailing_punct(tw).eq_ignore_ascii_case(pw));
                if window_match {
                    let matched: String = text_words[i..i + n].join(" ");
                    replacements.push((matched, entry.replacement.clone()));
                    i += n;
                } else {
                    i += 1;
                }
            }
        }

        let text_words: Vec<&str> = text.split_whitespace().collect();
        for word in &text_words {
            let bare = strip_trailing_punct(word);
            if let Some(entry) = single
                .iter()
                .find(|e| lev_le1_ci(bare.as_bytes(), e.phrase.as_bytes()))
            {
                replacements.push((bare.to_owned(), entry.replacement.clone()));
            }
        }

        replacements
    }

    /// Return up to `limit` entries most relevant to `text` for LLM injection.
    ///
    /// Relevance: entries whose phrase matches (exact or fuzzy Levenshtein ≤ 1)
    /// a word in `text`. Sorted by priority DESC; truncated to `limit`.
    pub fn top_relevant<'a>(&'a self, text: &str, limit: usize) -> Vec<&'a DictionaryEntry> {
        // Keep original-case words; lev_le1_ci handles case-insensitivity below.
        let text_words: Vec<&str> = text.split_whitespace().collect();

        let mut relevant: Vec<&DictionaryEntry> = self
            .cache
            .iter()
            .filter(|e| {
                // No-alloc ASCII CI substring match.
                if contains_ascii_ci(text, &e.phrase) {
                    return true;
                }
                // Fuzzy single-word match (no to_lowercase allocation).
                if !e.phrase.contains(char::is_whitespace) {
                    return text_words
                        .iter()
                        .any(|w| lev_le1_ci(w.as_bytes(), e.phrase.as_bytes()));
                }
                false
            })
            .collect();

        // Already sorted priority DESC by cache invariant; just truncate.
        relevant.truncate(limit);
        relevant
    }

    // ── Import / export ──────────────────────────────────────────────────

    /// Import entries from plain text (one per line: `spoken → replacement`).
    ///
    /// Lines that don't contain ` → ` are silently skipped.
    /// Returns the number of entries inserted.
    pub fn import_text(&mut self, text: &str) -> PersistResult<usize> {
        let mut count = 0;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((phrase, replacement)) = line.split_once(" → ") {
                self.insert(phrase.trim(), replacement.trim(), 5)?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Export all entries as plain text (`spoken → replacement` per line).
    pub fn export_text(&self) -> String {
        let capacity = self
            .cache
            .iter()
            .map(|e| e.phrase.len() + e.replacement.len() + 5)
            .sum();
        let mut out = String::with_capacity(capacity);
        for (i, e) in self.cache.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&e.phrase);
            out.push_str(" → ");
            out.push_str(&e.replacement);
        }
        out
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[cfg(feature = "dictionary")]
fn apply_multi_word(text: &str, entry: &DictionaryEntry) -> String {
    // Borrow phrase words directly — eq_ignore_ascii_case avoids Vec<String>.
    let phrase_words: Vec<&str> = entry.phrase.split_whitespace().collect();
    let n = phrase_words.len();
    let text_words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::with_capacity(text_words.len());
    let mut i = 0;
    while i < text_words.len() {
        if i + n <= text_words.len() {
            let window_match = text_words[i..i + n]
                .iter()
                .zip(&phrase_words)
                .all(|(tw, pw)| strip_trailing_punct(tw).eq_ignore_ascii_case(pw));
            if window_match {
                // Preserve any trailing punctuation from the last word of the window.
                let last = text_words[i + n - 1];
                let punct = &last[strip_trailing_punct(last).len()..];
                let mut s = String::with_capacity(entry.replacement.len() + punct.len());
                s.push_str(&entry.replacement);
                s.push_str(punct);
                out.push(s);
                i += n;
                continue;
            }
        }
        out.push(text_words[i].to_owned());
        i += 1;
    }
    out.join(" ")
}

#[cfg(feature = "dictionary")]
fn apply_single_words(text: &str, singles: &[&DictionaryEntry]) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::with_capacity(words.len());
    for word in words {
        let bare = strip_trailing_punct(word);
        let punct = &word[bare.len()..];
        // lev_le1_ci: O(n) CI check with no allocation, replaces to_lowercase pair.
        let replaced = singles
            .iter()
            .find(|e| lev_le1_ci(bare.as_bytes(), e.phrase.as_bytes()));
        match replaced {
            Some(e) => {
                let mut s = String::with_capacity(e.replacement.len() + punct.len());
                s.push_str(&e.replacement);
                s.push_str(punct);
                out.push(s);
            }
            None => out.push(word.to_owned()),
        }
    }
    out.join(" ")
}

/// Strip trailing ASCII punctuation for comparison purposes.
#[cfg(feature = "dictionary")]
#[inline]
fn strip_trailing_punct(s: &str) -> &str {
    s.trim_end_matches(['.', ',', '!', '?', ';', ':'])
}

/// ASCII case-insensitive substring search with no heap allocation.
#[cfg(feature = "dictionary")]
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

/// O(n) Levenshtein distance ≤ 1 check (byte-level, ASCII, case-insensitive).
///
/// Replaces the `levenshtein(a.to_lowercase(), b.to_lowercase()) <= 1` pattern
/// in hot paths: no allocation, no full DP table, early exit.
#[cfg(feature = "dictionary")]
#[inline]
fn lev_le1_ci(a: &[u8], b: &[u8]) -> bool {
    // Orient so `short` ≤ `long_s`.
    let (short, long_s) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    match long_s.len() - short.len() {
        0 => {
            // Same length: at most 1 substitution allowed.
            let mut diffs = 0u32;
            for (&x, &y) in short.iter().zip(long_s.iter()) {
                if !x.eq_ignore_ascii_case(&y) {
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
                if short[si].eq_ignore_ascii_case(&long_s[li]) {
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

/// Wagner-Fischer Levenshtein distance (byte-level; fine for ASCII ASR text).
/// Only the dictionary path uses it, so it is gated with its sole caller.
#[cfg(feature = "dictionary")]
pub fn levenshtein(a: &str, b: &str) -> usize {
    // Operate on bytes — avoids char collection, valid for ASCII input.
    let a = a.as_bytes();
    let b = b.as_bytes();
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        let mut row_min = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            if curr[j] < row_min {
                row_min = curr[j];
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(all(test, feature = "dictionary"))]
mod tests {
    use super::*;
    use lune_persist::Database;

    fn manager() -> DictionaryManager {
        let db = Database::open_in_memory().unwrap();
        DictionaryManager::new(db).unwrap()
    }

    // ── levenshtein ───────────────────────────────────────────────────────

    #[test]
    fn levenshtein_identical() {
        assert_eq!(levenshtein("rust", "rust"), 0);
    }

    #[test]
    fn levenshtein_one_deletion() {
        assert_eq!(levenshtein("rustt", "rust"), 1);
    }

    #[test]
    fn levenshtein_one_substitution() {
        assert_eq!(levenshtein("rast", "rust"), 1);
    }

    #[test]
    fn levenshtein_two_edits() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    // ── lev_le1_ci ───────────────────────────────────────────────────────

    #[test]
    fn lev_le1_ci_identical() {
        assert!(lev_le1_ci(b"rust", b"rust"));
    }

    #[test]
    fn lev_le1_ci_case_insensitive() {
        assert!(lev_le1_ci(b"Rust", b"rust"));
        assert!(lev_le1_ci(b"GITHUB", b"github"));
    }

    #[test]
    fn lev_le1_ci_one_sub() {
        assert!(lev_le1_ci(b"rast", b"rust"));
    }

    #[test]
    fn lev_le1_ci_one_insert() {
        assert!(lev_le1_ci(b"rustt", b"rust"));
    }

    #[test]
    fn lev_le1_ci_two_subs_rejected() {
        assert!(!lev_le1_ci(b"raxt", b"rust"));
    }

    #[test]
    fn lev_le1_ci_length_diff_gt1_rejected() {
        assert!(!lev_le1_ci(b"abc", b"abcde"));
    }

    // ── apply ─────────────────────────────────────────────────────────────

    #[test]
    fn apply_exact_single_word() {
        let mut mgr = manager();
        mgr.insert("rust lang", "Rust", 10).unwrap(); // multi-word — won't trigger here
        mgr.insert("github", "GitHub", 5).unwrap();

        let result = mgr.apply("I use github every day");
        assert_eq!(result, "I use GitHub every day");
    }

    #[test]
    fn apply_fuzzy_single_word() {
        let mut mgr = manager();
        mgr.insert("kubernetes", "Kubernetes", 10).unwrap();

        // "kubirnites" has 2 substitutions (e→i at pos 3, e→i at pos 6) — should NOT match.
        let result = mgr.apply("I use kubirnites");
        assert_eq!(result, "I use kubirnites");

        // "kubernetss" is 1 substitution away — should match.
        let result2 = mgr.apply("I use kubernetss");
        assert_eq!(result2, "I use Kubernetes");
    }

    #[test]
    fn apply_multi_word_phrase() {
        let mut mgr = manager();
        mgr.insert("lune voice", "Lune Voice", 10).unwrap();

        let result = mgr.apply("I am using lune voice today");
        assert_eq!(result, "I am using Lune Voice today");
    }

    #[test]
    fn apply_preserves_trailing_punct() {
        let mut mgr = manager();
        mgr.insert("github", "GitHub", 5).unwrap();

        let result = mgr.apply("visit github.");
        assert_eq!(result, "visit GitHub.");
    }

    // ── top_relevant ──────────────────────────────────────────────────────

    #[test]
    fn top_relevant_finds_match() {
        let mut mgr = manager();
        mgr.insert("github", "GitHub", 5).unwrap();
        mgr.insert("rust", "Rust", 10).unwrap();

        let relevant = mgr.top_relevant("I use github", 10);
        assert!(relevant.iter().any(|e| e.phrase == "github"));
    }

    #[test]
    fn top_relevant_respects_limit() {
        let mut mgr = manager();
        for i in 0..15u8 {
            mgr.insert(&format!("word{i}"), &format!("Word{i}"), i)
                .unwrap();
        }
        let text: String = (0..15)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let relevant = mgr.top_relevant(&text, 10);
        assert_eq!(relevant.len(), 10);
    }

    // ── import / export ───────────────────────────────────────────────────

    #[test]
    fn import_export_roundtrip() {
        let mut mgr = manager();
        let text = "github → GitHub\nrust lang → Rust\n# comment\n\nbad line";
        let count = mgr.import_text(text).unwrap();
        assert_eq!(count, 2);

        let exported = mgr.export_text();
        assert!(exported.contains("github → GitHub") || exported.contains("rust lang → Rust"));
        assert!(!exported.contains("# comment"));
        assert!(!exported.contains("bad line"));
    }
}
