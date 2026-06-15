//! Per-word capitalization rate: `P(capitalized | the word is not at a
//! sentence start)`, built from cased corpora by `scripts/sim/cap_rate.py` +
//! the `build_caprate` bin.
//!
//! The merge uses it to tell genuine proper nouns (`john`, p_cap≈1.0) from
//! common words a tiny correction model wrongly capitalizes mid-sentence
//! (`make`, p_cap≈0.04) — a distinction frequency cannot make
//! (`let's`=144K < `john`=240K in the case-folded freq table). ROC over a gold
//! set put perfect separation at τ=0.75 (proper nouns ≥0.96, common ≤0.71).

use std::collections::HashMap;
use std::sync::LazyLock;

/// Threshold separating proper nouns (keep the capital) from common words
/// (revert it). From the cap-rate simulation's ROC: Youden J = 1.0 at 0.75,
/// sitting in a wide gap (common ≤0.71, proper ≥0.96).
pub const CAP_KEEP_THRESHOLD: f32 = 0.75;

/// Stricter threshold for *removing* a capital (decasing mid-sentence).
/// Reverting a capital is riskier than keeping one — many proper nouns are
/// spelled like common words ("Code", "Voice", "Apple") — so only decase a word
/// that is capitalized less than half the time mid-sentence. Discourse over-caps
/// ("make" 0.04, "sure" 0.04, "so" 0.12) sit far below this; proper-noun-capable
/// words ("firefox" 0.73, "okay" 0.54) sit at or above it and are spared.
pub const CAP_DECASE_THRESHOLD: f32 = 0.5;

/// On-disk format version for `en_caprate.bin`. The payload is
/// `(CAPRATE_FORMAT_VERSION, Vec<(word, p_cap)>)`; a mismatch (a future bincode
/// or schema change) is rejected into the empty-table fallback instead of
/// silently misdecoding. Bump when the payload shape changes.
pub const CAPRATE_FORMAT_VERSION: u16 = 1;

pub struct CapRateDict {
    pcap: HashMap<Box<str>, f32>,
}

impl CapRateDict {
    fn from_bytes(bytes: &[u8]) -> Self {
        let pcap = Self::decode(bytes).unwrap_or_else(|| {
            // A malformed or version-mismatched blob disables cap-rate casing
            // rather than crashing the daemon: every lookup then returns `None`,
            // the conservative "leave the casing as-is" default.
            tracing::error!(
                len = bytes.len(),
                "en_caprate.bin failed to decode; cap-rate casing disabled"
            );
            HashMap::new()
        });
        Self { pcap }
    }

    /// Decode the versioned `(u16, Vec<(word, p_cap)>)` payload. `None` on a
    /// version mismatch or a malformed payload (so `from_bytes` can fall back);
    /// an empty blob decodes to an empty table.
    fn decode(bytes: &[u8]) -> Option<HashMap<Box<str>, f32>> {
        if bytes.is_empty() {
            return Some(HashMap::new());
        }
        let ((version, entries), _): ((u16, Vec<(String, f32)>), _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard()).ok()?;
        if version != CAPRATE_FORMAT_VERSION {
            return None;
        }
        let mut pcap = HashMap::with_capacity(entries.len());
        for (word, p) in entries {
            pcap.insert(word.into_boxed_str(), p);
        }
        Some(pcap)
    }

    #[inline]
    pub fn get(&self, word: &str) -> Option<f32> {
        self.pcap.get(word).copied()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pcap.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pcap.is_empty()
    }
}

static EN_CAPRATE: LazyLock<CapRateDict> =
    LazyLock::new(|| CapRateDict::from_bytes(include_bytes!("../data/en_caprate.bin")));

/// Lexical key characters: alphanumerics plus *internal* apostrophes and
/// hyphens, so `"Make,"` reduces to `"make"` while contractions and compounds
/// keep the punctuation the cap-rate table is keyed on (`"I'm"` -> `"i'm"`,
/// `"well-known"` -> `"well-known"`). A curly apostrophe folds to a straight one
/// because the corpus keys use straight quotes. Stripping the apostrophe here
/// was the bug that mis-resolved `"I'm"` to `"im"` (p_cap 0.64) and lowercased
/// the pronoun mid-sentence. Case is *not* folded here — callers do that.
fn key_chars(word: &str) -> String {
    word.chars()
        .filter_map(|c| match c {
            '\u{2019}' => Some('\''),
            c if c.is_alphanumeric() || c == '\'' || c == '-' => Some(c),
            _ => None,
        })
        .collect()
}

/// Lowercase lexical key for an arbitrary-cased token.
fn norm_key(word: &str) -> String {
    key_chars(word).to_lowercase()
}

/// `P(capitalized | mid-sentence)` for an English `word` (any casing /
/// surrounding punctuation). `None` when the word is unknown — callers treat
/// unknown as low (revert a spurious capital), the conservative default.
pub fn en_p_cap(word: &str) -> Option<f32> {
    let key = norm_key(word);
    if key.is_empty() {
        return None;
    }
    EN_CAPRATE.get(&key)
}

/// `en_p_cap` for input the caller has already lowercased (e.g. the lexicon's
/// `word.to_lowercase()`), skipping a redundant second lowercasing pass. The
/// key is identical because lowercasing already-lowercase text is the identity.
pub fn en_p_cap_lower(lower: &str) -> Option<f32> {
    let key = key_chars(lower);
    if key.is_empty() {
        return None;
    }
    EN_CAPRATE.get(&key)
}

/// True when `word` is capitalized often enough mid-sentence that an
/// LLM-introduced (or ASR-introduced) capital should be trusted as a proper
/// noun. Unknown words return `false` (revert), sparing the ~29k known proper
/// nouns while keeping the prior conservative default for everything else.
pub fn en_is_proper_cap(word: &str) -> bool {
    en_p_cap(word).is_some_and(|p| p >= CAP_KEEP_THRESHOLD)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proper_nouns_score_high() {
        for w in ["john", "John", "paris", "Paris.", "monday", "England"] {
            let p = en_p_cap(w).unwrap_or(0.0);
            assert!(p >= CAP_KEEP_THRESHOLD, "{w} p_cap={p} should be >= keep");
            assert!(en_is_proper_cap(w), "{w} should be proper");
        }
    }

    #[test]
    fn common_and_discourse_words_score_low() {
        for w in [
            "make", "Make", "sure", "Sure.", "so", "that", "really", "just",
        ] {
            let p = en_p_cap(w).unwrap_or(0.0);
            assert!(p < CAP_KEEP_THRESHOLD, "{w} p_cap={p} should be < keep");
            assert!(!en_is_proper_cap(w), "{w} should NOT be proper");
        }
    }

    #[test]
    fn unknown_word_is_not_proper() {
        assert_eq!(en_p_cap("zzqwxnonsense"), None);
        assert!(!en_is_proper_cap("zzqwxnonsense"));
    }

    #[test]
    fn table_loaded() {
        assert!(
            EN_CAPRATE.len() > 10_000,
            "caprate table should be populated"
        );
    }

    #[test]
    fn pronoun_contractions_keep_their_capital() {
        // Regression: norm_key used to strip the apostrophe, collapsing "I'm"
        // onto the unrelated key "im" (p_cap 0.64) and lowercasing the pronoun
        // mid-sentence. The contraction keys themselves are ~always capitalized.
        for w in ["I'm", "I'll", "I'd", "I've"] {
            let p = en_p_cap(w).unwrap_or(0.0);
            assert!(p >= CAP_KEEP_THRESHOLD, "{w} p_cap={p} should be >= keep");
            assert!(en_is_proper_cap(w), "{w} should stay capitalized");
        }
    }

    #[test]
    fn lowercase_contractions_resolve_as_common() {
        // Real keys (apostrophe preserved) that score low — a mid-sentence
        // capital on them should revert, not be treated as an unknown word.
        for w in ["don't", "it's", "can't", "didn't"] {
            let p = en_p_cap(w).expect("contraction should be a known key");
            assert!(p < CAP_KEEP_THRESHOLD, "{w} p_cap={p} should be < keep");
        }
    }

    #[test]
    fn hyphenated_compound_is_known() {
        // Previously unreachable: "well-known" collapsed to "wellknown" (absent).
        assert!(
            en_p_cap("well-known").is_some(),
            "hyphenated key must resolve"
        );
    }

    #[test]
    fn en_p_cap_lower_matches_en_p_cap_for_lowercase() {
        for w in ["make", "i'm", "well-known", "amazon", "zzznope"] {
            assert_eq!(en_p_cap_lower(w), en_p_cap(w), "mismatch for {w}");
        }
    }

    #[test]
    fn malformed_blob_falls_back_to_empty() {
        // A corrupt or version-mismatched bin disables casing, never panics.
        assert!(CapRateDict::from_bytes(b"\xde\xad\xbe\xef not bincode").is_empty());
        assert!(CapRateDict::from_bytes(&[]).is_empty());
    }
}
