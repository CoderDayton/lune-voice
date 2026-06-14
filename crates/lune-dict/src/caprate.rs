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

pub struct CapRateDict {
    pcap: HashMap<Box<str>, f32>,
}

impl CapRateDict {
    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                pcap: HashMap::new(),
            };
        }
        let (entries, _): (Vec<(String, f32)>, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())
                .expect("decode caprate bin");
        let mut pcap = HashMap::with_capacity(entries.len());
        for (word, p) in entries {
            pcap.insert(word.into_boxed_str(), p);
        }
        Self { pcap }
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

/// Lowercase, alphanumeric-only key — matches the merge's `norm_word`, so a
/// token like `"Make,"` or `"Sure."` resolves to its bare lexical form.
fn norm_key(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
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
}
