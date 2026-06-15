use std::collections::HashMap;
use std::sync::LazyLock;

use crate::freq::{self, FreqDict};
use crate::vocab;

const VOCAB_FREQ_FLOOR: u32 = 10_000;

#[derive(Clone, Debug, PartialEq)]
pub struct WordInfo {
    pub canonical: Box<str>,
    pub freq: u32,
    pub freq_score: f32,
    pub source: WordSource,
    /// P(capitalized | mid-sentence) for English words (from `caprate`); `None`
    /// for non-English or words absent from the cap-rate table.
    pub p_cap: Option<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WordSource {
    FreqMap,
    Vocab,
    Wordnik,
}

pub struct Lexicon {
    lang: Box<str>,
    dict: &'static FreqDict,
}

impl Lexicon {
    pub const CONFIDENT_FREQ: u32 = 5_000;

    pub fn for_lang(lang: &str) -> Self {
        Self {
            lang: lang.into(),
            dict: freq::dict_for(lang),
        }
    }

    #[inline]
    pub fn lang(&self) -> &str {
        &self.lang
    }

    #[inline]
    pub fn dict(&self) -> &'static FreqDict {
        self.dict
    }

    #[inline]
    pub fn is_english(&self) -> bool {
        freq::is_english(&self.lang)
    }

    pub fn lookup(&self, word: &str) -> Option<WordInfo> {
        let lower = word.to_lowercase();
        let p_cap = self
            .is_english()
            .then(|| crate::caprate::en_p_cap_lower(&lower))
            .flatten();
        if let Some(freq) = self.dict.get(&lower) {
            let canonical = VOCAB_INDEX
                .canonical_by_lower
                .get(lower.as_str())
                .cloned()
                .unwrap_or_else(|| lower.into_boxed_str());
            return Some(WordInfo {
                canonical,
                freq,
                freq_score: freq_score(freq),
                source: WordSource::FreqMap,
                p_cap,
            });
        }

        if let Some(canonical) = VOCAB_INDEX.canonical_by_lower.get(lower.as_str()) {
            return Some(WordInfo {
                canonical: canonical.clone(),
                freq: VOCAB_FREQ_FLOOR,
                freq_score: freq_score(VOCAB_FREQ_FLOOR),
                source: WordSource::Vocab,
                p_cap,
            });
        }

        if self.is_english() && wordnik_list::word_exists(&lower) {
            return Some(WordInfo {
                canonical: lower.into_boxed_str(),
                freq: 0,
                freq_score: 0.0,
                source: WordSource::Wordnik,
                p_cap,
            });
        }

        None
    }

    #[inline]
    pub fn is_known(&self, word: &str) -> bool {
        self.lookup(word).is_some()
    }

    pub fn is_confident(&self, word: &str) -> bool {
        self.lookup(word).is_some_and(|info| {
            info.freq >= Self::CONFIDENT_FREQ || info.source == WordSource::Vocab
        })
    }
}

pub fn freq_score(freq: u32) -> f32 {
    if freq == 0 {
        return 0.0;
    }
    ((freq as f32).ln() / 17.8).clamp(0.0, 1.0)
}

struct VocabIndex {
    canonical_by_lower: HashMap<Box<str>, Box<str>>,
}

impl VocabIndex {
    fn build() -> Self {
        let mut canonical_by_lower: HashMap<Box<str>, Box<str>> = HashMap::new();
        for word in vocab::bundled().words.keys().map(String::as_str) {
            let lower = word.to_ascii_lowercase();
            let canonical = word.into();
            match canonical_by_lower.get(lower.as_str()) {
                Some(existing) if existing.as_ref() != lower && word == lower => {}
                _ => {
                    canonical_by_lower.insert(lower.into_boxed_str(), canonical);
                }
            }
        }

        Self { canonical_by_lower }
    }
}

static VOCAB_INDEX: LazyLock<VocabIndex> = LazyLock::new(VocabIndex::build);

pub fn warm() {
    freq::warm();
    let _ = vocab::bundled().word_count();
    let _ = VOCAB_INDEX.canonical_by_lower.len();
}

/// Whether the curated vocab knows `lower` (an already-lowercased word) as a
/// brand whose canonical form carries an uppercase letter — "react" → "React",
/// "github" → "GitHub", "redux" → "Redux". Lets the deterministic decasing in
/// [`crate::formatting::decase_common_words`] spare brand names that happen to
/// be ordinary English words, which `p_cap` cannot distinguish on its own.
pub fn is_capitalized_brand(lower: &str) -> bool {
    VOCAB_INDEX
        .canonical_by_lower
        .get(lower)
        .is_some_and(|canonical| canonical.chars().any(|c| c.is_ascii_uppercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_tiers_are_case_insensitive() {
        let lexicon = Lexicon::for_lang("en");

        let the = lexicon.lookup("THE").unwrap();
        let algorithm = lexicon.lookup("algorithm").unwrap();
        assert_eq!(the.source, WordSource::FreqMap);
        assert!(the.freq > algorithm.freq);
        assert!(lexicon.is_known("hello"));
    }

    #[test]
    fn vocab_lookup_preserves_canonical_casing() {
        let lexicon = Lexicon::for_lang("en");
        let info = lexicon.lookup("websocket").unwrap();

        assert_eq!(info.source, WordSource::Vocab);
        assert_eq!(info.canonical.as_ref(), "WebSocket");
        assert!(lexicon.is_confident("websocket"));
    }

    #[test]
    fn wordnik_lookup_is_english_only() {
        let english = Lexicon::for_lang("en");
        let french = Lexicon::for_lang("fr");

        assert_eq!(
            english.lookup("zymurgy").unwrap().source,
            WordSource::Wordnik
        );
        assert!(french.lookup("zymurgy").is_none());
    }

    #[test]
    fn lookup_exposes_p_cap_for_english_only() {
        let english = Lexicon::for_lang("en");
        let make = english.lookup("make").unwrap();
        assert!(
            make.p_cap.is_some_and(|p| p < 0.5),
            "make p_cap={:?}",
            make.p_cap
        );
        let monday = english.lookup("monday").unwrap();
        assert!(
            monday.p_cap.is_some_and(|p| p > 0.75),
            "monday p_cap={:?}",
            monday.p_cap
        );
        // Non-English lexicons never carry a (English) p_cap.
        let french = Lexicon::for_lang("fr");
        if let Some(info) = french.lookup("le") {
            assert!(info.p_cap.is_none());
        }
    }
}
