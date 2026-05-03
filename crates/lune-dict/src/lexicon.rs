use std::collections::HashMap;
use std::sync::LazyLock;

use crate::freq::{self, FreqDict};
use crate::vocab;

const VOCAB_FREQ_FLOOR: u32 = 10_000;
const MAX_VOCAB_WORD_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct WordInfo {
    pub canonical: Box<str>,
    pub freq: u32,
    pub freq_score: f32,
    pub source: WordSource,
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
        if let Some(freq) = self.dict.get(&lower) {
            return Some(WordInfo {
                canonical: lower.into_boxed_str(),
                freq,
                freq_score: freq_score(freq),
                source: WordSource::FreqMap,
            });
        }

        if let Some(canonical) = VOCAB_INDEX.canonical_by_lower.get(lower.as_str()) {
            return Some(WordInfo {
                canonical: canonical.clone(),
                freq: VOCAB_FREQ_FLOOR,
                freq_score: freq_score(VOCAB_FREQ_FLOOR),
                source: WordSource::Vocab,
            });
        }

        if self.is_english() && wordnik_list::word_exists(&lower) {
            return Some(WordInfo {
                canonical: lower.into_boxed_str(),
                freq: 0,
                freq_score: 0.0,
                source: WordSource::Wordnik,
            });
        }

        None
    }

    pub(crate) fn candidate_freq_score(&self, word: &str) -> f32 {
        let lower = word.to_lowercase();
        self.candidate_freq_score_for_lowercase(&lower)
    }

    pub(crate) fn candidate_freq_score_for_lowercase(&self, lower: &str) -> f32 {
        if let Some(freq) = self.dict.get(lower) {
            return freq_score(freq);
        }
        if VOCAB_INDEX.canonical_by_lower.contains_key(lower) {
            return freq_score(VOCAB_FREQ_FLOOR);
        }
        0.0
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

pub(crate) fn vocab_candidates_by_len(lo: usize, hi: usize) -> impl Iterator<Item = &'static str> {
    VOCAB_INDEX.candidates_by_len(lo, hi)
}

struct VocabIndex {
    canonical_by_lower: HashMap<Box<str>, Box<str>>,
    by_len: [Vec<Box<str>>; MAX_VOCAB_WORD_LEN],
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

        let mut by_len: [Vec<Box<str>>; MAX_VOCAB_WORD_LEN] = std::array::from_fn(|_| Vec::new());
        for canonical in canonical_by_lower.values() {
            let len = canonical.len().min(MAX_VOCAB_WORD_LEN - 1);
            by_len[len].push(canonical.clone());
        }
        for bucket in &mut by_len {
            bucket.sort_unstable();
        }

        Self {
            canonical_by_lower,
            by_len,
        }
    }

    fn candidates_by_len(
        &'static self,
        lo: usize,
        hi: usize,
    ) -> impl Iterator<Item = &'static str> {
        let start = lo.min(MAX_VOCAB_WORD_LEN);
        let end = hi.min(MAX_VOCAB_WORD_LEN);
        (start..end).flat_map(move |len| self.by_len[len].iter().map(Box::as_ref))
    }
}

static VOCAB_INDEX: LazyLock<VocabIndex> = LazyLock::new(VocabIndex::build);

pub fn warm() {
    freq::warm();
    let _ = vocab::bundled().word_count();
    let _ = VOCAB_INDEX.canonical_by_lower.len();
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
    fn vocab_candidates_are_bucketed_by_canonical_length() {
        let candidates: Vec<_> = vocab_candidates_by_len(9, 10).collect();

        assert!(candidates.contains(&"WebSocket"));
        assert!(!candidates.contains(&"websocket"));
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
}
