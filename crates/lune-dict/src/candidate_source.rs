use std::collections::HashMap;

use crate::lexicon::Lexicon;

#[derive(Clone, Copy)]
pub struct CandidateEntry<'a> {
    pub canonical: &'a str,
    pub freq_score: f32,
    pub is_curated: bool,
    pub is_lowercase: bool,
}

pub struct CandidateSource<'a> {
    entries: Vec<CandidateEntry<'a>>,
    pos: usize,
}

impl<'a> CandidateSource<'a> {
    pub fn new(lexicon: &'a Lexicon, word: &str, lo: usize, hi: usize) -> Self {
        let lower = word.to_lowercase();
        let mut seen = HashMap::new();
        let mut entries = Vec::new();

        if lexicon.is_english() {
            for bucket_len in lo..hi {
                for candidate in wordnik_list::word_iterator_by_len(bucket_len) {
                    let freq_score = lexicon.candidate_freq_score_for_lowercase(candidate);
                    push_lowercase_candidate(
                        &mut entries,
                        &mut seen,
                        &lower,
                        candidate,
                        freq_score,
                    );
                }
            }
        } else {
            for candidate in lexicon.dict().candidates_iter(lo, hi) {
                let freq_score = lexicon.candidate_freq_score_for_lowercase(candidate);
                push_lowercase_candidate(&mut entries, &mut seen, &lower, candidate, freq_score);
            }
        }

        for candidate in crate::lexicon::vocab_candidates_by_len(lo, hi) {
            if !viable_vocab_candidate(&lower, candidate) {
                continue;
            }
            let freq_score = lexicon.candidate_freq_score(candidate);
            push_or_replace_candidate(&mut entries, &mut seen, &lower, candidate, freq_score);
        }

        Self { entries, pos: 0 }
    }
}

impl<'a> Iterator for CandidateSource<'a> {
    type Item = CandidateEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = *self.entries.get(self.pos)?;
        self.pos += 1;
        Some(entry)
    }
}

fn viable_vocab_candidate(original_lower: &str, candidate: &str) -> bool {
    let is_acronym = candidate.len() >= 2 && candidate.chars().all(|c| c.is_ascii_uppercase());
    if is_acronym && original_lower.chars().all(|c| c.is_ascii_lowercase()) {
        return false;
    }
    if original_lower.len() <= 3 && candidate.len() != original_lower.len() {
        return false;
    }
    true
}

fn push_lowercase_candidate<'a>(
    entries: &mut Vec<CandidateEntry<'a>>,
    seen: &mut HashMap<String, usize>,
    original_lower: &str,
    candidate: &'a str,
    freq_score: f32,
) {
    if candidate == original_lower || seen.contains_key(candidate) {
        return;
    }
    seen.insert(candidate.to_owned(), entries.len());
    entries.push(CandidateEntry {
        canonical: candidate,
        freq_score,
        is_curated: false,
        is_lowercase: true,
    });
}

fn push_or_replace_candidate<'a>(
    entries: &mut Vec<CandidateEntry<'a>>,
    seen: &mut HashMap<String, usize>,
    original_lower: &str,
    candidate: &'a str,
    freq_score: f32,
) {
    let candidate_lower = candidate.to_lowercase();
    let is_lowercase = candidate == candidate_lower;
    if candidate_lower == original_lower {
        return;
    }

    if let Some(&idx) = seen.get(&candidate_lower) {
        if entries[idx].canonical != candidate_lower && candidate == candidate_lower {
            return;
        }
        entries[idx] = CandidateEntry {
            canonical: candidate,
            freq_score,
            is_curated: true,
            is_lowercase,
        };
        return;
    }

    let idx = entries.len();
    seen.insert(candidate_lower, idx);
    entries.push(CandidateEntry {
        canonical: candidate,
        freq_score,
        is_curated: true,
        is_lowercase,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_candidates_include_wordnik_and_vocab_without_self() {
        let lexicon = Lexicon::for_lang("en");
        let candidates: Vec<_> = CandidateSource::new(&lexicon, "websoket", 7, 10).collect();

        assert!(
            candidates
                .iter()
                .any(|entry| entry.canonical == "WebSocket" && entry.is_curated)
        );
        assert!(
            !candidates
                .iter()
                .any(|entry| entry.canonical.eq_ignore_ascii_case("websoket"))
        );
    }

    #[test]
    fn non_english_candidates_include_freq_and_vocab() {
        let lexicon = Lexicon::for_lang("fr");
        let candidates: Vec<_> = CandidateSource::new(&lexicon, "bonjor", 5, 8).collect();

        assert!(candidates.iter().any(|entry| entry.canonical == "bonjour"));
        assert!(
            candidates
                .iter()
                .any(|entry| entry.canonical.eq_ignore_ascii_case("Roblox") && entry.is_curated)
        );
    }

    #[test]
    fn primary_candidates_are_marked_lowercase_not_curated() {
        let lexicon = Lexicon::for_lang("en");
        let candidates: Vec<_> = CandidateSource::new(&lexicon, "teh", 2, 4).collect();
        let the = candidates
            .iter()
            .find(|entry| entry.canonical == "the")
            .expect("wordnik candidate");

        assert!(!the.is_curated);
        assert!(the.is_lowercase);
    }
}
