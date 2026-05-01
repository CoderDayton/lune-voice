use crate::scorer::{self, BigramModel, FrequencyLookup};

const HEURISTIC_ACCEPT_THRESHOLD: f32 = 0.55;
const EDIT2_ACCEPT_THRESHOLD: f32 = 0.70;
const MAX_CANDIDATES: usize = 128;

pub struct CorrectionResult {
    pub corrected_text: String,
    pub unresolved_words: Vec<String>,
    pub all_resolved: bool,
}

pub struct HeuristicCorrector {
    bigrams: BigramModel,
}

impl Default for HeuristicCorrector {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicCorrector {
    pub fn new() -> Self {
        Self {
            bigrams: BigramModel::new(),
        }
    }

    pub fn correct(
        &self,
        text: &str,
        word_fix_scores: &[(String, f32)],
        dict_replacements: &[(String, String)],
        lang: &str,
    ) -> CorrectionResult {
        let en = lang.is_empty() || lang.starts_with("en");
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result_words: Vec<String> = Vec::with_capacity(words.len());
        let mut unresolved = Vec::new();

        for (i, &word) in words.iter().enumerate() {
            let stripped = strip_punctuation(word);
            if stripped.is_empty() {
                result_words.push(word.to_owned());
                continue;
            }

            let lower = stripped.to_ascii_lowercase();
            let is_unknown = !FrequencyLookup::is_known(&lower);
            let fix_score = word_fix_scores
                .iter()
                .find(|(w, _)| w.eq_ignore_ascii_case(stripped))
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            let flagged_by_herc = fix_score >= 0.40;

            if let Some((_, repl)) = dict_replacements
                .iter()
                .find(|(phrase, _)| phrase.eq_ignore_ascii_case(stripped))
            {
                let with_punct = restore_punctuation(word, repl);
                result_words.push(with_punct);
                continue;
            }

            let is_unknown = en && is_unknown;
            if !is_unknown && !flagged_by_herc {
                result_words.push(word.to_owned());
                continue;
            }

            // Skip unknown-only words that look like proper nouns or acronyms
            if is_unknown && !flagged_by_herc {
                let first_upper = stripped.starts_with(|c: char| c.is_ascii_uppercase());
                let all_upper =
                    stripped.len() >= 2 && stripped.chars().all(|c| c.is_ascii_uppercase());
                if first_upper || all_upper {
                    unresolved.push(stripped.to_owned());
                    result_words.push(word.to_owned());
                    continue;
                }
            }

            let both_signals = is_unknown && flagged_by_herc;
            let unknown_only = is_unknown && !flagged_by_herc;
            let prev: Vec<&str> = words[i.saturating_sub(2)..i].to_vec();
            let next_end = (i + 3).min(words.len());
            let next: Vec<&str> = words[i + 1..next_end].to_vec();

            if let Some((candidate, score, is_edit2)) =
                self.best_candidate(&lower, &prev, &next, both_signals)
            {
                let threshold = if is_edit2 {
                    EDIT2_ACCEPT_THRESHOLD
                } else if unknown_only {
                    0.60
                } else {
                    HEURISTIC_ACCEPT_THRESHOLD
                };

                if score >= threshold {
                    let corrected = preserve_casing(stripped, &candidate);
                    let with_punct = restore_punctuation(word, &corrected);
                    result_words.push(with_punct);
                    continue;
                }
            }

            unresolved.push(stripped.to_owned());
            result_words.push(word.to_owned());
        }

        let all_resolved = unresolved.is_empty();
        CorrectionResult {
            corrected_text: result_words.join(" "),
            unresolved_words: unresolved,
            all_resolved,
        }
    }

    fn best_candidate(
        &self,
        word: &str,
        prev: &[&str],
        next: &[&str],
        both_signals: bool,
    ) -> Option<(String, f32, bool)> {
        let mut candidates: Vec<(String, f32, bool)> = Vec::new();

        // Edit-1 candidates
        self.generate_edit1(word, &mut candidates);

        // Edit-2 only for words >= 5 chars
        let has_edit2 = word.len() >= 5;
        if has_edit2 {
            self.generate_edit2(word, &mut candidates);
        }

        candidates.truncate(MAX_CANDIDATES);

        let mut best: Option<(String, f32, bool)> = None;
        for (cand, _raw_wed, is_edit2) in &candidates {
            let score = self.rank_candidate(word, cand, prev, next, both_signals);
            if best.as_ref().is_none_or(|(_, bs, _)| score > *bs) {
                best = Some((cand.clone(), score, *is_edit2));
            }
        }

        best
    }

    fn rank_candidate(
        &self,
        original: &str,
        candidate: &str,
        prev: &[&str],
        next: &[&str],
        both_signals: bool,
    ) -> f32 {
        let wed_dist = scorer::wed(original, candidate);
        let freq_score = FrequencyLookup::freq_score(candidate);
        let dist_score = 1.0 / (wed_dist + 0.5);

        let mut raw = 0.40 * dist_score + 0.60 * freq_score;

        let freq = FrequencyLookup::freq(candidate);
        if freq == 0 {
            raw -= 0.30;
        }
        if candidate.len() < original.len() {
            raw -= 0.05;
        }
        if is_subsequence(original, candidate) {
            raw += 0.15;
        }
        if scorer::phonetic_key(original) == scorer::phonetic_key(candidate) {
            raw += 0.10;
        }

        let context_score = self.bigrams.context_score_wide(candidate, prev, next);
        let mut final_score = 0.75 * raw + 0.25 * context_score;

        if both_signals {
            final_score += 0.10;
        }

        final_score
    }

    fn generate_edit1(&self, word: &str, out: &mut Vec<(String, f32, bool)>) {
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();

        // Deletions
        for i in 0..n {
            let mut s = String::with_capacity(n - 1);
            for (j, &c) in chars.iter().enumerate() {
                if j != i {
                    s.push(c);
                }
            }
            if FrequencyLookup::is_known(&s) {
                let d = scorer::wed(word, &s);
                out.push((s, d, false));
            }
        }

        // Transpositions
        for i in 0..n.saturating_sub(1) {
            let mut s: Vec<char> = chars.clone();
            s.swap(i, i + 1);
            let s: String = s.into_iter().collect();
            if FrequencyLookup::is_known(&s) {
                let d = scorer::wed(word, &s);
                out.push((s, d, false));
            }
        }

        // Substitutions (a-z only)
        for i in 0..n {
            for c in 'a'..='z' {
                if c == chars[i] {
                    continue;
                }
                let mut s: Vec<char> = chars.clone();
                s[i] = c;
                let s: String = s.into_iter().collect();
                if FrequencyLookup::is_known(&s) {
                    let d = scorer::wed(word, &s);
                    out.push((s, d, false));
                }
            }
        }

        // Insertions (a-z)
        for i in 0..=n {
            for c in 'a'..='z' {
                let mut s = String::with_capacity(n + 1);
                for (j, &ch) in chars.iter().enumerate() {
                    if j == i {
                        s.push(c);
                    }
                    s.push(ch);
                }
                if i == n {
                    s.push(c);
                }
                if FrequencyLookup::is_known(&s) {
                    let d = scorer::wed(word, &s);
                    out.push((s, d, false));
                }
            }
        }
    }

    fn generate_edit2(&self, word: &str, out: &mut Vec<(String, f32, bool)>) {
        let mut edit1_words: Vec<String> = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();

        // Generate all edit-1 strings (not just known words)
        for i in 0..n {
            let mut s = String::with_capacity(n - 1);
            for (j, &c) in chars.iter().enumerate() {
                if j != i {
                    s.push(c);
                }
            }
            edit1_words.push(s);
        }
        for i in 0..n.saturating_sub(1) {
            let mut s: Vec<char> = chars.clone();
            s.swap(i, i + 1);
            edit1_words.push(s.into_iter().collect());
        }
        for i in 0..n {
            for c in 'a'..='z' {
                if c == chars[i] {
                    continue;
                }
                let mut s: Vec<char> = chars.clone();
                s[i] = c;
                edit1_words.push(s.into_iter().collect());
            }
        }

        let remaining = MAX_CANDIDATES.saturating_sub(out.len());
        let mut count = 0;
        for e1 in &edit1_words {
            if count >= remaining {
                break;
            }
            let e1_chars: Vec<char> = e1.chars().collect();
            let en = e1_chars.len();

            // Only substitutions and transpositions for edit-2 (cheaper)
            for i in 0..en {
                if count >= remaining {
                    break;
                }
                for c in 'a'..='z' {
                    if c == e1_chars[i] {
                        continue;
                    }
                    let mut s: Vec<char> = e1_chars.clone();
                    s[i] = c;
                    let s: String = s.into_iter().collect();
                    if FrequencyLookup::is_known(&s) && s != word {
                        let d = scorer::wed(word, &s);
                        out.push((s, d, true));
                        count += 1;
                        if count >= remaining {
                            break;
                        }
                    }
                }
            }
        }
    }
}

fn strip_punctuation(word: &str) -> &str {
    let s = word.trim_start_matches(|c: char| c.is_ascii_punctuation());
    s.trim_end_matches(|c: char| c.is_ascii_punctuation())
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

fn preserve_casing(original: &str, candidate: &str) -> String {
    if original.chars().all(|c| c.is_ascii_uppercase()) {
        return candidate.to_ascii_uppercase();
    }
    let mut chars = original.chars();
    if chars.next().is_some_and(|c| c.is_ascii_uppercase()) && chars.all(|c| c.is_ascii_lowercase())
    {
        let mut s = candidate.to_owned();
        if let Some(first) = s.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        return s;
    }
    candidate.to_owned()
}

fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut h = haystack.chars();
    for nc in needle.chars() {
        loop {
            match h.next() {
                Some(hc) if hc == nc => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_common_misspellings() {
        let c = HeuristicCorrector::new();
        let result = c.correct("teh quick brwon fox", &[], &[], "en");
        assert_eq!(result.corrected_text, "the quick brown fox");
    }

    #[test]
    fn preserve_casing_test() {
        assert_eq!(preserve_casing("THE", "the"), "THE");
        assert_eq!(preserve_casing("The", "the"), "The");
        assert_eq!(preserve_casing("the", "the"), "the");
    }

    #[test]
    fn punctuation_preserved() {
        let c = HeuristicCorrector::new();
        let result = c.correct("teh, quick", &[], &[], "en");
        assert!(result.corrected_text.contains("the,"));
    }
}
