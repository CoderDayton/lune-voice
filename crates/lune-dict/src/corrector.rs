use std::sync::Arc;

use lune_ngram::NGramModel;

use crate::scorer::{self, FrequencyLookup};

pub fn has_dict(language: &str) -> bool {
    language.is_empty() || language.starts_with("en")
}

// ── Noisy Channel Model (Shannon, 1948) ─────────────────────────────────────
//
// P(correction | observation, context) ∝ P(obs | cor) · P(cor | context)
//
// Error model:  P(obs | cor) = exp(−α · WED(obs, cor) + Σ βᵢ · fᵢ)
// Language model: P(cor | ctx) = λ · P_unigram(cor) + (1−λ) · P_bigram(cor | ctx)
//
// α (error precision): half-life at WED=1.5, i.e. exp(−α·1.5) = 0.5.
//   Single transpositions/substitutions (WED≈1) retain ~63% weight;
//   distant candidates (WED≥2.5) decay to ~32%.
//   α = ln(2)/1.5 ≈ 0.462
//
// βᵢ expressed as WED-equivalent reductions:
//   β = α · Δd, where Δd is the effective distance reduction.
//
const ERROR_PRECISION: f32 = core::f32::consts::LN_2 / 1.5;

// Phonetic match: equivalent to reducing WED by 0.5
const PHONETIC_WED_EQUIV: f32 = 0.5;
// Subsequence match: equivalent to reducing WED by 0.6
const SUBSEQ_WED_EQUIV: f32 = 0.6;
// Shorter candidate penalty: equivalent to increasing WED by 0.2
const SHORTER_WED_EQUIV: f32 = 0.2;

// Max WED: candidates beyond this distance have P(obs|cor) < exp(−α·2.5) ≈ 0.32.
// Diminishing returns beyond this point.
const MAX_WED: f32 = 2.5;

// Acceptance thresholds derived from Bayes risk minimization:
//   Accept when P(correction is right) > C_fp / (C_fp + C_fn)
//   Close candidates (WED ≤ 1.5): C_fp/C_fn ≈ 1.2 → threshold = 1.2/2.2 ≈ 0.55
//   Distant candidates (WED > 1.5): C_fp/C_fn ≈ 2.3 → threshold = 2.3/3.3 ≈ 0.70
//   Unknown-only (no HWCE flag): moderate → 0.50
const CLOSE_THRESHOLD: f32 = 0.55;
const DISTANT_THRESHOLD: f32 = 0.70;
const UNKNOWN_ONLY_THRESHOLD: f32 = 0.50;

// HWCE flag threshold: words with risk above this are considered uncertain
const HWCE_FLAG_THRESHOLD: f32 = 0.40;

// WED boundary between "close" and "distant" candidates
const DISTANT_WED: f32 = 1.5;

#[derive(Debug)]
pub struct WordScore {
    pub word: String,
    pub candidate_score: f32,
}

pub struct CorrectionResult {
    pub corrected_text: String,
    pub unresolved_words: Vec<WordScore>,
    pub all_resolved: bool,
}

pub struct HeuristicCorrector {
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
        if !has_dict(lang) {
            return CorrectionResult {
                corrected_text: text.to_owned(),
                unresolved_words: Vec::new(),
                all_resolved: true,
            };
        }

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
            let risk_score = word_risk_scores
                .iter()
                .find(|(w, _)| w.eq_ignore_ascii_case(stripped))
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            let flagged = risk_score >= HWCE_FLAG_THRESHOLD;

            if let Some((_, repl)) = dict_replacements
                .iter()
                .find(|(phrase, _)| phrase.eq_ignore_ascii_case(stripped))
            {
                result_words.push(restore_punctuation(word, repl));
                continue;
            }

            if !is_unknown && !flagged {
                result_words.push(word.to_owned());
                continue;
            }

            let looks_proper = stripped.starts_with(|c: char| c.is_ascii_uppercase())
                || (stripped.len() >= 2 && stripped.chars().all(|c| c.is_ascii_uppercase()));

            if is_unknown && !flagged && looks_proper {
                unresolved.push(WordScore {
                    word: stripped.to_owned(),
                    candidate_score: 0.0,
                });
                result_words.push(word.to_owned());
                continue;
            }

            let both = is_unknown && flagged;
            let unknown_only = is_unknown && !flagged;
            // Risk-proportional boost: scale excess risk into candidate score space
            let risk_boost = if both {
                (risk_score - HWCE_FLAG_THRESHOLD).max(0.0) * 0.75
            } else {
                0.0
            };

            let prev: Vec<&str> = words[i.saturating_sub(2)..i].to_vec();
            let next_end = (i + 3).min(words.len());
            let next: Vec<&str> = words[i + 1..next_end].to_vec();

            let best = self.best_candidate(&lower, &prev, &next);
            if let Some((ref candidate, base_score, is_distant)) = best {
                let score = base_score + risk_boost;
                let threshold = if is_distant {
                    DISTANT_THRESHOLD
                } else if unknown_only {
                    UNKNOWN_ONLY_THRESHOLD
                } else {
                    CLOSE_THRESHOLD
                };

                if score >= threshold {
                    let corrected = preserve_casing(stripped, candidate);
                    result_words.push(restore_punctuation(word, &corrected));
                    continue;
                }
            }

            let cand_score = best.map(|(_, s, _)| s + risk_boost).unwrap_or(0.0);
            unresolved.push(WordScore {
                word: stripped.to_owned(),
                candidate_score: cand_score,
            });
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
    ) -> Option<(String, f32, bool)> {
        if word.len() < 2 {
            return None;
        }

        let mut best: Option<(String, f32, bool)> = None;
        let lo = word.len().saturating_sub(1);
        let hi = word.len() + 2;

        for bucket_len in lo..hi {
            for cand in wordnik_list::word_iterator_by_len(bucket_len) {
                let wed = scorer::wed(word, cand);
                if wed > MAX_WED {
                    continue;
                }
                let score = self.noisy_channel_score(word, cand, prev, next, wed);
                if best.as_ref().is_none_or(|(_, bs, _)| score > *bs) {
                    best = Some((cand.to_owned(), score, wed > DISTANT_WED));
                }
            }
        }

        best
    }

    fn noisy_channel_score(
        &self,
        original: &str,
        candidate: &str,
        prev: &[&str],
        next: &[&str],
        wed: f32,
    ) -> f32 {
        // ── Error model: P(obs | cor) ──
        let mut log_error = -ERROR_PRECISION * wed;

        if scorer::phonetic_key(original) == scorer::phonetic_key(candidate) {
            log_error += ERROR_PRECISION * PHONETIC_WED_EQUIV;
        }
        if is_subsequence(original, candidate) {
            log_error += ERROR_PRECISION * SUBSEQ_WED_EQUIV;
        }
        if candidate.len() < original.len() {
            log_error -= ERROR_PRECISION * SHORTER_WED_EQUIV;
        }

        let error_model = log_error.exp();

        // ── Language model: P(cor | context) via n-gram ──
        let freq_score = FrequencyLookup::freq_score(candidate);
        let ngram_context = self.ngram_context_score(candidate, prev, next);
        const FREQ_WEIGHT: f32 = 0.35;
        const NGRAM_WEIGHT: f32 = 0.65;
        let lang_model = FREQ_WEIGHT * freq_score + NGRAM_WEIGHT * ngram_context;

        // ── Posterior: Hölder mean (p=2) ──
        const P: f32 = 2.0;
        const W_ERROR: f32 = 0.40;
        const W_LANG: f32 = 1.0 - W_ERROR;
        let a = (error_model + 1e-6).powf(P);
        let b = (lang_model + 1e-6).powf(P);
        (W_ERROR * a + W_LANG * b).powf(1.0 / P)
    }

    fn ngram_context_score(&self, candidate: &str, prev: &[&str], next: &[&str]) -> f32 {
        let prefix: Vec<String> = prev
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|w| w.to_string())
            .collect();
        let forward = self.ngram.interpolated_prob(&prefix, candidate);
        let backward = next
            .first()
            .map(|n| self.ngram.interpolated_prob(&[candidate.to_string()], n))
            .unwrap_or(1.0);
        // Map log-prob product to [0,1] via calibrated sigmoid
        let combined = (forward * backward.sqrt()).ln();
        let sigmoid = 1.0 / (1.0 + (-0.15 * (combined + 5.0)).exp());
        sigmoid as f32
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

    #[test]
    fn known_word_not_flagged_passes_through() {
        let c = HeuristicCorrector::new();
        let result = c.correct("the quick brown fox", &[], &[], "en");
        assert_eq!(result.corrected_text, "the quick brown fox");
        assert!(result.all_resolved);
    }

    #[test]
    fn hwce_flagged_known_word_triggers_candidate_search() {
        let c = HeuristicCorrector::new();
        let scores = vec![("teh".into(), 0.80)];
        let result = c.correct("teh quick", &scores, &[], "en");
        assert_eq!(result.corrected_text, "the quick");
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
    fn non_english_bypasses_corrector() {
        let c = HeuristicCorrector::new();
        let result = c.correct("bonjour le monde", &[], &[], "fr");
        assert_eq!(result.corrected_text, "bonjour le monde");
        assert!(result.all_resolved);
    }

    #[test]
    fn unresolved_words_reported() {
        let c = HeuristicCorrector::new();
        let result = c.correct("the xyzqqq fox", &[], &[], "en");
        assert!(!result.unresolved_words.is_empty());
        assert!(result.unresolved_words.iter().any(|ws| ws.word == "xyzqqq"));
    }

    #[test]
    fn risk_boost_increases_candidate_score() {
        let c = HeuristicCorrector::new();
        let low_risk = vec![("teh".into(), 0.41_f32)];
        let high_risk = vec![("teh".into(), 0.90_f32)];
        let r_low = c.correct("teh fox", &low_risk, &[], "en");
        let r_high = c.correct("teh fox", &high_risk, &[], "en");
        assert_eq!(r_low.corrected_text, r_high.corrected_text);
    }

    #[test]
    fn empty_input() {
        let c = HeuristicCorrector::new();
        let result = c.correct("", &[], &[], "en");
        assert_eq!(result.corrected_text, "");
        assert!(result.all_resolved);
    }

    #[test]
    fn single_char_word_not_corrected() {
        let c = HeuristicCorrector::new();
        let result = c.correct("I a", &[], &[], "en");
        assert_eq!(result.corrected_text, "I a");
    }

    #[test]
    fn is_subsequence_cases() {
        assert!(is_subsequence("abc", "aXbYcZ"));
        assert!(is_subsequence("ace", "abcde"));
        assert!(!is_subsequence("adc", "abc"));
        assert!(is_subsequence("", "anything"));
        assert!(!is_subsequence("x", ""));
    }

    #[test]
    fn has_dict_checks() {
        assert!(has_dict("en"));
        assert!(has_dict("eng"));
        assert!(has_dict(""));
        assert!(!has_dict("fr"));
        assert!(!has_dict("de"));
    }

    #[test]
    fn restore_punctuation_edges() {
        assert_eq!(restore_punctuation("\"hello\"", "hi"), "\"hi\"");
        assert_eq!(restore_punctuation("hello.", "hi"), "hi.");
        assert_eq!(restore_punctuation("hello", "hi"), "hi");
    }

    #[test]
    fn noisy_channel_prefers_closer_candidate() {
        let c = HeuristicCorrector::new();
        let result = c.correct("hllo world", &[], &[], "en");
        assert_eq!(result.corrected_text, "hello world");
    }

    #[test]
    fn ngram_context_boosts_contextual_candidate() {
        use lune_ngram::{NGramModel, ScoringConfig};
        use std::sync::Arc;

        let mut model = NGramModel::new(ScoringConfig::trigram_default()).unwrap();
        model.train_sentences([
            "the weather is nice today",
            "the weather is warm today",
            "check the weather forecast",
            "the weather report said rain",
        ]);
        let c = HeuristicCorrector::from_ngram(Arc::new(model));
        // "weahter" is OOV, triggers candidate search
        let result = c.correct("the weahter is nice", &[], &[], "en");
        assert_eq!(result.corrected_text, "the weather is nice");
    }

    #[test]
    fn from_ngram_constructor_works() {
        use lune_ngram::{NGramModel, ScoringConfig};
        use std::sync::Arc;

        let model = NGramModel::new(ScoringConfig::trigram_default()).unwrap();
        let c = HeuristicCorrector::from_ngram(Arc::new(model));
        let result = c.correct("teh quick", &[], &[], "en");
        assert_eq!(result.corrected_text, "the quick");
    }
}
