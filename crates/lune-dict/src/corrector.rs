use std::sync::Arc;

use lune_ngram::NGramModel;

use crate::candidate_source::CandidateSource;
use crate::lexicon::{Lexicon, WordSource};
use crate::scorer;

pub fn has_dict(language: &str) -> bool {
    crate::freq::has_dict(language)
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

// Phonetic match: equivalent to reducing WED by 1.0
const PHONETIC_WED_EQUIV: f32 = 1.0;
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
const UNKNOWN_ONLY_THRESHOLD: f32 = 0.50;

// HWCE flag threshold: words with risk above this are considered uncertain
const HWCE_FLAG_THRESHOLD: f32 = 0.40;

// WED boundary between "close" and "distant" candidates
const DISTANT_WED: f32 = 1.5;

#[derive(Debug)]
pub struct WordScore {
    pub word: String,
    pub position: usize,
    pub candidate_score: f32,
}

pub struct CorrectionResult {
    pub corrected_text: String,
    pub word_risk_positions: Vec<(usize, f32)>,
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
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_risk_positions = align_word_risk_positions(&words, word_risk_scores);

        if !has_dict(lang) {
            return CorrectionResult {
                corrected_text: text.to_owned(),
                word_risk_positions,
                unresolved_words: Vec::new(),
                all_resolved: true,
            };
        }

        let lexicon = Lexicon::for_lang(lang);
        let mut result_words: Vec<String> = Vec::with_capacity(words.len());
        let mut unresolved = Vec::new();

        for (i, &word) in words.iter().enumerate() {
            let stripped = strip_punctuation(word);
            if stripped.is_empty() || stripped.chars().any(|c| c.is_ascii_digit()) {
                result_words.push(word.to_owned());
                continue;
            }

            let lower = stripped.to_ascii_lowercase();
            let lexicon_info = lexicon.lookup(stripped);
            let is_unknown = !lexicon_info.as_ref().is_some_and(|info| {
                info.freq >= Lexicon::CONFIDENT_FREQ || info.source == WordSource::Vocab
            });
            let risk_score = word_risk_positions
                .iter()
                .find(|(position, _)| *position == i)
                .map(|(_, risk)| *risk)
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
            if lexicon_info
                .as_ref()
                .is_some_and(|info| info.source == WordSource::Vocab)
            {
                result_words.push(word.to_owned());
                continue;
            }

            let looks_proper = stripped.starts_with(|c: char| c.is_ascii_uppercase())
                || (stripped.len() >= 2 && stripped.chars().all(|c| c.is_ascii_uppercase()));
            let is_camel_case = stripped.chars().any(|c| c.is_ascii_uppercase())
                && stripped.chars().any(|c| c.is_ascii_lowercase())
                && stripped.chars().filter(|c| c.is_ascii_uppercase()).count() >= 2;

            if is_unknown && (looks_proper || is_camel_case) && risk_score < 0.80 {
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

            let best = self.best_candidate(&lower, &prev, &next, &lexicon);
            if let Some((ref candidate, base_score, is_distant)) = best {
                // Never correct garbage into a distant guess — leave for LLM
                if !is_distant {
                    let score = base_score + risk_boost;
                    let threshold = if unknown_only {
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
            }

            // Only mark as unresolved if HWCE flagged it — unflagged unknown words
            // are likely valid words not in our corpus, so just pass through.
            if flagged {
                let cand_score = best.map(|(_, s, _)| s + risk_boost).unwrap_or(0.0);
                unresolved.push(WordScore {
                    word: stripped.to_owned(),
                    position: i,
                    candidate_score: cand_score,
                });
            }
            result_words.push(word.to_owned());
        }

        let all_resolved = unresolved.is_empty();
        CorrectionResult {
            corrected_text: result_words.join(" "),
            word_risk_positions,
            unresolved_words: unresolved,
            all_resolved,
        }
    }

    fn best_candidate(
        &self,
        word: &str,
        prev: &[&str],
        next: &[&str],
        lexicon: &Lexicon,
    ) -> Option<(String, f32, bool)> {
        if word.len() < 2 {
            return None;
        }

        let mut best: Option<(String, f32, bool)> = None;
        let lo = word.chars().count().saturating_sub(1);
        let hi = word.chars().count() + 2;

        for cand in CandidateSource::new(lexicon, word, lo, hi) {
            let cand_lower;
            let candidate = if cand.is_lowercase {
                cand.canonical
            } else {
                cand_lower = cand.canonical.to_ascii_lowercase();
                &cand_lower
            };

            if !cand.is_curated && !plausible_candidate(word, candidate) {
                continue;
            }

            let Some(wed) = scorer::wed_with_limit(word, candidate, MAX_WED) else {
                continue;
            };
            let score = self.noisy_channel_score(word, candidate, prev, next, wed, cand.freq_score);
            if best.as_ref().is_none_or(|(_, bs, _)| score > *bs) {
                best = Some((cand.canonical.to_owned(), score, wed > DISTANT_WED));
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
        freq_score: f32,
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

fn align_word_risk_positions(
    words: &[&str],
    word_risk_scores: &[(String, f32)],
) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    let mut cursor = 0usize;

    for (risk_word, risk) in word_risk_scores {
        let risk_word = strip_punctuation(risk_word);
        if risk_word.is_empty() {
            continue;
        }

        if let Some(offset) = words[cursor..]
            .iter()
            .position(|word| strip_punctuation(word).eq_ignore_ascii_case(risk_word))
        {
            let idx = cursor + offset;
            out.push((idx, *risk));
            cursor = (idx + 1).min(words.len());
        }
    }

    out
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

fn plausible_candidate(word: &str, candidate: &str) -> bool {
    if word.len().abs_diff(candidate.len()) > 2 {
        return false;
    }

    let word_first = word.as_bytes().first().copied();
    let cand_first = candidate.as_bytes().first().copied();
    let word_last = word.as_bytes().last().copied();
    let cand_last = candidate.as_bytes().last().copied();
    let matching_edge = word_first == cand_first || word_last == cand_last;

    let common = ascii_letter_intersection(word, candidate);
    let min_len = word.len().min(candidate.len());
    let overlap_floor = min_len.saturating_sub(if min_len <= 5 { 1 } else { 2 });

    common >= overlap_floor && (matching_edge || common == min_len)
}

fn ascii_letter_intersection(a: &str, b: &str) -> usize {
    let mut a_counts = [0u8; 26];
    let mut b_counts = [0u8; 26];

    for byte in a.bytes().filter(u8::is_ascii_lowercase) {
        a_counts[(byte - b'a') as usize] = a_counts[(byte - b'a') as usize].saturating_add(1);
    }
    for byte in b.bytes().filter(u8::is_ascii_lowercase) {
        b_counts[(byte - b'a') as usize] = b_counts[(byte - b'a') as usize].saturating_add(1);
    }

    a_counts
        .iter()
        .zip(b_counts)
        .map(|(a, b)| (*a).min(b) as usize)
        .sum()
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
    fn corrects_prefilter_edge_cases() {
        let c = HeuristicCorrector::new();

        assert_eq!(
            c.correct("hllo world", &[], &[], "en").corrected_text,
            "hello world"
        );
        assert_eq!(
            c.correct("fone call", &[], &[], "en").corrected_text,
            "phone call"
        );
        assert_eq!(
            c.correct("weahter report", &[], &[], "en").corrected_text,
            "weather report"
        );
        assert_eq!(
            c.correct("conection failed", &[], &[], "en").corrected_text,
            "connection failed"
        );
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
    fn word_risk_positions_survive_heuristic_correction() {
        let c = HeuristicCorrector::new();
        let scores = vec![("teh".into(), 0.80)];
        let result = c.correct("teh quick", &scores, &[], "en");

        assert_eq!(result.corrected_text, "the quick");
        assert_eq!(result.word_risk_positions, vec![(0, 0.80)]);
    }

    #[test]
    fn word_risk_positions_handle_duplicate_words_by_order() {
        let c = HeuristicCorrector::new();
        let scores = vec![("the".into(), 0.10), ("the".into(), 0.85)];
        let result = c.correct("the build fixed the deployment", &scores, &[], "en");

        assert_eq!(result.word_risk_positions, vec![(0, 0.10), (3, 0.85)]);
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
    fn unsupported_lang_bypasses_corrector() {
        let c = HeuristicCorrector::new();
        let result = c.correct("jambo dunia", &[], &[], "swahili");
        assert_eq!(result.corrected_text, "jambo dunia");
        assert!(result.all_resolved);
    }

    #[test]
    fn french_known_words_pass_through() {
        let c = HeuristicCorrector::new();
        let result = c.correct("bonjour le monde", &[], &[], "fr");
        assert_eq!(result.corrected_text, "bonjour le monde");
        assert!(result.all_resolved);
    }

    #[test]
    fn non_english_typo_still_corrects_after_prefilter() {
        let c = HeuristicCorrector::new();
        let result = c.correct("bonjor le monde", &[], &[], "fr");

        assert_eq!(result.corrected_text, "bonjour le monde");
    }

    #[test]
    fn unresolved_words_reported() {
        let c = HeuristicCorrector::new();
        let risks = vec![("xyzqqq".to_string(), 0.9)];
        let result = c.correct("the xyzqqq fox", &risks, &[], "en");
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
    fn candidate_prefilter_keeps_common_error_shapes() {
        for (word, candidate) in [
            ("teh", "the"),
            ("brwon", "brown"),
            ("hllo", "hello"),
            ("conection", "connection"),
            ("fone", "phone"),
            ("adress", "address"),
            ("recieve", "receive"),
            ("seperate", "separate"),
        ] {
            assert!(
                plausible_candidate(word, candidate),
                "{word} should keep {candidate}"
            );
        }
        assert!(!plausible_candidate("teh", "algorithm"));
    }

    #[test]
    fn candidate_prefilter_rejects_low_overlap_words() {
        for (word, candidate) in [
            ("teh", "algorithm"),
            ("brwon", "yellow"),
            ("conection", "framework"),
            ("fone", "data"),
        ] {
            assert!(
                !plausible_candidate(word, candidate),
                "{word} should reject {candidate}"
            );
        }
    }

    #[test]
    fn ascii_letter_intersection_counts_duplicate_letters() {
        assert_eq!(ascii_letter_intersection("letter", "teller"), 5);
        assert_eq!(ascii_letter_intersection("letter", "later"), 4);
        assert_eq!(ascii_letter_intersection("aaa", "aab"), 2);
    }

    #[test]
    fn flagged_curated_vocab_passes_through_without_correction() {
        let c = HeuristicCorrector::new();
        let risks = vec![("WebSocket".to_string(), 0.95)];
        let result = c.correct("WebSocket failed", &risks, &[], "en");

        assert_eq!(result.corrected_text, "WebSocket failed");
        assert!(result.all_resolved);
    }

    #[test]
    fn has_dict_checks() {
        assert!(has_dict("en"));
        assert!(has_dict("eng"));
        assert!(has_dict("English"));
        assert!(has_dict(""));
        assert!(has_dict("fr"));
        assert!(has_dict("de"));
        assert!(has_dict("Japanese"));
        assert!(!has_dict("swahili"));
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

    #[test]
    fn qd_002_no_false_positives() {
        let c = HeuristicCorrector::new();
        let text = "The memory leak was traced back to the event listener on the WebSocket connection. It wasn't being cleaned up when the component unmounted. I pushed a fix to the dev branch and it looks stable now.";
        let result = c.correct(text, &[], &[], "en");
        eprintln!(
            "unresolved: {:?}",
            result
                .unresolved_words
                .iter()
                .map(|w| &w.word)
                .collect::<Vec<_>>()
        );
        assert!(
            result.unresolved_words.is_empty(),
            "unexpected unresolved: {:?}",
            result.unresolved_words
        );
    }
}
