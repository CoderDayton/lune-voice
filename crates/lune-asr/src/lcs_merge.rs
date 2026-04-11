//! LCS-based overlap stitching for chunked ASR transcripts.
//!
//! Shared between OrtQwen and Parakeet backends for merging overlapping
//! audio chunk transcriptions at word boundaries.
//!
//! Operates on [`WordConfidence`] pairs rather than bare strings so that
//! per-word confidence survives the merge. This matters for the downstream
//! LLM correction gate: with a scalar-average fallback, a >25s utterance
//! collapses the gate to a single Bernoulli decision for the whole segment
//! (either every word is marked `[?word?]` or none are). Threading per-word
//! confidence through the merge preserves ~log₂(N) bits of signal the gate
//! needs to be selective.

use lune_core::asr::WordConfidence;

/// Merge two word sequences from overlapping audio chunks using LCS with
/// position-biased scoring.
///
/// The bias prefers matches near the *overlap corner* — end of `a`'s tail
/// and start of `b`'s head — where the true overlap should land by
/// construction of the chunker. Matches far from that corner are weighted
/// lower, so when multiple equal-length LCS paths exist, the path that
/// runs along the true audio overlap wins the backtrack.
///
/// Scoring is `(lcs_length, position_bonus)` compared lexicographically.
/// `position_bonus` is strictly less than `lcs_length`, so the bias
/// *never* changes the LCS length; it only disambiguates ties.
///
/// Requires LCS ≥ 2 words to avoid false merges on coincidental matches.
pub(crate) fn lcs_merge_words(a: &[WordConfidence], b: &[WordConfidence]) -> Vec<WordConfidence> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    // Search window: overlap region ≈ 3-8 words depending on speech rate.
    // For Parakeet: 1s overlap × ~3 words/s ≈ 3 words; pad to 8 for safety.
    const OVERLAP_WINDOW: usize = 8;
    let tail_start = a.len().saturating_sub(OVERLAP_WINDOW);
    let tail_a = &a[tail_start..];
    let head_len = OVERLAP_WINDOW.min(b.len());
    let head_b = &b[..head_len];

    // Build LCS table with lexicographic score `(length, position_bonus)`.
    // `position_bonus` is the sum of per-match corner-proximity weights and
    // is bounded in `[0, lcs_length)`, so ordering by length first preserves
    // the standard LCS guarantee while using position as a tiebreaker.
    let (m, n) = (tail_a.len(), head_b.len());
    let mut dp = vec![vec![Score::zero(); n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            let diag = dp[i - 1][j - 1];
            let up = dp[i - 1][j];
            let left = dp[i][j - 1];

            if tail_a[i - 1].word.eq_ignore_ascii_case(&head_b[j - 1].word) {
                let bonus = corner_bonus(i - 1, j - 1, m, n);
                let match_score = Score {
                    length: diag.length + 1,
                    bonus: diag.bonus + bonus,
                };
                dp[i][j] = max_score(match_score, max_score(up, left));
            } else {
                dp[i][j] = max_score(up, left);
            }
        }
    }

    let lcs_len = dp[m][n].length as usize;
    if lcs_len < 2 {
        // LCS too short — likely coincidental match, not real overlap.
        // Concatenate without merging so no words (or confidences) are lost.
        let mut result = a.to_vec();
        result.extend_from_slice(b);
        return result;
    }

    // Backtrack: take a match iff the match-derived score is exactly the
    // cell's stored score. Otherwise follow the lexicographically-larger
    // neighbour. Ties resolve toward `up` (consistent with forward pass).
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        let here = dp[i][j];
        let match_possible = tail_a[i - 1].word.eq_ignore_ascii_case(&head_b[j - 1].word);
        if match_possible {
            let diag = dp[i - 1][j - 1];
            let bonus = corner_bonus(i - 1, j - 1, m, n);
            let match_score = Score {
                length: diag.length + 1,
                bonus: diag.bonus + bonus,
            };
            if score_eq(here, match_score) {
                i -= 1;
                j -= 1;
                continue;
            }
        }
        let up = dp[i - 1][j];
        let left = dp[i][j - 1];
        if score_cmp(up, left).is_ge() {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    // i = index in tail_a where LCS starts (0-based)
    // j = index in head_b where LCS starts (0-based)
    let split_a = tail_start + i;
    let mut result = Vec::with_capacity(split_a + (b.len() - j));
    result.extend_from_slice(&a[..split_a]);
    result.extend_from_slice(&b[j..]);
    result
}

// ── Position-biased scoring ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Score {
    length: u16,
    bonus: f32,
}

impl Score {
    const fn zero() -> Self {
        Self {
            length: 0,
            bonus: 0.0,
        }
    }
}

/// Lexicographic comparison: primary = length, secondary = bonus.
#[inline]
fn score_cmp(a: Score, b: Score) -> std::cmp::Ordering {
    a.length.cmp(&b.length).then(
        a.bonus
            .partial_cmp(&b.bonus)
            .unwrap_or(std::cmp::Ordering::Equal),
    )
}

#[inline]
fn score_eq(a: Score, b: Score) -> bool {
    a.length == b.length && (a.bonus - b.bonus).abs() < 1e-6
}

#[inline]
fn max_score(a: Score, b: Score) -> Score {
    if score_cmp(a, b).is_ge() { a } else { b }
}

/// Corner-proximity bonus for a match at `(i, j)` within the window
/// `(m, n)`. Returns a value in `[0, 1)` — strictly less than 1 so the
/// bonus is always a tiebreaker, never large enough to flip an LCS-length
/// comparison.
///
/// Peaks at `(m - 1, 0)` = end of `a`'s tail meeting start of `b`'s head,
/// which is exactly where the true audio overlap should land.
#[inline]
fn corner_bonus(i: usize, j: usize, m: usize, n: usize) -> f32 {
    debug_assert!(i < m && j < n);
    const MAX_BONUS: f32 = 0.9; // strictly < 1 so ties break cleanly
    let tail_end_ness = (i + 1) as f32 / m as f32;
    let head_start_ness = (n - j) as f32 / n as f32;
    MAX_BONUS * tail_end_ness * head_start_ness
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand for building a `WordConfidence` in tests.
    fn w(word: &str, conf: f32) -> WordConfidence {
        WordConfidence::new(word, conf)
    }

    fn seq(words: &[(&str, f32)]) -> Vec<WordConfidence> {
        words.iter().map(|(s, c)| w(s, *c)).collect()
    }

    fn words_of(xs: &[WordConfidence]) -> Vec<&str> {
        xs.iter().map(|x| x.word.as_str()).collect()
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(
            words_of(&lcs_merge_words(&[], &seq(&[("hello", 0.9)]))),
            vec!["hello"]
        );
        assert_eq!(
            words_of(&lcs_merge_words(&seq(&[("hello", 0.9)]), &[])),
            vec!["hello"]
        );
        assert!(lcs_merge_words(&[], &[]).is_empty());
    }

    #[test]
    fn no_overlap_concat() {
        let a = seq(&[("hello", 0.9), ("world", 0.8)]);
        let b = seq(&[("foo", 0.7), ("bar", 0.6)]);
        // LCS < 2 → concatenate, all confidences preserved.
        let merged = lcs_merge_words(&a, &b);
        assert_eq!(words_of(&merged), vec!["hello", "world", "foo", "bar"]);
        let confs: Vec<f32> = merged.iter().map(|w| w.confidence).collect();
        assert_eq!(confs, vec![0.9, 0.8, 0.7, 0.6]);
    }

    #[test]
    fn clean_overlap() {
        // a ends with "brown fox", b starts with "brown fox".
        // Merged result uses b's confidences in the overlap region.
        let a = seq(&[("the", 0.9), ("quick", 0.85), ("brown", 0.5), ("fox", 0.4)]);
        let b = seq(&[
            ("brown", 0.95),
            ("fox", 0.92),
            ("jumps", 0.8),
            ("over", 0.75),
        ]);
        let merged = lcs_merge_words(&a, &b);
        assert_eq!(
            words_of(&merged),
            vec!["the", "quick", "brown", "fox", "jumps", "over"]
        );
        // brown/fox confidences should come from b (0.95, 0.92), not a (0.5, 0.4).
        let confs: Vec<f32> = merged.iter().map(|w| w.confidence).collect();
        assert_eq!(confs, vec![0.9, 0.85, 0.95, 0.92, 0.8, 0.75]);
    }

    #[test]
    fn full_overlap() {
        let a = seq(&[("hello", 0.9), ("world", 0.8)]);
        let b = seq(&[("hello", 0.7), ("world", 0.6), ("again", 0.5)]);
        let merged = lcs_merge_words(&a, &b);
        assert_eq!(words_of(&merged), vec!["hello", "world", "again"]);
        // Overlap words take confidences from b.
        let confs: Vec<f32> = merged.iter().map(|w| w.confidence).collect();
        assert_eq!(confs, vec![0.7, 0.6, 0.5]);
    }

    #[test]
    fn case_insensitive() {
        let a = seq(&[("The", 0.9), ("Quick", 0.8)]);
        let b = seq(&[("the", 0.85), ("quick", 0.75), ("fox", 0.7)]);
        let merged = lcs_merge_words(&a, &b);
        assert_eq!(words_of(&merged), vec!["the", "quick", "fox"]);
    }

    #[test]
    fn overlap_corner_match_preferred_over_coincidental() {
        // Both tails contain "the" and "and" — coincidental duplicates away
        // from the overlap boundary. The true overlap is at the corner:
        // a ends with "red cat", b starts with "red cat". The unbiased LCS
        // could latch onto the early "the" in a and the late "the" in b,
        // producing a spurious split. The corner-biased LCS must choose
        // "red cat" because its position bonus dominates the tiebreaker.
        //
        // Windows after truncation to OVERLAP_WINDOW = 8:
        //   tail_a = ["the", "black", "dog", "saw", "the", "and", "red", "cat"]
        //   head_b = ["red", "cat", "the", "and", "that", "was", "the", "end"]
        // The unique-position LCS has length 4 (the, and, red, cat) but any
        // choice that crosses the corner properly should at least place the
        // split point so "red cat" is preserved at the junction.
        let a = seq(&[
            ("I", 0.9),
            ("saw", 0.9),
            ("the", 0.9),
            ("black", 0.9),
            ("dog", 0.9),
            ("saw", 0.9),
            ("the", 0.9),
            ("and", 0.9),
            ("red", 0.9),
            ("cat", 0.9),
        ]);
        let b = seq(&[
            ("red", 0.8),
            ("cat", 0.8),
            ("the", 0.8),
            ("and", 0.8),
            ("that", 0.8),
            ("was", 0.8),
            ("the", 0.8),
            ("end", 0.8),
        ]);
        let merged = lcs_merge_words(&a, &b);
        // The critical assertion: the transition from a-words → b-words must
        // happen at or before "red". Any later split would have dropped the
        // "red cat" overlap from b, corrupting the tail.
        let words = words_of(&merged);
        let red_pos = words.iter().position(|w| *w == "red").expect("red present");
        // After "red" we must see b's continuation ("cat", "the", "and",
        // "that", "was", "the", "end"), not a's original "cat" alone.
        assert_eq!(
            &words[red_pos..],
            &["red", "cat", "the", "and", "that", "was", "the", "end"],
            "corner-biased LCS did not pick the overlap boundary: {:?}",
            words
        );
        // The stitched "red" must carry b's fresher confidence, not a's.
        assert!(
            (merged[red_pos].confidence - 0.8).abs() < 1e-5,
            "red confidence not taken from b: {}",
            merged[red_pos].confidence
        );
    }

    #[test]
    fn per_word_confidence_survives_long_merge() {
        // Three chunks, pairwise merge. Without confidence threading, the
        // downstream gate only sees a scalar average. Here we assert that
        // each merged word retains its distinct per-chunk confidence.
        let c1 = seq(&[("alpha", 0.9), ("beta", 0.8), ("gamma", 0.3)]);
        let c2 = seq(&[("beta", 0.85), ("gamma", 0.95), ("delta", 0.4)]);
        let c3 = seq(&[("gamma", 0.92), ("delta", 0.6), ("epsilon", 0.7)]);

        let merged = lcs_merge_words(&c1, &c2);
        let merged = lcs_merge_words(&merged, &c3);

        assert_eq!(
            words_of(&merged),
            vec!["alpha", "beta", "gamma", "delta", "epsilon"]
        );
        // "gamma" should come from c2 (0.95) before the c3 merge, then
        // from c3 (0.92) after. "delta" comes from c3 (0.6).
        let confs: Vec<f32> = merged.iter().map(|w| w.confidence).collect();
        assert_eq!(confs, vec![0.9, 0.85, 0.92, 0.6, 0.7]);
    }
}
