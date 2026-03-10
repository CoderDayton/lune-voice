//! LCS-based overlap stitching for chunked ASR transcripts.
//!
//! Shared between OrtQwen and Parakeet backends for merging overlapping
//! audio chunk transcriptions at word boundaries.

/// Merge two word sequences from overlapping audio chunks using LCS.
///
/// Finds the longest common subsequence between the tail of `a` and the head
/// of `b`. If the LCS is long enough (≥2 words), stitches at the overlap
/// boundary: `a[..overlap_start] + b[overlap_start_in_b..]`.
/// The overlap words are taken from `b`'s transcription (fresher context).
///
/// Requires LCS ≥ 2 words to avoid false merges on coincidental word matches.
pub(crate) fn lcs_merge_words(a: &[String], b: &[String]) -> Vec<String> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    // Search window: overlap region ≈ 3-8 words depending on speech rate.
    let overlap_window = 8;
    let tail_start = a.len().saturating_sub(overlap_window);
    let tail_a = &a[tail_start..];
    let head_len = overlap_window.min(b.len());
    let head_b = &b[..head_len];

    // Build LCS table for the overlap region
    let (m, n) = (tail_a.len(), head_b.len());
    let mut dp = vec![vec![0u16; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if tail_a[i - 1].eq_ignore_ascii_case(&head_b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let lcs_len = dp[m][n] as usize;
    if lcs_len < 2 {
        // LCS too short — likely coincidental match, not real overlap.
        // Just concatenate without merging.
        let mut result = a.to_vec();
        result.extend_from_slice(b);
        return result;
    }

    // Backtrack to find where LCS starts in tail_a and head_b.
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        if tail_a[i - 1].eq_ignore_ascii_case(&head_b[j - 1]) {
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    // i = index in tail_a where LCS starts (0-based)
    // j = index in head_b where LCS starts (0-based)
    // Take a[..split_a] (before overlap) + b[j..] (overlap from b + rest of b)
    let split_a = tail_start + i;

    let mut result = Vec::with_capacity(split_a + (b.len() - j));
    result.extend_from_slice(&a[..split_a]);
    result.extend_from_slice(&b[j..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(lcs_merge_words(&[], &s(&["hello"])), s(&["hello"]));
        assert_eq!(lcs_merge_words(&s(&["hello"]), &[]), s(&["hello"]));
        assert_eq!(lcs_merge_words(&[], &[]), Vec::<String>::new());
    }

    #[test]
    fn no_overlap_concat() {
        let a = s(&["hello", "world"]);
        let b = s(&["foo", "bar"]);
        // LCS < 2 → concatenate
        assert_eq!(
            lcs_merge_words(&a, &b),
            s(&["hello", "world", "foo", "bar"])
        );
    }

    #[test]
    fn clean_overlap() {
        let a = s(&["the", "quick", "brown", "fox"]);
        let b = s(&["brown", "fox", "jumps", "over"]);
        let result = lcs_merge_words(&a, &b);
        assert_eq!(
            result,
            s(&["the", "quick", "brown", "fox", "jumps", "over"])
        );
    }

    #[test]
    fn full_overlap() {
        let a = s(&["hello", "world"]);
        let b = s(&["hello", "world", "again"]);
        let result = lcs_merge_words(&a, &b);
        assert_eq!(result, s(&["hello", "world", "again"]), "got: {:?}", result);
    }

    #[test]
    fn case_insensitive() {
        let a = s(&["The", "Quick"]);
        let b = s(&["the", "quick", "fox"]);
        let result = lcs_merge_words(&a, &b);
        assert_eq!(result, s(&["the", "quick", "fox"]));
    }
}
