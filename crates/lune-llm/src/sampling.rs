//! Token sampling strategies for ORT-based LLM inference.
//!
//! Token sampling implementations for ORT-based LLM inference:
//! greedy argmax, temperature scaling, top-k filtering, top-p (nucleus),
//! and repetition penalty.

use std::collections::VecDeque;

/// Sample a token from raw logits.
///
/// When `temperature <= 0.0`, performs greedy argmax.
/// Otherwise applies: repeat penalty → temperature → top-k → top-p → categorical.
///
/// `recent_tokens` is a sliding window of recently generated token IDs used
/// for repetition penalty.
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    recent_tokens: &VecDeque<i64>,
) -> i64 {
    debug_assert!(!logits.is_empty(), "empty logits slice");

    let mut working: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();

    // Apply repetition penalty to recently seen tokens.
    if repeat_penalty != 1.0 {
        for &tok in recent_tokens {
            let idx = tok as usize;
            if idx < working.len() {
                let logit = &mut working[idx].1;
                if *logit > 0.0 {
                    *logit /= repeat_penalty;
                } else {
                    *logit *= repeat_penalty;
                }
            }
        }
    }

    // Greedy: skip all stochastic steps.
    if temperature <= 0.0 {
        return argmax(&working);
    }

    // Temperature scaling.
    for (_, logit) in &mut working {
        *logit /= temperature;
    }

    // Top-k: keep only top_k highest logits.
    if top_k > 0 && top_k < working.len() {
        // Partial sort: move top_k elements to the front.
        working.select_nth_unstable_by(top_k, |a, b| b.1.total_cmp(&a.1));
        working.truncate(top_k);
    }

    // Softmax over remaining candidates.
    let max_logit = working
        .iter()
        .map(|(_, l)| *l)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);

    let mut sum = 0.0f32;
    for (_, logit) in &mut working {
        *logit = (*logit - max_logit).exp();
        sum += *logit;
    }
    for (_, prob) in &mut working {
        *prob /= sum;
    }

    // Top-p (nucleus): sort by probability descending, accumulate until >= top_p.
    if top_p < 1.0 {
        working.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut cumulative = 0.0f32;
        let cutoff = working
            .iter()
            .position(|&(_, p)| {
                cumulative += p;
                cumulative >= top_p
            })
            .map_or(working.len(), |pos| pos + 1);
        working.truncate(cutoff);

        // Re-normalize after truncation.
        let new_sum: f32 = working.iter().map(|(_, p)| p).sum();
        if new_sum > 0.0 {
            for (_, p) in &mut working {
                *p /= new_sum;
            }
        }
    }

    // Categorical sampling using a simple linear scan.
    // For vocab sizes after top-k filtering (~20 elements), this is faster
    // than building an alias table.
    let r = simple_random_f32();
    let mut cumulative = 0.0f32;
    for &(idx, prob) in &working {
        cumulative += prob;
        if r <= cumulative {
            return idx as i64;
        }
    }

    // Fallback: last token (rounding errors).
    working.last().map_or(0, |&(idx, _)| idx as i64)
}

/// Argmax over (index, logit) pairs.
#[inline]
fn argmax(items: &[(usize, f32)]) -> i64 {
    items
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |&(idx, _)| idx as i64)
}

/// Fast, non-cryptographic random f32 in [0, 1) for sampling.
///
/// Uses a thread-local xorshift64 seeded from OS entropy via `getrandom`.
/// Not suitable for security — only for token sampling jitter.
fn simple_random_f32() -> f32 {
    use std::cell::Cell;

    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let mut buf = [0u8; 8];
            getrandom::fill(&mut buf).expect("getrandom seed");
            let seed = u64::from_le_bytes(buf);
            // Ensure non-zero state for xorshift.
            if seed == 0 { 1 } else { seed }
        });
    }

    STATE.with(|cell| {
        let mut s = cell.get();
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        cell.set(s);
        // Map to [0, 1) — upper 23 bits give uniform mantissa.
        (s >> 41) as f32 / (1u64 << 23) as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let recent = VecDeque::new();
        let token = sample_token(&logits, 0.0, 0, 1.0, 1.0, &recent);
        assert_eq!(token, 2); // index of 0.9
    }

    #[test]
    fn greedy_with_negative_temp() {
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let recent = VecDeque::new();
        let token = sample_token(&logits, -1.0, 0, 1.0, 1.0, &recent);
        assert_eq!(token, 2);
    }

    #[test]
    fn repeat_penalty_suppresses() {
        // Token 0 has highest logit but is in recent_tokens with high penalty.
        let logits = vec![10.0, 9.0, 1.0];
        let mut recent = VecDeque::new();
        recent.push_back(0);
        // With penalty=100.0, token 0's logit becomes 10.0/100.0 = 0.1
        let token = sample_token(&logits, 0.0, 0, 1.0, 100.0, &recent);
        assert_eq!(token, 1); // token 1 (logit=9.0) now wins
    }

    #[test]
    fn top_k_limits_candidates() {
        // With top_k=1 and temperature, should always pick the max.
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let recent = VecDeque::new();
        for _ in 0..10 {
            let token = sample_token(&logits, 1.0, 1, 1.0, 1.0, &recent);
            assert_eq!(token, 2);
        }
    }

    #[test]
    fn simple_random_produces_valid_range() {
        for _ in 0..100 {
            let r = simple_random_f32();
            assert!((0.0..1.0).contains(&r), "random out of range: {r}");
        }
    }
}
