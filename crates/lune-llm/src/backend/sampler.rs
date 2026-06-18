use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use half::f16;
use ort::value::DynValue;
use rand::Rng;

use lune_core::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub eos_token_id: Option<u32>,
    pub stop_token_ids: Box<[u32]>,
}

impl Default for SamplingParams {
    /// Penalized greedy: deterministic argmax (temperature 0, top_k 1) taken
    /// *after* the repetition/presence penalties reshape the logits.
    ///
    /// A grid search over the sampling space (`benchmarks/tuning/sampling_tune.rs`)
    /// found no config that reliably beats greedy on correction quality — a
    /// sampled generation was ~3x more likely to hurt than help. So we keep
    /// greedy's quality and determinism, while the mild penalties still demote
    /// already-emitted tokens to break any repeat/echo loop. (rep > ~1.1 starts
    /// to degrade quality, so 1.1 is the ceiling here.)
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.5,
            eos_token_id: None,
            stop_token_ids: Box::new([]),
        }
    }
}

#[derive(Default, Debug)]
pub struct DecodeState {
    pub generated: Vec<u32>,
    pub token_counts: HashMap<u32, u16>,
    /// Per-emitted-token decision margin, populated only when the caller opts
    /// in (see `OrtStyleLlm::generate_scored`). Stays empty on the default
    /// path. Invariant: `margins.is_empty() || margins.len() == generated.len()`.
    pub margins: Vec<f32>,
}

impl DecodeState {
    pub fn clear(&mut self) {
        self.generated.clear();
        self.token_counts.clear();
        self.margins.clear();
    }

    pub fn push(&mut self, token_id: u32) {
        self.generated.push(token_id);
        let c = self.token_counts.entry(token_id).or_insert(0);
        *c = c.saturating_add(1);
    }

    pub fn should_stop(&self, token_id: u32, params: &SamplingParams) -> bool {
        if params.eos_token_id == Some(token_id) {
            return true;
        }
        params.stop_token_ids.contains(&token_id)
    }
}

#[derive(Default, Debug)]
pub struct SamplerWorkspace {
    last_logits: Vec<f32>,
    probs: Vec<f32>,
    kept: Vec<Candidate>,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: u32,
    logit: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem {
    id: u32,
    logit: f32,
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.logit
            .total_cmp(&other.logit)
            .then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn e(msg: impl std::fmt::Display) -> Error {
    Error::config(msg.to_string())
}

pub fn sample_last_token(
    value: &DynValue,
    params: &SamplingParams,
    state: &DecodeState,
    ws: &mut SamplerWorkspace,
    rng: &mut impl Rng,
) -> Result<u32> {
    if params.top_k == 1 && penalties_are_identity(params) {
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            return argmax_last_logits_f32(shape.as_ref(), data);
        }
        let (shape, data) = value.try_extract_tensor::<f16>().map_err(e)?;
        return argmax_last_logits_f16(shape.as_ref(), data);
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        copy_last_logits_f32(shape.as_ref(), data, &mut ws.last_logits)?;
        return sample_from_logits(
            &mut ws.last_logits,
            params,
            state,
            &mut ws.probs,
            &mut ws.kept,
            rng,
        );
    }

    let (shape, data) = value.try_extract_tensor::<f16>().map_err(e)?;
    copy_last_logits_f16(shape.as_ref(), data, &mut ws.last_logits)?;
    sample_from_logits(
        &mut ws.last_logits,
        params,
        state,
        &mut ws.probs,
        &mut ws.kept,
        rng,
    )
}

/// Returns true if all penalties are effectively identity (no practical effect).
///
/// Uses a relative tolerance appropriate for logit-space differences.
/// A penalty effect < 1e-4 in logit space is essentially invisible to the model.
fn penalties_are_identity(params: &SamplingParams) -> bool {
    const TOL: f32 = 1e-4;
    // repetition_penalty == 1.0 means no scaling
    (params.repetition_penalty - 1.0).abs() < TOL
        && params.frequency_penalty.abs() < TOL
        && params.presence_penalty.abs() < TOL
}

fn last_logits_range(shape: &[i64], data_len: usize) -> Result<std::ops::Range<usize>> {
    if shape.len() != 3 {
        return Err(e(format!(
            "expected logits shape [batch, seq, vocab], got {shape:?}"
        )));
    }
    let batch = shape[0] as usize;
    let seq = shape[1] as usize;
    let vocab = shape[2] as usize;
    if batch != 1 || seq == 0 || vocab == 0 {
        return Err(e(format!("unsupported logits shape {shape:?}")));
    }
    let offset = (seq - 1) * vocab;
    if offset + vocab > data_len {
        return Err(e(format!(
            "logits data shorter than shape: len={data_len}, shape={shape:?}"
        )));
    }
    Ok(offset..offset + vocab)
}

fn argmax_last_logits_f32(shape: &[i64], data: &[f32]) -> Result<u32> {
    let range = last_logits_range(shape, data.len())?;
    Ok(argmax_finite(&data[range]) as u32)
}

fn argmax_last_logits_f16(shape: &[i64], data: &[f16]) -> Result<u32> {
    let range = last_logits_range(shape, data.len())?;
    let slice = &data[range];
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let mut any_finite = false;
    for (i, &v) in slice.iter().enumerate() {
        let f = v.to_f32();
        if f.is_finite() && f > best_val {
            best_val = f;
            best_idx = i;
            any_finite = true;
        }
    }
    if !any_finite {
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in slice.iter().enumerate() {
            let f = v.to_f32();
            if f > max_val {
                max_val = f;
                max_idx = i;
            }
        }
        return Ok(max_idx as u32);
    }
    Ok(best_idx as u32)
}

fn copy_last_logits_f32(shape: &[i64], data: &[f32], out: &mut Vec<f32>) -> Result<()> {
    let range = last_logits_range(shape, data.len())?;
    out.clear();
    out.extend_from_slice(&data[range]);
    Ok(())
}

fn copy_last_logits_f16(shape: &[i64], data: &[f16], out: &mut Vec<f32>) -> Result<()> {
    let range = last_logits_range(shape, data.len())?;
    out.clear();
    out.extend(data[range].iter().map(|x| x.to_f32()));
    Ok(())
}

fn sample_from_logits(
    logits: &mut [f32],
    params: &SamplingParams,
    state: &DecodeState,
    probs: &mut Vec<f32>,
    kept: &mut Vec<Candidate>,
    rng: &mut impl Rng,
) -> Result<u32> {
    apply_penalties(logits, state, params);

    if params.temperature <= 0.0 {
        return Ok(argmax(logits) as u32);
    }

    build_candidates(logits, params.top_k, kept);

    if kept.is_empty() {
        return Ok(argmax(logits) as u32);
    }
    if kept.len() == 1 {
        return Ok(kept[0].id);
    }

    softmax_candidates(kept, params.temperature, probs);

    if params.min_p > 0.0 {
        apply_min_p(probs, params.min_p);
    }

    if params.top_p > 0.0 && params.top_p < 1.0 {
        apply_top_p(probs, params.top_p);
    }

    renormalize(probs);

    if probs.iter().all(|&p| p <= 0.0 || !p.is_finite()) {
        return Ok(kept[0].id);
    }

    let r = rng.random::<f32>();
    let mut cdf = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r <= cdf {
            return Ok(kept[i].id);
        }
    }

    // Fallback: CDF may not sum to 1.0 due to floating point. Return highest-probability.
    kept.first()
        .map(|c| c.id)
        .ok_or_else(|| Error::llm("sampler: no valid candidates after filtering"))
}

fn apply_penalties(logits: &mut [f32], state: &DecodeState, params: &SamplingParams) {
    // Early return if penalties are identity (no practical effect)
    if penalties_are_identity(params) {
        return;
    }

    let rep = params.repetition_penalty;
    let freq = params.frequency_penalty;
    let pres = params.presence_penalty;

    let apply_rep = (rep - 1.0).abs() >= f32::EPSILON;
    let apply_freq = freq != 0.0;
    let apply_pres = pres != 0.0;

    for (&token_id, &count) in &state.token_counts {
        let idx = token_id as usize;
        if idx >= logits.len() {
            continue;
        }

        let logit = &mut logits[idx];

        if apply_rep {
            if *logit < 0.0 {
                *logit *= rep;
            } else {
                *logit /= rep;
            }
        }
        if apply_freq {
            *logit -= freq * (count as f32);
        }
        if apply_pres {
            *logit -= pres;
        }
    }
}

fn build_candidates(logits: &[f32], top_k: usize, kept: &mut Vec<Candidate>) {
    kept.clear();

    if top_k == 0 || top_k >= logits.len() {
        kept.extend(logits.iter().enumerate().filter_map(|(i, &logit)| {
            logit.is_finite().then_some(Candidate {
                id: i as u32,
                logit,
            })
        }));
        kept.sort_unstable_by(|a, b| b.logit.total_cmp(&a.logit));
        return;
    }

    let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::with_capacity(top_k + 1);

    for (i, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        let item = Reverse(HeapItem {
            id: i as u32,
            logit,
        });

        if heap.len() < top_k {
            heap.push(item);
        } else if let Some(smallest) = heap.peek()
            && item.0.logit > smallest.0.logit
        {
            heap.pop();
            heap.push(item);
        }
    }

    kept.extend(heap.into_iter().map(|Reverse(x)| Candidate {
        id: x.id,
        logit: x.logit,
    }));
    kept.sort_unstable_by(|a, b| b.logit.total_cmp(&a.logit));
}

fn softmax_candidates(kept: &[Candidate], temperature: f32, probs: &mut Vec<f32>) {
    probs.clear();
    probs.reserve(kept.len());

    let max_logit = kept[0].logit;
    let temp = temperature.max(1e-6);

    let mut sum = 0.0f32;
    for c in kept {
        let p = ((c.logit - max_logit) / temp).exp();
        probs.push(p);
        sum += p;
    }

    if sum > 0.0 && sum.is_finite() {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn apply_min_p(probs: &mut [f32], min_p: f32) {
    // probs[0] is the largest mass — softmax is monotone in logit, and
    // build_candidates returns `kept` sorted by descending logit. Skips
    // the O(n) max scan.
    let Some(&max_p) = probs.first() else {
        return;
    };
    let cutoff = max_p * min_p;
    for p in probs.iter_mut() {
        if *p < cutoff {
            *p = 0.0;
        }
    }
}

fn apply_top_p(probs: &mut [f32], top_p: f32) {
    // Nucleus sampling (Holtzman et al. 2019): keep the smallest prefix whose
    // cumulative mass first reaches top_p — *inclusive* of the crossing token.
    // Truncating the crossing token (older bug) leaves total mass < top_p,
    // contradicting the threshold semantics.
    let mut cumulative = 0.0f32;
    let mut crossed = false;
    for p in probs.iter_mut() {
        if *p <= 0.0 {
            continue;
        }
        if crossed {
            *p = 0.0;
            continue;
        }
        cumulative += *p;
        if cumulative >= top_p {
            crossed = true;
        }
    }
}

fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().copied().sum();
    if sum > 0.0 && sum.is_finite() {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn argmax_finite(logits: &[f32]) -> usize {
    let mut best_finite_i: Option<usize> = None;
    let mut best_finite_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_finite_v {
            best_finite_v = v;
            best_finite_i = Some(i);
        }
    }
    best_finite_i.unwrap_or_else(|| argmax(logits))
}

// ─── scored decode: emit the per-token decision margin alongside the token ───
//
// The margin is `logit(emitted) − max logit(other)` in logit space, on the
// exact logits used for the decision. It is observational: every scored path
// below selects the *same* token as its unscored twin for the same inputs and
// rng (enforced by `selection_identity` test). The downstream conformal gate
// only uses margin *ordering*, so no normalization is needed.

/// Margin when there is no runner-up (single finite candidate): treated as
/// maximally confident by the calibrator.
const SATURATED_MARGIN: f32 = 1.0e4;

fn margin_or_saturated(top1: f32, top2: f32) -> f32 {
    if top2.is_finite() {
        top1 - top2
    } else {
        SATURATED_MARGIN
    }
}

/// Top-1 index + margin, raw (no finiteness filter) — mirrors [`argmax`].
fn argmax2_raw(logits: &[f32]) -> (usize, f32) {
    let (mut bi, mut b1, mut b2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > b1 {
            b2 = b1;
            b1 = v;
            bi = i;
        } else if v > b2 {
            b2 = v;
        }
    }
    (bi, margin_or_saturated(b1, b2))
}

/// Top-1 index + margin, finite-aware — mirrors [`argmax_finite`].
fn argmax2_finite(logits: &[f32]) -> (usize, f32) {
    let (mut bi, mut b1, mut b2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut any = false;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        any = true;
        if v > b1 {
            b2 = b1;
            b1 = v;
            bi = i;
        } else if v > b2 {
            b2 = v;
        }
    }
    if !any {
        return (argmax(logits), SATURATED_MARGIN);
    }
    (bi, margin_or_saturated(b1, b2))
}

fn argmax2_last_logits_f32(shape: &[i64], data: &[f32]) -> Result<(u32, f32)> {
    let range = last_logits_range(shape, data.len())?;
    let (idx, m) = argmax2_finite(&data[range]);
    Ok((idx as u32, m))
}

fn argmax2_last_logits_f16(shape: &[i64], data: &[f16]) -> Result<(u32, f32)> {
    let range = last_logits_range(shape, data.len())?;
    let slice = &data[range];
    let (mut bi, mut b1, mut b2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut any = false;
    for (i, &v) in slice.iter().enumerate() {
        let f = v.to_f32();
        if !f.is_finite() {
            continue;
        }
        any = true;
        if f > b1 {
            b2 = b1;
            b1 = f;
            bi = i;
        } else if f > b2 {
            b2 = f;
        }
    }
    if !any {
        // Mirror argmax_last_logits_f16's non-finite fallback (raw argmax).
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in slice.iter().enumerate() {
            let f = v.to_f32();
            if f > max_val {
                max_val = f;
                max_idx = i;
            }
        }
        return Ok((max_idx as u32, SATURATED_MARGIN));
    }
    Ok((bi as u32, margin_or_saturated(b1, b2)))
}

/// Scored twin of [`sample_last_token`]: returns `(token_id, decision_margin)`.
/// Selects identically to `sample_last_token` for the same value/params/state/rng.
pub fn sample_last_token_scored(
    value: &DynValue,
    params: &SamplingParams,
    state: &DecodeState,
    ws: &mut SamplerWorkspace,
    rng: &mut impl Rng,
) -> Result<(u32, f32)> {
    if params.top_k == 1 && penalties_are_identity(params) {
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            return argmax2_last_logits_f32(shape.as_ref(), data);
        }
        let (shape, data) = value.try_extract_tensor::<f16>().map_err(e)?;
        return argmax2_last_logits_f16(shape.as_ref(), data);
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        copy_last_logits_f32(shape.as_ref(), data, &mut ws.last_logits)?;
        return sample_from_logits_scored(
            &mut ws.last_logits,
            params,
            state,
            &mut ws.probs,
            &mut ws.kept,
            rng,
        );
    }

    let (shape, data) = value.try_extract_tensor::<f16>().map_err(e)?;
    copy_last_logits_f16(shape.as_ref(), data, &mut ws.last_logits)?;
    sample_from_logits_scored(
        &mut ws.last_logits,
        params,
        state,
        &mut ws.probs,
        &mut ws.kept,
        rng,
    )
}

/// Scored twin of [`sample_from_logits`]. Replays the exact same control flow
/// and rng draws (so the chosen id matches), then derives the emitted token's
/// margin against the best alternative.
fn sample_from_logits_scored(
    logits: &mut [f32],
    params: &SamplingParams,
    state: &DecodeState,
    probs: &mut Vec<f32>,
    kept: &mut Vec<Candidate>,
    rng: &mut impl Rng,
) -> Result<(u32, f32)> {
    apply_penalties(logits, state, params);

    if params.temperature <= 0.0 {
        let (idx, m) = argmax2_raw(logits);
        return Ok((idx as u32, m));
    }

    build_candidates(logits, params.top_k, kept);

    if kept.is_empty() {
        let (idx, m) = argmax2_raw(logits);
        return Ok((idx as u32, m));
    }
    if kept.len() == 1 {
        return Ok((kept[0].id, SATURATED_MARGIN));
    }

    softmax_candidates(kept, params.temperature, probs);

    if params.min_p > 0.0 {
        apply_min_p(probs, params.min_p);
    }
    if params.top_p > 0.0 && params.top_p < 1.0 {
        apply_top_p(probs, params.top_p);
    }
    renormalize(probs);

    // `kept` is sorted by descending logit (build_candidates), so the emitted
    // token's best alternative is kept[0] (or kept[1] when it *is* kept[0]).
    let emitted_pos = if probs.iter().all(|&p| p <= 0.0 || !p.is_finite()) {
        0 // mirrors `return Ok(kept[0].id)`
    } else {
        let r = rng.random::<f32>();
        let mut cdf = 0.0f32;
        let mut pos = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            cdf += p;
            if r <= cdf {
                pos = i;
                break;
            }
        }
        pos // CDF-underflow fallback lands on 0, matching kept.first()
    };

    let emit_logit = kept[emitted_pos].logit;
    let best_other = if emitted_pos == 0 {
        kept.get(1).map(|c| c.logit)
    } else {
        Some(kept[0].logit)
    };
    let margin = best_other.map_or(SATURATED_MARGIN, |o| emit_logit - o);
    Ok((kept[emitted_pos].id, margin))
}

#[cfg(test)]
mod scored_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn random_logits(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| lcg(&mut s) * 8.0).collect()
    }

    #[test]
    fn argmax2_matches_argmax() {
        for seed in 0..200u64 {
            let v = random_logits(seed + 1, 64);
            assert_eq!(
                argmax2_raw(&v).0,
                argmax(&v),
                "raw idx mismatch seed {seed}"
            );
            assert_eq!(
                argmax2_finite(&v).0,
                argmax_finite(&v),
                "finite idx mismatch seed {seed}"
            );
        }
    }

    #[test]
    fn margin_equals_top1_minus_top2() {
        for seed in 0..200u64 {
            let v = random_logits(seed + 7, 48);
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let (_, m) = argmax2_raw(&v);
            assert!(
                (m - (sorted[0] - sorted[1])).abs() < 1e-4,
                "margin mismatch seed {seed}"
            );
        }
    }

    #[test]
    fn finite_aware_skips_inf_and_nan() {
        let v = vec![f32::NEG_INFINITY, 2.0, f32::NAN, 5.0, 1.0];
        assert_eq!(argmax2_finite(&v).0, 3);
        assert!((argmax2_finite(&v).1 - (5.0 - 2.0)).abs() < 1e-4);
    }

    #[test]
    fn single_finite_candidate_saturates() {
        let v = vec![f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY];
        assert_eq!(argmax2_finite(&v).1, SATURATED_MARGIN);
    }

    /// The core no-downgrade proof: the scored sampler picks the same token as
    /// the plain sampler across greedy and sampling configs, for the same rng.
    #[test]
    fn selection_identity() {
        let state = DecodeState::default();
        let configs = [
            SamplingParams::default(), // penalized greedy
            SamplingParams {
                temperature: 0.0,
                top_k: 1,
                repetition_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                ..SamplingParams::default()
            }, // pure greedy fast path
            SamplingParams {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.95,
                min_p: 0.02,
                ..SamplingParams::default()
            }, // sampling
        ];
        for cfg in &configs {
            for seed in 0..300u64 {
                let base = random_logits(seed + 11, 128);
                let mut a = base.clone();
                let mut b = base.clone();
                let mut wa = SamplerWorkspace::default();
                let mut wb = SamplerWorkspace::default();
                let mut ra = StdRng::seed_from_u64(seed);
                let mut rb = StdRng::seed_from_u64(seed);
                let plain =
                    sample_from_logits(&mut a, cfg, &state, &mut wa.probs, &mut wa.kept, &mut ra)
                        .unwrap();
                let (scored, _m) = sample_from_logits_scored(
                    &mut b,
                    cfg,
                    &state,
                    &mut wb.probs,
                    &mut wb.kept,
                    &mut rb,
                )
                .unwrap();
                assert_eq!(plain, scored, "selection diverged: cfg={cfg:?} seed={seed}");
            }
        }
    }
}
