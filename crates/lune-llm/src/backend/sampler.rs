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
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 1.5,
            eos_token_id: None,
            stop_token_ids: Box::new([]),
        }
    }
}

#[derive(Default, Debug)]
pub struct DecodeState {
    pub generated: Vec<u32>,
    pub token_counts: HashMap<u32, u16>,
}

impl DecodeState {
    pub fn clear(&mut self) {
        self.generated.clear();
        self.token_counts.clear();
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
