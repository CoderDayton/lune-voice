//! Compressed KV cache with frozen prefix support.
//!
//! Prunes, rotates, and quantizes a prefix KV state for reuse across requests.
//! Decompresses on demand back to ONNX-compatible DynValue tensors.

use half::f16;
use ndarray::Array4;
use ort::value::{DynValue, Value};
use rand::Rng;
use tracing::debug;

use lune_core::error::{Error, Result};

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Config(msg.to_string())
}

// ── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub is_f16: bool,
    pub sink_tokens: usize,
    pub prune_keep_ratio: f32,
    pub quant_bits: u8,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 0,
            num_kv_heads: 8,
            head_dim: 128,
            is_f16: false,
            sink_tokens: 4,
            prune_keep_ratio: 1.0,
            quant_bits: 4,
        }
    }
}

// ── Internals ───────────────────────────────────────────────────────────

struct QuantizedTensor {
    codes: Vec<u8>,
    scales: Vec<f32>,
    mins: Vec<f32>,
}

struct QuantizedLayer {
    key_q: QuantizedTensor,
    val_q: QuantizedTensor,
    sink_keys: Vec<f32>,
    sink_vals: Vec<f32>,
    kept_tokens: usize,
    sink_count: usize,
}

struct FrozenPrefix {
    layers: Vec<QuantizedLayer>,
    total_seq_len: usize,
}

struct RotationMatrix {
    data: Vec<f32>,
    dim: usize,
}

// ── Rotation ────────────────────────────────────────────────────────────

impl RotationMatrix {
    fn random(dim: usize) -> Self {
        let mut rng = rand::rng();
        let mut data = Vec::with_capacity(dim * dim);
        for _ in 0..dim * dim {
            let u: f32 = rng.random();
            let v: f32 = rng.random();
            data.push((-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos());
        }
        // Modified Gram-Schmidt for orthogonalization
        for j in 0..dim {
            for i in 0..j {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += data[k * dim + j] * data[k * dim + i];
                }
                for k in 0..dim {
                    data[k * dim + j] -= dot * data[k * dim + i];
                }
            }
            let mut norm = 0.0f32;
            for k in 0..dim {
                norm += data[k * dim + j] * data[k * dim + j];
            }
            let inv = 1.0 / norm.sqrt().max(1e-12);
            for k in 0..dim {
                data[k * dim + j] *= inv;
            }
        }
        Self { data, dim }
    }

    fn apply(&self, src: &[f32], tokens: usize, dst: &mut [f32]) {
        let d = self.dim;
        for t in 0..tokens {
            let row = t * d;
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in 0..d {
                    sum += src[row + k] * self.data[k * d + j];
                }
                dst[row + j] = sum;
            }
        }
    }

    fn apply_inverse(&self, src: &[f32], tokens: usize, dst: &mut [f32]) {
        let d = self.dim;
        for t in 0..tokens {
            let row = t * d;
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in 0..d {
                    sum += src[row + k] * self.data[j * d + k];
                }
                dst[row + j] = sum;
            }
        }
    }
}

// ── Quantization ────────────────────────────────────────────────────────

fn quantize_4bit(data: &[f32], tokens: usize, dim: usize) -> QuantizedTensor {
    let levels = 15.0f32;
    let mut scales = vec![0.0f32; dim];
    let mut mins = vec![0.0f32; dim];

    for d in 0..dim {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for t in 0..tokens {
            let v = data[t * dim + d];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let range = hi - lo;
        scales[d] = if range < 1e-10 { 1.0 } else { range / levels };
        mins[d] = lo;
    }

    let total_nibbles = tokens * dim;
    let mut codes = vec![0u8; total_nibbles.div_ceil(2)];
    for t in 0..tokens {
        for d in 0..dim {
            let idx = t * dim + d;
            let q = ((data[idx] - mins[d]) / scales[d])
                .round()
                .clamp(0.0, levels) as u8;
            if idx % 2 == 0 {
                codes[idx / 2] |= q;
            } else {
                codes[idx / 2] |= q << 4;
            }
        }
    }
    QuantizedTensor {
        codes,
        scales,
        mins,
    }
}

fn dequantize_4bit(qt: &QuantizedTensor, tokens: usize, dim: usize, out: &mut [f32]) {
    for t in 0..tokens {
        for d in 0..dim {
            let idx = t * dim + d;
            let byte = qt.codes[idx / 2];
            let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[idx] = qt.mins[d] + nibble as f32 * qt.scales[d];
        }
    }
}

// ── Extraction helpers ──────────────────────────────────────────────────

fn extract_f32(dv: &DynValue) -> Result<(Vec<usize>, Vec<f32>)> {
    if let Ok((shape, data)) = dv.try_extract_tensor::<f32>() {
        let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        Ok((dims, data.to_vec()))
    } else if let Ok((shape, data)) = dv.try_extract_tensor::<f16>() {
        let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        Ok((dims, f32_data))
    } else {
        Err(err("KV tensor: unsupported dtype"))
    }
}

fn to_dyn_value(data: &[f32], shape: [usize; 4], is_f16: bool) -> Result<DynValue> {
    if is_f16 {
        let f16_data: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        let arr = Array4::from_shape_vec(shape, f16_data).map_err(err)?;
        Value::from_array(arr).map(|v| v.into_dyn()).map_err(err)
    } else {
        let arr = Array4::from_shape_vec(shape, data.to_vec()).map_err(err)?;
        Value::from_array(arr).map(|v| v.into_dyn()).map_err(err)
    }
}

// ── Score / gather helpers ───────────────────────────────────────────────

/// Accumulate per-token key L2-norm scores across all layers and heads.
///
/// When `factors` is `Some`, each layer's contribution is scaled by
/// `factors[li]` before accumulation (used for the normalised re-score pass).
/// `scores` must be pre-zeroed by the caller.
#[allow(clippy::too_many_arguments)]
fn accumulate_scores(
    layer_data: &[(Vec<f32>, Vec<f32>, usize)],
    scores: &mut [f32],
    layer_norms: &mut [f32],
    h: usize,
    d: usize,
    seq_len: usize,
    sink: usize,
    factors: Option<&[f32]>,
) {
    for (li, (key_data, _, _)) in layer_data.iter().enumerate() {
        let factor = factors.map_or(1.0, |f| f[li]);
        let mut layer_sum = 0.0f32;
        for head in 0..h {
            for (t, score) in scores.iter_mut().enumerate() {
                let pos = t + sink;
                let base = head * seq_len * d + pos * d;
                let norm_sq: f32 = key_data[base..base + d].iter().map(|x| x * x).sum();
                *score += norm_sq * factor;
                layer_sum += norm_sq * factor;
            }
        }
        layer_norms[li] = layer_sum;
    }
}

/// Gather `keep` tokens (selected by `kept_indices`) from `src` into a
/// freshly-allocated `keep * d` buffer.  Each index is relative to the
/// prunable window; add `sink` to obtain the absolute position.
fn gather_tokens(
    src: &[f32],
    kept_indices: &[usize],
    head: usize,
    seq_len: usize,
    sink: usize,
    d: usize,
) -> Vec<f32> {
    let keep = kept_indices.len();
    let mut gathered = vec![0.0f32; keep * d];
    for (out_t, &src_t) in kept_indices.iter().enumerate() {
        let src_pos = src_t + sink;
        let src_base = head * seq_len * d + src_pos * d;
        let dst_base = out_t * d;
        gathered[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
    }
    gathered
}

// ── Public API ──────────────────────────────────────────────────────────

pub struct KvCache {
    config: KvCacheConfig,
    rotation: RotationMatrix,
    prefix: Option<FrozenPrefix>,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Self {
        let rotation = RotationMatrix::random(config.head_dim);
        Self {
            config,
            rotation,
            prefix: None,
        }
    }

    pub fn has_prefix(&self) -> bool {
        self.prefix.is_some()
    }

    pub fn clear_prefix(&mut self) {
        self.prefix = None;
    }

    /// Total token count stored in the frozen prefix (sinks + kept).
    pub fn prefix_len(&self) -> usize {
        self.prefix.as_ref().map_or(0, |p| p.total_seq_len)
    }

    /// Freeze raw KV outputs into a compressed, reusable prefix.
    ///
    /// `kv` is interleaved `[layer0_key, layer0_val, layer1_key, …]`.
    pub fn freeze_prefix(&mut self, kv: Vec<DynValue>) -> Result<()> {
        let c = &self.config;
        let expected = c.num_layers * 2;
        if kv.len() != expected {
            return Err(err(format!(
                "expected {expected} KV tensors, got {}",
                kv.len()
            )));
        }

        // Extract all layers to f32 and read sequence length from first tensor.
        let mut layer_data: Vec<(Vec<f32>, Vec<f32>, usize)> = Vec::with_capacity(c.num_layers);
        let mut seq_len = 0usize;
        for layer in 0..c.num_layers {
            let (ks, kd) = extract_f32(&kv[layer * 2])?;
            let (_, vd) = extract_f32(&kv[layer * 2 + 1])?;
            if layer == 0 {
                seq_len = ks[2]; // shape: [1, heads, seq, dim]
            }
            layer_data.push((kd, vd, seq_len));
        }

        if seq_len == 0 {
            return Err(err("cannot freeze empty KV"));
        }

        let h = c.num_kv_heads;
        let d = c.head_dim;
        let sink = c.sink_tokens.min(seq_len);
        let prunable = seq_len - sink;

        // Score tokens for pruning (aggregate key L2 norms across layers+heads).
        let mut scores = vec![0.0f32; prunable];
        let mut layer_norms = vec![0.0f32; c.num_layers];

        accumulate_scores(
            &layer_data,
            &mut scores,
            &mut layer_norms,
            h,
            d,
            seq_len,
            sink,
            None,
        );

        // Normalize per-layer contribution.
        if c.num_layers > 1 {
            let mean_norm: f32 = layer_norms.iter().sum::<f32>() / c.num_layers as f32;
            if mean_norm > 1e-12 {
                // Re-score with normalization. First reset.
                scores.fill(0.0);
                let factors: Vec<f32> = layer_norms
                    .iter()
                    .map(|&ln| if ln > 1e-12 { mean_norm / ln } else { 1.0 })
                    .collect();
                layer_norms.fill(0.0);
                accumulate_scores(
                    &layer_data,
                    &mut scores,
                    &mut layer_norms,
                    h,
                    d,
                    seq_len,
                    sink,
                    Some(&factors),
                );
            }
        }

        // Select top-K tokens to keep.
        let keep = if prunable == 0 {
            0
        } else {
            ((c.prune_keep_ratio * prunable as f32).ceil() as usize)
                .max(1)
                .min(prunable)
        };
        let mut ranked: Vec<usize> = (0..prunable).collect();
        ranked.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let mut kept_indices: Vec<usize> = ranked[..keep].to_vec();
        kept_indices.sort_unstable(); // restore positional order

        let total_seq = sink + keep;

        debug!(
            seq_len,
            sink, prunable, keep, total_seq, "freezing KV prefix"
        );

        // Compress each layer.
        let mut layers = Vec::with_capacity(c.num_layers);
        let mut rot_buf = vec![0.0f32; keep.max(1) * d];

        for (key_data, val_data, _) in &layer_data {
            let mut sink_keys = Vec::with_capacity(h * sink * d);
            let mut sink_vals = Vec::with_capacity(h * sink * d);
            let mut key_q_all_codes = Vec::new();
            let mut key_q_all_scales = Vec::new();
            let mut key_q_all_mins = Vec::new();
            let mut val_q_all_codes = Vec::new();
            let mut val_q_all_scales = Vec::new();
            let mut val_q_all_mins = Vec::new();

            for head in 0..h {
                // Sink tokens: store at full precision.
                for t in 0..sink {
                    let base = head * seq_len * d + t * d;
                    sink_keys.extend_from_slice(&key_data[base..base + d]);
                    sink_vals.extend_from_slice(&val_data[base..base + d]);
                }

                if keep > 0 {
                    // Gather kept tokens.
                    let gathered_k = gather_tokens(key_data, &kept_indices, head, seq_len, sink, d);
                    let gathered_v = gather_tokens(val_data, &kept_indices, head, seq_len, sink, d);

                    // Rotate then quantize keys.
                    self.rotation
                        .apply(&gathered_k, keep, &mut rot_buf[..keep * d]);
                    let kq = quantize_4bit(&rot_buf[..keep * d], keep, d);
                    key_q_all_codes.extend_from_slice(&kq.codes);
                    key_q_all_scales.extend_from_slice(&kq.scales);
                    key_q_all_mins.extend_from_slice(&kq.mins);

                    // Rotate then quantize values.
                    self.rotation
                        .apply(&gathered_v, keep, &mut rot_buf[..keep * d]);
                    let vq = quantize_4bit(&rot_buf[..keep * d], keep, d);
                    val_q_all_codes.extend_from_slice(&vq.codes);
                    val_q_all_scales.extend_from_slice(&vq.scales);
                    val_q_all_mins.extend_from_slice(&vq.mins);
                }
            }

            layers.push(QuantizedLayer {
                key_q: QuantizedTensor {
                    codes: key_q_all_codes,
                    scales: key_q_all_scales,
                    mins: key_q_all_mins,
                },
                val_q: QuantizedTensor {
                    codes: val_q_all_codes,
                    scales: val_q_all_scales,
                    mins: val_q_all_mins,
                },
                sink_keys,
                sink_vals,
                kept_tokens: keep,
                sink_count: sink,
            });
        }

        self.prefix = Some(FrozenPrefix {
            layers,
            total_seq_len: total_seq,
        });
        Ok(())
    }

    /// Decompress the frozen prefix back into ONNX-compatible KV tensors.
    ///
    /// Returns interleaved `[layer0_key, layer0_val, layer1_key, …]`.
    pub fn prefix_kv(&self) -> Result<Vec<DynValue>> {
        let fp = self
            .prefix
            .as_ref()
            .ok_or_else(|| err("no frozen prefix"))?;
        let c = &self.config;
        let h = c.num_kv_heads;
        let d = c.head_dim;
        let sink = fp.layers[0].sink_count;
        let keep = fp.layers[0].kept_tokens;
        let total = fp.total_seq_len;

        let nibbles_per_head = (keep * d).div_ceil(2);
        let mut deq_buf = vec![0.0f32; keep.max(1) * d];
        let mut inv_buf = vec![0.0f32; keep.max(1) * d];
        let mut result = Vec::with_capacity(c.num_layers * 2);

        for layer in &fp.layers {
            let mut key_out = vec![0.0f32; h * total * d];
            let mut val_out = vec![0.0f32; h * total * d];

            for head in 0..h {
                // Copy sink tokens directly.
                let sink_base = head * sink * d;
                let dst_base = head * total * d;
                key_out[dst_base..dst_base + sink * d]
                    .copy_from_slice(&layer.sink_keys[sink_base..sink_base + sink * d]);
                val_out[dst_base..dst_base + sink * d]
                    .copy_from_slice(&layer.sink_vals[sink_base..sink_base + sink * d]);

                if keep > 0 {
                    // Dequantize keys for this head.
                    let code_off = head * nibbles_per_head;
                    let scale_off = head * d;
                    let head_qt = QuantizedTensor {
                        codes: layer.key_q.codes[code_off..code_off + nibbles_per_head].to_vec(),
                        scales: layer.key_q.scales[scale_off..scale_off + d].to_vec(),
                        mins: layer.key_q.mins[scale_off..scale_off + d].to_vec(),
                    };
                    dequantize_4bit(&head_qt, keep, d, &mut deq_buf[..keep * d]);
                    self.rotation.apply_inverse(
                        &deq_buf[..keep * d],
                        keep,
                        &mut inv_buf[..keep * d],
                    );
                    let k_dst = head * total * d + sink * d;
                    key_out[k_dst..k_dst + keep * d].copy_from_slice(&inv_buf[..keep * d]);

                    // Dequantize values for this head.
                    let head_vt = QuantizedTensor {
                        codes: layer.val_q.codes[code_off..code_off + nibbles_per_head].to_vec(),
                        scales: layer.val_q.scales[scale_off..scale_off + d].to_vec(),
                        mins: layer.val_q.mins[scale_off..scale_off + d].to_vec(),
                    };
                    dequantize_4bit(&head_vt, keep, d, &mut deq_buf[..keep * d]);
                    self.rotation.apply_inverse(
                        &deq_buf[..keep * d],
                        keep,
                        &mut inv_buf[..keep * d],
                    );
                    let v_dst = head * total * d + sink * d;
                    val_out[v_dst..v_dst + keep * d].copy_from_slice(&inv_buf[..keep * d]);
                }
            }

            let shape = [1, h, total, d];
            result.push(to_dyn_value(&key_out, shape, c.is_f16)?);
            result.push(to_dyn_value(&val_out, shape, c.is_f16)?);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_kv(layers: usize, heads: usize, seq: usize, dim: usize) -> Vec<DynValue> {
        let mut out = Vec::with_capacity(layers * 2);
        let mut rng = rand::rng();
        for _ in 0..layers {
            for _ in 0..2 {
                let data: Vec<f32> = (0..heads * seq * dim)
                    .map(|_| rng.random::<f32>())
                    .collect();
                let arr = Array4::from_shape_vec((1, heads, seq, dim), data).unwrap();
                out.push(Value::from_array(arr).unwrap().into_dyn());
            }
        }
        out
    }

    #[test]
    fn roundtrip_preserves_shape() {
        let layers = 2;
        let heads = 4;
        let seq = 32;
        let dim = 16;
        let kv = make_dummy_kv(layers, heads, seq, dim);

        let mut cache = KvCache::new(KvCacheConfig {
            num_layers: layers,
            num_kv_heads: heads,
            head_dim: dim,
            is_f16: false,
            sink_tokens: 4,
            prune_keep_ratio: 0.3,
            quant_bits: 4,
        });

        cache.freeze_prefix(kv).unwrap();
        assert!(cache.has_prefix());

        let expected_len = 4 + ((seq - 4) as f32 * 0.3).ceil() as usize;
        assert_eq!(cache.prefix_len(), expected_len);

        let restored = cache.prefix_kv().unwrap();
        assert_eq!(restored.len(), layers * 2);

        // Check shapes of restored tensors.
        for dv in &restored {
            let (shape, _) = extract_f32(dv).unwrap();
            assert_eq!(shape[0], 1);
            assert_eq!(shape[1], heads);
            assert_eq!(shape[2], expected_len);
            assert_eq!(shape[3], dim);
        }
    }

    #[test]
    fn roundtrip_accuracy() {
        let layers = 1;
        let heads = 2;
        let seq = 20;
        let dim = 8;
        let kv = make_dummy_kv(layers, heads, seq, dim);

        let (_, original_keys) = extract_f32(&kv[0]).unwrap();

        let mut cache = KvCache::new(KvCacheConfig {
            num_layers: layers,
            num_kv_heads: heads,
            head_dim: dim,
            is_f16: false,
            sink_tokens: 4,
            prune_keep_ratio: 1.0, // keep all to test quant error
            quant_bits: 4,
        });

        cache.freeze_prefix(kv).unwrap();
        let restored = cache.prefix_kv().unwrap();
        let (_, restored_keys) = extract_f32(&restored[0]).unwrap();

        // Sink tokens (first 4) should be exact.
        for head in 0..heads {
            for t in 0..4 {
                for d in 0..dim {
                    let orig = original_keys[head * seq * dim + t * dim + d];
                    let rest = restored_keys[head * seq * dim + t * dim + d];
                    assert!(
                        (orig - rest).abs() < 1e-6,
                        "sink mismatch at head={head} t={t} d={d}: {orig} vs {rest}"
                    );
                }
            }
        }

        // Non-sink tokens: quantization error should be bounded.
        let mut max_err = 0.0f32;
        for head in 0..heads {
            for t in 4..seq {
                for d in 0..dim {
                    let orig = original_keys[head * seq * dim + t * dim + d];
                    let rest = restored_keys[head * seq * dim + t * dim + d];
                    max_err = max_err.max((orig - rest).abs());
                }
            }
        }
        assert!(max_err < 0.15, "max quantization error too high: {max_err}");
    }

    #[test]
    fn clear_invalidates() {
        let kv = make_dummy_kv(1, 2, 10, 8);
        let mut cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            is_f16: false,
            sink_tokens: 2,
            prune_keep_ratio: 0.5,
            quant_bits: 4,
        });
        cache.freeze_prefix(kv).unwrap();
        assert!(cache.has_prefix());
        cache.clear_prefix();
        assert!(!cache.has_prefix());
    }
}
