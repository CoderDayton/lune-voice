use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

/// RMSNorm (no bias, weight-only).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        x_norm.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// Rotary Position Embeddings.
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(DType::F32)?;
        // [max_len, half]
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        // [max_len, head_dim]
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos_cache = emb.cos()?;
        let sin_cache = emb.sin()?;
        Ok(Self { cos_cache, sin_cache })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos_cache.i(offset..offset + seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin_cache.i(offset..offset + seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let q_rot = apply_rotary(q, &cos, &sin)?;
        let k_rot = apply_rotary(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.to_dtype(x.dtype())?;
    let sin = sin.to_dtype(x.dtype())?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

/// Grouped-query attention with optional sliding window and causal masking.
#[derive(Debug)]
pub struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    pub fn load(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_bias: bool,
        k_bias: bool,
        v_bias: bool,
        o_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let load_linear = |name: &str, in_d: usize, out_d: usize, bias: bool| -> Result<candle_nn::Linear> {
            let w = vb.get((out_d, in_d), &format!("{name}.weight"))?;
            let b = if bias {
                Some(vb.get(out_d, &format!("{name}.bias"))?)
            } else {
                None
            };
            Ok(candle_nn::Linear::new(w, b))
        };
        let q_proj = load_linear("q_proj", hidden, num_heads * head_dim, q_bias)?;
        let k_proj = load_linear("k_proj", hidden, num_kv_heads * head_dim, k_bias)?;
        let v_proj = load_linear("v_proj", hidden, num_kv_heads * head_dim, v_bias)?;
        let o_proj = load_linear("o_proj", num_heads * head_dim, hidden, o_bias)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(x)?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(x)?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rope.apply(&q, &k, offset)?;

        // KV cache append.
        let (k, v) = match &self.kv_cache {
            Some((pk, pv)) => {
                let k = Tensor::cat(&[pk, &k], 2)?;
                let v = Tensor::cat(&[pv, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA: repeat KV heads.
        let kv_groups = self.num_heads / self.num_kv_heads;
        let k = if kv_groups > 1 { repeat_kv(&k, kv_groups)? } else { k };
        let v = if kv_groups > 1 { repeat_kv(&v, kv_groups)? } else { v };

        // Scaled dot-product attention.
        let kv_len = k.dim(2)?;
        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        // Causal + sliding window mask.
        let mask = build_causal_mask(seq_len, kv_len, offset, sliding_window, x.device())?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq_len, ()))?;
        self.o_proj.forward(&out)
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache = None;
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, heads, seq, dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, heads, n_rep, seq, dim))?
        .reshape((b, heads * n_rep, seq, dim))
}

fn build_causal_mask(
    q_len: usize,
    kv_len: usize,
    _offset: usize,
    sliding_window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let q_pos = kv_len - q_len + i;
        for j in 0..kv_len {
            let causal_ok = j <= q_pos;
            let window_ok = match sliding_window {
                Some(w) => q_pos.saturating_sub(w) < j + 1,
                None => true,
            };
            if !(causal_ok && window_ok) {
                mask[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
}

/// Gated MLP: gate_proj * SiLU(up_proj(x)) -> down_proj.
#[derive(Debug)]
pub struct GatedMlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl GatedMlp {
    pub fn load(hidden: usize, intermediate: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let load = |name: &str, i: usize, o: usize| -> Result<candle_nn::Linear> {
            let w = vb.get((o, i), &format!("{name}.weight"))?;
            let b = if bias { Some(vb.get(o, &format!("{name}.bias"))?) } else { None };
            Ok(candle_nn::Linear::new(w, b))
        };
        Ok(Self {
            gate_proj: load("gate_proj", hidden, intermediate)?,
            up_proj: load("up_proj", hidden, intermediate)?,
            down_proj: load("down_proj", intermediate, hidden)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Causal Conv1d with left-padding for streaming.
#[derive(Debug)]
pub struct CausalConv1d {
    conv: Conv1d,
    left_pad: usize,
}

impl CausalConv1d {
    pub fn load(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let w = vb.get((out_c, in_c, kernel), "weight")?;
        let b = vb.get(out_c, "bias")?;
        let cfg = Conv1dConfig {
            stride,
            padding: 0,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = Conv1d::new(w, Some(b), cfg);
        let left_pad = kernel - stride;
        Ok(Self { conv, left_pad })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = x.pad_with_zeros(D::Minus1, self.left_pad, 0)?;
        self.conv.forward(&padded)
    }
}

/// Adaptive RMSNorm used in decoder layers: Linear(h→32) → GELU → Linear(32→h).
#[derive(Debug)]
pub struct AdaRmsNorm {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
}

impl AdaRmsNorm {
    pub fn load(hidden: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = vb.get((32, hidden), "linear1.weight")?;
        let w2 = vb.get((hidden, 32), "linear2.weight")?;
        Ok(Self {
            linear1: candle_nn::Linear::new(w1, None),
            linear2: candle_nn::Linear::new(w2, None),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?.gelu()?;
        self.linear2.forward(&h)
    }
}

/// Sinusoidal time embedding for delay conditioning.
#[derive(Debug)]
pub struct TimeEmbedding {
    inv_freq: Tensor,
}

impl TimeEmbedding {
    pub fn new(dim: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (-theta.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        Ok(Self { inv_freq })
    }

    pub fn forward(&self, time: f32, device: &Device) -> Result<Tensor> {
        let t = Tensor::new(&[time], device)?;
        let emb = t.broadcast_mul(&self.inv_freq)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Tensor::cat(&[&cos, &sin], D::Minus1)
    }
}
