use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::AudioEncoderConfig;
use super::layers::{Attention, CausalConv1d, GatedMlp, RmsNorm, RotaryEmbedding};

/// Conv stem: CausalConv1d(mel→hidden) → GELU → CausalConv1d(hidden→hidden, stride=2) → GELU.
#[derive(Debug)]
struct Embedder {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
}

impl Embedder {
    fn load(cfg: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let conv1 = CausalConv1d::load(cfg.num_mel_bins, cfg.hidden_size, 3, 1, vb.pp("conv1"))?;
        let conv2 = CausalConv1d::load(cfg.hidden_size, cfg.hidden_size, 3, 2, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(mel)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        // [B, hidden, T] → [B, T, hidden]
        x.transpose(1, 2)
    }
}

/// Single encoder transformer layer.
struct EncoderLayer {
    self_attn: Attention,
    self_attn_norm: RmsNorm,
    final_norm: RmsNorm,
    mlp: GatedMlp,
}

impl EncoderLayer {
    fn load(cfg: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Encoder attention: q,v,o have bias; k has no bias.
        let self_attn = Attention::load(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            true,  // q_bias
            false, // k_bias
            true,  // v_bias
            true,  // o_bias
            vb.pp("self_attn"),
        )?;
        let self_attn_norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("self_attn_layer_norm"))?;
        let final_norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("final_layer_norm"))?;
        // Encoder MLP has bias on down_proj.
        let mlp = GatedMlp::load(cfg.hidden_size, cfg.intermediate_size, false, vb.pp("mlp"))?;
        Ok(Self { self_attn, self_attn_norm, final_norm, mlp })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        // Pre-norm self-attention.
        let residual = x.clone();
        let h = self.self_attn_norm.forward(x)?;
        let h = self.self_attn.forward(&h, rope, offset, sliding_window)?;
        let x = (residual + h)?;
        // Pre-norm MLP.
        let residual = x.clone();
        let h = self.final_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }

    fn reset_cache(&mut self) {
        self.self_attn.reset_cache();
    }
}

/// Voxtral Realtime audio encoder.
pub struct AudioEncoder {
    embedder: Embedder,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    rope: RotaryEmbedding,
    sliding_window: Option<usize>,
}

impl AudioEncoder {
    pub fn load(cfg: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embedder = Embedder::load(cfg, vb.pp("embedder"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(cfg, vb.pp(format!("layers.{i}")))?);
        }
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rope = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_parameters.rope_theta,
            vb.device(),
        )?;
        let sw = if cfg.sliding_window > 0 { Some(cfg.sliding_window) } else { None };
        Ok(Self { embedder, layers, norm, rope, sliding_window: sw })
    }

    /// Forward pass: mel features [B, mel_bins, T] → hidden states [B, T', hidden].
    pub fn forward(&mut self, mel: &Tensor, offset: usize) -> Result<Tensor> {
        let mut x = self.embedder.forward(mel)?;
        for layer in &mut self.layers {
            x = layer.forward(&x, &self.rope, offset, self.sliding_window)?;
        }
        self.norm.forward(&x)
    }

    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }
}
