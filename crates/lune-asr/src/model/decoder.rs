use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

use crate::config::TextDecoderConfig;
use super::layers::{AdaRmsNorm, Attention, GatedMlp, RmsNorm, RotaryEmbedding};

/// Single text decoder layer with adaptive RMSNorm conditioning.
struct DecoderLayer {
    self_attn: Attention,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    ada_rms_norm: AdaRmsNorm,
    mlp: GatedMlp,
}

impl DecoderLayer {
    fn load(cfg: &TextDecoderConfig, vb: VarBuilder) -> Result<Self> {
        // Decoder attention: no bias on any projection.
        let self_attn = Attention::load(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            false, false, false, false,
            vb.pp("self_attn"),
        )?;
        let input_norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attn_norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let ada_rms_norm = AdaRmsNorm::load(cfg.hidden_size, vb.pp("ada_rms_norm"))?;
        let mlp = GatedMlp::load(cfg.hidden_size, cfg.intermediate_size, false, vb.pp("mlp"))?;
        Ok(Self { self_attn, input_norm, post_attn_norm, ada_rms_norm, mlp })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        sliding_window: Option<usize>,
        t_cond: &Tensor,
    ) -> Result<Tensor> {
        // Pre-norm self-attention.
        let residual = x.clone();
        let h = self.input_norm.forward(x)?;
        let h = self.self_attn.forward(&h, rope, offset, sliding_window)?;
        let x = (residual + h)?;
        // Pre-norm MLP with adaptive conditioning.
        let residual = x.clone();
        let h = self.post_attn_norm.forward(&x)?;
        let ada_scale = self.ada_rms_norm.forward(t_cond)?;
        let h = (h * (ada_scale + 1.0)?)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }

    fn reset_cache(&mut self) {
        self.self_attn.reset_cache();
    }
}

/// Voxtral Realtime text decoder (causal LM).
pub struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head_tied: bool,
    lm_head: Option<candle_nn::Linear>,
    rope: RotaryEmbedding,
    sliding_window: Option<usize>,
    vocab_size: usize,
}

impl TextDecoder {
    pub fn load(cfg: &TextDecoderConfig, vb: VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, model_vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(cfg, model_vb.pp(format!("layers.{i}")))?);
        }
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, model_vb.pp("norm"))?;
        let rope = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings.min(8192), // pre-allocate reasonable amount
            cfg.rope_parameters.rope_theta,
            vb.device(),
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            let w = vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?;
            Some(candle_nn::Linear::new(w, None))
        };
        let sw = cfg.sliding_window;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_tied: cfg.tie_word_embeddings,
            lm_head,
            rope,
            sliding_window: sw,
            vocab_size: cfg.vocab_size,
        })
    }

    /// Forward pass on token IDs, returns logits [B, seq_len, vocab_size].
    pub fn forward_ids(
        &mut self,
        ids: &Tensor,
        offset: usize,
        t_cond: &Tensor,
    ) -> Result<Tensor> {
        let embeds = self.embed_tokens.forward(ids)?;
        self.forward_embeds(&embeds, offset, t_cond)
    }

    /// Forward pass on pre-built embeddings, returns logits.
    pub fn forward_embeds(
        &mut self,
        embeds: &Tensor,
        offset: usize,
        t_cond: &Tensor,
    ) -> Result<Tensor> {
        let mut x = embeds.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x, &self.rope, offset, self.sliding_window, t_cond)?;
        }
        x = self.norm.forward(&x)?;
        self.lm_head_forward(&x)
    }

    fn lm_head_forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.lm_head {
            Some(lm) => lm.forward(x),
            None => {
                // Tied weights: use embed_tokens weight transposed.
                let w = self.embed_tokens.embeddings();
                let b = x.dim(0)?;
                let w = w.broadcast_left(b)?;
                x.matmul(&w.t()?)
            }
        }
    }

    /// Get token embedding for a given ID.
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }
}
