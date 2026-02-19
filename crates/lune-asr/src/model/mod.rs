pub mod layers;
pub mod encoder;
pub mod decoder;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::config::VoxtralRealtimeConfig;
use encoder::AudioEncoder;
use decoder::TextDecoder;
use layers::TimeEmbedding;

/// Multi-modal projector: downsample 4x then project to LLM hidden size.
#[derive(Debug)]
struct MultiModalProjector {
    linear_1: candle_nn::Linear,
    linear_2: candle_nn::Linear,
}

impl MultiModalProjector {
    fn load(encoder_hidden: usize, downsample: usize, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let in_dim = encoder_hidden * downsample;
        let w1 = vb.get((text_hidden, in_dim), "linear_1.weight")?;
        let w2 = vb.get((text_hidden, text_hidden), "linear_2.weight")?;
        Ok(Self {
            linear_1: candle_nn::Linear::new(w1, None),
            linear_2: candle_nn::Linear::new(w2, None),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear_1.forward(x)?.gelu()?;
        self.linear_2.forward(&h)
    }
}

/// Top-level Voxtral Realtime model: encoder + projector + decoder.
pub struct VoxtralRealtime {
    pub encoder: AudioEncoder,
    projector: MultiModalProjector,
    pub decoder: TextDecoder,
    time_emb: TimeEmbedding,
    downsample_factor: usize,
    encoder_hidden: usize,
}

impl VoxtralRealtime {
    /// Load model from safetensors via VarBuilder.
    pub fn load(cfg: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = AudioEncoder::load(&cfg.audio_config, vb.pp("audio_tower"))?;
        let projector = MultiModalProjector::load(
            cfg.audio_config.hidden_size,
            cfg.downsample_factor,
            cfg.text_config.hidden_size,
            vb.pp("multi_modal_projector"),
        )?;
        let decoder = TextDecoder::load(&cfg.text_config, vb.pp("language_model"))?;
        let time_emb = TimeEmbedding::new(
            cfg.text_config.hidden_size,
            10000.0,
            vb.device(),
        )?;
        Ok(Self {
            encoder,
            projector,
            decoder,
            time_emb,
            downsample_factor: cfg.downsample_factor,
            encoder_hidden: cfg.audio_config.hidden_size,
        })
    }

    /// Encode audio mel features → projected audio embeddings ready for the LLM.
    pub fn encode_audio(&mut self, mel: &Tensor) -> Result<Tensor> {
        let hidden = self.encoder.forward(mel, 0)?;
        // Downsample by concatenating consecutive frames.
        let (b, t, h) = hidden.dims3()?;
        let t_ds = t / self.downsample_factor;
        let hidden = hidden.narrow(1, 0, t_ds * self.downsample_factor)?;
        let hidden = hidden.reshape((b, t_ds, h * self.downsample_factor))?;
        self.projector.forward(&hidden)
    }

    /// Get time conditioning vector for the decoder.
    pub fn time_conditioning(&self, delay_tokens: usize, device: &Device) -> Result<Tensor> {
        self.time_emb.forward(delay_tokens as f32, device)
    }

    pub fn reset_cache(&mut self) {
        self.encoder.reset_cache();
        self.decoder.reset_cache();
    }
}
