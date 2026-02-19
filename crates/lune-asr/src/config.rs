use serde::Deserialize;

/// Top-level config for Voxtral-Mini-4B-Realtime-2602.
/// Mirrors the HuggingFace `config.json` structure.
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeConfig {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextDecoderConfig,
    pub projector_hidden_act: String,
    pub downsample_factor: usize,
    pub audio_length_per_tok: usize,
    pub default_num_delay_tokens: usize,
    pub hidden_size: usize,
    #[serde(default = "default_vocab")]
    pub vocab_size: usize,
}

fn default_vocab() -> usize {
    131072
}

/// Audio encoder config (voxtral_realtime_encoder).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_mel_bins: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_sliding_window_enc")]
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub activation_function: String,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub attention_dropout: f64,
}

fn default_sliding_window_enc() -> usize {
    750
}

/// Text decoder config (voxtral_realtime_text).
#[derive(Debug, Clone, Deserialize)]
pub struct TextDecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_sliding_window_dec")]
    pub sliding_window: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub hidden_act: String,
    pub vocab_size: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_dropout: f64,
}

fn default_sliding_window_dec() -> Option<usize> {
    Some(8192)
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_rope_type() -> String {
    "default".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_json() {
        let json = r#"{
            "audio_config": {
                "activation_function": "gelu",
                "attention_dropout": 0.0,
                "head_dim": 64,
                "hidden_act": "silu",
                "hidden_size": 1280,
                "initializer_range": 0.02,
                "intermediate_size": 5120,
                "max_position_embeddings": 1500,
                "model_type": "voxtral_realtime_encoder",
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "num_key_value_heads": 32,
                "num_mel_bins": 128,
                "rms_norm_eps": 1e-05,
                "rope_parameters": { "rope_theta": 1000000.0, "rope_type": "default" },
                "sliding_window": 750,
                "vocab_size": 131072
            },
            "audio_length_per_tok": 8,
            "default_num_delay_tokens": 6,
            "downsample_factor": 4,
            "hidden_size": 3072,
            "model_type": "voxtral_realtime",
            "projector_hidden_act": "gelu",
            "text_config": {
                "attention_dropout": 0.0,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 3072,
                "intermediate_size": 9216,
                "max_position_embeddings": 131072,
                "model_type": "voxtral_realtime_text",
                "num_attention_heads": 32,
                "num_hidden_layers": 26,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-05,
                "rope_parameters": { "rope_theta": 1000000.0, "rope_type": "default" },
                "sliding_window": 8192,
                "tie_word_embeddings": true,
                "use_cache": true,
                "vocab_size": 131072
            }
        }"#;
        let cfg: VoxtralRealtimeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.audio_config.hidden_size, 1280);
        assert_eq!(cfg.audio_config.num_hidden_layers, 32);
        assert_eq!(cfg.audio_config.num_mel_bins, 128);
        assert_eq!(cfg.audio_config.sliding_window, 750);
        assert_eq!(cfg.text_config.hidden_size, 3072);
        assert_eq!(cfg.text_config.num_hidden_layers, 26);
        assert_eq!(cfg.text_config.num_key_value_heads, 8);
        assert_eq!(cfg.downsample_factor, 4);
    }
}
