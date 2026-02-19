use std::path::PathBuf;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::error::{Error, Result};

use crate::config::VoxtralRealtimeConfig;
use crate::mel::{MelConfig, MelSpectrogram};
use crate::model::VoxtralRealtime;
use crate::segment::SegmentTracker;

/// Configuration for the Voxtral local backend.
pub struct VoxtralConfig {
    /// Directory containing model safetensors, config.json, and tokenizer.
    pub model_dir: PathBuf,
    pub use_gpu: bool,
    /// Sliding context window size in seconds.
    pub context_secs: f32,
    /// Transcription delay in tokens (default 6 = 480ms).
    pub delay_tokens: usize,
}

/// A session managed by VoxtralLocalBackend.
pub struct VoxtralSession {
    mel: MelSpectrogram,
    tracker: SegmentTracker,
    audio_buffer: Vec<f32>,
    max_context_samples: usize,
    sample_rate: u32,
    total_samples: u64,
}

/// Local Voxtral ASR backend using candle for safetensors inference.
///
/// Architecture: mel spectrogram → causal audio encoder → 4x downsample
/// projector → Mistral-like LLM decoder → text tokens.
///
/// Loads Voxtral-Mini-4B-Realtime-2602 weights from safetensors.
pub struct VoxtralLocalBackend {
    model: Mutex<VoxtralRealtime>,
    tokenizer: Tokenizer,
    device: Device,
    config: VoxtralConfig,
    model_config: VoxtralRealtimeConfig,
}

impl VoxtralLocalBackend {
    /// Create backend by loading model from `model_dir`.
    ///
    /// Expected files in `model_dir`:
    /// - `config.json` (model config)
    /// - `model*.safetensors` (model weights)
    /// - `tekken.json` or `tokenizer.json` (tokenizer)
    pub fn new(config: VoxtralConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir.clone()));
        }

        let device = if config.use_gpu {
            Device::cuda_if_available(0).map_err(|e| Error::Asr(format!("CUDA init: {e}")))?
        } else {
            Device::Cpu
        };

        // Load config.json.
        let config_path = config.model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Asr(format!("read config.json: {e}")))?;
        let model_config: VoxtralRealtimeConfig = serde_json::from_str(&config_str)
            .map_err(|e| Error::Asr(format!("parse config.json: {e}")))?;

        // Load safetensors.
        let st_files = find_safetensors(&config.model_dir)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&st_files, DType::BF16, &device)
                .map_err(|e| Error::Asr(format!("load safetensors: {e}")))?
        };

        // Load model.
        let model = VoxtralRealtime::load(&model_config, vb)
            .map_err(|e| Error::Asr(format!("load model: {e}")))?;

        // Load tokenizer.
        let tok_path = find_tokenizer(&config.model_dir)?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Asr(format!("load tokenizer: {e}")))?;

        tracing::info!(
            model_dir = %config.model_dir.display(),
            device = ?device,
            delay_tokens = config.delay_tokens,
            "initialized Voxtral Realtime backend"
        );

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            config,
            model_config,
        })
    }

    /// Create backend without loading weights (for testing pipeline infrastructure).
    pub fn new_without_model(use_gpu: bool) -> Result<Self> {
        let device = if use_gpu {
            Device::cuda_if_available(0).map_err(|e| Error::Asr(format!("CUDA init: {e}")))?
        } else {
            Device::Cpu
        };

        // Minimal config for testing.
        let json = include_str!("default_config.json");
        let model_config: VoxtralRealtimeConfig = serde_json::from_str(json)
            .map_err(|e| Error::Asr(format!("parse default config: {e}")))?;

        // Build model with zero-initialized weights.
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = VoxtralRealtime::load(&model_config, vb)
            .map_err(|e| Error::Asr(format!("load zero model: {e}")))?;

        let tokenizer = Tokenizer::from_bytes(MINIMAL_TOKENIZER)
            .map_err(|e| Error::Asr(format!("minimal tokenizer: {e}")))?;

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            config: VoxtralConfig {
                model_dir: PathBuf::new(),
                use_gpu,
                context_secs: 30.0,
                delay_tokens: 6,
            },
            model_config,
        })
    }

    /// Run inference on mel features → text.
    fn decode_mel(
        &self,
        model: &mut VoxtralRealtime,
        mel_data: &[f32],
        n_frames: usize,
        n_mels: usize,
    ) -> Result<Option<String>> {
        if n_frames == 0 {
            return Ok(None);
        }

        let mel = Tensor::from_vec(
            mel_data.to_vec(),
            (1, n_mels, n_frames),
            &self.device,
        ).map_err(|e| Error::Asr(format!("mel tensor: {e}")))?;

        let mel = mel.to_dtype(DType::F32)
            .map_err(|e| Error::Asr(format!("mel dtype: {e}")))?;

        model.reset_cache();

        // Encode audio → projected embeddings [1, T', text_hidden].
        let audio_embeds = model.encode_audio(&mel)
            .map_err(|e| Error::Asr(format!("encode audio: {e}")))?;

        // Time conditioning for delay.
        let t_cond = model.time_conditioning(self.config.delay_tokens, &self.device)
            .map_err(|e| Error::Asr(format!("time cond: {e}")))?
            .unsqueeze(0)
            .map_err(|e| Error::Asr(format!("t_cond unsqueeze: {e}")))?
            .unsqueeze(0)
            .map_err(|e| Error::Asr(format!("t_cond unsqueeze2: {e}")))?;

        // Feed audio embeddings into decoder.
        let logits = model.decoder.forward_embeds(&audio_embeds, 0, &t_cond)
            .map_err(|e| Error::Asr(format!("decoder fwd: {e}")))?;

        // Greedy decode all positions.
        let logits = logits.squeeze(0)
            .map_err(|e| Error::Asr(format!("squeeze: {e}")))?;
        let token_ids = logits.argmax(candle_core::D::Minus1)
            .map_err(|e| Error::Asr(format!("argmax: {e}")))?
            .to_vec1::<u32>()
            .map_err(|e| Error::Asr(format!("to_vec1: {e}")))?;

        if token_ids.is_empty() {
            return Ok(None);
        }

        let text = self.tokenizer.decode(&token_ids, true)
            .map_err(|e| Error::Asr(format!("tokenizer decode: {e}")))?;

        let text = text.trim().to_string();
        if text.is_empty() { Ok(None) } else { Ok(Some(text)) }
    }
}

#[async_trait::async_trait]
impl AsrBackend for VoxtralLocalBackend {
    type SessionHandle = Mutex<VoxtralSession>;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        let sample_rate = config.sample_rate;
        let max_context_samples = (self.config.context_secs * sample_rate as f32) as usize;
        let mel_config = MelConfig {
            n_mels: self.model_config.audio_config.num_mel_bins,
            n_fft: 400,
            hop_length: 160,
            sample_rate,
        };
        let session = VoxtralSession {
            mel: MelSpectrogram::new(mel_config),
            tracker: SegmentTracker::new(500),
            audio_buffer: Vec::with_capacity(max_context_samples),
            max_context_samples,
            sample_rate,
            total_samples: 0,
        };
        tracing::info!(sample_rate, "ASR session started");
        Ok(Mutex::new(session))
    }

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut session = session.lock()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;

        session.audio_buffer.extend_from_slice(pcm);
        session.total_samples += pcm.len() as u64;

        // Sliding window.
        if session.audio_buffer.len() > session.max_context_samples {
            let excess = session.audio_buffer.len() - session.max_context_samples;
            session.audio_buffer.drain(..excess);
        }

        let mel_output = session.mel.compute(&session.audio_buffer);
        if mel_output.n_frames == 0 {
            return Ok(());
        }

        let timestamp_ms = session.total_samples * 1000 / session.sample_rate as u64;

        // Note: mel.compute returns data in [n_frames, n_mels] row-major.
        // The encoder expects [B, n_mels, n_frames] (channel-first).
        // We need to transpose the data.
        let mut transposed = vec![0f32; mel_output.n_frames * mel_output.n_mels];
        for f in 0..mel_output.n_frames {
            for m in 0..mel_output.n_mels {
                transposed[m * mel_output.n_frames + f] = mel_output.data[f * mel_output.n_mels + m];
            }
        }

        let mut model = self.model.lock()
            .map_err(|e| Error::Asr(format!("model lock: {e}")))?;

        if let Some(text) = self.decode_mel(
            &mut model,
            &transposed,
            mel_output.n_frames,
            mel_output.n_mels,
        )? {
            if !text.is_empty() {
                session.tracker.push_partial(text, Some(timestamp_ms));
            }
        }

        session.tracker.check_silence_finalization();
        Ok(())
    }

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let mut session = session.lock()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
        Ok(session.tracker.drain())
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        let mut session = session.into_inner()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
        session.tracker.finalize_all();
        let remaining = session.tracker.drain();
        tracing::info!(
            segments = remaining.len(),
            total_samples = session.total_samples,
            "ASR session ended"
        );
        Ok(())
    }
}

fn find_safetensors(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error::Asr(format!("read model dir: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::Asr(format!("no .safetensors files in {}", dir.display())));
    }
    Ok(files)
}

fn find_tokenizer(dir: &std::path::Path) -> Result<PathBuf> {
    for name in &["tekken.json", "tokenizer.json"] {
        let p = dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(Error::Asr(format!("no tokenizer found in {}", dir.display())))
}

/// Minimal tokenizer JSON for testing without real model files.
const MINIMAL_TOKENIZER: &[u8] = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{},"merges":[]}}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_rejects_missing_dir() {
        let config = VoxtralConfig {
            model_dir: PathBuf::from("/nonexistent/model"),
            use_gpu: false,
            context_secs: 30.0,
            delay_tokens: 6,
        };
        let result = VoxtralLocalBackend::new(config);
        assert!(result.is_err());
    }
}
