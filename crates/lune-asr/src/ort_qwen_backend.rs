//! ORT-based Qwen ASR backend.
//!
//! Uses three ONNX Runtime sessions (encoder, embed_tokens, decoder) to
//! replicate the streaming ASR pipeline from the `qwen-asr` crate.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::session::Session;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::error::{Error, Result};

use crate::ort_forced_aligner::{FaConfig, OrtForcedAligner};
use crate::ort_qwen_stream::{self, OrtStreamState, StreamConfig};
use crate::ort_utils::build_ort_session;

// ── Configuration ───────────────────────────────────────────────────────────

/// Runtime configuration for the ORT Qwen ASR backend.
#[derive(Debug, Clone)]
pub struct OrtQwenAsrConfig {
    /// Directory containing qwen3-asr-encoder.onnx, qwen3-asr-embed-tokens.onnx, qwen3-asr-decoder.onnx, tokenizer.json.
    pub model_dir: PathBuf,
    /// Whether to use GPU (CUDA EP registered globally via init_ort_runtime).
    pub use_gpu: bool,
    /// Streaming chunk size in seconds.
    pub stream_chunk_sec: f32,
    /// Number of unstable tail tokens kept for rollback.
    pub stream_rollback: i32,
    /// Number of chunks kept "unfixed" before committing text.
    pub stream_unfixed_chunks: i32,
    /// Decoder cap per chunk.
    pub stream_max_new_tokens: i32,
    /// Whether to condition each chunk on previously committed text.
    pub past_text_conditioning: bool,
    /// Maximum audio window (seconds) passed to the encoder per chunk.
    /// Shorter windows keep encoder features short, which matters because the
    /// model was fine-tuned on short clips — long encoder sequences push it
    /// out-of-distribution and cause second-half hallucinations.
    pub max_encoder_sec: f32,
}

impl Default for OrtQwenAsrConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            use_gpu: false,
            // 1.0s chunks give the decoder more audio context per step, reducing
            // early-EOS probability on unfamiliar phrases (e.g. "AN OBJECT OF WORSHIP").
            stream_chunk_sec: 1.0,
            // rollback=3: allows 4-token decodes to advance stable cursor.
            // rollback=5 caused stalls whenever the decoder generated exactly 5 tokens.
            stream_rollback: 3,
            stream_unfixed_chunks: 2,
            stream_max_new_tokens: 48,
            past_text_conditioning: true,
            // 8s window: model is trained on short clips; longer windows cause
            // second-half hallucinations despite correct first-half transcription.
            max_encoder_sec: 8.0,
        }
    }
}

// ── Backend ─────────────────────────────────────────────────────────────────

/// ORT-based Qwen ASR backend using three ONNX sessions.
pub struct OrtQwenAsrBackend {
    encoder: Arc<Mutex<Session>>,
    embed_tokens: Arc<Mutex<Session>>,
    decoder: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    config: OrtQwenAsrConfig,
}

impl OrtQwenAsrBackend {
    pub fn new(config: OrtQwenAsrConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir));
        }

        let use_gpu = config.use_gpu;
        let dir = &config.model_dir;

        let encoder_path = dir.join("qwen3-asr-encoder.onnx");
        let embed_path = dir.join("qwen3-asr-embed-tokens.onnx");
        let decoder_path = dir.join("qwen3-asr-decoder.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        let mut encoder = build_ort_session(&encoder_path, use_gpu)?;
        let mut embed_tokens = build_ort_session(&embed_path, use_gpu)?;
        let mut decoder = build_ort_session(&decoder_path, use_gpu)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Asr(format!("tokenizer load: {e}")))?;

        // Warmup: prime CUDA kernels for all three sessions.
        warmup_asr(&mut encoder, &mut embed_tokens, &mut decoder)?;

        tracing::info!(
            model_dir = %dir.display(),
            chunk_sec = config.stream_chunk_sec,
            rollback = config.stream_rollback,
            unfixed_chunks = config.stream_unfixed_chunks,
            max_new_tokens = config.stream_max_new_tokens,
            past_text = config.past_text_conditioning,
            "initialized ORT Qwen ASR backend"
        );

        Ok(Self {
            encoder: Arc::new(Mutex::new(encoder)),
            embed_tokens: Arc::new(Mutex::new(embed_tokens)),
            decoder: Mutex::new(decoder),
            tokenizer,
            config,
        })
    }
}

// ── Session state ───────────────────────────────────────────────────────────

struct SessionState {
    config: AsrSessionConfig,
    audio_buf: Vec<f32>,
    stream: OrtStreamState,
    stream_config: StreamConfig,
    next_id: u64,
}

pub struct OrtQwenAsrSession {
    state: Mutex<SessionState>,
}

// ── Language mapping ────────────────────────────────────────────────────────

#[inline]
#[allow(dead_code)] // Will be used when language forcing is wired up.
fn map_language_tag(language: &str) -> Option<&'static str> {
    let lower = language.trim().to_ascii_lowercase();
    let base = lower.split(['-', '_']).next().unwrap_or_default();
    match base {
        "zh" => Some("Chinese"),
        "en" => Some("English"),
        "yue" => Some("Cantonese"),
        "ar" => Some("Arabic"),
        "de" => Some("German"),
        "fr" => Some("French"),
        "es" => Some("Spanish"),
        "pt" => Some("Portuguese"),
        "id" => Some("Indonesian"),
        "it" => Some("Italian"),
        "ko" => Some("Korean"),
        "ru" => Some("Russian"),
        "th" => Some("Thai"),
        "vi" => Some("Vietnamese"),
        "ja" => Some("Japanese"),
        "tr" => Some("Turkish"),
        "hi" => Some("Hindi"),
        "ms" => Some("Malay"),
        "nl" => Some("Dutch"),
        "sv" => Some("Swedish"),
        "da" => Some("Danish"),
        "fi" => Some("Finnish"),
        "pl" => Some("Polish"),
        "cs" => Some("Czech"),
        "fil" | "tl" => Some("Filipino"),
        "fa" => Some("Persian"),
        "el" => Some("Greek"),
        "ro" => Some("Romanian"),
        "hu" => Some("Hungarian"),
        "mk" => Some("Macedonian"),
        _ => None,
    }
}

// ── Warmup ──────────────────────────────────────────────────────────────────

/// Run minimal dummy inference through each session to prime CUDA kernels.
#[cold]
fn warmup_asr(
    encoder: &mut Session,
    embed_tokens: &mut Session,
    decoder: &mut Session,
) -> Result<()> {
    use ndarray::{Array2, Array3, ArrayD, IxDyn};
    use ort::value::{DynValue, Value};
    use ort_qwen_stream::{HEAD_DIM, HIDDEN_DIM, NUM_DECODER_LAYERS, NUM_KV_HEADS};

    // 1) Encoder: tiny mel [1, 128, 2] — model expects f16.
    let mel = Array3::from_elem((1, 128, 2), half::f16::ZERO);
    let mel_val: DynValue = Value::from_array(mel.into_dyn())
        .map_err(|e| Error::Asr(format!("warmup mel: {e}")))?
        .into();
    let _enc = encoder
        .run(vec![("mel".to_string(), mel_val)])
        .map_err(|e| Error::Asr(format!("warmup encoder: {e}")))?;

    // 2) Embed_tokens: single token
    let ids = Array2::from_shape_vec((1, 1), vec![0i64]).unwrap();
    let ids_val: DynValue = Value::from_array(ids)
        .map_err(|e| Error::Asr(format!("warmup ids: {e}")))?
        .into();
    let _emb = embed_tokens
        .run(vec![("input_ids".to_string(), ids_val)])
        .map_err(|e| Error::Asr(format!("warmup embed_tokens: {e}")))?;

    // 3) Decoder: single-token input_embeds with empty KV cache — model expects f16.
    let embeds = ArrayD::from_elem(IxDyn(&[1, 1, HIDDEN_DIM]), half::f16::ZERO);
    let pos = Array3::from_shape_vec((3, 1, 1), vec![0i64, 0, 0]).unwrap();
    let empty_kv = ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), half::f16::ZERO);

    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_DECODER_LAYERS * 2);
    inputs.push((
        "inputs_embeds".into(),
        Value::from_array(embeds).unwrap().into(),
    ));
    inputs.push((
        "position_ids".into(),
        Value::from_array(pos).unwrap().into(),
    ));
    for layer in 0..NUM_DECODER_LAYERS {
        for kind in &["key", "value"] {
            inputs.push((
                format!("past_key_values.{layer}.{kind}"),
                Value::from_array(empty_kv.clone()).unwrap().into(),
            ));
        }
    }
    let _dec = decoder
        .run(inputs)
        .map_err(|e| Error::Asr(format!("warmup decoder: {e}")))?;

    tracing::debug!("ASR warmup complete");
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[cold]
#[inline(never)]
fn lock_err(context: &'static str, e: impl std::fmt::Display) -> Error {
    Error::Asr(format!("{context}: {e}"))
}

// ── AsrBackend implementation ───────────────────────────────────────────────

#[async_trait::async_trait]
impl AsrBackend for OrtQwenAsrBackend {
    type SessionHandle = OrtQwenAsrSession;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        if config.sample_rate != 16_000 {
            return Err(Error::Asr(format!(
                "ORT Qwen ASR expects 16000 Hz mono PCM, got {} Hz",
                config.sample_rate
            )));
        }

        let chunk_samples = (self.config.stream_chunk_sec * config.sample_rate as f32) as usize;

        let max_encoder_samples = if self.config.max_encoder_sec > 0.0 {
            Some((self.config.max_encoder_sec * config.sample_rate as f32) as usize)
        } else {
            None
        };
        let stream_config = StreamConfig {
            chunk_samples,
            rollback: self.config.stream_rollback.max(0) as usize,
            unfixed_chunks: self.config.stream_unfixed_chunks,
            max_new_tokens: self.config.stream_max_new_tokens.max(1) as usize,
            past_text_conditioning: self.config.past_text_conditioning,
            max_encoder_samples,
        };

        Ok(OrtQwenAsrSession {
            state: Mutex::new(SessionState {
                config,
                audio_buf: Vec::with_capacity(16_000 * 8),
                stream: OrtStreamState::new(),
                stream_config,
                next_id: 1,
            }),
        })
    }

    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;
        state.audio_buf.extend_from_slice(pcm);
        Ok(())
    }

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;

        if state.audio_buf.is_empty() {
            return Ok(vec![]);
        }

        let mut encoder = self
            .encoder
            .lock()
            .map_err(|e| lock_err("encoder lock", e))?;
        let mut embed_tokens = self
            .embed_tokens
            .lock()
            .map_err(|e| lock_err("embed_tokens lock", e))?;
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|e| lock_err("decoder lock", e))?;

        let sample_rate = state.config.sample_rate;
        let SessionState {
            ref audio_buf,
            ref mut stream,
            ref stream_config,
            ..
        } = *state;

        let start_samples = stream.audio_cursor();
        let prev_stable_len = stream.stable_text_tokens.len();

        let text = ort_qwen_stream::stream_push_audio(
            &mut encoder,
            &mut embed_tokens,
            &mut decoder,
            &self.tokenizer,
            audio_buf,
            stream,
            stream_config,
            false,
        )?;

        let confidence = stream.confidence_since(prev_stable_len);
        let min_token_confidence = stream.min_confidence_since(prev_stable_len);
        let end_samples = stream.audio_cursor();
        let id = state.next_id;
        state.next_id = id.saturating_add(1);

        let text = text.trim();
        if text.is_empty() {
            return Ok(vec![]);
        }

        let start_ms = (start_samples as u64).saturating_mul(1000) / sample_rate as u64;
        let end_ms = (end_samples as u64).saturating_mul(1000) / sample_rate as u64;

        Ok(vec![AsrSegment {
            id,
            confidence,
            min_token_confidence,
            is_final: true,
            start_ms: Some(start_ms),
            end_ms: Some(end_ms),
            text: text.to_owned(),
        }])
    }

    async fn end_session(&self, _session: Self::SessionHandle) -> Result<()> {
        Ok(())
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;
        state.audio_buf.clear();
        state.stream.reset();
        Ok(())
    }

    fn segment_capacity_hint(&self) -> usize {
        1
    }
}

impl OrtQwenAsrBackend {
    /// Build an `OrtForcedAligner` that reuses this backend's encoder and
    /// embed_tokens sessions (shared via `Arc`). Only the FA decoder is loaded
    /// from `fa_model_dir`, which must contain:
    ///   - `qwen3-fa-decoder.onnx` (+ `qwen3-fa-decoder.onnx_data`)
    ///   - `fa_config.json` (`{ timestamp_token_id, timestamp_segment_ms, timestamp_classes }`)
    pub fn make_forced_aligner(&self, fa_model_dir: &Path) -> Result<OrtForcedAligner> {
        let decoder_path = fa_model_dir.join("qwen3-fa-decoder.onnx");
        let config_path = fa_model_dir.join("fa_config.json");

        let fa_decoder = build_ort_session(&decoder_path, self.config.use_gpu)?;

        let cfg_bytes = std::fs::read(&config_path)
            .map_err(|e| Error::Asr(format!("fa_config.json read: {e}")))?;
        let fa_config: FaConfig = serde_json::from_slice(&cfg_bytes)
            .map_err(|e| Error::Asr(format!("fa_config.json parse: {e}")))?;

        OrtForcedAligner::new(
            Arc::clone(&self.encoder),
            Arc::clone(&self.embed_tokens),
            fa_decoder,
            self.tokenizer.clone(),
            fa_config,
        )
    }
}
