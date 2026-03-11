//! ORT-based Qwen ASR backend.
//!
//! Uses three ONNX Runtime sessions (encoder, embed_tokens, decoder) for
//! batch transcription. Audio accumulates via `push_audio`; transcription
//! happens in a single `transcribe_oneshot` call on `poll_final`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ort::session::Session;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig, WordConfidence};
use lune_core::error::{Error, Result};

use crate::ort_forced_aligner::{FaConfig, OrtForcedAligner};
use crate::ort_qwen_stream;
use crate::ort_utils::build_ort_session;

// ── Configuration ───────────────────────────────────────────────────────────

/// Maximum audio the encoder will process in a single pass.
/// The ONNX model supports dynamic mel_frames up to 6000 (=60s).
/// The encoder uses windowed attention internally (8s windows) —
/// this is NOT a segmentation boundary. The decoder sees all
/// encoder outputs via cross-attention for global coherence.
const MAX_ENCODER_SAMPLES: usize = 60 * 16_000;

/// Decoder stays grounded within 8s (matches encoder's internal attention window).
const CHUNK_SAMPLES: usize = 8 * 16_000;
/// Only chunk audio longer than this (oneshot is fine for short clips).
const CHUNK_THRESHOLD: usize = 10 * 16_000;

/// Runtime configuration for the ORT Qwen ASR backend.
#[derive(Debug, Clone)]
pub struct OrtQwenAsrConfig {
    /// Directory containing qwen3-asr-encoder.onnx, qwen3-asr-embed-tokens.onnx, qwen3-asr-decoder.onnx, tokenizer.json.
    pub model_dir: PathBuf,
    /// Whether to use GPU (CUDA EP registered globally via init_ort_runtime).
    pub use_gpu: bool,
}

impl Default for OrtQwenAsrConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            use_gpu: false,
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
    /// Whether the decoder ONNX model has an `attention_mask` input.
    has_attention_mask: bool,
    /// Optional forced aligner for timestamp-based chunk merging.
    /// Attached after construction via `attach_aligner()`.
    aligner: OnceLock<Arc<OrtForcedAligner>>,
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

        // Detect whether decoder accepts attention_mask.
        let has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");

        // Warmup: prime CUDA kernels for all three sessions.
        warmup_asr(
            &mut encoder,
            &mut embed_tokens,
            &mut decoder,
            has_attention_mask,
        )?;

        tracing::info!(
            model_dir = %dir.display(),
            "initialized ORT Qwen ASR backend (batch mode)"
        );

        Ok(Self {
            encoder: Arc::new(Mutex::new(encoder)),
            embed_tokens: Arc::new(Mutex::new(embed_tokens)),
            decoder: Mutex::new(decoder),
            tokenizer,
            config,
            has_attention_mask,
            aligner: OnceLock::new(),
        })
    }
}

// ── Session state ───────────────────────────────────────────────────────────

struct SessionState {
    audio_buf: Vec<f32>,
    audio_start: usize,
    next_id: u64,
    sample_rate: u32,
    forced_language: Option<String>,
}

impl SessionState {
    #[inline]
    fn audio(&self) -> &[f32] {
        &self.audio_buf[self.audio_start..]
    }

    #[inline]
    fn audio_len(&self) -> usize {
        self.audio_buf.len().saturating_sub(self.audio_start)
    }

    fn append_audio(&mut self, pcm: &[f32]) {
        self.audio_buf.extend_from_slice(pcm);

        let trim = self.audio_len().saturating_sub(MAX_ENCODER_SAMPLES);
        if trim == 0 {
            return;
        }

        self.audio_start += trim;

        // Compact only when the dead prefix is large enough to pay for the copy.
        if self.audio_start >= self.audio_buf.capacity() / 4 || self.audio_start >= self.audio_len()
        {
            self.compact();
        }
    }

    fn clear(&mut self) {
        self.audio_buf.clear();
        self.audio_start = 0;
    }

    fn compact(&mut self) {
        if self.audio_start == 0 {
            return;
        }
        let len = self.audio_len();
        self.audio_buf.copy_within(self.audio_start.., 0);
        self.audio_buf.truncate(len);
        self.audio_start = 0;
    }
}

pub struct OrtQwenAsrSession {
    state: Mutex<SessionState>,
}

// ── Language mapping ────────────────────────────────────────────────────────

#[inline]
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
    has_attention_mask: bool,
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

    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(4 + NUM_DECODER_LAYERS * 2);
    inputs.push((
        "inputs_embeds".into(),
        Value::from_array(embeds).unwrap().into(),
    ));
    inputs.push((
        "position_ids".into(),
        Value::from_array(pos).unwrap().into(),
    ));
    // attention_mask: [1, past_len + seq_len] = [1, 0 + 1] = [1, 1]
    // Only add if the model supports it (new export with attention_mask input).
    if has_attention_mask {
        let attn_mask = Array2::from_shape_vec((1, 1), vec![1i64]).unwrap();
        inputs.push((
            "attention_mask".into(),
            Value::from_array(attn_mask).unwrap().into(),
        ));
    }
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

/// Average confidence from log-probs: exp(mean(log_probs)).
fn avg_confidence(log_probs: &[f32]) -> Option<f32> {
    if log_probs.is_empty() {
        return None;
    }
    Some((log_probs.iter().sum::<f32>() / log_probs.len() as f32).exp())
}

/// Minimum confidence from log-probs: exp(min(log_probs)).
fn min_confidence(log_probs: &[f32]) -> Option<f32> {
    log_probs
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .map(|lp| lp.exp())
}

// ── Per-word confidence ─────────────────────────────────────────────────────

/// Aggregate BPE subword tokens into per-word confidence values.
///
/// Word boundary = raw BPE token starts with Ġ (U+0120) or is the first token.
/// Per-word confidence = minimum exp(log_prob) across constituent subwords
/// (conservative: one uncertain subword flags the whole word).
fn tokens_to_word_confidences(
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[i64],
    log_probs: &[f32],
) -> Vec<WordConfidence> {
    debug_assert_eq!(token_ids.len(), log_probs.len());
    if token_ids.is_empty() {
        return vec![];
    }

    let mut words: Vec<WordConfidence> = Vec::new();
    let mut current_word = String::new();
    let mut current_min_prob = f32::INFINITY;

    for (i, (&tok_id, &lp)) in token_ids.iter().zip(log_probs.iter()).enumerate() {
        let raw_token = match tokenizer.id_to_token(tok_id as u32) {
            Some(t) => t,
            None => continue,
        };

        // Word boundary: first token or starts with Ġ (GPT-2 BPE word-initial marker).
        let is_boundary = i == 0 || raw_token.starts_with('Ġ');

        if is_boundary && !current_word.is_empty() {
            // Push previous word.
            words.push(WordConfidence {
                word: std::mem::take(&mut current_word),
                confidence: current_min_prob.exp(),
            });
            current_min_prob = f32::INFINITY;
        }

        // Decode the token to get clean text (strips Ġ prefix).
        let decoded = tokenizer.decode(&[tok_id as u32], true).unwrap_or_default();
        let trimmed = decoded.trim();
        if trimmed.is_empty() {
            continue;
        }

        current_word.push_str(trimmed);
        current_min_prob = current_min_prob.min(lp);
    }

    // Push final word.
    if !current_word.is_empty() {
        words.push(WordConfidence {
            word: current_word,
            confidence: current_min_prob.exp(),
        });
    }

    words
}

// ── Chunked inference ───────────────────────────────────────────────────────

impl OrtQwenAsrBackend {
    /// Attach a forced aligner for timestamp-based chunk merging.
    /// Called after construction when the FA model is available.
    pub fn attach_aligner(&self, fa: Arc<OrtForcedAligner>) {
        let _ = self.aligner.set(fa);
    }

    /// Split long audio into overlapping 8s chunks, decode each independently,
    /// and merge using FA timestamps (preferred) or LCS fallback.
    ///
    /// Locks encoder/embed_tokens/decoder per-chunk so FA can re-acquire
    /// the same Arc<Mutex> sessions between chunks.
    fn transcribe_chunked(
        &self,
        pcm: &[f32],
        forced_language: Option<&str>,
    ) -> Result<(String, Vec<WordConfidence>)> {
        let mut all_text = String::new();
        let mut all_word_confs: Vec<WordConfidence> = Vec::new();
        // Accumulated token IDs from previous chunks — gives the decoder
        // linguistic context for continuation (what was already transcribed).
        let mut past_text_tokens: Vec<i64> = Vec::new();

        let mut chunk_idx = 0usize;
        let mut offset = 0usize;
        while offset < pcm.len() {
            // No overlap: past_text_tokens provides decoder context, so each
            // chunk covers a unique audio region. Step = chunk size.
            let end = (offset + CHUNK_SAMPLES).min(pcm.len());
            let chunk = &pcm[offset..end];

            let result = {
                let mut enc = self
                    .encoder
                    .lock()
                    .map_err(|e| lock_err("encoder lock", e))?;
                let mut emb = self
                    .embed_tokens
                    .lock()
                    .map_err(|e| lock_err("embed_tokens lock", e))?;
                let mut dec = self
                    .decoder
                    .lock()
                    .map_err(|e| lock_err("decoder lock", e))?;
                ort_qwen_stream::transcribe_oneshot(
                    &mut enc,
                    &mut emb,
                    &mut dec,
                    &self.tokenizer,
                    chunk,
                    forced_language,
                    self.has_attention_mask,
                    &past_text_tokens,
                )?
            }; // locks dropped here

            let text = result.text.trim().to_owned();
            tracing::info!(
                chunk = chunk_idx,
                offset_s = offset as f32 / 16_000.0,
                end_s = end as f32 / 16_000.0,
                tokens = result.token_ids.len(),
                text = %text,
                "chunked ASR decode"
            );

            if !text.is_empty() {
                // Accumulate tokens for next chunk's past_text conditioning.
                past_text_tokens.extend_from_slice(&result.token_ids);

                // Simple concatenation — each chunk produces only new text.
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(&text);

                let wc = tokens_to_word_confidences(
                    &self.tokenizer,
                    &result.token_ids,
                    &result.log_probs,
                );
                all_word_confs.extend(wc);
            }

            chunk_idx += 1;
            offset = end; // step = chunk size (no overlap)
        }

        tracing::info!(
            chunks = chunk_idx,
            total_words = all_text.split_whitespace().count(),
            text = %all_text,
            "chunked ASR merged"
        );

        Ok((all_text, all_word_confs))
    }
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

        let forced_language = match config
            .language
            .as_deref()
            .map(str::trim)
            .filter(|language| !language.is_empty())
        {
            Some(language) => {
                let mapped = map_language_tag(language).ok_or_else(|| {
                    Error::Asr(format!(
                        "ORT Qwen ASR does not support forced language hint `{language}`"
                    ))
                })?;
                Some(format!("Transcribe the {mapped} audio into text."))
            }
            None => None,
        };

        Ok(OrtQwenAsrSession {
            state: Mutex::new(SessionState {
                audio_buf: Vec::with_capacity(16_000 * 8),
                audio_start: 0,
                next_id: 1,
                sample_rate: config.sample_rate,
                forced_language,
            }),
        })
    }

    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;
        state.append_audio(pcm);
        Ok(())
    }

    async fn poll_results(&self, _session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        // Batch mode: accumulate audio, defer all transcription to poll_final.
        Ok(vec![])
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;

        if state.audio_len() == 0 {
            tracing::info!("poll_final: audio buffer empty, no segments");
            return Ok(vec![]);
        }

        let id = state.next_id;
        state.next_id = id.saturating_add(1);
        let audio = state.audio();
        let total = audio.len();
        let sample_rate = state.sample_rate;

        let (text, word_confidences, confidence, min_token_confidence) = if total > CHUNK_THRESHOLD
        {
            // Chunked path: locks per-chunk internally, returns merged word confidences.
            let (text, wc) = self.transcribe_chunked(audio, state.forced_language.as_deref())?;
            // Derive segment-level confidence from per-word confidences.
            let conf = if wc.is_empty() {
                None
            } else {
                let sum: f32 = wc.iter().map(|w| w.confidence).sum();
                Some(sum / wc.len() as f32)
            };
            let min_conf = wc
                .iter()
                .map(|w| w.confidence)
                .min_by(|a, b| a.partial_cmp(b).unwrap());
            (text, wc, conf, min_conf)
        } else {
            // Short audio: single-pass oneshot.
            let result = {
                let mut enc = self
                    .encoder
                    .lock()
                    .map_err(|e| lock_err("encoder lock", e))?;
                let mut emb = self
                    .embed_tokens
                    .lock()
                    .map_err(|e| lock_err("embed_tokens lock", e))?;
                let mut dec = self
                    .decoder
                    .lock()
                    .map_err(|e| lock_err("decoder lock", e))?;
                ort_qwen_stream::transcribe_oneshot(
                    &mut enc,
                    &mut emb,
                    &mut dec,
                    &self.tokenizer,
                    audio,
                    state.forced_language.as_deref(),
                    self.has_attention_mask,
                    &[],
                )?
            };
            let wc =
                tokens_to_word_confidences(&self.tokenizer, &result.token_ids, &result.log_probs);
            let conf = avg_confidence(&result.log_probs);
            let min_conf = min_confidence(&result.log_probs);
            (result.text, wc, conf, min_conf)
        };

        let text = text.trim().to_owned();
        tracing::info!(
            audio_samples = total,
            audio_secs = total as f32 / 16_000.0,
            text_len = text.len(),
            text = %text,
            "poll_final: transcription complete"
        );
        if text.is_empty() {
            return Ok(vec![]);
        }

        let end_ms = (total as u64).saturating_mul(1000) / sample_rate as u64;

        Ok(vec![AsrSegment {
            id,
            confidence,
            min_token_confidence,
            word_confidences,
            is_final: true,
            start_ms: Some(0),
            end_ms: Some(end_ms),
            text,
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
        state.clear();
        Ok(())
    }

    fn segment_capacity_hint(&self) -> usize {
        1
    }
}

impl OrtQwenAsrBackend {
    /// Build an `OrtForcedAligner` using a wav2vec2 CTC model.
    ///
    /// `fa_model_dir` must contain:
    ///   - `model.onnx` (+ `model.onnx.data`) — wav2vec2 CTC ONNX model
    ///   - `fa_config.json` — labels, blank_id, stride_ms, etc.
    pub fn make_forced_aligner(&self, fa_model_dir: &Path) -> Result<OrtForcedAligner> {
        let model_path = fa_model_dir.join("model.onnx");
        let config_path = fa_model_dir.join("fa_config.json");

        let fa_model = build_ort_session(&model_path, self.config.use_gpu)?;

        let cfg_bytes = std::fs::read(&config_path)
            .map_err(|e| Error::Asr(format!("fa_config.json read: {e}")))?;
        let fa_config: FaConfig = serde_json::from_slice(&cfg_bytes)
            .map_err(|e| Error::Asr(format!("fa_config.json parse: {e}")))?;

        OrtForcedAligner::new(fa_model, fa_config)
    }
}
