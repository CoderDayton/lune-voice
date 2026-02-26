//! ORT-based LLM backend for text post-processing.
//!
//! Uses two ONNX sessions exported via `torch.onnx.export`:
//!   - `qwen3-0.6b-embed-tokens.onnx`: input_ids → embeds (FP16)
//!   - `qwen3-0.6b-decoder.onnx`: inputs_embeds + KV cache → logits + KV cache (FP16)
//!
//! All I/O tensors (embeds, logits, KV cache) are native FP16.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ndarray::{Array2, ArrayD, IxDyn};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, Value};
use tokenizers::Tokenizer;

use lune_core::error::{Error, Result};
use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

use crate::decode_utils::{StreamingState, check_stop_sequences, trim_partial_stop_sequences};
use crate::prompt;
use crate::sampling;

/// Configuration for the ORT LLM backend.
#[derive(Debug, Clone)]
pub struct OrtLlmConfig {
    /// Directory containing decoder + embed-tokens ONNX files and `tokenizer.json`.
    pub model_dir: PathBuf,
    /// Use GPU acceleration (requires global `ort::init()` with CUDA EP).
    pub use_gpu: bool,
    /// Context window size in tokens.
    pub n_ctx: u32,
    /// Maximum tokens to generate per request.
    pub max_output_tokens: u32,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-p sampling.
    pub top_p: f32,
    /// Top-k sampling.
    pub top_k: i32,
    /// Repetition penalty multiplier (1.0 = off).
    pub repeat_penalty: f32,
    /// Number of recent tokens to consider for repetition penalty.
    pub repeat_last_n: i32,
}

impl Default for OrtLlmConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            use_gpu: true,
            n_ctx: 2048,
            max_output_tokens: 256,
            temperature: 0.0,
            top_p: 0.8,
            top_k: 20,
            repeat_penalty: 1.15,
            repeat_last_n: 64,
        }
    }
}

/// ORT LLM backend using ONNX Runtime for text post-processing.
pub struct OrtLlmBackend {
    inner: Arc<Mutex<OrtLlmInner>>,
    config: Arc<OrtLlmConfig>,
}

struct OrtLlmInner {
    embed_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
}

// SAFETY: OrtLlmInner is only accessed under Mutex.
unsafe impl Send for OrtLlmInner {}

impl OrtLlmBackend {
    /// Load ONNX models and tokenizer from `model_dir`.
    pub fn new(config: OrtLlmConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir.clone()));
        }

        let embed_path = find_model(&config.model_dir, &EMBED_CANDIDATES)?;
        let decoder_path = find_model(&config.model_dir, &DECODER_CANDIDATES)?;
        let tokenizer_path = config.model_dir.join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(Error::Llm(format!(
                "tokenizer.json not found in {}",
                config.model_dir.display()
            )));
        }

        let mut embed_session = build_session(&embed_path, config.use_gpu)?;
        let mut decoder_session = build_session(&decoder_path, config.use_gpu)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Llm(format!("tokenizer: {e}")))?;

        warmup_decoder(&mut decoder_session, &mut embed_session)?;

        tracing::info!(
            embed = %embed_path.display(),
            decoder = %decoder_path.display(),
            n_ctx = config.n_ctx,
            gpu = config.use_gpu,
            "initialized ORT LLM backend"
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(OrtLlmInner {
                embed_session,
                decoder_session,
                tokenizer,
            })),
            config: Arc::new(config),
        })
    }
}

const EMBED_CANDIDATES: [&str; 2] = ["qwen3-0.6b-embed-tokens.onnx", "embed_tokens.onnx"];

const DECODER_CANDIDATES: [&str; 4] = [
    "qwen3-0.6b-decoder.onnx",
    "qwen3-0.6b.onnx",
    "decoder_model_merged.onnx",
    "model.onnx",
];

/// Find an ONNX model matching one of the candidate names.
#[cold]
fn find_model(dir: &PathBuf, candidates: &[&str]) -> Result<PathBuf> {
    for c in candidates {
        let p = dir.join(c);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(Error::Llm(format!(
        "no ONNX model found in {} (tried: {})",
        dir.display(),
        candidates.join(", ")
    )))
}

/// Build an ORT session. EPs inherited from global `ort::init()`.
fn build_session(path: &std::path::Path, _use_gpu: bool) -> Result<Session> {
    Session::builder()
        .map_err(|e| Error::Llm(format!("session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Llm(format!("optimization level: {e}")))?
        .with_intra_threads(4)
        .map_err(|e| Error::Llm(format!("intra threads: {e}")))?
        .commit_from_file(path)
        .map_err(|e| Error::Llm(format!("load model {}: {e}", path.display())))
}

const NUM_KV_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
/// Embed input_ids via the embed_tokens session.
fn embed_tokens(session: &mut Session, token_ids: &[i64]) -> Result<DynValue> {
    let ids = Array2::from_shape_vec((1, token_ids.len()), token_ids.to_vec())
        .map_err(|e| Error::Llm(format!("embed input_ids: {e}")))?;
    let inputs: Vec<(String, DynValue)> = vec![(
        "input_ids".into(),
        Value::from_array(ids)
            .map_err(|e| Error::Llm(format!("embed value: {e}")))?
            .into(),
    )];
    let mut outputs = session
        .run(inputs)
        .map_err(|e| Error::Llm(format!("embed_tokens: {e}")))?;
    outputs
        .remove("embeds")
        .ok_or_else(|| Error::Llm("embed_tokens: missing 'embeds' output".into()))
}

/// Run a minimal warmup through both sessions.
#[cold]
fn warmup_decoder(decoder: &mut Session, embed: &mut Session) -> Result<()> {
    let embeds = embed_tokens(embed, &[0i64])?;
    let position_ids = Array2::from_shape_vec((1, 1), vec![0i64]).unwrap();
    let empty_kv = ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), half::f16::ZERO);

    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(2 + NUM_KV_LAYERS * 2);
    inputs.push(("inputs_embeds".into(), embeds));
    inputs.push((
        "position_ids".into(),
        Value::from_array(position_ids).unwrap().into(),
    ));
    for layer in 0..NUM_KV_LAYERS {
        for kind in &["key", "value"] {
            inputs.push((
                format!("past_key_values.{layer}.{kind}"),
                Value::from_array(empty_kv.clone()).unwrap().into(),
            ));
        }
    }

    let _outputs = decoder
        .run(inputs)
        .map_err(|e| Error::Llm(format!("warmup: {e}")))?;

    tracing::debug!("LLM warmup complete");
    Ok(())
}

/// Run inference synchronously, returning generated text.
fn generate_sync(
    inner: &Mutex<OrtLlmInner>,
    config: &OrtLlmConfig,
    prompt_text: &str,
    chunk_tx: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<String> {
    let mut guard = inner.lock().map_err(|e| Error::Llm(format!("lock: {e}")))?;
    let OrtLlmInner {
        ref mut embed_session,
        ref mut decoder_session,
        ref tokenizer,
    } = *guard;

    // Tokenize prompt.
    let encoding = tokenizer
        .encode(prompt_text, false)
        .map_err(|e| Error::Llm(format!("tokenize: {e}")))?;
    let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

    if token_ids.is_empty() {
        return Err(Error::Llm("prompt tokenized to zero tokens".into()));
    }

    let prompt_len = token_ids.len();
    if prompt_len as u32 >= config.n_ctx {
        return Err(Error::Llm(format!(
            "prompt ({prompt_len} tokens) exceeds context window ({})",
            config.n_ctx
        )));
    }

    let available = config.n_ctx as usize - prompt_len;
    if (available as u32) < config.max_output_tokens {
        tracing::warn!(
            prompt_tokens = prompt_len,
            available,
            max_output = config.max_output_tokens,
            "prompt leaves only {available} tokens for output; generation may truncate"
        );
    }

    // ── Prefill ──────────────────────────────────────────────────────────

    let embeds = embed_tokens(embed_session, &token_ids)?;

    let position_ids = Array2::from_shape_vec((1, prompt_len), (0..prompt_len as i64).collect())
        .map_err(|e| Error::Llm(format!("position_ids: {e}")))?;

    let empty_kv = ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), half::f16::ZERO);

    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(2 + NUM_KV_LAYERS * 2);
    inputs.push(("inputs_embeds".into(), embeds));
    inputs.push((
        "position_ids".into(),
        Value::from_array(position_ids)
            .map_err(|e| Error::Llm(format!("position_ids: {e}")))?
            .into(),
    ));
    for layer in 0..NUM_KV_LAYERS {
        for kind in &["key", "value"] {
            inputs.push((
                format!("past_key_values.{layer}.{kind}"),
                Value::from_array(empty_kv.clone())
                    .map_err(|e| Error::Llm(format!("kv tensor: {e}")))?
                    .into(),
            ));
        }
    }

    let (last_logits, mut kv_cache) = {
        let mut outputs = decoder_session
            .run(inputs)
            .map_err(|e| Error::Llm(format!("prefill: {e}")))?;

        let logits_array: ndarray::ArrayViewD<half::f16> = outputs[0]
            .try_extract_array()
            .map_err(|e| Error::Llm(format!("extract logits: {e}")))?;
        let seq_len = logits_array.shape()[1];

        let last_logits: Vec<f32> = logits_array
            .slice(ndarray::s![0, seq_len - 1, ..])
            .iter()
            .map(|v| v.to_f32())
            .collect();

        let mut kv_cache: Vec<DynValue> = Vec::with_capacity(NUM_KV_LAYERS * 2);
        for layer in 0..NUM_KV_LAYERS {
            for kind in &["key", "value"] {
                let name = format!("present_key_values.{layer}.{kind}");
                kv_cache.push(
                    outputs
                        .remove(&name)
                        .ok_or_else(|| Error::Llm(format!("missing output: {name}")))?,
                );
            }
        }

        (last_logits, kv_cache)
    };

    // ── Autoregressive decode ────────────────────────────────────────────

    let capacity = (config.max_output_tokens as usize).saturating_mul(4);
    let mut output = String::with_capacity(capacity);
    let mut streaming = StreamingState::new(chunk_tx.is_some());
    let mut recent_tokens: VecDeque<i64> = VecDeque::with_capacity(config.repeat_last_n as usize);
    let max_gen = config.max_output_tokens as usize;
    let mut current_logits = last_logits;

    let eos_id = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(151643) as i64;

    for step in 0..max_gen {
        let next_token = sampling::sample_token(
            &current_logits,
            config.temperature,
            config.top_k.max(0) as usize,
            config.top_p,
            config.repeat_penalty,
            &recent_tokens,
        );

        recent_tokens.push_back(next_token);
        if recent_tokens.len() > config.repeat_last_n as usize {
            recent_tokens.pop_front();
        }

        if next_token == eos_id {
            break;
        }

        let piece = tokenizer
            .decode(&[next_token as u32], false)
            .map_err(|e| Error::Llm(format!("decode token: {e}")))?;

        if !piece.is_empty() {
            let piece_len = piece.len();
            output.push_str(&piece);

            if check_stop_sequences(&mut output, piece_len) {
                break;
            }

            streaming.check_thinking_done(&output, piece_len);

            if let Some(ref tx) = chunk_tx {
                streaming.try_emit_chunk(&output, tx);
            }
        }

        // ── Single-token decode step ─────────────────────────────────────

        let step_embeds = embed_tokens(embed_session, &[next_token])?;

        let step_pos = Array2::from_shape_vec((1, 1), vec![(prompt_len + step) as i64])
            .map_err(|e| Error::Llm(format!("step position_ids: {e}")))?;

        let mut step_inputs: Vec<(String, DynValue)> = Vec::with_capacity(2 + NUM_KV_LAYERS * 2);

        step_inputs.push(("inputs_embeds".into(), step_embeds));
        step_inputs.push((
            "position_ids".into(),
            Value::from_array(step_pos)
                .map_err(|e| Error::Llm(format!("step position_ids: {e}")))?
                .into(),
        ));

        for (i, kv) in std::mem::take(&mut kv_cache).into_iter().enumerate() {
            let layer = i / 2;
            let kind = if i % 2 == 0 { "key" } else { "value" };
            step_inputs.push((format!("past_key_values.{layer}.{kind}"), kv));
        }

        let mut step_outputs = decoder_session
            .run(step_inputs)
            .map_err(|e| Error::Llm(format!("decode step {step}: {e}")))?;

        let step_logits: ndarray::ArrayViewD<half::f16> = step_outputs[0]
            .try_extract_array()
            .map_err(|e| Error::Llm(format!("step logits: {e}")))?;

        current_logits = step_logits
            .slice(ndarray::s![0, 0, ..])
            .iter()
            .map(|v| v.to_f32())
            .collect();

        kv_cache = Vec::with_capacity(NUM_KV_LAYERS * 2);
        for layer in 0..NUM_KV_LAYERS {
            for kind in &["key", "value"] {
                let name = format!("present_key_values.{layer}.{kind}");
                kv_cache.push(
                    step_outputs
                        .remove(&name)
                        .ok_or_else(|| Error::Llm(format!("missing output: {name}")))?,
                );
            }
        }
    }

    if output.len() >= capacity.saturating_sub(4) {
        trim_partial_stop_sequences(&mut output);
    }

    if let Some(ref tx) = chunk_tx {
        let cleaned = streaming.flush(&output, tx, |s| prompt::strip_thinking(s));
        return Ok(cleaned);
    }

    Ok(output)
}

#[async_trait::async_trait]
impl TextPostProcessor for OrtLlmBackend {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let prompt_text = prompt::build_chatml(input);

        tracing::debug!(
            prompt_len = prompt_text.len(),
            raw_text_len = input.raw_text.len(),
            style = %input.style_preset.id,
            "ORT LLM post-process request"
        );

        let inner = Arc::clone(&self.inner);
        let config = Arc::clone(&self.config);
        let raw_output =
            tokio::task::spawn_blocking(move || generate_sync(&inner, &config, &prompt_text, None))
                .await
                .map_err(|e| Error::Llm(format!("spawn_blocking join: {e}")))??;

        let cleaned = prompt::strip_thinking(&raw_output);

        tracing::debug!(output_len = cleaned.len(), "ORT LLM post-process complete");

        Ok(PostProcessOutput {
            cleaned_text: cleaned,
            metadata: None,
        })
    }

    async fn transform_streaming(
        &self,
        input: &PostProcessInput,
        chunk_tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<PostProcessOutput> {
        let prompt_text = prompt::build_chatml(input);

        tracing::debug!(
            prompt_len = prompt_text.len(),
            raw_text_len = input.raw_text.len(),
            style = %input.style_preset.id,
            "ORT LLM streaming post-process request"
        );

        let inner = Arc::clone(&self.inner);
        let config = Arc::clone(&self.config);

        let cleaned = tokio::task::spawn_blocking(move || {
            generate_sync(&inner, &config, &prompt_text, Some(chunk_tx))
        })
        .await
        .map_err(|e| Error::Llm(format!("spawn_blocking join: {e}")))??;

        tracing::debug!(
            output_len = cleaned.len(),
            "ORT LLM streaming post-process complete"
        );

        Ok(PostProcessOutput {
            cleaned_text: cleaned,
            metadata: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let cfg = OrtLlmConfig::default();
        assert_eq!(cfg.n_ctx, 2048);
        assert_eq!(cfg.max_output_tokens, 256);
        assert!((cfg.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_missing_model_dir() {
        let cfg = OrtLlmConfig {
            model_dir: PathBuf::from("/nonexistent/ort-model"),
            ..Default::default()
        };
        let result = OrtLlmBackend::new(cfg);
        assert!(result.is_err());
    }
}
