//! ORT-based LLM backend for text post-processing.
//!
//! ORT-based local LLM backend using ONNX Runtime inference.
//! Uses `decoder_model_merged.onnx` (Qwen3-0.6B exported via optimum-cli) which
//! provides `input_ids` + `past_key_values` → `logits` + `present_key_values`.
//!
//! Advantages: shared GPU memory arena with ASR sessions,
//! no C++ FFI compilation, unified ORT optimization pipeline.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ndarray::{Array2, Array4};
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
    /// Directory containing `decoder_model_merged.onnx` and `tokenizer.json`.
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
    session: Session,
    tokenizer: Tokenizer,
}

// SAFETY: OrtLlmInner is only accessed under Mutex.
unsafe impl Send for OrtLlmInner {}

impl OrtLlmBackend {
    /// Load an ONNX model and tokenizer from `model_dir`.
    pub fn new(config: OrtLlmConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir.clone()));
        }

        let model_path = find_onnx_model(&config.model_dir)?;
        let tokenizer_path = config.model_dir.join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(Error::Llm(format!(
                "tokenizer.json not found in {}",
                config.model_dir.display()
            )));
        }

        let mut session = build_llm_session(&model_path, config.use_gpu)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Llm(format!("tokenizer: {e}")))?;

        // Warmup: run a 1-token dummy inference to prime CUDA kernels.
        warmup_llm(&mut session)?;

        tracing::info!(
            model = %model_path.display(),
            n_ctx = config.n_ctx,
            gpu = config.use_gpu,
            "initialized ORT LLM backend"
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(OrtLlmInner { session, tokenizer })),
            config: Arc::new(config),
        })
    }
}

/// Find the decoder ONNX model in the model directory.
#[cold]
fn find_onnx_model(dir: &PathBuf) -> Result<PathBuf> {
    let candidates = [
        "qwen3-0.6b.onnx",
        "decoder_model_merged.onnx",
        "model.onnx",
        "decoder_model.onnx",
    ];
    for c in &candidates {
        let p = dir.join(c);
        if p.exists() {
            return Ok(p);
        }
    }
    // Fallback: find any .onnx file
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name.ends_with(".onnx") {
                    return Ok(p);
                }
            }
        }
    }
    Err(Error::Llm(format!(
        "no ONNX model found in {}",
        dir.display()
    )))
}

/// Build an ORT session for the LLM decoder model.
fn build_llm_session(path: &std::path::Path, use_gpu: bool) -> Result<Session> {
    let builder = Session::builder()
        .map_err(|e| Error::Llm(format!("session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Llm(format!("optimization level: {e}")))?
        .with_intra_threads(4)
        .map_err(|e| Error::Llm(format!("intra threads: {e}")))?;

    let builder = if use_gpu {
        #[cfg(feature = "cuda")]
        {
            builder
                .with_execution_providers([
                    ort::ep::CUDA::default().build(),
                    ort::ep::CPU::default().build().error_on_failure(),
                ])
                .map_err(|e| Error::Llm(format!("CUDA provider: {e}")))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::warn!("use_gpu=true but cuda feature not enabled, falling back to CPU");
            builder
        }
    } else {
        builder
    };

    builder
        .commit_from_file(path)
        .map_err(|e| Error::Llm(format!("load model {}: {e}", path.display())))
}

/// Number of KV cache layers for Qwen3-0.6B (28 layers × 2 for key+value).
const NUM_KV_LAYERS: usize = 28;
/// Number of KV attention heads.
const NUM_KV_HEADS: usize = 8;
/// Dimension per KV head.
const HEAD_DIM: usize = 128;

/// Run a minimal 1-token prefill to prime CUDA JIT compilation and memory pools.
#[cold]
fn warmup_llm(session: &mut Session) -> Result<()> {
    let input_ids = Array2::from_shape_vec((1, 1), vec![0i64]).unwrap();
    let attention_mask = Array2::<i64>::ones((1, 1));
    let position_ids = Array2::from_shape_vec((1, 1), vec![0i64]).unwrap();
    let empty_kv = Array4::<f32>::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM));

    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_KV_LAYERS * 2);
    inputs.push((
        "input_ids".into(),
        Value::from_array(input_ids).unwrap().into(),
    ));
    inputs.push((
        "attention_mask".into(),
        Value::from_array(attention_mask).unwrap().into(),
    ));
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

    let _outputs = session
        .run(inputs)
        .map_err(|e| Error::Llm(format!("warmup: {e}")))?;

    tracing::debug!("LLM warmup complete");
    Ok(())
}

/// Run inference synchronously, returning generated text.
///
/// Free function for `spawn_blocking` compatibility.
fn generate_sync(
    inner: &Mutex<OrtLlmInner>,
    config: &OrtLlmConfig,
    prompt_text: &str,
    chunk_tx: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<String> {
    let mut guard = inner.lock().map_err(|e| Error::Llm(format!("lock: {e}")))?;
    let OrtLlmInner {
        ref mut session,
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
    // First forward pass: full prompt, no KV cache.

    let input_ids = Array2::from_shape_vec((1, prompt_len), token_ids.clone())
        .map_err(|e| Error::Llm(format!("input_ids array: {e}")))?;

    let attention_mask = Array2::<i64>::ones((1, prompt_len));

    // Empty KV cache tensors for prefill (seq_len=0).
    let empty_kv = Array4::<f32>::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM));

    let mut kv_inputs: Vec<DynValue> = Vec::with_capacity(NUM_KV_LAYERS * 2);
    for _ in 0..NUM_KV_LAYERS * 2 {
        kv_inputs.push(
            Value::from_array(empty_kv.clone())
                .map_err(|e| Error::Llm(format!("kv tensor: {e}")))?
                .into(),
        );
    }

    // Position IDs for prefill: [0, 1, 2, ..., prompt_len-1].
    let position_ids = Array2::from_shape_vec((1, prompt_len), (0..prompt_len as i64).collect())
        .map_err(|e| Error::Llm(format!("position_ids array: {e}")))?;

    // Build input map for prefill.
    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_KV_LAYERS * 2);
    inputs.push((
        "input_ids".into(),
        Value::from_array(input_ids)
            .map_err(|e| Error::Llm(format!("input_ids value: {e}")))?
            .into(),
    ));
    inputs.push((
        "attention_mask".into(),
        Value::from_array(attention_mask)
            .map_err(|e| Error::Llm(format!("attention_mask value: {e}")))?
            .into(),
    ));
    inputs.push((
        "position_ids".into(),
        Value::from_array(position_ids)
            .map_err(|e| Error::Llm(format!("position_ids value: {e}")))?
            .into(),
    ));

    for (i, kv) in kv_inputs.into_iter().enumerate() {
        let layer = i / 2;
        let kind = if i % 2 == 0 { "key" } else { "value" };
        inputs.push((format!("past_key_values.{layer}.{kind}"), kv));
    }

    let (last_logits, mut kv_cache) = {
        let mut outputs = session
            .run(inputs)
            .map_err(|e| Error::Llm(format!("prefill: {e}")))?;

        // Extract logits from prefill — last position only.
        let logits_array: ndarray::ArrayViewD<f32> = outputs[0]
            .try_extract_array()
            .map_err(|e| Error::Llm(format!("extract logits: {e}")))?;
        let logits_shape = logits_array.shape();
        let seq_len = logits_shape[1];

        let last_logits: Vec<f32> = logits_array.slice(ndarray::s![0, seq_len - 1, ..]).to_vec();

        // Take ownership of KV cache outputs — zero-copy GPU passthrough.
        let mut kv_cache: Vec<DynValue> = Vec::with_capacity(NUM_KV_LAYERS * 2);
        for layer in 0..NUM_KV_LAYERS {
            for kind in &["key", "value"] {
                let name = format!("present.{layer}.{kind}");
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
        .unwrap_or(151643) as i64; // Qwen3 default EOS

    for step in 0..max_gen {
        // Sample next token.
        let next_token = sampling::sample_token(
            &current_logits,
            config.temperature,
            config.top_k.max(0) as usize,
            config.top_p,
            config.repeat_penalty,
            &recent_tokens,
        );

        // Track for repetition penalty.
        recent_tokens.push_back(next_token);
        if recent_tokens.len() > config.repeat_last_n as usize {
            recent_tokens.pop_front();
        }

        if next_token == eos_id {
            break;
        }

        // Decode token to text.
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

        let single_id = Array2::from_shape_vec((1, 1), vec![next_token])
            .map_err(|e| Error::Llm(format!("next id array: {e}")))?;

        // Attention mask grows by 1 each step: prompt_len + step + 1
        let total_seq = prompt_len + step + 1;
        let mask = Array2::<i64>::ones((1, total_seq));

        // Position ID for this decode step: prompt_len + step.
        let step_pos = Array2::from_shape_vec((1, 1), vec![(prompt_len + step) as i64])
            .map_err(|e| Error::Llm(format!("step position_ids: {e}")))?;

        let mut step_inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_KV_LAYERS * 2);

        step_inputs.push((
            "input_ids".into(),
            Value::from_array(single_id)
                .map_err(|e| Error::Llm(format!("step input_ids: {e}")))?
                .into(),
        ));
        step_inputs.push((
            "attention_mask".into(),
            Value::from_array(mask)
                .map_err(|e| Error::Llm(format!("step attention_mask: {e}")))?
                .into(),
        ));
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

        let mut step_outputs = session
            .run(step_inputs)
            .map_err(|e| Error::Llm(format!("decode step {step}: {e}")))?;

        // Extract logits (single position).
        let step_logits: ndarray::ArrayViewD<f32> = step_outputs[0]
            .try_extract_array()
            .map_err(|e| Error::Llm(format!("step logits: {e}")))?;

        current_logits = step_logits.slice(ndarray::s![0, 0, ..]).to_vec();

        // Take ownership of KV outputs — zero-copy GPU passthrough.
        kv_cache = Vec::with_capacity(NUM_KV_LAYERS * 2);
        for layer in 0..NUM_KV_LAYERS {
            for kind in &["key", "value"] {
                let name = format!("present.{layer}.{kind}");
                kv_cache.push(
                    step_outputs
                        .remove(&name)
                        .ok_or_else(|| Error::Llm(format!("missing output: {name}")))?,
                );
            }
        }
    }

    // Trim partial stop sequences on max-token exit.
    if output.len() >= capacity.saturating_sub(4) {
        trim_partial_stop_sequences(&mut output);
    }

    // Final flush for streaming.
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
