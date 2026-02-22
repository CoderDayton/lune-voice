use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};

use lune_core::error::{Error, Result};
use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

use crate::prompt;

/// Length of the longest stop sequence token (`<|im_end|>`).
/// Used to bound the suffix window scanned for stop sequences each decode step.
const MAX_STOP_SEQ_LEN: usize = 10;

/// Configuration for the local LLM backend.
#[derive(Debug, Clone)]
pub struct LocalLlmConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// Context window size in tokens.
    pub n_ctx: u32,
    /// Maximum tokens to generate per request.
    pub max_output_tokens: u32,
    /// Sampling temperature (0.7 recommended for non-thinking mode).
    pub temperature: f32,
    /// Top-p sampling.
    pub top_p: f32,
    /// Top-k sampling.
    pub top_k: i32,
}

impl Default for LocalLlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/Qwen3-0.6B-Q5_K_M.gguf"),
            use_gpu: true,
            n_ctx: 16384,
            max_output_tokens: 512,
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
        }
    }
}

/// Local LLM backend using llama-cpp-2 for GGUF inference.
///
/// Runs Qwen3-0.6B (or compatible) for text post-processing:
/// grammar correction, disfluency removal, punctuation.
pub struct LocalLlmBackend {
    inner: Arc<Mutex<LlmInner>>,
    /// Shared immutably across all `transform` calls; Arc makes the per-call
    /// clone an atomic increment rather than a heap allocation.
    config: Arc<LocalLlmConfig>,
}

struct LlmInner {
    backend: LlamaBackend,
    model: LlamaModel,
}

// SAFETY: LlamaBackend and LlamaModel are thread-safe behind our Mutex.
// llama.cpp's inference context is created per-call inside the Mutex.
unsafe impl Send for LlmInner {}

impl LocalLlmBackend {
    /// Load a GGUF model from disk.
    pub fn new(config: LocalLlmConfig) -> Result<Self> {
        if !config.model_path.exists() {
            return Err(Error::ModelNotFound(config.model_path.clone()));
        }

        // Route llama.cpp logs through tracing so they respect RUST_LOG filtering.
        // Without this, llama.cpp dumps INFO-level messages (CPU_REPACK, async
        // upload warnings) directly to stderr.
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));

        let backend =
            LlamaBackend::init().map_err(|e| Error::Llm(format!("llama backend init: {e}")))?;

        let gpu = config.use_gpu && backend.supports_gpu_offload();
        let mut model_params = LlamaModelParams::default();
        if gpu {
            model_params = model_params.with_n_gpu_layers(1000);
        }

        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .map_err(|e| Error::Llm(format!("load model: {e}")))?;

        // Pre-warm: allocate + drop a context to force KV cache memory allocation
        // so the first real inference call doesn't pay the cost.
        {
            let n_ctx = NonZeroU32::new(config.n_ctx)
                .ok_or_else(|| Error::Llm("n_ctx must be > 0".into()))?;
            let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
            let _warm = model
                .new_context(&backend, ctx_params)
                .map_err(|e| Error::Llm(format!("pre-warm context: {e}")))?;
        }

        tracing::info!(
            model_path = %config.model_path.display(),
            n_ctx = config.n_ctx,
            gpu,
            "initialized local LLM backend"
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(LlmInner { backend, model })),
            config: Arc::new(config),
        })
    }
}

/// Run inference on a prompt string, returning the generated text.
///
/// Free function so it can be called from `spawn_blocking` (which requires
/// `'static` + `Send` — the `Arc<Mutex<LlmInner>>` satisfies both).
fn generate_sync(
    inner: &Mutex<LlmInner>,
    config: &LocalLlmConfig,
    prompt_text: &str,
) -> Result<String> {
    let inner = inner.lock().map_err(|e| Error::Llm(format!("lock: {e}")))?;

    let n_ctx =
        NonZeroU32::new(config.n_ctx).ok_or_else(|| Error::Llm("n_ctx must be > 0".into()))?;
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    let mut ctx = inner
        .model
        .new_context(&inner.backend, ctx_params)
        .map_err(|e| Error::Llm(format!("create context: {e}")))?;

    // Tokenize prompt. Qwen3 has no BOS token.
    let tokens = inner
        .model
        .str_to_token(prompt_text, AddBos::Never)
        .map_err(|e| Error::Llm(format!("tokenize: {e}")))?;

    let prompt_len = tokens.len() as i32;
    if prompt_len as u32 >= config.n_ctx {
        return Err(Error::Llm(format!(
            "prompt ({prompt_len} tokens) exceeds context window ({})",
            config.n_ctx
        )));
    }

    // Build batch with prompt tokens.
    let batch_size = (prompt_len as usize).max(512);
    let mut batch = LlamaBatch::new(batch_size, 1);
    let last_idx = prompt_len - 1;
    // Use i32 range directly to avoid repeated `as i32` casts in the hot path.
    for (i, token) in (0i32..prompt_len).zip(tokens) {
        batch
            .add(token, i, &[0], i == last_idx)
            .map_err(|e| Error::Llm(format!("batch add: {e}")))?;
    }

    // Prefill.
    ctx.decode(&mut batch)
        .map_err(|e| Error::Llm(format!("decode prefill: {e}")))?;

    // Set up sampler chain.
    let mut sampler = if config.temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(config.top_k),
            LlamaSampler::top_p(config.top_p, 1),
            LlamaSampler::temp(config.temperature),
            LlamaSampler::dist(42),
        ])
    };

    // Autoregressive decode.
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    // Pre-allocate for expected output size: max_output_tokens * ~4 bytes average UTF-8.
    let mut output = String::with_capacity((config.max_output_tokens as usize) * 4);
    let mut n_cur = batch.n_tokens();
    let max_total = prompt_len + config.max_output_tokens as i32;
    let eos = inner.model.token_eos();

    while n_cur < max_total {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if token == eos {
            break;
        }

        let piece = inner
            .model
            .token_to_piece(token, &mut decoder, true, None)
            .unwrap_or_default();

        if !piece.is_empty() {
            output.push_str(&piece);

            // Optimised stop-sequence detection: scan only the suffix window that
            // can possibly contain a newly-completed marker.  A marker straddling
            // the old/new boundary requires at most (MAX_STOP_SEQ_LEN - 1) bytes
            // of old output, so total scan is O(piece.len() + MAX_STOP_SEQ_LEN)
            // rather than O(output.len()).
            let win_end = output.len();
            let win_start = win_end.saturating_sub(piece.len() + MAX_STOP_SEQ_LEN - 1);
            // Snap backward to a valid UTF-8 char boundary (≤3 bytes for well-formed input).
            let win_start = {
                let mut s = win_start;
                while s > 0 && !output.is_char_boundary(s) {
                    s -= 1;
                }
                s
            };
            let window = &output[win_start..];
            if window.contains("<|im_end|>") || window.contains("<think>") {
                break;
            }
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| Error::Llm(format!("batch add decode: {e}")))?;
        n_cur += 1;

        ctx.decode(&mut batch)
            .map_err(|e| Error::Llm(format!("decode step: {e}")))?;
    }

    Ok(output)
}

#[async_trait::async_trait]
impl TextPostProcessor for LocalLlmBackend {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let prompt_text = prompt::build_chatml(input);

        tracing::debug!(
            prompt_len = prompt_text.len(),
            raw_text_len = input.raw_text.len(),
            style = %input.style_preset.id,
            "LLM post-process request"
        );

        // Run inference on a blocking thread to avoid starving the async runtime.
        // spawn_blocking (not block_in_place) is required for current_thread + LocalSet.
        let inner = Arc::clone(&self.inner);
        // Arc clone: O(1) atomic increment, not a heap allocation.
        let config = Arc::clone(&self.config);
        let raw_output =
            tokio::task::spawn_blocking(move || generate_sync(&inner, &config, &prompt_text))
                .await
                .map_err(|e| Error::Llm(format!("spawn_blocking join: {e}")))??;

        let cleaned = prompt::strip_thinking(&raw_output);

        tracing::debug!(output_len = cleaned.len(), "LLM post-process complete");

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
        let cfg = LocalLlmConfig::default();
        assert_eq!(cfg.n_ctx, 16384);
        assert_eq!(cfg.max_output_tokens, 512);
        assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_missing_model() {
        let cfg = LocalLlmConfig {
            model_path: PathBuf::from("/nonexistent/model.gguf"),
            ..Default::default()
        };
        let result = LocalLlmBackend::new(cfg);
        assert!(result.is_err());
    }
}
