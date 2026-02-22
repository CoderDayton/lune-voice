use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use lune_core::error::{Error, Result};
use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

use crate::prompt;

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
            n_ctx: 2048,
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
    config: LocalLlmConfig,
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

        let backend =
            LlamaBackend::init().map_err(|e| Error::Llm(format!("llama backend init: {e}")))?;

        let mut model_params = LlamaModelParams::default();
        if config.use_gpu {
            model_params = model_params.with_n_gpu_layers(1000);
        }

        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .map_err(|e| Error::Llm(format!("load model: {e}")))?;

        tracing::info!(
            model_path = %config.model_path.display(),
            n_ctx = config.n_ctx,
            gpu = config.use_gpu,
            "initialized local LLM backend"
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(LlmInner { backend, model })),
            config,
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
    let batch_size = prompt_len.max(512) as usize;
    let mut batch = LlamaBatch::new(batch_size, 1);
    let last_idx = prompt_len - 1;
    for (i, token) in tokens.into_iter().enumerate() {
        let is_last = i as i32 == last_idx;
        batch
            .add(token, i as i32, &[0], is_last)
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
    let mut output = String::with_capacity(512);
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
        output.push_str(&piece);

        // Stop on <|im_end|> in the output.
        if output.contains("<|im_end|>") {
            break;
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
        let config = self.config.clone();
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
        assert_eq!(cfg.n_ctx, 2048);
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
