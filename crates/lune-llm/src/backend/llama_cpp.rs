//! Local llama.cpp styling LLM backend.
//!
//! The real implementation (feature `llama-cpp`) wraps [`llama_cpp_2`], which
//! compiles llama.cpp. When the feature is off, a stub with the same public
//! surface keeps the rest of the crate compiling — its constructor returns a
//! clear "rebuild with the feature" error.
//!
//! Design notes:
//! - The model ([`llama_cpp_2::model::LlamaModel`]) is loaded **once** at backend
//!   construction; its weights are read-only (`Send + Sync`), so it is shared via
//!   `Arc` across calls. A fresh [`llama_cpp_2::context::LlamaContext`] is created
//!   per `complete()` call — styling is a stateless one-shot, and `LlamaContext`
//!   is not `Send`, so it must never cross the async boundary.
//! - All inference runs on a blocking thread (`spawn_blocking`); no llama handle
//!   is held across `.await`.
//! - The styling model (Qwen3.5) defaults to "thinking" mode; thinking is
//!   hard-disabled by pre-filling an empty `<think></think>` block in the
//!   assistant turn (see [`build_chatml`]), and any residual think block is
//!   stripped from the output so styling emits only the rewritten text.

use std::path::PathBuf;

/// Construction parameters for [`LlamaCppBackend`].
pub struct LlamaCppConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Offload to GPU. Only meaningful when built with `llama-cuda`; on a
    /// CPU-only build llama.cpp runs on the CPU regardless.
    pub use_gpu: bool,
    /// Hard cap on generated tokens per call.
    pub max_new_tokens: usize,
}

#[cfg(feature = "llama-cpp")]
mod imp {
    use super::LlamaCppConfig;

    use std::num::NonZeroU32;
    use std::sync::{Arc, OnceLock};

    use async_trait::async_trait;
    use lune_core::error::{Error, Result};
    use lune_core::llm::LlmService;

    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::token::logit_bias::LlamaLogitBias;

    /// Context window. Dictation styling prompts are short; 4k is ample and
    /// keeps per-call context allocation small.
    const N_CTX: u32 = 4096;

    /// Process-global llama.cpp backend. `LlamaBackend::init()` must run exactly
    /// once per process; this also routes llama.cpp's noisy stderr logging into
    /// `tracing` (disabled by default).
    fn backend() -> Result<&'static LlamaBackend> {
        static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
        if let Some(b) = BACKEND.get() {
            return Ok(b);
        }
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );
        let b = LlamaBackend::init()
            .map_err(|e| Error::config(format!("llama.cpp backend init: {e}")))?;
        // Racing threads: whichever loses keeps its local `b` dropped; the winner
        // is stored. `get_or_init` would be cleaner but `init()` is fallible.
        let _ = BACKEND.set(b);
        BACKEND
            .get()
            .ok_or_else(|| Error::config("llama.cpp backend unavailable".to_string()))
    }

    pub struct LlamaCppBackend {
        model: Arc<LlamaModel>,
        max_new_tokens: usize,
    }

    impl LlamaCppBackend {
        #[cold]
        pub fn new(config: LlamaCppConfig) -> Result<Self> {
            let backend = backend()?;
            // 999 → offload all layers (llama.cpp clamps to the model's count);
            // 0 → CPU only.
            let n_gpu_layers: u32 = if config.use_gpu { 999 } else { 0 };
            let model = LlamaModel::load_from_file(
                backend,
                &config.model_path,
                &LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers),
            )
            .map_err(|e| Error::config(format!("load LLM {}: {e}", config.model_path.display())))?;
            tracing::info!(
                model = %config.model_path.display(),
                use_gpu = config.use_gpu,
                "llama.cpp styling LLM loaded"
            );
            Ok(Self {
                model: Arc::new(model),
                max_new_tokens: config.max_new_tokens.max(1),
            })
        }
    }

    /// Build a Qwen ChatML prompt with thinking hard-disabled by pre-filling an
    /// empty `<think></think>` block at the start of the assistant turn. The
    /// model then continues *after* the closed block, so it skips reasoning and
    /// the think tags never appear in the generated output (more reliable than
    /// the `/no_think` directive). The system block is omitted when empty.
    fn build_chatml(system: &str, user: &str) -> String {
        let system = system.trim();
        let mut prompt = String::with_capacity(system.len() + user.len() + 96);
        if !system.is_empty() {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(system);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(user);
        prompt.push_str("<|im_end|>\n");
        prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        prompt
    }

    /// Blocking generation. Creates a fresh context, greedily decodes up to
    /// `max_new_tokens`, and returns the decoded text with `<think>` stripped.
    fn generate(
        model: &LlamaModel,
        system: &str,
        user: &str,
        max_new_tokens: usize,
    ) -> Result<String> {
        let backend = backend()?;
        let mut ctx = model
            .new_context(
                backend,
                LlamaContextParams::default().with_n_ctx(NonZeroU32::new(N_CTX)),
            )
            .map_err(|e| Error::config(format!("llama context: {e}")))?;

        let prompt = build_chatml(system, user);
        let mut tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| Error::config(format!("tokenize: {e}")))?;

        // Guard: keep prompt within the window, leaving room to generate.
        let budget = (N_CTX as usize).saturating_sub(max_new_tokens).max(1);
        if tokens.len() > budget {
            let cut = tokens.len() - budget;
            tokens.drain(0..cut);
        }
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut batch = LlamaBatch::new(512.max(tokens.len()), 1);
        let last = tokens.len() - 1;
        for (i, tok) in tokens.iter().enumerate() {
            batch
                .add(*tok, i as i32, &[0], i == last)
                .map_err(|e| Error::config(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Error::config(format!("decode prompt: {e}")))?;

        // The empty `<think></think>` prefill above only pre-fills the prompt;
        // nothing stops the model from *generating* a fresh `<think>` token as
        // its first real output, re-entering reasoning mode with no matching
        // close before the token budget runs out (CPU inference isn't
        // bit-exact across runs, so this isn't reliably reproducible — the
        // model can tip into it on some decodes and not others). A closed
        // reasoning block gets stripped by `strip_think_tags`, but a reopened,
        // never-closed one swallows the real answer along with it. Hard-ban
        // the `<think>` token itself so generation can't re-enter thinking at
        // all, rather than trying to clean it up after the fact.
        let think_open_token = model
            .str_to_token("<think>", AddBos::Never)
            .ok()
            .filter(|toks| toks.len() == 1)
            .map(|toks| toks[0]);
        let mut chain_samplers = Vec::with_capacity(3);
        if let Some(tok) = think_open_token {
            chain_samplers.push(LlamaSampler::logit_bias(
                model.n_vocab(),
                &[LlamaLogitBias::new(tok, f32::NEG_INFINITY)],
            ));
        }
        // Mild repetition penalty (llama.cpp's historical 1.1 default over the
        // last 64 generated tokens). Pure greedy makes the small styler
        // collapse into degenerate repetition loops on free-form generation
        // (few-shot example synthesis burned its whole token budget on one
        // repeated phrase). Only generated tokens are penalized — the prompt
        // is never `accept`ed — so legit repetition in the user's text is
        // unaffected, and the decode stays deterministic.
        chain_samplers.push(LlamaSampler::penalties(64, 1.1, 0.0, 0.0));
        chain_samplers.push(LlamaSampler::greedy());
        let mut sampler = LlamaSampler::chain_simple(chain_samplers);
        let mut out = String::new();
        let mut n_cur = tokens.len() as i32;
        // Shared decoder so multi-byte UTF-8 split across tokens reassembles.
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        for _ in 0..max_new_tokens {
            let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(tok);
            if model.is_eog_token(tok) {
                break;
            }
            // `special = false`: never render control tokens into the output.
            if let Ok(piece) = model.token_to_piece(tok, &mut decoder, false, None) {
                out.push_str(&piece);
            }
            batch.clear();
            batch
                .add(tok, n_cur, &[0], true)
                .map_err(|e| Error::config(format!("batch add: {e}")))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| Error::config(format!("decode step: {e}")))?;
        }

        Ok(crate::prompt::strip_think_tags(&out).trim().to_string())
    }

    #[async_trait]
    impl LlmService for LlamaCppBackend {
        async fn complete(&self, system: &str, user: &str) -> Result<String> {
            let model = Arc::clone(&self.model);
            let system = system.to_owned();
            let user = user.to_owned();
            let max_new_tokens = self.max_new_tokens;
            tokio::task::spawn_blocking(move || generate(&model, &system, &user, max_new_tokens))
                .await
                .map_err(|e| Error::config(format!("llama join: {e}")))?
        }
    }
}

#[cfg(feature = "llama-cpp")]
pub use imp::LlamaCppBackend;

#[cfg(not(feature = "llama-cpp"))]
mod stub {
    use super::LlamaCppConfig;
    use async_trait::async_trait;
    use lune_core::error::{Error, Result};
    use lune_core::llm::LlmService;

    pub struct LlamaCppBackend;

    impl LlamaCppBackend {
        pub fn new(_config: LlamaCppConfig) -> Result<Self> {
            Err(Error::config(
                "llama.cpp backend not compiled in; rebuild with --features llama-cpp".to_string(),
            ))
        }
    }

    // Mirror the real backend's public surface so callers compile on a build
    // without the feature. `new` always errors, so this is never constructed.
    #[async_trait]
    impl LlmService for LlamaCppBackend {
        async fn complete(&self, _system: &str, _user: &str) -> Result<String> {
            Err(Error::config(
                "llama.cpp backend not compiled in; rebuild with --features llama-cpp".to_string(),
            ))
        }
    }
}

#[cfg(not(feature = "llama-cpp"))]
pub use stub::LlamaCppBackend;
