//! mistral.rs backend for ASR transcript correction via GGUF models.
//!
//! Uses `GgufModelBuilder` to load any GGUF model (e.g. Qwen2.5-1.5B-Instruct
//! Q4_K_M) and the chat-completion API for inference. The model is loaded once
//! and shared across calls.

use std::path::PathBuf;
use std::sync::Arc;

use mistralrs::{
    GgufModelBuilder, Model, RequestBuilder, StopTokens, TextMessageRole, TextMessages,
};

use crate::prompt;
use lune_core::error::{Error, Result};
use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

/// Configuration for the mistral.rs GGUF backend.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Path to the `.gguf` model file.
    pub model_path: PathBuf,
    /// GPU layers to offload (0 = CPU only, 1000 = all GPU).
    /// Note: mistralrs handles GPU detection automatically.
    pub n_gpu_layers: u32,
    /// Context window size in tokens.
    pub n_ctx: u32,
    /// Number of inference threads (unused by mistralrs, kept for config compat).
    pub n_threads: u32,
    /// Maximum output tokens to generate per call.
    pub max_output_tokens: u32,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_gpu_layers: 999,
            n_ctx: 2048,
            n_threads: 4,
            max_output_tokens: 256,
        }
    }
}

/// mistral.rs GGUF backend for ASR correction.
pub struct LlamaCppBackend {
    model: Arc<Model>,
    config: Arc<LlamaCppConfig>,
}

impl LlamaCppBackend {
    /// Load a GGUF model from disk.
    pub async fn new(config: LlamaCppConfig) -> Result<Self> {
        if !config.model_path.exists() {
            return Err(Error::ModelNotFound(config.model_path.clone()));
        }

        let model_dir = config
            .model_path
            .parent()
            .ok_or_else(|| Error::Llm("model_path has no parent directory".into()))?;
        let model_file = config
            .model_path
            .file_name()
            .ok_or_else(|| Error::Llm("model_path has no filename".into()))?
            .to_string_lossy()
            .to_string();

        let mut builder = GgufModelBuilder::new(
            model_dir.to_string_lossy().to_string(),
            vec![model_file.clone()],
        );

        if config.n_gpu_layers == 0 {
            builder = builder.with_force_cpu();
        }

        let model = builder
            .build()
            .await
            .map_err(|e| Error::Llm(format!("load model {}: {e}", config.model_path.display())))?;

        tracing::info!(
            model = %config.model_path.display(),
            gpu_layers = config.n_gpu_layers,
            max_output_tokens = config.max_output_tokens,
            "initialized mistral.rs LLM backend"
        );

        Ok(Self {
            model: Arc::new(model),
            config: Arc::new(config),
        })
    }
}

#[async_trait::async_trait]
impl TextPostProcessor for LlamaCppBackend {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        // N-best rescoring path (ProGRes-style). When the ASR backend has
        // surfaced ≥2 full-sentence hypotheses, switch to constrained
        // hypothesis-selection — strictly additive: the result is always
        // verbatim from `hypotheses`, so worst case is the argmax.
        if input.hypotheses.len() >= 2 {
            return self.rescore_n_best(input).await;
        }

        let input_words = input.raw_text.split_whitespace().count();
        let fallback = prompt::strip_markers(&input.raw_text);

        // User message is just the raw text — no "Transcript:" prefix.
        // The fine-tuned model expects raw [?word?]-marked text directly.
        let user_msg = &input.raw_text;

        let messages = TextMessages::new()
            .add_message(TextMessageRole::System, prompt::SYSTEM)
            .add_message(TextMessageRole::User, user_msg);

        let request = RequestBuilder::from(messages)
            .set_deterministic_sampler()
            .set_sampler_topk(1)
            .set_sampler_max_len(self.config.max_output_tokens as usize)
            .set_sampler_stop_toks(StopTokens::Seqs(vec!["<|im_end|>".into()]));

        let response = self
            .model
            .send_chat_request(request)
            .await
            .map_err(|e| Error::Llm(format!("inference: {e}")))?;

        let result = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| prompt::post_process(s))
            .unwrap_or_default();

        // Guard: empty output → fall back to stripped raw text
        if result.is_empty() {
            tracing::warn!("LLM returned empty output — using stripped raw text");
            return Ok(PostProcessOutput {
                cleaned_text: fallback,
                metadata: None,
            });
        }

        let output_words = result.split_whitespace().count();

        // Guard: output too long (hallucination)
        if output_words > input_words + 10 {
            tracing::warn!(
                input_words,
                output_words,
                "LLM hallucination guard (too long) — using stripped raw text"
            );
            return Ok(PostProcessOutput {
                cleaned_text: fallback,
                metadata: None,
            });
        }

        // Guard: output too short (content dropped)
        // Threshold: output must be at least 1/3 of input. The LLM does span
        // correction ([?word?] → corrected word), not summarization, so output
        // length should be close to input. Using 1/3 instead of 1/2 to avoid
        // false positives when input contains natural repetition.
        if input_words > 8 && output_words < input_words / 3 {
            tracing::warn!(
                input_words,
                output_words,
                "LLM truncation guard (too short) — using stripped raw text"
            );
            return Ok(PostProcessOutput {
                cleaned_text: fallback,
                metadata: None,
            });
        }

        // Guard: word preservation — verify that every unmarked word from
        // the input appears in the output in order. This catches the most
        // dangerous failure mode: the LLM silently changing correct words.
        if !prompt::validate_preservation(&input.raw_text, &result) {
            tracing::warn!(
                input = %input.raw_text,
                output = %result,
                "LLM preservation guard — unmarked words changed, using stripped raw text"
            );
            return Ok(PostProcessOutput {
                cleaned_text: fallback,
                metadata: None,
            });
        }

        Ok(PostProcessOutput {
            cleaned_text: result,
            metadata: None,
        })
    }
}

impl LlamaCppBackend {
    /// ProGRes-style N-best rescoring. The LLM receives a numbered
    /// candidate list and must wrap its choice in `<...>`. Output is
    /// parsed via `prompt::post_process_n_best` which guarantees the
    /// result is verbatim one of the hypotheses — no hallucination,
    /// no unmarked-word corruption, no format drift.
    ///
    /// Uses the dedicated `N_BEST_SYSTEM` prompt instead of the
    /// marker-correction `SYSTEM`, so the model sees a clear list-
    /// selection task with no novel syntax.
    async fn rescore_n_best(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        debug_assert!(
            input.hypotheses.len() >= 2,
            "rescore_n_best requires ≥2 hypotheses"
        );

        let argmax = input.hypotheses[0].clone();

        // Build the user message: ranked candidate list.
        let mut user_msg = String::with_capacity(256 + input.hypotheses.len() * 64);
        user_msg.push_str("Transcription candidates in order of ASR confidence:\n");
        for (i, hyp) in input.hypotheses.iter().enumerate() {
            user_msg.push_str(&format!("{}. {}\n", i + 1, hyp));
        }
        user_msg.push_str("\nMost fluent candidate:");

        let messages = TextMessages::new()
            .add_message(TextMessageRole::System, prompt::N_BEST_SYSTEM)
            .add_message(TextMessageRole::User, &user_msg);

        // Short output budget — the answer is one candidate wrapped in
        // brackets. Cap at the longest hypothesis + 16 tokens for the
        // brackets and any model preamble. This prevents runaway
        // generation and keeps latency low.
        let max_hyp_words = input
            .hypotheses
            .iter()
            .map(|h| h.split_whitespace().count())
            .max()
            .unwrap_or(0);
        let out_budget = (max_hyp_words * 2 + 16)
            .min(self.config.max_output_tokens as usize)
            .max(32);

        let request = RequestBuilder::from(messages)
            .set_deterministic_sampler()
            .set_sampler_topk(1)
            .set_sampler_max_len(out_budget)
            .set_sampler_stop_toks(StopTokens::Seqs(vec!["<|im_end|>".into(), "\n\n".into()]));

        let response = self
            .model
            .send_chat_request(request)
            .await
            .map_err(|e| Error::Llm(format!("n-best inference: {e}")))?;

        let raw_output = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .cloned()
            .unwrap_or_default();

        // Parse → one of the hypotheses verbatim, or argmax fallback.
        // This is the guarantee that makes the corrector strictly
        // additive: we can never output a sentence that isn't already
        // in the candidate list.
        let chosen = prompt::post_process_n_best(&raw_output, &input.hypotheses);

        // Diagnostic: was it the argmax or an alternate?
        if chosen != argmax {
            tracing::debug!(
                argmax = %argmax,
                chosen = %chosen,
                "[LLM n-best] picked alternate"
            );
        } else {
            tracing::debug!("[LLM n-best] kept argmax");
        }

        Ok(PostProcessOutput {
            cleaned_text: chosen,
            metadata: None,
        })
    }
}
