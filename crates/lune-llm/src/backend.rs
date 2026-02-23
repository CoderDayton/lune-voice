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

/// Length of the longest stop sequence string (`<|im_end|>` = 10 bytes).
/// Used to bound the suffix window scanned for stop sequences each decode step.
const MAX_STOP_SEQ_LEN: usize = 10;

/// All stop sequences checked during decoding.  Order is irrelevant for
/// correctness; we scan a small window (~20 bytes) so iteration cost is
/// dominated by `memchr`, not sequence count.
const STOP_SEQUENCES: &[&str] = &["<think>", "</think>", "<|im_end|>"];

// Compile-time guarantee that MAX_STOP_SEQ_LEN stays in sync with STOP_SEQUENCES
// AND that every stop sequence is pure ASCII (required by byte-level scanning).
const _: () = {
    let mut i = 0;
    let mut max = 0;
    while i < STOP_SEQUENCES.len() {
        let bytes = STOP_SEQUENCES[i].as_bytes();
        let len = bytes.len();
        if len > max {
            max = len;
        }
        // Assert all bytes are ASCII so byte-level memchr + starts_with is safe.
        let mut j = 0;
        while j < len {
            assert!(bytes[j] < 0x80, "stop sequence contains non-ASCII byte");
            j += 1;
        }
        i += 1;
    }
    assert!(
        max == MAX_STOP_SEQ_LEN,
        "MAX_STOP_SEQ_LEN does not match longest STOP_SEQUENCES entry"
    );
};

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
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-p sampling.
    pub top_p: f32,
    /// Top-k sampling.
    pub top_k: i32,
    /// Repetition penalty multiplier (1.0 = off, 1.1 = mild, 1.3 = strong).
    pub repeat_penalty: f32,
    /// Number of recent tokens to consider for repetition penalty.
    pub repeat_last_n: i32,
}

impl Default for LocalLlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/Qwen3-0.6B-Q4_K_M.gguf"),
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

/// Scan the tail of `output` for any stop sequence that may straddle the
/// old/new boundary. Returns `true` (and truncates `output`) if found.
///
/// Operates entirely on bytes — all stop sequences are compile-time-verified
/// ASCII, so byte comparisons are equivalent to string comparisons. This
/// avoids `floor_char_boundary` and UTF-8 validation overhead per decode step.
#[inline]
fn check_stop_sequences(output: &mut String, piece_len: usize) -> bool {
    let bytes = output.as_bytes();
    let win_end = bytes.len();
    let win_start = win_end.saturating_sub(piece_len + MAX_STOP_SEQ_LEN - 1);
    let window = &bytes[win_start..];

    // All stop sequences start with '<'. Find each '<' and test all sequences
    // at that position. First match = earliest occurrence, always correct for
    // straddling detection.
    for lt_off in memchr::memchr_iter(b'<', window) {
        let remaining = &window[lt_off..];
        for seq in STOP_SEQUENCES {
            if remaining.len() >= seq.len() && remaining[..seq.len()] == *seq.as_bytes() {
                output.truncate(win_start + lt_off);
                return true;
            }
        }
    }
    false
}

/// Trim any partial stop sequence that straddles the output boundary when
/// generation hits max tokens. Without this, output can end with a prefix
/// like `<|im_en` which leaks control markup.
///
/// Uses a single `memrchr` for the last `<` instead of O(sequences * prefix_len)
/// nested iteration — all stop sequences are ASCII and start with `<`, so any
/// partial prefix must begin at the rightmost `<` in the tail.
fn trim_partial_stop_sequences(output: &mut String) {
    let bytes = output.as_bytes();
    let tail_start = bytes.len().saturating_sub(MAX_STOP_SEQ_LEN);
    let tail = &bytes[tail_start..];

    // All stop sequences start with '<'; a partial prefix must begin at one.
    let Some(lt_off) = memchr::memrchr(b'<', tail) else {
        return;
    };

    let candidate = &bytes[tail_start + lt_off..];

    // If the substring from the last '<' to the end is a strict prefix of any
    // stop sequence, it's a dangling partial — truncate it.
    for seq in STOP_SEQUENCES {
        let seq_bytes = seq.as_bytes();
        if candidate.len() < seq_bytes.len() && seq_bytes[..candidate.len()] == *candidate {
            output.truncate(tail_start + lt_off);
            return;
        }
    }
}

/// Run inference on a prompt string, returning the generated text.
///
/// When `chunk_tx` is `Some`, word-boundary chunks are emitted via the channel
/// as tokens are decoded (streaming mode). When `None`, generation completes
/// silently (batch mode).
///
/// Free function so it can be called from `spawn_blocking` (which requires
/// `'static` + `Send` — the `Arc<Mutex<LlmInner>>` satisfies both).
fn generate_sync(
    inner: &Mutex<LlmInner>,
    config: &LocalLlmConfig,
    prompt_text: &str,
    chunk_tx: Option<tokio::sync::mpsc::Sender<String>>,
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

    if tokens.is_empty() {
        return Err(Error::Llm("prompt tokenized to zero tokens".into()));
    }

    // Use u32 for token counts to avoid i32 overflow on large prompts.
    let prompt_len = u32::try_from(tokens.len())
        .map_err(|_| Error::Llm(format!("prompt too large: {} tokens", tokens.len())))?;
    if prompt_len >= config.n_ctx {
        return Err(Error::Llm(format!(
            "prompt ({prompt_len} tokens) exceeds context window ({})",
            config.n_ctx
        )));
    }

    // Safe cast: prompt_len < n_ctx <= u32::MAX, and n_ctx is a practical context
    // window size (well under i32::MAX), so this won't wrap.
    let available = config.n_ctx - prompt_len;
    if available < config.max_output_tokens {
        tracing::warn!(
            prompt_tokens = prompt_len,
            available,
            max_output = config.max_output_tokens,
            "prompt leaves only {available} tokens for output; generation may truncate"
        );
    }

    let prompt_len_i32 = i32::try_from(prompt_len)
        .map_err(|_| Error::Llm(format!("prompt too large for i32: {prompt_len}")))?;

    // Build batch sized exactly for the prompt — during autoregressive decode
    // we clear and add 1 token, so the prefill size is always sufficient.
    let mut batch = LlamaBatch::new(prompt_len as usize, 1);
    let last_idx = prompt_len_i32 - 1;
    for (i, token) in (0i32..).zip(tokens) {
        batch
            .add(token, i, &[0], i == last_idx)
            .map_err(|e| Error::Llm(format!("batch add: {e}")))?;
    }

    // Prefill.
    ctx.decode(&mut batch)
        .map_err(|e| Error::Llm(format!("decode prefill: {e}")))?;

    // Set up sampler chain — repetition penalty always applied to prevent loops.
    let mut sampler = if config.temperature <= 0.0 {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(config.repeat_last_n, config.repeat_penalty, 0.0, 0.0),
            LlamaSampler::greedy(),
        ])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(config.repeat_last_n, config.repeat_penalty, 0.0, 0.0),
            LlamaSampler::top_k(config.top_k),
            LlamaSampler::top_p(config.top_p, 1),
            LlamaSampler::temp(config.temperature),
            LlamaSampler::dist(42),
        ])
    };

    // Autoregressive decode.
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    // Pre-allocate for expected output size: max_output_tokens * ~4 bytes avg
    // (tokens ~ 3-4 chars, English UTF-8 ~ 1.1 bytes/char).
    let capacity = (config.max_output_tokens as usize).saturating_mul(4);
    let mut output = String::with_capacity(capacity);
    let mut n_cur = prompt_len_i32;
    // .min() guarantees value fits in i32; cast is infallible.
    let max_output_i32 = config.max_output_tokens.min(i32::MAX as u32) as i32;
    // Saturating add prevents i32 overflow on absurd configs.
    let max_total = n_cur.saturating_add(max_output_i32);
    let eos = inner.model.token_eos();

    // Streaming state: bytes of `output` already sent to the channel.
    let mut emitted_up_to: usize = 0;

    // After prefill, sample from the last logit position (last_idx).
    // After each decode step the batch has exactly 1 token so index is always 0.
    let mut sample_idx = last_idx;

    while n_cur < max_total {
        let token = sampler.sample(&ctx, sample_idx);
        sampler.accept(token);

        if token == eos {
            break;
        }

        match inner.model.token_to_piece(token, &mut decoder, true, None) {
            Ok(ref piece) if !piece.is_empty() => {
                let piece_len = piece.len();
                output.push_str(piece);

                if check_stop_sequences(&mut output, piece_len) {
                    break;
                }

                // Streaming chunk emission (only when channel is provided).
                if let Some(ref tx) = chunk_tx {
                    let safe_end = output.len().saturating_sub(MAX_STOP_SEQ_LEN);
                    if safe_end > emitted_up_to + MIN_CHUNK_BYTES {
                        let safe_slice = &output.as_bytes()[emitted_up_to..safe_end];
                        if let Some(ws_off) = memchr::memrchr(b' ', safe_slice) {
                            let boundary = emitted_up_to + ws_off + 1;
                            let chunk = output[emitted_up_to..boundary].to_string();
                            emitted_up_to = boundary;
                            let _ = tx.blocking_send(chunk);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::trace!(token = token.0, error = %e, "token decode failed");
            }
            _ => {} // empty token piece — skip
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| Error::Llm(format!("batch add decode: {e}")))?;
        n_cur += 1;
        // After clear+add, batch always has exactly 1 token → index 0.
        sample_idx = 0;

        ctx.decode(&mut batch)
            .map_err(|e| Error::Llm(format!("decode step: {e}")))?;
    }

    // If we hit max_total without a stop sequence, trim any partial stop
    // sequence prefix that may be dangling at the end.
    if n_cur >= max_total {
        trim_partial_stop_sequences(&mut output);
    }

    // Final flush: emit everything remaining.
    if let Some(ref tx) = chunk_tx
        && emitted_up_to < output.len()
    {
        let _ = tx.blocking_send(output[emitted_up_to..].to_string());
    }

    Ok(output)
}

/// Minimum bytes past `emitted_up_to` before emitting a chunk.
/// Set to 1 so chunks fire at every word boundary — one word at a time.
const MIN_CHUNK_BYTES: usize = 1;

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
            tokio::task::spawn_blocking(move || generate_sync(&inner, &config, &prompt_text, None))
                .await
                .map_err(|e| Error::Llm(format!("spawn_blocking join: {e}")))??;

        let cleaned = prompt::strip_thinking(&raw_output);

        tracing::debug!(output_len = cleaned.len(), "LLM post-process complete");

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
            "LLM streaming post-process request"
        );

        let inner = Arc::clone(&self.inner);
        let config = Arc::clone(&self.config);

        let output = tokio::task::spawn_blocking(move || {
            generate_sync(&inner, &config, &prompt_text, Some(chunk_tx))
        })
        .await
        .map_err(|e| Error::Llm(format!("spawn_blocking join: {e}")))??;

        let cleaned = prompt::strip_thinking(&output);

        tracing::debug!(
            output_len = cleaned.len(),
            "LLM streaming post-process complete"
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
        let cfg = LocalLlmConfig::default();
        assert_eq!(cfg.n_ctx, 2048);
        assert_eq!(cfg.max_output_tokens, 256);
        assert!((cfg.temperature - 0.0).abs() < f32::EPSILON);
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

    #[test]
    fn stop_sequence_detection() {
        let mut s = String::from("Hello world<|im_end|>");
        assert!(check_stop_sequences(&mut s, 10));
        assert_eq!(s, "Hello world");
    }

    #[test]
    fn stop_sequence_think_open() {
        let mut s = String::from("result<think>");
        assert!(check_stop_sequences(&mut s, 7));
        assert_eq!(s, "result");
    }

    #[test]
    fn stop_sequence_think_close() {
        let mut s = String::from("leaked</think>");
        assert!(check_stop_sequences(&mut s, 8));
        assert_eq!(s, "leaked");
    }

    #[test]
    fn stop_sequence_finds_earliest_match() {
        // If a stop sequence straddles the boundary AND a second occurrence
        // exists entirely in the new piece, `find` (forward) catches the
        // earliest one. `rfind` would miss the straddling occurrence.
        let mut s = String::from("text<think>more<think>");
        // piece_len = 17 covers "nk>more<think>" plus overlap
        assert!(check_stop_sequences(&mut s, 17));
        assert_eq!(s, "text");
    }

    #[test]
    fn partial_stop_sequence_trimming() {
        let mut s = String::from("output text<|im_en");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "output text");
    }

    #[test]
    fn partial_stop_sequence_trimming_longest_wins() {
        // `<|im_end` is an 8-byte prefix of `<|im_end|>`.
        // The old code matched shortest-first (`<` = 1 byte) and returned early,
        // leaving `<|im_end` minus 1 char. Verify the longest prefix is trimmed.
        let mut s = String::from("output text<|im_end");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "output text");
    }

    #[test]
    fn no_partial_trim_when_clean() {
        let mut s = String::from("clean output");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "clean output");
    }

    #[test]
    fn stop_sequence_early_exit_no_angle_bracket() {
        // No '<' in the window — check_stop_sequences should short-circuit.
        let mut s = String::from("hello world");
        assert!(!check_stop_sequences(&mut s, 5));
        assert_eq!(s, "hello world");
    }

    #[test]
    fn partial_trim_single_angle_bracket() {
        // A lone '<' at the end is a 1-byte prefix of all stop sequences.
        let mut s = String::from("output<");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "output");
    }
}
