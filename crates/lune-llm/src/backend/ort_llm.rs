//! ONNX style LLM with auto-detected model parameters and KV-cache decode.
//!
//! Supports any causal LM exported via onnxruntime-genai (Llama, Qwen, etc.)
//! by probing the session's input metadata at load time.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use half::f16;
use ndarray::{Array2, Array4};
use ort::logging::LogLevel;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, TensorRef, Value};
use tokenizers::Tokenizer;
use tracing::info;

use lune_asr::ort_utils::{make_shrink_run_options, quiet_ort_logger};
use lune_core::error::{Error, Result};

const DEFAULT_MAX_TOKENS: usize = 512;

pub struct OrtStyleLlmConfig {
    pub model_dir: PathBuf,
    pub use_gpu: bool,
    /// True when the user explicitly chose CUDA — registers the EP with
    /// `error_on_failure()` so the session fails to load instead of silently
    /// running on CPU.
    pub cuda_required: bool,
    pub max_new_tokens: usize,
}

impl Default for OrtStyleLlmConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            use_gpu: true,
            cuda_required: false,
            max_new_tokens: DEFAULT_MAX_TOKENS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ChatFormat {
    Llama,  // <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
    ChatML, // <|im_start|>role\n...<|im_end|>
}

struct ChatTokenIds {
    im_start: i64,
    im_end: i64,
    newline: i64,
}

/// Exact, GPU-resident system-prefix KV: the present outputs of a one-time
/// prefix prefill, bound by reference on every subsequent call. Unlike the
/// quantized `KvCache` (the CPU-mode path, where RAM is the constraint),
/// this is lossless and costs zero per-call work — no dequantization, no
/// re-upload.
struct DevicePrefix {
    kv: Vec<DynValue>,
    len: usize,
}

struct ModelParams {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    has_position_ids: bool,
    kv_is_f16: bool,
    eos_ids: Vec<u32>,
    chat_format: ChatFormat,
    chat_tokens: Option<ChatTokenIds>,
}

pub struct OrtStyleLlm {
    /// Declared before `session`: fields drop in declaration order, and these
    /// ORT-owned device values must be released while their session's
    /// allocator still exists.
    device_prefix: std::sync::Mutex<Option<DevicePrefix>>,
    session: Mutex<Session>,
    /// Session output names in declaration order: `[0]` = logits, `[1..]` =
    /// present-KV tensors matching the past-KV input order. Captured once at
    /// load so the IO-binding decode loop can bind/extract by name.
    output_names: Vec<String>,
    embed_session: Option<Mutex<Session>>,
    tokenizer: Tokenizer,
    max_new_tokens: usize,
    params: ModelParams,
    sampling_params: super::sampler::SamplingParams,
    kv_cache: std::sync::Mutex<super::kv_cache::KvCache>,
    cached_prefix_hash: std::sync::Mutex<u64>,
    use_gpu: bool,
}

fn e(msg: impl std::fmt::Display) -> Error {
    Error::config(msg.to_string())
}

fn dyn_i64(arr: Array2<i64>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
}
fn dyn_f16_4(arr: Array4<f16>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
}
fn dyn_f32_4(arr: Array4<f32>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
}

fn llm_kv_names(num_layers: usize) -> &'static [String] {
    use std::sync::OnceLock;
    // Max 64 layers covers all current LLM architectures.
    const MAX_LAYERS: usize = 64;
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    let all = NAMES.get_or_init(|| {
        let mut v = Vec::with_capacity(MAX_LAYERS * 2);
        for i in 0..MAX_LAYERS {
            v.push(format!("past_key_values.{i}.key"));
            v.push(format!("past_key_values.{i}.value"));
        }
        v
    });
    &all[..num_layers * 2]
}

fn empty_kv(heads: usize, dim: usize, is_f16: bool) -> Result<(DynValue, DynValue)> {
    if is_f16 {
        let z = Array4::<f16>::zeros((1, heads, 0, dim));
        Ok((dyn_f16_4(z.clone())?, dyn_f16_4(z)?))
    } else {
        let z = Array4::<f32>::zeros((1, heads, 0, dim));
        Ok((dyn_f32_4(z.clone())?, dyn_f32_4(z)?))
    }
}

fn probe_model_params(session: &Session, tokenizer: &Tokenizer) -> ModelParams {
    let inputs = session.inputs();
    let has_position_ids = inputs.iter().any(|i| i.name() == "position_ids");

    let mut num_layers = 0usize;
    let mut num_kv_heads = 8;
    let mut head_dim = 128;
    let mut kv_is_f16 = false;
    for inp in inputs {
        let name = inp.name();
        if let Some(rest) = name.strip_prefix("past_key_values.")
            && let Some(layer_str) = rest.strip_suffix(".key")
        {
            if let Ok(layer) = layer_str.parse::<usize>() {
                num_layers = num_layers.max(layer + 1);
            }
            if let Some(shape) = inp.dtype().tensor_shape()
                && shape.len() == 4
            {
                num_kv_heads = shape[1] as usize;
                head_dim = shape[3] as usize;
            }
            if inp.dtype().tensor_type() == Some(ort::value::TensorElementType::Float16) {
                kv_is_f16 = true;
            }
        }
    }

    let mut eos_ids: Vec<u32> = Vec::new();
    let vocab = tokenizer.get_vocab(true);
    for name in [
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "</s>",
        "<|end|>",
    ] {
        if let Some(&id) = vocab.get(name) {
            eos_ids.push(id);
        }
    }
    if eos_ids.is_empty() {
        eos_ids.push(2);
    }

    let chat_tokens = if let (Some(&im_start), Some(&im_end)) =
        (vocab.get("<|im_start|>"), vocab.get("<|im_end|>"))
    {
        let newline = vocab.get("\n").copied().unwrap_or(198) as i64;
        Some(ChatTokenIds {
            im_start: im_start as i64,
            im_end: im_end as i64,
            newline,
        })
    } else {
        None
    };

    let chat_format = if chat_tokens.is_some() {
        ChatFormat::ChatML
    } else if vocab.contains_key("<|begin_of_text|>") {
        ChatFormat::Llama
    } else {
        ChatFormat::ChatML
    };

    info!(
        num_layers,
        num_kv_heads,
        head_dim,
        has_position_ids,
        kv_is_f16,
        eos_count = eos_ids.len(),
        ?chat_format,
        "auto-detected model params"
    );

    ModelParams {
        num_layers,
        num_kv_heads,
        head_dim,
        has_position_ids,
        kv_is_f16,
        eos_ids,
        chat_format,
        chat_tokens,
    }
}

impl OrtStyleLlm {
    pub fn new(config: OrtStyleLlmConfig) -> Result<Self> {
        let files = find_model(&config.model_dir)?;

        let logger = quiet_ort_logger();
        let intra_threads = std::thread::available_parallelism()
            .map(|n| n.get().clamp(2, 8))
            .unwrap_or(4);

        let cuda_required = config.cuda_required;
        let make_builder = |gpu: bool, path: &Path| -> Result<Session> {
            let mut b = Session::builder()
                .map_err(e)?
                .with_logger(logger.clone())
                .map_err(e)?
                .with_log_level(LogLevel::Error)
                .map_err(e)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(e)?
                .with_intra_threads(intra_threads)
                .map_err(e)?;
            // Deliberately NOT `with_env_allocators()`: the LLM keeps its own
            // CUDA arena so that the TTL unload actually returns VRAM to the
            // driver. With the env-shared arena, freed LLM memory only went
            // back to the arena the always-resident ASR sessions keep alive —
            // "unloading" never lowered the GPU footprint.
            if gpu {
                // Use ORT default arena strategy (kNextPowerOfTwo). Variable
                // KV-cache shapes per decode step fragment a SameAsRequested
                // arena across runs; power-of-two buckets reuse cleanly.
                // Always error_on_failure: a silent CPU fallback turns a
                // ~0.5 s correction into multi-second latency with no signal.
                // The caller below decides whether a CPU retry is allowed.
                let cuda_ep = ort::ep::CUDA::default()
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                    .with_conv_max_workspace(false)
                    .with_tf32(true)
                    .build()
                    .error_on_failure();
                b = b.with_execution_providers([cuda_ep]).map_err(e)?;
            } else {
                let opt_path = path.with_extension("onnx.opt");
                b = b.with_optimized_model_path(&opt_path).map_err(e)?;
            }
            b.commit_from_file(path).map_err(e)
        };

        // `gpu_active` tracks where the session actually lives — it drives
        // the IO-binding device target, so it must never disagree with the EP.
        let (session, gpu_active) = match make_builder(config.use_gpu, &files.decoder) {
            Ok(s) => (s, config.use_gpu),
            Err(err) if config.use_gpu && !cuda_required => {
                tracing::error!(
                    error = %err,
                    "CUDA unavailable for the correction LLM — retrying on CPU. \
                     Corrections will be slower until the GPU is back."
                );
                (make_builder(false, &files.decoder)?, false)
            }
            Err(err) => return Err(err),
        };
        info!(gpu = gpu_active, "LLM session execution provider resolved");
        let embed_session = files
            .embed_tokens
            .as_ref()
            .map(|p| make_builder(gpu_active, p))
            .transpose()?;

        let tokenizer = Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
            .map_err(|e| Error::config(format!("tokenizer: {e}")))?;

        let params = probe_model_params(&session, &tokenizer);

        // Transcript correction is a copy-edit task, not chat: decode greedily
        // and deterministically. The chat defaults (temp 0.7, top-p 0.8,
        // presence_penalty 1.5) are actively destructive here — presence
        // penalty in particular punishes the model for re-emitting the input's
        // own tokens, forcing paraphrase drift that grows with utterance
        // length. top_k=1 + identity penalties also hits the sampler's fused
        // argmax fast path (no softmax workspace per step).
        let sampling_params = super::sampler::SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            eos_token_id: params.eos_ids.first().copied(),
            stop_token_ids: params.eos_ids.clone().into_boxed_slice(),
        };

        let kv_cache = super::kv_cache::KvCache::new(super::kv_cache::KvCacheConfig {
            num_layers: params.num_layers,
            num_kv_heads: params.num_kv_heads,
            head_dim: params.head_dim,
            is_f16: params.kv_is_f16,
            ..Default::default()
        });

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if output_names.is_empty() {
            return Err(Error::config("LLM model has no outputs"));
        }

        info!(model = %files.decoder.display(), split = embed_session.is_some(), "style LLM loaded");
        Ok(Self {
            session: Mutex::new(session),
            output_names,
            embed_session: embed_session.map(Mutex::new),
            tokenizer,
            max_new_tokens: config.max_new_tokens,
            params,
            sampling_params,
            kv_cache: std::sync::Mutex::new(kv_cache),
            device_prefix: std::sync::Mutex::new(None),
            cached_prefix_hash: std::sync::Mutex::new(0),
            use_gpu: gpu_active,
        })
    }

    pub fn user_prompt_with_examples(
        transcript: &str,
        language: &str,
        examples: &[(String, String)],
        context: Option<&str>,
        uncertain_words: &[(String, usize)],
    ) -> String {
        // The `language` field arrives in inconsistent forms — the runtime
        // sometimes hands us a BCP-47 code ("en") and sometimes a full
        // English name from the ASR's language head ("English", "Dutch").
        // Only inject a recognizable label; otherwise omit it entirely so
        // the model isn't shown an ungrammatical header like
        // "the following en transcript".
        let label = lune_core::lang::display_name(language);
        let mut header = match label {
            Some(name) => format!("Correct the following {name} transcript:"),
            None => "Correct the following transcript:".to_string(),
        };
        // Confidence-guided correction (arXiv:2509.25048): name the words the
        // ASR flagged as low-confidence so the model focuses edits where
        // errors are likely. Embedded in the instruction header, parenthesized
        // and away from the transcript — a small model echoes a free-standing
        // hint list at the end of the prompt straight into its output, and
        // the echoed copies then scramble the validator's LCS alignment.
        if !uncertain_words.is_empty() {
            header.push_str(" (likely misheard: ");
            for (i, (word, _)) in uncertain_words.iter().take(8).enumerate() {
                if i > 0 {
                    header.push_str(", ");
                }
                header.push_str(word);
            }
            header.push(')');
        }
        let mut prompt = String::with_capacity(header.len() + transcript.len() + 128);
        prompt.push_str(&header);
        if let Some(ctx) = context.filter(|c| !c.trim().is_empty()) {
            // Cross-utterance context: lets the model keep names, spellings,
            // and topic vocabulary consistent with what the user just
            // dictated. Reference-only — the validator discards any of it
            // that leaks into the output.
            prompt
                .push_str("\n\nEarlier dictation (context only — do not repeat or correct it):\n");
            prompt.push_str(ctx);
        }
        if !examples.is_empty() {
            prompt.push_str("\n\nExamples:");
            for (input, output) in examples {
                prompt.push_str("\nInput: ");
                prompt.push_str(input);
                prompt.push_str("\nOutput: ");
                prompt.push_str(output);
            }
        }
        prompt.push('\n');
        prompt.push_str(transcript);
        prompt.push_str("\n\n/no_think");
        prompt
    }

    pub fn apply_correction(
        &self,
        transcript: &str,
        language: &str,
        uncertain_words: &[(String, usize)],
    ) -> Result<String> {
        use std::hash::{Hash, Hasher};
        let system = lune_core::config::BASE_SYSTEM_PROMPT.replace("{lang}", language);
        let user_text =
            Self::user_prompt_with_examples(transcript, language, &[], None, uncertain_words);

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        system.hash(&mut hasher);
        let sys_hash = hasher.finish();

        {
            let mut prev = self.cached_prefix_hash.lock().map_err(e)?;
            if *prev != sys_hash {
                self.kv_cache.lock().map_err(e)?.clear_prefix();
                self.device_prefix.lock().map_err(e)?.take();
                *prev = sys_hash;
            }
        }

        let (ids, sys_prefix_len) = self.build_chat_ids(&system, &user_text)?;
        let out = self.generate(&ids, sys_prefix_len)?;
        let text = self
            .tokenizer
            .decode(&out, true)
            .map_err(|err| Error::config(format!("detokenize: {err}")))?;
        let text = strip_think_tags(text.trim());
        Ok(text.to_string())
    }

    /// Run a styling-only pass with no system prompt (sys_prefix_len=0).
    /// Keeps the correction KV cache untouched.
    pub fn apply_styling(
        &self,
        corrected_text: &str,
        style: &lune_core::config::StylePreset,
    ) -> Result<String> {
        let user_text = style.build_styling_prompt(corrected_text);
        let (ids, _sys_prefix_len) = self.build_chat_ids("", &user_text)?;
        let out = self.generate(&ids, 0)?;
        let text = self
            .tokenizer
            .decode(&out, true)
            .map_err(|err| Error::config(format!("detokenize: {err}")))?;
        let text = strip_think_tags(text.trim());
        Ok(text.to_string())
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| Error::config(format!("tokenize: {err}")))?;
        Ok(enc.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Returns `(token_ids, system_prefix_len)`.
    fn build_chat_ids(&self, system: &str, user_text: &str) -> Result<(Vec<i64>, usize)> {
        match (&self.params.chat_format, &self.params.chat_tokens) {
            (ChatFormat::ChatML, Some(ct)) => {
                let sys_ids = self.tokenize_text(system)?;
                let user_ids = self.tokenize_text(user_text)?;

                // <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
                let mut ids = Vec::with_capacity(10 + sys_ids.len() + user_ids.len());
                ids.push(ct.im_start);
                ids.extend(self.tokenize_text("system")?);
                ids.push(ct.newline);
                ids.extend(sys_ids);
                ids.push(ct.im_end);
                ids.push(ct.newline);
                let sys_prefix_len = ids.len();
                ids.push(ct.im_start);
                ids.extend(self.tokenize_text("user")?);
                ids.push(ct.newline);
                ids.extend(user_ids);
                ids.push(ct.im_end);
                ids.push(ct.newline);
                ids.push(ct.im_start);
                ids.extend(self.tokenize_text("assistant")?);
                ids.push(ct.newline);
                // Template-level thinking suppression: pre-fill a closed,
                // empty think block so decoding starts after it — the same
                // mechanism as Qwen3's official enable_thinking=false
                // template. The quantized model ignores the soft `/no_think`
                // directive and otherwise burns the entire copy-edit decode
                // budget inside an unterminated <think> block, which made
                // every correction pass return its input unchanged.
                ids.extend(self.tokenize_text("<think>\n\n</think>\n\n")?);
                Ok((ids, sys_prefix_len))
            }
            _ => {
                let sys_prompt = format!(
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                     {system}<|eot_id|>"
                );
                let sys_ids = self.tokenize_text(&sys_prompt)?;
                let sys_prefix_len = sys_ids.len();
                let user_prompt = format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n\
                     {user_text}<|eot_id|>\
                     <|start_header_id|>assistant<|end_header_id|>\n\n"
                );
                let user_ids = self.tokenize_text(&user_prompt)?;
                let mut ids = sys_ids;
                ids.extend(user_ids);
                Ok((ids, sys_prefix_len))
            }
        }
    }

    fn embed_ids(&self, ids: &[i64]) -> Result<Option<DynValue>> {
        let embed = match &self.embed_session {
            Some(s) => s,
            None => return Ok(None),
        };
        let mut sess = embed.lock().map_err(e)?;
        let input_ids = Array2::from_shape_vec((1, ids.len()), ids.to_vec()).map_err(e)?;
        let inputs: Inputs = vec![("input_ids".into(), dyn_i64(input_ids)?)];
        let outputs = sess.run(inputs).map_err(e)?;
        // Surface a clean error rather than panic if the embed session
        // somehow returns zero outputs (treat empty as a model contract bug).
        let (_, value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| e("embed session returned no outputs"))?;
        Ok(Some(value))
    }

    fn generate(&self, prompt_ids: &[i64], sys_prefix_len: usize) -> Result<Vec<u32>> {
        use super::sampler::{DecodeState, SamplerWorkspace};

        let p = &self.params;
        let mut session = self.session.lock().map_err(e)?;
        let names = llm_kv_names(p.num_layers);

        let mut state = DecodeState::default();
        let mut ws = SamplerWorkspace::default();
        let mut rng = rand::rng();

        self.ensure_prefix_kv(&mut session, names, prompt_ids, sys_prefix_len)?;
        let Some((kv, total_prefill)) = self.prefill_user(
            &mut session,
            names,
            prompt_ids,
            sys_prefix_len,
            &mut state,
            &mut ws,
            &mut rng,
        )?
        else {
            return Ok(vec![]);
        };
        // Copy-edit budget: the output of a correction/styling pass is
        // bounded by its input, so cap decode at the user-portion length
        // plus headroom (punctuation expansion, contractions, the empty
        // think-block stub) instead of always allowing `max_new_tokens`.
        // Greedy decode stops at EOS anyway; this bounds the runaway case
        // where a degenerate output would otherwise burn the full budget.
        let user_len = prompt_ids.len() - sys_prefix_len;
        let budget = (user_len + user_len / 4 + 48).min(self.max_new_tokens);
        self.decode_loop(
            &mut session,
            names,
            kv,
            total_prefill,
            budget,
            &mut state,
            &mut ws,
            &mut rng,
        )?;
        // No arena janitor here, deliberately: the LLM's decode ladder lives
        // in its private arena, which is destroyed wholesale at TTL unload
        // (see make_builder — no env allocators). Shrinking per correction
        // measured +0.35 s on long corrections for ~0.5 GB that is only held
        // during the short loaded window — a bad trade. The ASR decoder,
        // which is resident forever, does run a post-utterance janitor.
        Ok(state.generated)
    }

    /// Phase 1: on first call, prefill the system prefix alone and freeze
    /// the resulting KV. GPU mode keeps the present outputs device-resident
    /// and exact; CPU mode compresses into the quantized cache. No-op on
    /// subsequent calls or when there is no prefix to freeze.
    fn ensure_prefix_kv(
        &self,
        session: &mut Session,
        names: &'static [String],
        prompt_ids: &[i64],
        sys_prefix_len: usize,
    ) -> Result<()> {
        if sys_prefix_len == 0 {
            return Ok(());
        }
        if self.use_gpu {
            return self.ensure_prefix_kv_gpu(session, names, prompt_ids, sys_prefix_len);
        }
        let p = &self.params;
        let mut cache = self.kv_cache.lock().map_err(e)?;
        if cache.has_prefix() {
            return Ok(());
        }
        let split = self.embed_session.is_some();
        let prefix_ids = &prompt_ids[..sys_prefix_len];
        let pn = prefix_ids.len();
        let mut inputs: Inputs<'static> =
            Vec::with_capacity(input_capacity(p.num_layers, p.has_position_ids));
        if split {
            let embeds = self
                .embed_ids(prefix_ids)?
                .ok_or_else(|| e("split decode requires embed session"))?;
            build_embeds_inputs_into(&mut inputs, embeds, pn, p.has_position_ids)?;
        } else {
            build_inputs_into(&mut inputs, prefix_ids, pn, p.has_position_ids)?;
        }
        for i in 0..p.num_layers {
            let (k, v) = empty_kv(p.num_kv_heads, p.head_dim, p.kv_is_f16)?;
            inputs.push((std::borrow::Cow::Borrowed(names[i * 2].as_str()), k));
            inputs.push((std::borrow::Cow::Borrowed(names[i * 2 + 1].as_str()), v));
        }
        let outputs = session.run(inputs).map_err(e)?;
        let raw_kv: Vec<DynValue> = outputs.into_iter().skip(1).map(|(_, v)| v).collect();
        cache.freeze_prefix(raw_kv)?;
        Ok(())
    }

    /// GPU variant of phase 1: prefill the prefix via IO binding so its
    /// present-KV lands (and stays) in device memory, stored exactly.
    fn ensure_prefix_kv_gpu(
        &self,
        session: &mut Session,
        names: &'static [String],
        prompt_ids: &[i64],
        sys_prefix_len: usize,
    ) -> Result<()> {
        let p = &self.params;
        {
            let dp = self.device_prefix.lock().map_err(e)?;
            if dp.is_some() {
                return Ok(());
            }
        }
        let split = self.embed_session.is_some();
        let prefix_ids = &prompt_ids[..sys_prefix_len];
        let pn = prefix_ids.len();

        let logits_name = self.output_names[0].clone();
        let present_names: Vec<String> = self.output_names[1..].to_vec();

        let mut binding = session.create_binding().map_err(e)?;
        let ids_val;
        let emb_val;
        if split {
            emb_val = self
                .embed_ids(prefix_ids)?
                .ok_or_else(|| e("split decode requires embed session"))?;
            binding.bind_input("inputs_embeds", &emb_val).map_err(e)?;
        } else {
            ids_val = dyn_i64(Array2::from_shape_vec((1, pn), prefix_ids.to_vec()).map_err(e)?)?;
            binding.bind_input("input_ids", &ids_val).map_err(e)?;
        }
        let mask_val = dyn_i64(Array2::<i64>::ones((1, pn)))?;
        binding.bind_input("attention_mask", &mask_val).map_err(e)?;
        let pos_val;
        if p.has_position_ids {
            let positions: Vec<i64> = (0..pn as i64).collect();
            pos_val = dyn_i64(Array2::from_shape_vec((1, pn), positions).map_err(e)?)?;
            binding.bind_input("position_ids", &pos_val).map_err(e)?;
        }
        let mut empties: Vec<DynValue> = Vec::with_capacity(p.num_layers * 2);
        for _ in 0..p.num_layers {
            let (k, v) = empty_kv(p.num_kv_heads, p.head_dim, p.kv_is_f16)?;
            empties.push(k);
            empties.push(v);
        }
        for (i, v) in empties.iter().enumerate() {
            binding.bind_input(names[i].as_str(), v).map_err(e)?;
        }
        self.bind_outputs(&mut binding, &logits_name, &present_names)?;

        let prefill_opts = make_shrink_run_options(self.use_gpu)?;
        let mut out = session
            .run_binding_with_options(&binding, &prefill_opts)
            .map_err(e)?;
        let mut kv = Vec::with_capacity(present_names.len());
        for name in &present_names {
            kv.push(
                out.remove(name.as_str())
                    .ok_or_else(|| e(format!("missing present output {name}")))?,
            );
        }
        *self.device_prefix.lock().map_err(e)? = Some(DevicePrefix { kv, len: pn });
        info!(prefix_tokens = pn, "system prefix KV frozen on device");
        Ok(())
    }

    /// GPU variant of phase 2: user prefill bound against the device-resident
    /// prefix; present outputs stay on device for the decode loop.
    #[allow(clippy::too_many_arguments)]
    fn prefill_user_gpu(
        &self,
        session: &mut Session,
        names: &'static [String],
        prompt_ids: &[i64],
        sys_prefix_len: usize,
        state: &mut super::sampler::DecodeState,
        ws: &mut super::sampler::SamplerWorkspace,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Result<Option<(Vec<DynValue>, usize)>> {
        use super::sampler::sample_last_token;

        let p = &self.params;
        let split = self.embed_session.is_some();
        let logits_name = self.output_names[0].clone();
        let present_names: Vec<String> = self.output_names[1..].to_vec();

        // Hold the guard across the run: binding borrows the prefix values.
        let dp_guard = self.device_prefix.lock().map_err(e)?;
        let (prefill_ids, prefix_seq_len) = match (dp_guard.as_ref(), sys_prefix_len) {
            (Some(dp), n) if n > 0 => (&prompt_ids[n..], dp.len),
            _ => (prompt_ids, 0),
        };
        let user_len = prefill_ids.len();
        let total_prefill = prefix_seq_len + user_len;

        let mut binding = session.create_binding().map_err(e)?;
        let ids_val;
        let emb_val;
        if split {
            emb_val = self
                .embed_ids(prefill_ids)?
                .ok_or_else(|| e("split decode requires embed session"))?;
            binding.bind_input("inputs_embeds", &emb_val).map_err(e)?;
        } else {
            ids_val =
                dyn_i64(Array2::from_shape_vec((1, user_len), prefill_ids.to_vec()).map_err(e)?)?;
            binding.bind_input("input_ids", &ids_val).map_err(e)?;
        }
        let mask_val = dyn_i64(Array2::<i64>::ones((1, total_prefill)))?;
        binding.bind_input("attention_mask", &mask_val).map_err(e)?;
        let pos_val;
        if p.has_position_ids {
            let positions: Vec<i64> = (prefix_seq_len as i64..total_prefill as i64).collect();
            pos_val = dyn_i64(Array2::from_shape_vec((1, user_len), positions).map_err(e)?)?;
            binding.bind_input("position_ids", &pos_val).map_err(e)?;
        }
        let mut empties: Vec<DynValue> = Vec::new();
        if let Some(dp) = dp_guard.as_ref().filter(|_| prefix_seq_len > 0) {
            for (i, v) in dp.kv.iter().enumerate() {
                binding.bind_input(names[i].as_str(), v).map_err(e)?;
            }
        } else {
            empties.reserve(p.num_layers * 2);
            for _ in 0..p.num_layers {
                let (k, v) = empty_kv(p.num_kv_heads, p.head_dim, p.kv_is_f16)?;
                empties.push(k);
                empties.push(v);
            }
            for (i, v) in empties.iter().enumerate() {
                binding.bind_input(names[i].as_str(), v).map_err(e)?;
            }
        }
        self.bind_outputs(&mut binding, &logits_name, &present_names)?;

        let prefill_opts = make_shrink_run_options(self.use_gpu)?;
        let mut out = session
            .run_binding_with_options(&binding, &prefill_opts)
            .map_err(e)?;
        drop(dp_guard);

        let first = sample_last_token(
            &out[logits_name.as_str()],
            &self.sampling_params,
            state,
            ws,
            rng,
        )?;
        if state.should_stop(first, &self.sampling_params) {
            return Ok(None);
        }
        state.push(first);

        let mut kv = Vec::with_capacity(present_names.len());
        for name in &present_names {
            kv.push(
                out.remove(name.as_str())
                    .ok_or_else(|| e(format!("missing present output {name}")))?,
            );
        }
        Ok(Some((kv, total_prefill)))
    }

    /// Phase 2: prefill the user portion of the prompt seeded with the
    /// cached prefix KV. Samples the first generated token; returns
    /// `None` if that token already signals stop. On success, returns the
    /// initial KV outputs and `total_prefill` (prefix + user lengths) for
    /// the decode loop.
    #[allow(clippy::too_many_arguments)]
    fn prefill_user(
        &self,
        session: &mut Session,
        names: &'static [String],
        prompt_ids: &[i64],
        sys_prefix_len: usize,
        state: &mut super::sampler::DecodeState,
        ws: &mut super::sampler::SamplerWorkspace,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Result<Option<(Vec<DynValue>, usize)>> {
        use super::sampler::sample_last_token;

        let p = &self.params;
        let split = self.embed_session.is_some();

        if self.use_gpu {
            return self.prefill_user_gpu(
                session,
                names,
                prompt_ids,
                sys_prefix_len,
                state,
                ws,
                rng,
            );
        }

        // ── Acquire prefix KV (or build empty when no system prefix) ───
        let cache = self.kv_cache.lock().map_err(e)?;
        let prefix_seq_len;
        let (prefill_ids, initial_kv) = if cache.has_prefix() && sys_prefix_len > 0 {
            let cached_kv = cache.prefix_kv()?;
            prefix_seq_len = cache.prefix_len();
            (&prompt_ids[sys_prefix_len..], cached_kv)
        } else {
            prefix_seq_len = 0;
            let mut empty = Vec::with_capacity(p.num_layers * 2);
            for _ in 0..p.num_layers {
                let (k, v) = empty_kv(p.num_kv_heads, p.head_dim, p.kv_is_f16)?;
                empty.push(k);
                empty.push(v);
            }
            (prompt_ids, empty)
        };
        drop(cache);

        let user_len = prefill_ids.len();
        let total_prefill = prefix_seq_len + user_len;

        let mut inputs: Inputs<'static> =
            Vec::with_capacity(input_capacity(p.num_layers, p.has_position_ids));
        if split {
            let embeds = self
                .embed_ids(prefill_ids)?
                .ok_or_else(|| e("split decode requires embed session"))?;
            build_embeds_inputs_with_past_into(
                &mut inputs,
                embeds,
                user_len,
                prefix_seq_len,
                p.has_position_ids,
            )?;
        } else {
            build_inputs_with_past_into(
                &mut inputs,
                prefill_ids,
                user_len,
                prefix_seq_len,
                p.has_position_ids,
            )?;
        }
        for (i, kv_val) in initial_kv.into_iter().enumerate() {
            inputs.push((std::borrow::Cow::Borrowed(names[i].as_str()), kv_val));
        }

        // Apply CUDA arena shrinkage on this prefill run (once per generate
        // call). Releases unused chunks from the previous call's
        // variable-shape decode loop before this call grows the arena.
        // Doing it here, not in the per-step decode loop, avoids per-step
        // shrink/grow churn.
        let prefill_opts = make_shrink_run_options(self.use_gpu)?;
        let outputs = session.run_with_options(inputs, &prefill_opts).map_err(e)?;
        let first = sample_last_token(&outputs[0], &self.sampling_params, state, ws, rng)?;
        if state.should_stop(first, &self.sampling_params) {
            return Ok(None);
        }
        state.push(first);

        // Reused across the decode loop to avoid reallocating `Vec<DynValue>`
        // every step (28 layers × 2 = 56 entries × up to ~448 tokens).
        let mut kv: Vec<DynValue> = Vec::with_capacity(p.num_layers * 2);
        kv.extend(outputs.into_iter().skip(1).map(|(_, v)| v));
        Ok(Some((kv, total_prefill)))
    }

    /// Phase 3: autoregressive decode via IO binding.
    ///
    /// Plain `session.run` fetches every output to host memory — with the
    /// CUDA EP that round-trips the entire KV cache (num_layers × 2 tensors,
    /// growing with sequence length) over PCIe **per token**, O(seq²) traffic
    /// that dominated long-utterance latency. Binding the present-KV outputs
    /// to device memory keeps the cache GPU-resident across steps; only the
    /// logits land on the host for sampling. The attention mask is a
    /// preallocated all-ones row bound as a per-step prefix view — no
    /// per-token allocation or upload of a growing mask.
    #[allow(clippy::too_many_arguments)]
    fn decode_loop(
        &self,
        session: &mut Session,
        names: &'static [String],
        mut kv: Vec<DynValue>,
        total_prefill: usize,
        max_tokens: usize,
        state: &mut super::sampler::DecodeState,
        ws: &mut super::sampler::SamplerWorkspace,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Result<()> {
        use super::sampler::sample_last_token;

        let p = &self.params;
        let split = self.embed_session.is_some();
        let logits_name = self.output_names[0].clone();
        let present_names: Vec<String> = self.output_names[1..].to_vec();
        if present_names.len() != kv.len() {
            return Err(e(format!(
                "present outputs ({}) != past inputs ({})",
                present_names.len(),
                kv.len()
            )));
        }

        // Worst-case mask width: prefill + every token we may generate.
        let mask_ones = Array2::<i64>::ones((1, total_prefill + max_tokens + 1));
        let mut pos_arr = Array2::<i64>::zeros((1, 1));
        let mut binding = session.create_binding().map_err(e)?;

        for _ in 1..max_tokens {
            let tok = *state.generated.last().unwrap() as i64;
            let total = total_prefill + state.generated.len();

            let ids_val;
            let emb_val;
            binding.clear();
            if split {
                emb_val = self
                    .embed_ids(&[tok])?
                    .ok_or_else(|| e("split decode requires embed session"))?;
                binding.bind_input("inputs_embeds", &emb_val).map_err(e)?;
            } else {
                ids_val = dyn_i64(Array2::from_elem((1, 1), tok))?;
                binding.bind_input("input_ids", &ids_val).map_err(e)?;
            }
            let mask_ref =
                TensorRef::from_array_view(mask_ones.slice(ndarray::s![.., ..total])).map_err(e)?;
            binding.bind_input("attention_mask", &mask_ref).map_err(e)?;
            let pos_val;
            if p.has_position_ids {
                pos_arr.fill((total - 1) as i64);
                pos_val = dyn_i64(pos_arr.clone())?;
                binding.bind_input("position_ids", &pos_val).map_err(e)?;
            }
            // Feed the previous step's (device-resident) KV straight back in.
            for (i, v) in kv.drain(..).enumerate() {
                binding.bind_input(names[i].as_str(), &v).map_err(e)?;
            }
            self.bind_outputs(&mut binding, &logits_name, &present_names)?;

            let mut out = session.run_binding(&binding).map_err(e)?;
            let next = sample_last_token(
                &out[logits_name.as_str()],
                &self.sampling_params,
                state,
                ws,
                rng,
            )?;
            if state.should_stop(next, &self.sampling_params) {
                break;
            }
            state.push(next);
            // kv was emptied by drain above; refill in place from this step's
            // present outputs (still on device when use_gpu).
            for name in &present_names {
                kv.push(
                    out.remove(name.as_str())
                        .ok_or_else(|| e(format!("missing present output {name}")))?,
                );
            }
        }
        Ok(())
    }

    /// Bind logits to host (the sampler reads them) and present-KV outputs to
    /// device memory when running on GPU, so the cache never crosses PCIe.
    fn bind_outputs(
        &self,
        binding: &mut ort::session::IoBinding,
        logits_name: &str,
        present_names: &[String],
    ) -> Result<()> {
        let cpu = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(e)?;
        binding
            .bind_output_to_device(logits_name, &cpu)
            .map_err(e)?;
        let kv_mem = if self.use_gpu {
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(e)?
        } else {
            cpu
        };
        for name in present_names {
            binding
                .bind_output_to_device(name.as_str(), &kv_mem)
                .map_err(e)?;
        }
        Ok(())
    }
}

/// Function words the LLM may legitimately insert into ASR output.
/// Deliberately excludes "not" and "no": inserting a negation inverts the
/// sentence's meaning — never a safe unsupervised edit, whatever the model's
/// confidence.
const SAFE_INSERTIONS: &[&str] = &[
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "from", "with", "by", "as", "is", "are",
    "am", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have", "had", "will",
    "would", "could", "should", "can", "may", "might", "shall", "and", "but", "or", "so", "if",
    "that", "i", "it", "he", "she", "we", "they", "me", "him", "her", "us", "them", "you", "my",
    "your", "his", "its", "our", "their", "this",
];

/// Known ASR homophones / confusion pairs the validator may accept.
const ALLOWED_SUBS: &[(&str, &str)] = &[
    ("bye", "buy"),
    ("buy", "by"),
    ("by", "buy"),
    ("two", "to"),
    ("too", "to"),
    ("to", "too"),
    ("to", "two"),
    ("meat", "meet"),
    ("meet", "meat"),
    ("there", "their"),
    ("their", "there"),
    ("theyre", "they're"),
    ("they're", "there"),
    ("they're", "their"),
    ("hear", "here"),
    ("here", "hear"),
    ("right", "write"),
    ("write", "right"),
    ("no", "know"),
    ("know", "no"),
    ("lets", "let's"),
    ("cloth", "clothes"),
    ("clothe", "clothes"),
    ("brought", "bought"),
    ("bough", "bought"),
    ("weather", "whether"),
    ("whether", "weather"),
    ("brake", "break"),
    ("break", "brake"),
    ("peace", "piece"),
    ("piece", "peace"),
    ("bare", "bear"),
    ("bear", "bare"),
    ("where", "wear"),
    ("wear", "where"),
    ("wont", "won't"),
    ("dont", "don't"),
    ("cant", "can't"),
    ("im", "i'm"),
    ("ive", "i've"),
    ("youre", "you're"),
    ("hes", "he's"),
    ("shes", "she's"),
    // NOTE: deliberately absent: ("your","you're"), ("its","it's"),
    // ("were","we're"), ("let","let's"). Those expand a usually-correct
    // word into a contraction — the LLM applies them far more often as
    // grammar damage ("your dog" → "you're dog") than as repair.
    ("would", "wood"),
    ("wood", "would"),
    ("wait", "weight"),
    ("weight", "wait"),
    ("seen", "scene"),
    ("scene", "seen"),
    ("week", "weak"),
    ("weak", "week"),
    ("threw", "through"),
    ("through", "threw"),
    ("whole", "hole"),
    ("hole", "whole"),
    ("waste", "waist"),
    ("waist", "waste"),
    ("led", "lead"),
    ("lead", "led"),
    ("course", "coarse"),
    ("coarse", "course"),
];

fn norm_word(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Whether `s` ends with a sentence-terminating punctuation mark.
/// Trailing closing quotes/brackets are stripped before checking.
fn ends_sentence(s: &str) -> bool {
    let trimmed = s.trim_end_matches(['"', '\'', ')', ']', '}', '»']);
    trimmed
        .chars()
        .last()
        .is_some_and(|c| matches!(c, '.' | '!' | '?' | '…'))
}

/// First alphabetic character is uppercase.
fn starts_uppercase_alpha(s: &str) -> bool {
    s.chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(|c| c.is_uppercase())
}

/// First alphabetic character is lowercase.
fn starts_lowercase_alpha(s: &str) -> bool {
    s.chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(|c| c.is_lowercase())
}

/// Any uppercase letter after the first alphabetic character — acronyms
/// ("NASA"), mixed-case brands ("GitHub", "iPhone" via 'P').
fn has_interior_uppercase(s: &str) -> bool {
    s.chars()
        .filter(|c| c.is_alphabetic())
        .skip(1)
        .any(|c| c.is_uppercase())
}

/// Transplant the input word's letter casing onto the output word, keeping the
/// output's punctuation. Only meaningful when both share the same alphanumeric
/// core ignoring case (which `Op::Match` guarantees).
fn transplant_case(inp_w: &str, out_w: &str) -> String {
    let mut inp_alnum = inp_w.chars().filter(|c| c.is_alphanumeric());
    out_w
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                inp_alnum.next().unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Lowercase the first alphabetic character of `s`, leaving everything else
/// (leading punctuation, interior letters, trailing punctuation) intact.
fn lowercase_first_letter(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut done = false;
    for c in s.chars() {
        if !done && c.is_alphabetic() {
            out.extend(c.to_lowercase());
            done = true;
        } else {
            out.push(c);
        }
    }
    out
}

/// Substitutions that must never be accepted, even when they look like a
/// 1-edit spelling repair — expanding a correct word into a contraction is
/// grammar damage far more often than repair.
const BLOCKED_SUBS: &[(&str, &str)] =
    &[("your", "youre"), ("there", "theyre"), ("their", "theyre")];

fn is_allowed_sub(from: &str, to: &str) -> bool {
    ALLOWED_SUBS.iter().any(|(a, b)| from == *a && to == *b)
}

fn is_blocked_sub(from: &str, to: &str) -> bool {
    BLOCKED_SUBS.iter().any(|(a, b)| from == *a && to == *b)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut costs: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut last = i;
        costs[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let old = costs[j + 1];
            costs[j + 1] = if ca == cb {
                last
            } else {
                1 + last.min(costs[j]).min(costs[j + 1])
            };
            last = old;
        }
    }
    costs[b.len()]
}

fn looks_like_spelling_repair(from: &str, to: &str) -> bool {
    let max_len = from.len().max(to.len());
    let dist = levenshtein(from, to);
    max_len >= 4 && (dist <= 1 || (dist <= 2 && max_len >= 7))
}

pub(crate) fn contains_forbidden_output(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        "here is",
        "corrected transcript",
        "the corrected",
        "i changed",
        "i fixed",
        "input:",
        "output:",
        "possibly misheard",
    ]
    .iter()
    .any(|m| lower.contains(m))
}

/// Token-level validator: align input/output words via LCS, accept safe
/// edits (case, punctuation, function-word insertions, known substitutions,
/// spelling repairs), reject everything else.
///
/// Dictionary-mandated substitutions are not handled here — the pipeline
/// pre-seeds those into the input before the LLM runs, so they arrive at
/// this validator as `Op::Match` already.
pub(crate) fn validate_and_merge(input: &str, llm_output: &str, lang: &str) -> String {
    let inp: Vec<&str> = input.split_whitespace().collect();
    let out: Vec<&str> = llm_output.split_whitespace().collect();

    if inp.is_empty() {
        return llm_output.to_string();
    }
    if out.is_empty() {
        return input.to_string();
    }

    let m = inp.len();
    let n = out.len();

    // Pre-normalize once. `norm_word` allocates, and the original code
    // called it on every cell of the O(m*n) DP fill plus again in the
    // backtrack. Memoizing collapses both passes to O(m+n) allocations.
    let inp_norm: Vec<String> = inp.iter().map(|w| norm_word(w)).collect();
    let out_norm: Vec<String> = out.iter().map(|w| norm_word(w)).collect();

    // LCS table — flat 1-D row-major Vec instead of Vec<Vec<u16>>.
    // For a 20-word utterance this collapses 21 inner allocations to 1
    // and gives the inner loop sequential cache access.
    let stride = n + 1;
    let mut dp = vec![0u16; (m + 1) * stride];
    let dp_at = |dp: &[u16], i: usize, j: usize| dp[i * stride + j];
    for i in 1..=m {
        for j in 1..=n {
            let v = if inp_norm[i - 1] == out_norm[j - 1] {
                dp_at(&dp, i - 1, j - 1) + 1
            } else {
                dp_at(&dp, i - 1, j).max(dp_at(&dp, i, j - 1))
            };
            dp[i * stride + j] = v;
        }
    }

    // If fewer than 55% of input words survived, or the word count moved by
    // more than ~50%, the output is suspect (translation, paraphrase, heavy
    // censoring, hallucination) — fall back entirely. Correction edits are
    // local; they never legitimately rewrite half the utterance.
    let lcs_len = dp_at(&dp, m, n) as usize;
    if lcs_len * 100 / m.max(1) < 55 {
        return input.to_string();
    }
    if n > m + m / 2 + 2 || m > n + n / 2 + 2 {
        return input.to_string();
    }

    // Backtrack into an alignment (reversed, then flipped).
    #[derive(Debug)]
    enum Op<'a> {
        Match(&'a str, &'a str),
        Insert(&'a str),
        Delete(&'a str),
    }

    let mut ops: Vec<Op<'_>> = Vec::with_capacity(m + n);
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && inp_norm[i - 1] == out_norm[j - 1] {
            ops.push(Op::Match(inp[i - 1], out[j - 1]));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp_at(&dp, i, j - 1) >= dp_at(&dp, i - 1, j)) {
            ops.push(Op::Insert(out[j - 1]));
            j -= 1;
        } else {
            ops.push(Op::Delete(inp[i - 1]));
            i -= 1;
        }
    }
    ops.reverse();

    let is_en = matches!(lune_core::lang::display_name(lang), Some("English"));

    // Detect the "second-word capitalization" failure mode: the LLM invents
    // a mid-utterance sentence break (period + Title-cased next word) where
    // the input had none. Both signals must hold to suppress the cap:
    //   * LLM-side previous word ended with `.!?` (it thinks new sentence)
    //   * Input-side previous word did NOT end with `.!?` (no real break)
    // This preserves legitimate mid-sentence proper-noun cap like "John",
    // because in that case the previous LLM word ("met") has no terminator.
    let mut result: Vec<String> = Vec::with_capacity(ops.len());
    let mut prev_inp_terminated = true; // utterance start = sentence-initial
    let mut prev_out_terminated = true;
    // Precompute the last op that consumes an input token. Any Insert past
    // this position is trailing (no following input) — drop it to suppress
    // function-word hallucinations at the tail of the utterance.
    let last_input_op_idx = ops
        .iter()
        .rposition(|op| matches!(op, Op::Match(_, _) | Op::Delete(_)));
    let mut idx = 0;
    while idx < ops.len() {
        match &ops[idx] {
            Op::Match(inp_w, out_w) => {
                let spurious_break = prev_out_terminated && !prev_inp_terminated;
                // The LLM removed a sentence break the input really had; its
                // downcasing of the now-mid-sentence word is legitimate.
                let legit_debreak = prev_inp_terminated && !prev_out_terminated;
                let emitted = if is_en
                    && spurious_break
                    && starts_lowercase_alpha(inp_w)
                    && starts_uppercase_alpha(out_w)
                {
                    lowercase_first_letter(out_w)
                } else if has_interior_uppercase(inp_w)
                    || (starts_uppercase_alpha(inp_w)
                        && starts_lowercase_alpha(out_w)
                        && !legit_debreak)
                {
                    // ASR casing is the source of truth for acronyms, brands,
                    // and proper nouns — never let the LLM flatten "GitHub"
                    // to "github" or "Dayton" to "dayton".
                    transplant_case(inp_w, out_w)
                } else {
                    (*out_w).to_string()
                };
                prev_inp_terminated = ends_sentence(inp_w);
                prev_out_terminated = ends_sentence(out_w);
                result.push(emitted);
                idx += 1;
            }
            Op::Delete(inp_w) => {
                // Check if next op is Insert (potential replacement).
                if let Some(Op::Insert(out_w)) = ops.get(idx + 1) {
                    let from = norm_word(inp_w);
                    let to = norm_word(out_w);
                    if is_en
                        && !is_blocked_sub(&from, &to)
                        && (is_allowed_sub(&from, &to) || looks_like_spelling_repair(&from, &to))
                    {
                        let spurious_break = prev_out_terminated && !prev_inp_terminated;
                        let emitted = if spurious_break
                            && starts_lowercase_alpha(inp_w)
                            && starts_uppercase_alpha(out_w)
                        {
                            lowercase_first_letter(out_w)
                        } else {
                            (*out_w).to_string()
                        };
                        prev_inp_terminated = ends_sentence(inp_w);
                        prev_out_terminated = ends_sentence(out_w);
                        result.push(emitted);
                        idx += 2;
                        continue;
                    }
                }
                result.push((*inp_w).to_string());
                prev_inp_terminated = ends_sentence(inp_w);
                prev_out_terminated = ends_sentence(inp_w);
                idx += 1;
            }
            Op::Insert(w) => {
                let has_following_input = last_input_op_idx.is_some_and(|last| idx < last);
                if is_en && has_following_input && SAFE_INSERTIONS.contains(&norm_word(w).as_str())
                {
                    // Inserted function words ("the", "a", "to", …) are never
                    // sentence-initial in any meaningful sense — keep as-is.
                    result.push((*w).to_string());
                    prev_out_terminated = ends_sentence(w);
                }
                idx += 1;
            }
        }
    }

    result.join(" ")
}

#[async_trait::async_trait]
impl lune_core::llm::LlmService for OrtStyleLlm {
    async fn complete(&self, system: &str, user: &str) -> Result<String> {
        let (ids, sys_prefix_len) = self.build_chat_ids(system, user)?;
        // Reuse the frozen system-prefix KV across calls — the correction
        // system prompt is ~700 tokens of rules/examples/dictionary and
        // dominates prefill cost. Hash-invalidate when the system text
        // changes (e.g. dictionary edits). The styling pass sends an empty
        // system; bypass the cache there so it can't evict the correction
        // prefix between corrections.
        let prefix_len = if system.is_empty() {
            0
        } else {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            system.hash(&mut hasher);
            let sys_hash = hasher.finish();
            let mut prev = self.cached_prefix_hash.lock().map_err(e)?;
            if *prev != sys_hash {
                self.kv_cache.lock().map_err(e)?.clear_prefix();
                self.device_prefix.lock().map_err(e)?.take();
                *prev = sys_hash;
            }
            sys_prefix_len
        };
        let out_ids = self.generate(&ids, prefix_len)?;
        let text = self
            .tokenizer
            .decode(&out_ids, true)
            .map_err(|e| Error::llm(format!("decode: {e}")))?;
        Ok(strip_think_tags(&text).trim().to_owned())
    }
}

type Inputs<'a> = Vec<(std::borrow::Cow<'a, str>, DynValue)>;

fn push_position_ids(out: &mut Inputs<'static>, positions: Vec<i64>) -> Result<()> {
    let len = positions.len();
    let pos = Array2::from_shape_vec((1, len), positions).map_err(e)?;
    out.push(("position_ids".into(), dyn_i64(pos)?));
    Ok(())
}

/// Capacity hint for prefill-style input vectors:
/// inputs_embeds/input_ids + attention_mask + optional position_ids + KV pairs.
#[inline]
fn input_capacity(num_layers: usize, position_ids: bool) -> usize {
    let base = if position_ids { 3 } else { 2 };
    base + num_layers * 2
}

fn build_inputs_into(
    out: &mut Inputs<'static>,
    ids: &[i64],
    seq_len: usize,
    position_ids: bool,
) -> Result<()> {
    let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec()).map_err(e)?;
    let mask = Array2::<i64>::ones((1, seq_len));
    out.push(("input_ids".into(), dyn_i64(input_ids)?));
    out.push(("attention_mask".into(), dyn_i64(mask)?));
    if position_ids {
        push_position_ids(out, (0..seq_len as i64).collect())?;
    }
    Ok(())
}

fn build_embeds_inputs_into(
    out: &mut Inputs<'static>,
    embeds: DynValue,
    seq_len: usize,
    position_ids: bool,
) -> Result<()> {
    let mask = Array2::<i64>::ones((1, seq_len));
    out.push(("inputs_embeds".into(), embeds));
    out.push(("attention_mask".into(), dyn_i64(mask)?));
    if position_ids {
        push_position_ids(out, (0..seq_len as i64).collect())?;
    }
    Ok(())
}

fn build_inputs_with_past_into(
    out: &mut Inputs<'static>,
    ids: &[i64],
    seq_len: usize,
    past_len: usize,
    position_ids: bool,
) -> Result<()> {
    let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec()).map_err(e)?;
    let mask = Array2::<i64>::ones((1, past_len + seq_len));
    out.push(("input_ids".into(), dyn_i64(input_ids)?));
    out.push(("attention_mask".into(), dyn_i64(mask)?));
    if position_ids {
        push_position_ids(
            out,
            (past_len..past_len + seq_len).map(|i| i as i64).collect(),
        )?;
    }
    Ok(())
}

fn build_embeds_inputs_with_past_into(
    out: &mut Inputs<'static>,
    embeds: DynValue,
    seq_len: usize,
    past_len: usize,
    position_ids: bool,
) -> Result<()> {
    let mask = Array2::<i64>::ones((1, past_len + seq_len));
    out.push(("inputs_embeds".into(), embeds));
    out.push(("attention_mask".into(), dyn_i64(mask)?));
    if position_ids {
        push_position_ids(
            out,
            (past_len..past_len + seq_len).map(|i| i as i64).collect(),
        )?;
    }
    Ok(())
}

struct ModelFiles {
    decoder: PathBuf,
    embed_tokens: Option<PathBuf>,
}

fn find_model(dir: &Path) -> Result<ModelFiles> {
    // q4f16 deliberately outranks int8: probing the Qwen3-0.6B exports with
    // the production correction prompt (scripts/probe_llm_quant.py) showed
    // the int8 variant ignores `/no_think` — it streams `<think>` reasoning
    // into the transcript and leaks few-shot examples — while q4f16 follows
    // the contract. Quantization level is not the quality axis here;
    // instruction-following integrity is.
    let quant_priority = ["q4f16", "int8", "fp16", "q4", "quantized", ""];
    let suffix = |q: &&str| -> String {
        if q.is_empty() {
            String::new()
        } else {
            format!("_{q}")
        }
    };

    for q in &quant_priority {
        let s = suffix(q);
        for base in [dir.join("onnx"), dir.to_path_buf()] {
            let decoder = base.join(format!("decoder_model_merged{s}.onnx"));
            let embed = base.join(format!("embed_tokens{s}.onnx"));
            if decoder.exists() && embed.exists() {
                return Ok(ModelFiles {
                    decoder,
                    embed_tokens: Some(embed),
                });
            }
        }
    }

    for q in &quant_priority {
        let s = suffix(q);
        let name = format!("model{s}.onnx");
        let p = dir.join(&name);
        if p.exists() {
            return Ok(ModelFiles {
                decoder: p,
                embed_tokens: None,
            });
        }
        let nested = dir.join("onnx").join(&name);
        if nested.exists() {
            return Ok(ModelFiles {
                decoder: nested,
                embed_tokens: None,
            });
        }
    }

    Err(Error::config(format!("no ONNX model in {}", dir.display())))
}

fn strip_think_tags(text: &str) -> &str {
    if let Some(end) = text.find("</think>") {
        text[end + 8..].trim_start()
    } else if text.starts_with("<think>") {
        ""
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── user_prompt ──────────────────────────────────────────────────────

    #[test]
    fn user_prompt_appends_no_think() {
        let prompt = OrtStyleLlm::user_prompt_with_examples("hello world", "en", &[], None, &[]);
        assert!(prompt.contains("hello world"));
        assert!(prompt.contains("English"));
        assert!(prompt.ends_with("/no_think"));
    }

    #[test]
    fn user_prompt_preserves_transcript() {
        let prompt =
            OrtStyleLlm::user_prompt_with_examples("das ist ein test", "de", &[], None, &[]);
        assert!(prompt.contains("das ist ein test"));
        assert!(prompt.contains("German"));
        assert!(prompt.ends_with("/no_think"));
    }

    #[test]
    fn user_prompt_normalizes_full_english_name_from_asr() {
        let prompt =
            OrtStyleLlm::user_prompt_with_examples("das ist ein test", "German", &[], None, &[]);
        assert!(prompt.contains("German transcript"));
    }

    #[test]
    fn user_prompt_drops_language_when_auto_or_unknown() {
        let auto = OrtStyleLlm::user_prompt_with_examples("hello", "auto", &[], None, &[]);
        assert!(auto.contains("Correct the following transcript:"));
        assert!(!auto.contains("auto"));

        let empty = OrtStyleLlm::user_prompt_with_examples("hello", "", &[], None, &[]);
        assert!(empty.contains("Correct the following transcript:"));

        let unknown = OrtStyleLlm::user_prompt_with_examples("hello", "klingon", &[], None, &[]);
        assert!(unknown.contains("Correct the following transcript:"));
        assert!(!unknown.contains("klingon"));
    }

    #[test]
    fn user_prompt_includes_examples_when_provided() {
        let examples = vec![(
            "I said Barolo in the meeting.".to_string(),
            "I said Boorloo in the meeting.".to_string(),
        )];
        let prompt = OrtStyleLlm::user_prompt_with_examples(
            "I went the store and brought some wrld",
            "en",
            &examples,
            None,
            &[],
        );
        assert!(prompt.contains("I went the store"));
        assert!(prompt.ends_with("/no_think"));
        assert!(prompt.contains("Examples:"));
        assert!(prompt.contains("Input: I said Barolo in the meeting."));
        assert!(prompt.contains("Output: I said Boorloo in the meeting."));
    }

    #[test]
    fn user_prompt_includes_uncertain_word_hints() {
        let uncertain = vec![("computor".to_string(), 5), ("no".to_string(), 2)];
        let prompt = OrtStyleLlm::user_prompt_with_examples(
            "i dont no why the computor is broken",
            "en",
            &[],
            None,
            &uncertain,
        );
        assert!(prompt.contains("(likely misheard: computor, no)"));
        // Hint phrasing must never trip the forbidden-output guard if echoed.
        assert!(!contains_forbidden_output("(likely misheard: x)"));
        let no_hints = OrtStyleLlm::user_prompt_with_examples("hello", "en", &[], None, &[]);
        assert!(!no_hints.contains("likely misheard"));
    }

    #[test]
    fn user_prompt_includes_context_block_when_provided() {
        let prompt = OrtStyleLlm::user_prompt_with_examples(
            "send it to dayton",
            "en",
            &[],
            Some("My colleague Dayton joined the project."),
            &[],
        );
        assert!(prompt.contains("Earlier dictation"));
        assert!(prompt.contains("My colleague Dayton joined the project."));
        assert!(prompt.ends_with("/no_think"));
        // Empty/whitespace context must not emit the block.
        let no_ctx = OrtStyleLlm::user_prompt_with_examples("hi", "en", &[], Some("  "), &[]);
        assert!(!no_ctx.contains("Earlier dictation"));
    }

    // ── strip_think_tags ─────────────────────────────────────────────────

    const THINK_OPEN: &str = "<think>";
    const THINK_CLOSE: &str = "</think>";

    #[test]
    fn strip_think_tags_no_tags_returns_as_is() {
        assert_eq!(strip_think_tags("hello world"), "hello world");
    }

    #[test]
    fn strip_think_tags_removes_thinking_block() {
        let input = format!("{THINK_OPEN}thinking here{THINK_CLOSE} result");
        assert_eq!(strip_think_tags(&input), "result");
    }

    #[test]
    fn strip_think_tags_opening_only_returns_empty() {
        assert_eq!(strip_think_tags(&format!("{THINK_OPEN}some thinking")), "");
    }

    #[test]
    fn strip_think_tags_leading_then_content() {
        let input = format!("{THINK_OPEN}{THINK_CLOSE}answer");
        assert_eq!(strip_think_tags(&input), "answer");
    }

    #[test]
    fn strip_think_tags_empty_string() {
        assert_eq!(strip_think_tags(""), "");
    }

    #[test]
    fn strip_think_tags_tag_in_middle() {
        let input = format!("before {THINK_CLOSE}after");
        assert_eq!(strip_think_tags(&input), "after");
    }

    // ── validate_and_merge ───────────────────────────────────────────────

    #[test]
    fn merge_accepts_case_and_punctuation() {
        assert_eq!(
            validate_and_merge("how are you doing today", "How are you doing today?", "en"),
            "How are you doing today?"
        );
    }

    #[test]
    fn merge_accepts_function_word_insertion() {
        assert_eq!(
            validate_and_merge("i went store", "I went to the store.", "en"),
            "I went to the store."
        );
    }

    #[test]
    fn merge_rejects_translation() {
        assert_eq!(
            validate_and_merge("je suis allé au magasin", "I went to the store", "fr"),
            "je suis allé au magasin"
        );
    }

    #[test]
    fn merge_rejects_censoring() {
        let merged = validate_and_merge("what the fuck is going on", "what is going on", "en");
        assert!(
            merged.contains("fuck"),
            "censored word must survive: {merged}"
        );
    }

    #[test]
    fn merge_accepts_allowed_substitution() {
        assert_eq!(
            validate_and_merge("i brought some groceries", "I bought some groceries.", "en"),
            "I bought some groceries."
        );
    }

    #[test]
    fn merge_accepts_homophone_fix() {
        assert_eq!(
            validate_and_merge("we should meat at noon", "We should meet at noon.", "en"),
            "We should meet at noon."
        );
    }

    #[test]
    fn merge_rejects_unrelated_replacement() {
        assert_eq!(
            validate_and_merge("the cat sat on the mat", "The dog sat on the mat.", "en"),
            "The cat sat on the mat."
        );
    }

    #[test]
    fn merge_accepts_spelling_repair() {
        assert_eq!(
            validate_and_merge("i went to the storee", "I went to the store.", "en"),
            "I went to the store."
        );
    }

    #[test]
    fn merge_keeps_identical() {
        assert_eq!(
            validate_and_merge("Hello world", "Hello world", "en"),
            "Hello world"
        );
    }

    #[test]
    fn merge_empty_input_returns_llm() {
        assert_eq!(validate_and_merge("", "Hello.", "en"), "Hello.");
    }

    #[test]
    fn merge_empty_llm_returns_input() {
        assert_eq!(validate_and_merge("hello world", "", "en"), "hello world");
    }

    #[test]
    fn merge_rejects_hallucinated_content() {
        let merged = validate_and_merge("hello world", "Hello world, this is extra content.", "en");
        assert!(
            !merged.contains("extra"),
            "hallucinated content must be dropped: {merged}"
        );
    }

    #[test]
    fn merge_rejects_trailing_function_word_insertion() {
        assert_eq!(
            validate_and_merge(
                "I gave a talk in Boorloo about transcription.",
                "I gave a talk in Boorloo about transcription in",
                "en",
            ),
            "I gave a talk in Boorloo about transcription"
        );
    }

    #[test]
    fn merge_handles_contraction_fix() {
        assert_eq!(
            validate_and_merge("dont worry about it", "Don't worry about it.", "en"),
            "Don't worry about it."
        );
    }

    #[test]
    fn merge_strips_spurious_mid_utterance_capitalization() {
        // The LLM hallucinated sentence breaks the user did not produce.
        // The merge must keep punctuation but suppress the false title-case
        // it injected on every "new sentence" head.
        let merged = validate_and_merge("hello there friend", "Hello. There. Friend.", "en");
        assert_eq!(merged, "Hello. there. friend.");
    }

    #[test]
    fn merge_preserves_legitimate_sentence_initial_cap() {
        // Input has its own sentence terminator — the LLM's capitalization
        // of the next sentence is correct and must survive.
        let merged = validate_and_merge("hello. how are you", "Hello. How are you?", "en");
        assert_eq!(merged, "Hello. How are you?");
    }

    #[test]
    fn merge_preserves_proper_noun_capitalization() {
        // Mid-sentence proper noun capitalization (no LLM-introduced sentence
        // break before it) is legitimate and must survive.
        let merged = validate_and_merge("i met john today", "I met John today.", "en");
        assert_eq!(merged, "I met John today.");
    }

    #[test]
    fn merge_strips_cap_on_replacement_after_spurious_break() {
        // Replacement words (Delete+Insert) must follow the same rule:
        // if the previous input word didn't end a sentence, suppress the
        // LLM's spurious title-case on the replacement.
        let merged = validate_and_merge("we should meat at noon", "We should. Meet at noon.", "en");
        // "Meet" replaces "meat" (homophone repair) but mid-sentence after a
        // spurious "." → leading cap suppressed.
        assert_eq!(merged, "We should. meet at noon.");
    }

    #[test]
    fn merge_preserves_acronym_and_brand_casing() {
        // The LLM flattened mixed-case identifiers; ASR casing must win.
        assert_eq!(
            validate_and_merge("i use GitHub at NASA", "I use github at nasa.", "en"),
            "I use GitHub at NASA."
        );
    }

    #[test]
    fn merge_preserves_proper_noun_against_downcase() {
        // Mid-sentence proper noun downcased by the LLM with no sentence
        // restructuring — input casing must survive.
        assert_eq!(
            validate_and_merge("my name is Dayton", "My name is dayton.", "en"),
            "My name is Dayton."
        );
    }

    #[test]
    fn merge_allows_downcase_when_llm_merges_sentences() {
        // Input had a real break; the LLM legitimately merged the sentences,
        // so downcasing the former sentence head is correct.
        assert_eq!(
            validate_and_merge(
                "i like it. However it failed",
                "I like it, however it failed.",
                "en"
            ),
            "I like it, however it failed."
        );
    }

    #[test]
    fn merge_rejects_your_to_youre_expansion() {
        assert_eq!(
            validate_and_merge("is that your dog", "Is that you're dog?", "en"),
            "Is that your dog?"
        );
    }

    #[test]
    fn merge_rejects_word_count_blowup() {
        // Output more than ~1.5× the input word count is a paraphrase, not a
        // correction — fall back entirely.
        assert_eq!(
            validate_and_merge(
                "send the report",
                "Please make sure to send over the full report as soon as possible.",
                "en",
            ),
            "send the report"
        );
    }

    #[test]
    fn merge_blocks_insertion_for_non_english() {
        assert_eq!(
            validate_and_merge("la découverte du paquet", "la a découverte du paquet", "fr"),
            "la découverte du paquet"
        );
    }

    // ── forbidden output ────────────────────────────────────────────────

    #[test]
    fn forbidden_output_catches_meta_text() {
        assert!(contains_forbidden_output(
            "Here is the corrected transcript: Hello."
        ));
        assert!(contains_forbidden_output("I fixed the spelling errors."));
        assert!(!contains_forbidden_output("Hello world."));
    }

    // ── helpers ──────────────────────────────────────────────────────────

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("store", "storee"), 1);
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn spelling_repair_accepts_close_words() {
        assert!(looks_like_spelling_repair("storee", "store"));
        assert!(looks_like_spelling_repair("grocerys", "groceries"));
        assert!(!looks_like_spelling_repair("cat", "dog"));
        assert!(!looks_like_spelling_repair("ab", "cd"));
    }
}
