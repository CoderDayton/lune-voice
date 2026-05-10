//! ONNX style LLM with auto-detected model parameters and KV-cache decode.
//!
//! Supports any causal LM exported via onnxruntime-genai (Llama, Qwen, etc.)
//! by probing the session's input metadata at load time.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use half::f16;
use ndarray::{Array2, Array4};
use ort::logging::{LogLevel, LoggerFunction};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use ort::value::{DynValue, Value};
use tokenizers::Tokenizer;
use tracing::info;

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
    session: Mutex<Session>,
    embed_session: Option<Mutex<Session>>,
    tokenizer: Tokenizer,
    max_new_tokens: usize,
    params: ModelParams,
    sampling_params: super::sampler::SamplingParams,
    kv_cache: std::sync::Mutex<super::kv_cache::KvCache>,
    cached_prefix_hash: std::sync::Mutex<u64>,
}

fn e(msg: impl std::fmt::Display) -> Error {
    Error::Config(msg.to_string())
}

fn quiet_ort_logger() -> LoggerFunction {
    Arc::new(|level, _category, id, location, message| {
        if level >= LogLevel::Error {
            tracing::error!(
                target: "ort::logging",
                ort_id = %id,
                ort_location = %location,
                "{message}"
            );
        }
    })
}

/// Build a `RunOptions` that asks ORT to shrink CPU and CUDA arenas at the
/// end of the run. Used at the prefill step of each LLM generate call to
/// reclaim leftover chunks from the prior call's variable-shape decode loop.
fn make_shrink_run_options() -> Result<RunOptions> {
    let mut opts = RunOptions::new().map_err(e)?;
    opts.add_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0;gpu:0")
        .map_err(e)?;
    Ok(opts)
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
                .map_err(e)?
                // Share the env-registered CUDA arena with the ASR sessions.
                .with_env_allocators()
                .map_err(e)?;
            if gpu {
                // Use ORT default arena strategy (kNextPowerOfTwo). Variable
                // KV-cache shapes per decode step fragment a SameAsRequested
                // arena across runs; power-of-two buckets reuse cleanly.
                let cuda_ep = ort::ep::CUDA::default()
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                    .with_conv_max_workspace(false)
                    .with_tf32(true)
                    .build();
                let cuda_ep = if cuda_required {
                    cuda_ep.error_on_failure()
                } else {
                    cuda_ep
                };
                b = b.with_execution_providers([cuda_ep]).map_err(e)?;
            } else {
                let opt_path = path.with_extension("onnx.opt");
                b = b.with_optimized_model_path(&opt_path).map_err(e)?;
            }
            b.commit_from_file(path).map_err(e)
        };

        let session = make_builder(config.use_gpu, &files.decoder)?;
        let embed_session = files
            .embed_tokens
            .as_ref()
            .map(|p| make_builder(config.use_gpu, p))
            .transpose()?;

        let tokenizer = Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
            .map_err(|e| Error::Config(format!("tokenizer: {e}")))?;

        let params = probe_model_params(&session, &tokenizer);

        let sampling_params = super::sampler::SamplingParams {
            eos_token_id: params.eos_ids.first().copied(),
            stop_token_ids: params.eos_ids.clone().into_boxed_slice(),
            ..Default::default()
        };

        let kv_cache = super::kv_cache::KvCache::new(super::kv_cache::KvCacheConfig {
            num_layers: params.num_layers,
            num_kv_heads: params.num_kv_heads,
            head_dim: params.head_dim,
            is_f16: params.kv_is_f16,
            ..Default::default()
        });

        info!(model = %files.decoder.display(), split = embed_session.is_some(), "style LLM loaded");
        Ok(Self {
            session: Mutex::new(session),
            embed_session: embed_session.map(Mutex::new),
            tokenizer,
            max_new_tokens: config.max_new_tokens,
            params,
            sampling_params,
            kv_cache: std::sync::Mutex::new(kv_cache),
            cached_prefix_hash: std::sync::Mutex::new(0),
        })
    }

    pub fn user_prompt(
        transcript: &str,
        language: &str,
        _uncertain_words: &[(String, usize)],
    ) -> String {
        // The `language` field arrives in inconsistent forms — the runtime
        // sometimes hands us a BCP-47 code ("en") and sometimes a full
        // English name from the ASR's language head ("English", "Dutch").
        // Only inject a recognizable label; otherwise omit it entirely so
        // the model isn't shown an ungrammatical header like
        // "the following en transcript".
        let label = lune_core::lang::display_name(language);
        let header = match label {
            Some(name) => format!("Correct the following {name} transcript:"),
            None => "Correct the following transcript:".to_string(),
        };
        format!("{header}\n{transcript}\n\n/no_think")
    }

    pub fn apply_correction(
        &self,
        transcript: &str,
        language: &str,
        uncertain_words: &[(String, usize)],
    ) -> Result<String> {
        use std::hash::{Hash, Hasher};
        let system = lune_core::config::BASE_SYSTEM_PROMPT.replace("{lang}", language);
        let user_text = Self::user_prompt(transcript, language, uncertain_words);

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        system.hash(&mut hasher);
        let sys_hash = hasher.finish();

        {
            let mut prev = self.cached_prefix_hash.lock().map_err(e)?;
            if *prev != sys_hash {
                self.kv_cache.lock().map_err(e)?.clear_prefix();
                *prev = sys_hash;
            }
        }

        let (ids, sys_prefix_len) = self.build_chat_ids(&system, &user_text)?;
        let out = self.generate(&ids, sys_prefix_len)?;
        let text = self
            .tokenizer
            .decode(&out, true)
            .map_err(|err| Error::Config(format!("detokenize: {err}")))?;
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
            .map_err(|err| Error::Config(format!("detokenize: {err}")))?;
        let text = strip_think_tags(text.trim());
        Ok(text.to_string())
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| Error::Config(format!("tokenize: {err}")))?;
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
        use super::sampler::{DecodeState, SamplerWorkspace, sample_last_token};

        let p = &self.params;
        let mut session = self.session.lock().map_err(e)?;
        let split = self.embed_session.is_some();

        let mut state = DecodeState::default();
        let mut ws = SamplerWorkspace::default();
        let mut rng = rand::rng();
        let names = llm_kv_names(p.num_layers);

        // ── Acquire or build prefix KV ──────────────────────────────────
        let mut cache = self.kv_cache.lock().map_err(e)?;
        let prefix_seq_len;

        if !cache.has_prefix() && sys_prefix_len > 0 {
            // First call: prefill system prefix alone to build the frozen cache.
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
        }

        // ── Prefill user portion with cached prefix KV ──────────────────
        let has_prefix = cache.has_prefix();
        let (prefill_ids, initial_kv) = if has_prefix && sys_prefix_len > 0 {
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
        // call). Releases unused chunks from the previous call's variable-shape
        // decode loop before this call grows the arena. Doing it here, not in
        // the per-step decode loop, avoids per-step shrink/grow churn.
        let prefill_opts = make_shrink_run_options()?;
        let outputs = session.run_with_options(inputs, &prefill_opts).map_err(e)?;
        let first = sample_last_token(
            &outputs[0],
            &self.sampling_params,
            &state,
            &mut ws,
            &mut rng,
        )?;
        if state.should_stop(first, &self.sampling_params) {
            return Ok(vec![]);
        }
        state.push(first);

        // Reused across the decode loop to avoid reallocating `Vec<DynValue>`
        // every step (28 layers × 2 = 56 entries × up to ~448 tokens).
        let mut kv: Vec<DynValue> = Vec::with_capacity(p.num_layers * 2);
        kv.extend(outputs.into_iter().skip(1).map(|(_, v)| v));

        // ── Decode ──────────────────────────────────────────────────────
        let max = self.max_new_tokens;
        let inputs_cap = input_capacity(p.num_layers, p.has_position_ids);
        for _ in 1..max {
            let tok = *state.generated.last().unwrap() as i64;
            let total = total_prefill + state.generated.len();

            // ORT's session.run consumes `inputs`; we can't keep the Vec
            // across iterations, but we can pre-size it so the in-loop
            // pushes never reallocate.
            let mut inputs: Inputs<'static> = Vec::with_capacity(inputs_cap);
            if split {
                let embeds = self
                    .embed_ids(&[tok])?
                    .ok_or_else(|| e("split decode requires embed session"))?;
                build_embeds_decode_inputs_into(&mut inputs, embeds, total, p.has_position_ids)?;
            } else {
                build_decode_inputs_into(&mut inputs, tok, total, p.has_position_ids)?;
            }
            for (i, v) in kv.drain(..).enumerate() {
                inputs.push((std::borrow::Cow::Borrowed(names[i].as_str()), v));
            }

            let outputs = session.run(inputs).map_err(e)?;
            let next = sample_last_token(
                &outputs[0],
                &self.sampling_params,
                &state,
                &mut ws,
                &mut rng,
            )?;
            if state.should_stop(next, &self.sampling_params) {
                break;
            }
            state.push(next);
            // kv was emptied by drain above; refill in place.
            kv.extend(outputs.into_iter().skip(1).map(|(_, v)| v));
        }

        Ok(state.generated)
    }
}

/// Function words the LLM may legitimately insert into ASR output.
const SAFE_INSERTIONS: &[&str] = &[
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "from", "with", "by", "as", "is", "are",
    "am", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have", "had", "will",
    "would", "could", "should", "can", "may", "might", "shall", "and", "but", "or", "so", "if",
    "that", "not", "no", "i", "it", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "you", "my", "your", "his", "its", "our", "their", "this",
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
    ("let", "let's"),
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
    ("were", "we're"),
    ("hes", "he's"),
    ("shes", "she's"),
    ("your", "you're"),
    ("its", "it's"),
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

fn is_allowed_sub(from: &str, to: &str) -> bool {
    ALLOWED_SUBS.iter().any(|(a, b)| from == *a && to == *b)
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

    // If fewer than 40% of input words survived, the output is suspect
    // (translation, heavy censoring, hallucination) — fall back entirely.
    let lcs_len = dp_at(&dp, m, n) as usize;
    if lcs_len * 100 / m.max(1) < 40 {
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

    let mut result: Vec<&str> = Vec::with_capacity(ops.len());
    let mut idx = 0;
    while idx < ops.len() {
        match &ops[idx] {
            Op::Match(_inp_w, out_w) => {
                result.push(out_w);
                idx += 1;
            }
            Op::Delete(inp_w) => {
                // Check if next op is Insert (potential replacement).
                if let Some(Op::Insert(out_w)) = ops.get(idx + 1) {
                    let from = norm_word(inp_w);
                    let to = norm_word(out_w);
                    if is_en
                        && (is_allowed_sub(&from, &to) || looks_like_spelling_repair(&from, &to))
                    {
                        result.push(out_w);
                        idx += 2;
                        continue;
                    }
                }
                result.push(inp_w);
                idx += 1;
            }
            Op::Insert(w) => {
                if is_en && SAFE_INSERTIONS.contains(&norm_word(w).as_str()) {
                    result.push(w);
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
        let (ids, _sys_prefix_len) = self.build_chat_ids(system, user)?;
        let out_ids = self.generate(&ids, 0)?;
        let text = self
            .tokenizer
            .decode(&out_ids, true)
            .map_err(|e| Error::Llm(format!("decode: {e}")))?;
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

fn build_embeds_decode_inputs_into(
    out: &mut Inputs<'static>,
    embeds: DynValue,
    total_len: usize,
    position_ids: bool,
) -> Result<()> {
    let mask = Array2::<i64>::ones((1, total_len));
    out.push(("inputs_embeds".into(), embeds));
    out.push(("attention_mask".into(), dyn_i64(mask)?));
    if position_ids {
        push_position_ids(out, vec![(total_len - 1) as i64])?;
    }
    Ok(())
}

fn build_decode_inputs_into(
    out: &mut Inputs<'static>,
    token: i64,
    total_len: usize,
    position_ids: bool,
) -> Result<()> {
    let ids = Array2::from_shape_vec((1, 1), vec![token]).map_err(e)?;
    let mask = Array2::<i64>::ones((1, total_len));
    out.push(("input_ids".into(), dyn_i64(ids)?));
    out.push(("attention_mask".into(), dyn_i64(mask)?));
    if position_ids {
        push_position_ids(out, vec![(total_len - 1) as i64])?;
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

    Err(Error::Config(format!("no ONNX model in {}", dir.display())))
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
        let prompt = OrtStyleLlm::user_prompt("hello world", "en", &[]);
        assert!(prompt.contains("hello world"));
        assert!(prompt.contains("English"));
        assert!(prompt.ends_with("/no_think"));
    }

    #[test]
    fn user_prompt_preserves_transcript() {
        let prompt = OrtStyleLlm::user_prompt("das ist ein test", "de", &[]);
        assert!(prompt.contains("das ist ein test"));
        assert!(prompt.contains("German"));
        assert!(prompt.ends_with("/no_think"));
    }

    #[test]
    fn user_prompt_normalizes_full_english_name_from_asr() {
        let prompt = OrtStyleLlm::user_prompt("das ist ein test", "German", &[]);
        assert!(prompt.contains("German transcript"));
    }

    #[test]
    fn user_prompt_drops_language_when_auto_or_unknown() {
        let auto = OrtStyleLlm::user_prompt("hello", "auto", &[]);
        assert!(auto.contains("Correct the following transcript:"));
        assert!(!auto.contains("auto"));

        let empty = OrtStyleLlm::user_prompt("hello", "", &[]);
        assert!(empty.contains("Correct the following transcript:"));

        let unknown = OrtStyleLlm::user_prompt("hello", "klingon", &[]);
        assert!(unknown.contains("Correct the following transcript:"));
        assert!(!unknown.contains("klingon"));
    }

    #[test]
    fn user_prompt_ignores_uncertain_words() {
        let uncertain = vec![("brought".into(), 6usize), ("wrld".into(), 1usize)];
        let prompt =
            OrtStyleLlm::user_prompt("I went the store and brought some wrld", "en", &uncertain);
        assert!(prompt.contains("I went the store"));
        assert!(prompt.ends_with("/no_think"));
        assert!(!prompt.contains("misheard"));
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
    fn merge_handles_contraction_fix() {
        assert_eq!(
            validate_and_merge("dont worry about it", "Don't worry about it.", "en"),
            "Don't worry about it."
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
