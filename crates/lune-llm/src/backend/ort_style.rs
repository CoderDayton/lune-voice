//! ONNX style LLM with auto-detected model parameters and KV-cache decode.
//!
//! Supports any causal LM exported via onnxruntime-genai (Llama, Qwen, etc.)
//! by probing the session's input metadata at load time.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use half::f16;
use ndarray::{Array2, Array4};
use ort::logging::{LogLevel, LoggerFunction};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, Value};
use tokenizers::Tokenizer;
use tracing::info;

use lune_core::error::{Error, Result};
use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

const DEFAULT_MAX_TOKENS: usize = 256;

pub struct OrtStyleLlmConfig {
    pub model_dir: PathBuf,
    pub use_gpu: bool,
    pub max_new_tokens: usize,
}

impl Default for OrtStyleLlmConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            use_gpu: true,
            max_new_tokens: DEFAULT_MAX_TOKENS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ChatFormat {
    Llama,  // <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
    ChatML, // <|im_start|>role\n...<|im_end|>
}

struct ModelParams {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    has_position_ids: bool,
    kv_is_f16: bool,
    eos_ids: Vec<u32>,
    chat_format: ChatFormat,
}

pub struct OrtStyleLlm {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_new_tokens: usize,
    params: ModelParams,
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

fn dyn_i64(arr: Array2<i64>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
}
fn dyn_f16_4(arr: Array4<f16>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
}
fn dyn_f32_4(arr: Array4<f32>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
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

    let chat_format = if vocab.contains_key("<|im_start|>") {
        ChatFormat::ChatML
    } else {
        ChatFormat::Llama
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
    }
}

impl OrtStyleLlm {
    pub fn new(config: OrtStyleLlmConfig) -> Result<Self> {
        let model_path = find_model(&config.model_dir)?;
        let opt_path = model_path.with_extension("onnx.opt");

        let mut builder = Session::builder()
            .map_err(e)?
            .with_logger(quiet_ort_logger())
            .map_err(e)?
            .with_log_level(LogLevel::Error)
            .map_err(e)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(e)?
            .with_intra_threads(4)
            .map_err(e)?;

        if config.use_gpu {
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .map_err(e)?;
        } else {
            builder = builder.with_optimized_model_path(&opt_path).map_err(e)?;
        }

        let session = builder.commit_from_file(&model_path).map_err(e)?;

        let tokenizer = Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
            .map_err(|e| Error::Config(format!("tokenizer: {e}")))?;

        let params = probe_model_params(&session, &tokenizer);

        info!(model = %model_path.display(), "style LLM loaded");
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_new_tokens: config.max_new_tokens,
            params,
        })
    }

    pub fn user_prompt(transcript: &str) -> String {
        format!(
            "Correct this ASR transcript. Output corrected text only.\n\n{transcript}\n\n/no_think",
        )
    }

    pub fn apply_style_with_fix_words(
        &self,
        _system_prompt: &str,
        transcript: &str,
        fix_words: &[String],
        language: &str,
    ) -> Result<String> {
        let fix_str = if fix_words.is_empty() {
            String::new()
        } else {
            format!("Correct these words: {}", fix_words.join(", "))
        };
        let base = lune_core::config::base_system_prompt(language, &fix_str);
        let user_prompt = Self::user_prompt(transcript);
        let prompt = match self.params.chat_format {
            ChatFormat::ChatML => format!(
                "<|im_start|>system\n\
                 {base}<|im_end|>\n\
                 <|im_start|>user\n\
                 {user_prompt}<|im_end|>\n\
                 <|im_start|>assistant\n\
                 <think>\n\n</think>\n\n"
            ),
            ChatFormat::Llama => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                 {base}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
                 {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
        };
        let enc = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|err| Error::Config(format!("tokenize: {err}")))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        let out = self.generate(&ids)?;
        let text = self
            .tokenizer
            .decode(&out, true)
            .map_err(|err| Error::Config(format!("detokenize: {err}")))?;
        let text = strip_think_tags(text.trim());
        Ok(text.to_string())
    }

    fn generate(&self, prompt_ids: &[i64]) -> Result<Vec<u32>> {
        let p = &self.params;
        let mut session = self.session.lock().map_err(e)?;
        let n = prompt_ids.len();

        // ── Prefill ─────────────────────────────────────────────────────
        let mut inputs = build_inputs(prompt_ids, n, p.has_position_ids)?;
        for i in 0..p.num_layers {
            let (k, v) = empty_kv(p.num_kv_heads, p.head_dim, p.kv_is_f16)?;
            inputs.push((format!("past_key_values.{i}.key").into(), k));
            inputs.push((format!("past_key_values.{i}.value").into(), v));
        }

        let outputs = session.run(inputs).map_err(e)?;
        let first = argmax_last(&outputs[0])?;
        if p.eos_ids.contains(&first) {
            return Ok(vec![]);
        }

        let mut generated = vec![first];
        let mut kv: Vec<DynValue> = outputs.into_iter().skip(1).map(|(_, v)| v).collect();

        // ── Decode ──────────────────────────────────────────────────────
        let max = self.max_new_tokens.min(n);
        for _ in 1..max {
            let tok = *generated.last().unwrap() as i64;
            let total = n + generated.len();

            let mut inputs = build_decode_inputs(tok, total, p.has_position_ids)?;
            for (i, v) in kv.drain(..).enumerate() {
                let layer = i / 2;
                let kind = if i % 2 == 0 { "key" } else { "value" };
                inputs.push((format!("past_key_values.{layer}.{kind}").into(), v));
            }

            let outputs = session.run(inputs).map_err(e)?;
            let next = argmax_last(&outputs[0])?;
            if p.eos_ids.contains(&next) {
                break;
            }
            generated.push(next);
            kv = outputs.into_iter().skip(1).map(|(_, v)| v).collect();
        }

        Ok(generated)
    }
}

#[async_trait::async_trait]
impl TextPostProcessor for OrtStyleLlm {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let system_prompt = &input.style_preset.style_instructions;
        let text = self.apply_style_with_fix_words(
            system_prompt,
            &input.raw_text,
            &input.fix_words,
            &input.language,
        )?;
        let text = if text.is_empty() {
            crate::itn::normalize(&input.raw_text)
        } else {
            text
        };
        let text = crate::prompt::strip_markers(&text);
        Ok(PostProcessOutput {
            cleaned_text: text,
            metadata: None,
        })
    }
}

type Inputs<'a> = Vec<(std::borrow::Cow<'a, str>, DynValue)>;

fn build_inputs(ids: &[i64], seq_len: usize, position_ids: bool) -> Result<Inputs<'static>> {
    let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec()).map_err(e)?;
    let mask = Array2::<i64>::ones((1, seq_len));
    let mut out = vec![
        ("input_ids".into(), dyn_i64(input_ids)?),
        ("attention_mask".into(), dyn_i64(mask)?),
    ];
    if position_ids {
        let pos = Array2::from_shape_vec((1, seq_len), (0..seq_len as i64).collect()).map_err(e)?;
        out.push(("position_ids".into(), dyn_i64(pos)?));
    }
    Ok(out)
}

fn build_decode_inputs(
    token: i64,
    total_len: usize,
    position_ids: bool,
) -> Result<Inputs<'static>> {
    let ids = Array2::from_shape_vec((1, 1), vec![token]).map_err(e)?;
    let mask = Array2::<i64>::ones((1, total_len));
    let mut out = vec![
        ("input_ids".into(), dyn_i64(ids)?),
        ("attention_mask".into(), dyn_i64(mask)?),
    ];
    if position_ids {
        let pos = Array2::from_shape_vec((1, 1), vec![(total_len - 1) as i64]).map_err(e)?;
        out.push(("position_ids".into(), dyn_i64(pos)?));
    }
    Ok(out)
}

fn argmax_last(value: &DynValue) -> Result<u32> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return argmax_slice(shape.as_ref(), data);
    }
    let (shape, data) = value.try_extract_tensor::<f16>().map_err(e)?;
    let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    argmax_slice(shape.as_ref(), &f32_data)
}

fn argmax_slice(shape: &[i64], data: &[f32]) -> Result<u32> {
    let vocab = shape[2] as usize;
    let last_pos = shape[1] as usize - 1;
    let offset = last_pos * vocab;
    let slice = &data[offset..offset + vocab];
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in slice.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_id = i as u32;
        }
    }
    Ok(best_id)
}

fn find_model(dir: &Path) -> Result<PathBuf> {
    for name in [
        "model_int8.onnx",
        "model_q4f16.onnx",
        "model_fp16.onnx",
        "model_q4.onnx",
        "model.onnx",
    ] {
        let p = dir.join(name);
        if p.exists() {
            return Ok(p);
        }
        let nested = dir.join("onnx").join(name);
        if nested.exists() {
            return Ok(nested);
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
