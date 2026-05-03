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
use lune_core::llm::{CorrectionSpan, PostProcessInput, PostProcessOutput, TextPostProcessor};

const DEFAULT_MAX_TOKENS: usize = 512;

pub struct OrtStyleLlmConfig {
    pub model_dir: PathBuf,
    pub use_gpu: bool,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for OrtStyleLlmConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            use_gpu: true,
            max_new_tokens: DEFAULT_MAX_TOKENS,
            temperature: 0.0,
            top_p: 1.0,
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
    embed_session: Option<Mutex<Session>>,
    tokenizer: Tokenizer,
    max_new_tokens: usize,
    params: ModelParams,
    sampling_params: super::sampler::SamplingParams,
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
        let files = find_model(&config.model_dir)?;

        let make_builder = |gpu: bool, path: &Path| -> Result<Session> {
            let opt_path = path.with_extension("onnx.opt");
            let mut b = Session::builder()
                .map_err(e)?
                .with_logger(quiet_ort_logger())
                .map_err(e)?
                .with_log_level(LogLevel::Error)
                .map_err(e)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(e)?
                .with_intra_threads(4)
                .map_err(e)?;
            if gpu {
                b = b
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .map_err(e)?;
            } else {
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
            temperature: config.temperature,
            top_p: config.top_p,
            eos_token_id: params.eos_ids.first().copied(),
            stop_token_ids: params.eos_ids.clone().into_boxed_slice(),
            ..Default::default()
        };

        info!(model = %files.decoder.display(), split = embed_session.is_some(), "style LLM loaded");
        Ok(Self {
            session: Mutex::new(session),
            embed_session: embed_session.map(Mutex::new),
            tokenizer,
            max_new_tokens: config.max_new_tokens,
            params,
            sampling_params,
        })
    }

    pub fn user_prompt(
        transcript: &str,
        language: &str,
        _style_instructions: &str,
        correction_spans: &[CorrectionSpan],
        uncertain_words: &[(String, usize)],
    ) -> String {
        let mut instructions = String::new();
        if !correction_spans.is_empty() {
            let spans: Vec<String> = correction_spans
                .iter()
                .map(|span| {
                    format!(
                        "#{}..{} \"{}\" risk={:.2}",
                        span.start_word, span.end_word, span.text, span.word_error_risk
                    )
                })
                .collect();
            instructions.push_str("Editable spans (0-indexed, end exclusive): [");
            instructions.push_str(&spans.join(", "));
            instructions.push_str("]\n");
        }
        if !uncertain_words.is_empty() {
            let words: Vec<String> = uncertain_words
                .iter()
                .map(|(w, pos)| format!("{w} (#{pos})"))
                .collect();
            let hint = words.join(", ");
            instructions.push_str("Uncertain words (name and 0-indexed position): [");
            instructions.push_str(&hint);
            instructions.push_str("]\n");
        }
        if !instructions.is_empty() {
            instructions.push_str("Only change text inside editable spans or listed uncertain words. Copy everything else exactly.\n");
        }
        instructions.push_str(&format!(
            "Correct this {language} transcript:\n{transcript}"
        ));
        instructions
    }

    pub fn apply_style(
        &self,
        transcript: &str,
        language: &str,
        style: &lune_core::config::StylePreset,
        correction_spans: &[CorrectionSpan],
        uncertain_words: &[(String, usize)],
    ) -> Result<String> {
        let system = lune_core::config::BASE_SYSTEM_PROMPT.replace("{lang}", language);

        let user_text = Self::user_prompt(
            transcript,
            language,
            &style.style_instructions,
            correction_spans,
            uncertain_words,
        );

        let prompt = match self.params.chat_format {
            ChatFormat::ChatML => {
                format!(
                    "<|im_start|>system\n{system}<|im_end|>\n\
                     <|im_start|>user\n{user_text}<|im_end|>\n\
                     <|im_start|>assistant\n"
                )
            }
            ChatFormat::Llama => {
                format!(
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                     {system}<|eot_id|>\
                     <|start_header_id|>user<|end_header_id|>\n\n\
                     {user_text}<|eot_id|>\
                     <|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            }
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

    fn embed_ids(&self, ids: &[i64]) -> Result<Option<DynValue>> {
        let embed = match &self.embed_session {
            Some(s) => s,
            None => return Ok(None),
        };
        let mut sess = embed.lock().map_err(e)?;
        let input_ids = Array2::from_shape_vec((1, ids.len()), ids.to_vec()).map_err(e)?;
        let inputs: Inputs = vec![("input_ids".into(), dyn_i64(input_ids)?)];
        let outputs = sess.run(inputs).map_err(e)?;
        Ok(Some(outputs.into_iter().next().unwrap().1))
    }

    fn generate(&self, prompt_ids: &[i64]) -> Result<Vec<u32>> {
        use super::sampler::{DecodeState, SamplerWorkspace, sample_last_token};

        let p = &self.params;
        let mut session = self.session.lock().map_err(e)?;
        let n = prompt_ids.len();
        let split = self.embed_session.is_some();

        let mut state = DecodeState::default();
        let mut ws = SamplerWorkspace::default();
        let mut rng = rand::rng();

        // ── Prefill ─────────────────────────────────────────────────────
        let mut inputs = if split {
            let embeds = self.embed_ids(prompt_ids)?.unwrap();
            build_embeds_inputs(embeds, n, p.has_position_ids)?
        } else {
            build_inputs(prompt_ids, n, p.has_position_ids)?
        };
        for i in 0..p.num_layers {
            let (k, v) = empty_kv(p.num_kv_heads, p.head_dim, p.kv_is_f16)?;
            inputs.push((format!("past_key_values.{i}.key").into(), k));
            inputs.push((format!("past_key_values.{i}.value").into(), v));
        }

        let outputs = session.run(inputs).map_err(e)?;
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

        let mut kv: Vec<DynValue> = outputs.into_iter().skip(1).map(|(_, v)| v).collect();

        // ── Decode ──────────────────────────────────────────────────────
        let max = self.max_new_tokens;
        for _ in 1..max {
            let tok = *state.generated.last().unwrap() as i64;
            let total = n + state.generated.len();

            let mut inputs = if split {
                let embeds = self.embed_ids(&[tok])?.unwrap();
                build_embeds_decode_inputs(embeds, total, p.has_position_ids)?
            } else {
                build_decode_inputs(tok, total, p.has_position_ids)?
            };
            for (i, v) in kv.drain(..).enumerate() {
                let layer = i / 2;
                let kind = if i % 2 == 0 { "key" } else { "value" };
                inputs.push((format!("past_key_values.{layer}.{kind}").into(), v));
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
            kv = outputs.into_iter().skip(1).map(|(_, v)| v).collect();
        }

        Ok(state.generated)
    }
}

#[async_trait::async_trait]
impl TextPostProcessor for OrtStyleLlm {
    async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
        let mut text = self.apply_style(
            &input.raw_text,
            &input.language,
            &input.style_preset,
            &input.correction_spans,
            &input.uncertain_words,
        )?;

        text = crate::prompt::fix_all_number_spaces(&text);

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

fn build_embeds_inputs(
    embeds: DynValue,
    seq_len: usize,
    position_ids: bool,
) -> Result<Inputs<'static>> {
    let mask = Array2::<i64>::ones((1, seq_len));
    let mut out = vec![
        ("inputs_embeds".into(), embeds),
        ("attention_mask".into(), dyn_i64(mask)?),
    ];
    if position_ids {
        let pos = Array2::from_shape_vec((1, seq_len), (0..seq_len as i64).collect()).map_err(e)?;
        out.push(("position_ids".into(), dyn_i64(pos)?));
    }
    Ok(out)
}

fn build_embeds_decode_inputs(
    embeds: DynValue,
    total_len: usize,
    position_ids: bool,
) -> Result<Inputs<'static>> {
    let mask = Array2::<i64>::ones((1, total_len));
    let mut out = vec![
        ("inputs_embeds".into(), embeds),
        ("attention_mask".into(), dyn_i64(mask)?),
    ];
    if position_ids {
        let pos = Array2::from_shape_vec((1, 1), vec![(total_len - 1) as i64]).map_err(e)?;
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

struct ModelFiles {
    decoder: PathBuf,
    embed_tokens: Option<PathBuf>,
}

fn find_model(dir: &Path) -> Result<ModelFiles> {
    let quant_priority = ["q4f16", "int8", "fp16", "q4", "quantized", ""];

    for q in &quant_priority {
        let suffix = if q.is_empty() {
            String::new()
        } else {
            format!("_{q}")
        };
        for base in [dir.join("onnx"), dir.to_path_buf()] {
            let decoder = base.join(format!("decoder_model_merged{suffix}.onnx"));
            let embed = base.join(format!("embed_tokens{suffix}.onnx"));
            if decoder.exists() && embed.exists() {
                return Ok(ModelFiles {
                    decoder,
                    embed_tokens: Some(embed),
                });
            }
        }
    }

    for q in &quant_priority {
        let suffix = if q.is_empty() {
            String::new()
        } else {
            format!("_{q}")
        };
        {
            let name = format!("model{suffix}.onnx");
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

    #[test]
    fn user_prompt_lists_editable_spans_and_copy_constraint() {
        let prompt = OrtStyleLlm::user_prompt(
            "call with Jennifer Martinez to discuss interesting their OAuth deployment on AWS",
            "en",
            "",
            &[CorrectionSpan {
                start_word: 5,
                end_word: 10,
                text: "discuss interesting their OAuth deployment".into(),
                confidence: 0.18,
                word_error_risk: 0.82,
            }],
            &[],
        );

        assert!(prompt.contains("Editable spans"));
        assert!(prompt.contains("#5..10"));
        assert!(prompt.contains("Copy everything else exactly"));
        assert!(prompt.contains("Correct this en transcript:"));
    }

    #[test]
    fn user_prompt_contains_transcript() {
        let prompt = OrtStyleLlm::user_prompt("das ist ein test", "de", "", &[], &[]);

        assert!(prompt.contains("das ist ein test"));
        assert!(prompt.contains("transcript:"));
    }
}
