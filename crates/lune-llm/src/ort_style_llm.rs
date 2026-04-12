//! ONNX style LLM — Llama 3.2 1B Instruct with KV-cache decode.
//!
//! Applies style presets to ASR transcripts via a lightweight local LLM.
//! Uses greedy decoding with KV cache for fast autoregressive generation.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, Value};
use tokenizers::Tokenizer;
use tracing::info;

use lune_core::error::{Error, Result};

const NUM_LAYERS: usize = 16;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const EOS_IDS: [u32; 3] = [128001, 128008, 128009];
const DEFAULT_MAX_TOKENS: usize = 128;

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

pub struct OrtStyleLlm {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_new_tokens: usize,
}

fn e(msg: impl std::fmt::Display) -> Error {
    Error::Config(msg.to_string())
}

fn dyn_i64(arr: Array2<i64>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
}
fn dyn_f32_4(arr: Array4<f32>) -> Result<DynValue> {
    Value::from_array(arr).map(|v| v.into_dyn()).map_err(e)
}

impl OrtStyleLlm {
    pub fn new(config: OrtStyleLlmConfig) -> Result<Self> {
        let model_path = find_model(&config.model_dir)?;
        let opt_path = model_path.with_extension("onnx.opt");

        let session = Session::builder()
            .map_err(e)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(e)?
            .with_intra_threads(4)
            .map_err(e)?
            .with_optimized_model_path(&opt_path)
            .map_err(e)?
            .commit_from_file(&model_path)
            .map_err(e)?;

        let tokenizer = Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
            .map_err(|e| Error::Config(format!("tokenizer: {e}")))?;

        info!(model = %model_path.display(), "style LLM loaded");
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_new_tokens: config.max_new_tokens,
        })
    }

    pub fn apply_style(&self, system_prompt: &str, transcript: &str) -> Result<String> {
        let prompt = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
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
        Ok(text.trim().to_string())
    }

    fn generate(&self, prompt_ids: &[i64]) -> Result<Vec<u32>> {
        let mut session = self.session.lock().map_err(e)?;
        let n = prompt_ids.len();

        // ── Prefill ─────────────────────────────────────────────────────
        let mut inputs = build_inputs(prompt_ids, n, 0)?;
        let empty = Array4::<f32>::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM));
        for i in 0..NUM_LAYERS {
            inputs.push((
                format!("past_key_values.{i}.key").into(),
                dyn_f32_4(empty.clone())?,
            ));
            inputs.push((
                format!("past_key_values.{i}.value").into(),
                dyn_f32_4(empty.clone())?,
            ));
        }

        let outputs = session.run(inputs).map_err(e)?;
        let first = argmax_last(&outputs[0])?;
        if EOS_IDS.contains(&first) {
            return Ok(vec![]);
        }

        let mut generated = vec![first];
        let mut kv: Vec<DynValue> = outputs.into_iter().skip(1).map(|(_, v)| v).collect();

        // ── Decode ──────────────────────────────────────────────────────
        let max = self.max_new_tokens.min(n * 2);
        for _ in 1..max {
            let tok = *generated.last().unwrap() as i64;
            let total = n + generated.len();

            let mut inputs = build_decode_inputs(tok, total)?;
            for (i, v) in kv.drain(..).enumerate() {
                let layer = i / 2;
                let kind = if i % 2 == 0 { "key" } else { "value" };
                inputs.push((format!("past_key_values.{layer}.{kind}").into(), v));
            }

            let outputs = session.run(inputs).map_err(e)?;
            let next = argmax_last(&outputs[0])?;
            if EOS_IDS.contains(&next) {
                break;
            }
            generated.push(next);
            kv = outputs.into_iter().skip(1).map(|(_, v)| v).collect();
        }

        Ok(generated)
    }
}

type Inputs<'a> = Vec<(std::borrow::Cow<'a, str>, DynValue)>;

fn build_inputs(ids: &[i64], seq_len: usize, _offset: usize) -> Result<Inputs<'static>> {
    let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec()).map_err(e)?;
    let mask = Array2::<i64>::ones((1, seq_len));
    let pos = Array2::from_shape_vec((1, seq_len), (0..seq_len as i64).collect()).map_err(e)?;
    Ok(vec![
        ("input_ids".into(), dyn_i64(input_ids)?),
        ("attention_mask".into(), dyn_i64(mask)?),
        ("position_ids".into(), dyn_i64(pos)?),
    ])
}

fn build_decode_inputs(token: i64, total_len: usize) -> Result<Inputs<'static>> {
    let ids = Array2::from_shape_vec((1, 1), vec![token]).map_err(e)?;
    let mask = Array2::<i64>::ones((1, total_len));
    let pos = Array2::from_shape_vec((1, 1), vec![(total_len - 1) as i64]).map_err(e)?;
    Ok(vec![
        ("input_ids".into(), dyn_i64(ids)?),
        ("attention_mask".into(), dyn_i64(mask)?),
        ("position_ids".into(), dyn_i64(pos)?),
    ])
}

fn argmax_last(value: &DynValue) -> Result<u32> {
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(e)?;
    let shape = shape.as_ref();
    // shape: [1, seq_len, vocab_size]
    let vocab = shape[2] as usize;
    let last_pos = shape[1] as usize - 1;
    let offset = last_pos * vocab; // skip to last position's logits
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
    let onnx_dir = dir.join("onnx");
    let search = if onnx_dir.is_dir() { &onnx_dir } else { dir };
    for name in [
        "model_q4f16.onnx",
        "model_q4.onnx",
        "model_fp16.onnx",
        "model.onnx",
    ] {
        let p = search.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(Error::Config(format!("no ONNX model in {}", dir.display())))
}
