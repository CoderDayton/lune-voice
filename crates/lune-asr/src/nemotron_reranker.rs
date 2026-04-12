//! NVIDIA Llama-Nemotron Reranker 1B v2 — ONNX cross-encoder.
//!
//! Loads the `LlamaBidirectionalForSequenceClassification` ONNX export of
//! [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2)
//! and scores `(query, document)` pairs. Used by the session layer as a
//! replacement for the generative n-best rescoring path: when the ASR
//! emits ≥2 full-sentence hypotheses, we feed them through this cross-
//! encoder, rank by logit, and return the top-1 verbatim.
//!
//! Model layout:
//! - Input: `input_ids [batch, seq]` + `attention_mask [batch, seq]`
//!   (both int64), seq ≤ 512 for low latency, ≤ 8192 for long docs
//! - Output: `logits [batch, 1]` — raw relevance score per pair
//!
//! The model was fine-tuned with contrastive learning and bidirectional
//! attention from Llama-3.2-1B. Scores are unbounded reals; apply sigmoid
//! for probabilities, or just argmax directly for the top candidate.
//!
//! The prompt template matches the NVIDIA reference:
//!
//! ```text
//! question:{query} \n \n passage:{document}
//! ```
//!
//! Left-padding is used so the classifier head sees the document tail in
//! the same position across a batch.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::Session;
use tokenizers::Tokenizer;

use lune_core::error::{Error, Result};

use crate::ort_utils::build_ort_session;

/// Maximum tokens per (query, document) pair. Nemotron was trained at
/// 8192 but our use case is very short sentences — cap at 256 for
/// latency. Anything longer is truncated.
const MAX_SEQ_LEN: usize = 256;

/// Left-padding token id from `config.json` (`pad_token_id = 128001`).
/// Left-padding matches the NVIDIA reference inference script so the
/// classifier head sees the document tail at a fixed position.
const PAD_TOKEN_ID: u32 = 128001;

/// Constructor config for [`NemotronReranker`].
pub struct NemotronRerankerConfig {
    /// Directory containing the ONNX model + `tokenizer.json`.
    pub model_dir: PathBuf,
    /// Prefer CUDA EP when available.
    pub use_gpu: bool,
}

/// Cross-encoder reranker over (query, document) pairs.
///
/// Thread-safe: the inner `Session` is wrapped in a `Mutex` so multiple
/// async tasks can share one reranker. Scoring is batched — pass a slice
/// of documents and the encoder runs them in a single forward pass.
pub struct NemotronReranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl NemotronReranker {
    /// Load the reranker from a directory containing:
    /// - `model_int4_full.onnx` (preferred) or `model_int8.onnx` or
    ///   `model.onnx` (+ `model.onnx_data` sidecar)
    /// - `tokenizer.json`
    ///
    /// Falls through in that order of quantization preference — smaller
    /// first for lower VRAM, higher precision only if int4 is missing.
    pub fn new(config: NemotronRerankerConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir.clone()));
        }

        let model_path = find_onnx(&config.model_dir)?;
        let tok_path = config.model_dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(Error::Asr(format!(
                "tokenizer.json not found in {}",
                config.model_dir.display()
            )));
        }

        let mut tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Asr(format!("load tokenizer {}: {e}", tok_path.display())))?;

        // Left-pad + truncate match the NVIDIA reference template. We
        // always pad/truncate to the same length per batch, and the
        // left-pad ensures the classifier head sees the passage tail at
        // a stable relative position regardless of query length.
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            pad_to_multiple_of: None,
            pad_id: PAD_TOKEN_ID,
            pad_type_id: 0,
            pad_token: "<|end_of_text|>".to_string(),
        }));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ_LEN,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| Error::Asr(format!("set truncation: {e}")))?;

        let session = build_ort_session(&model_path, config.use_gpu)?;
        tracing::info!(
            model = %model_path.display(),
            tokenizer = %tok_path.display(),
            use_gpu = config.use_gpu,
            "initialized Nemotron reranker"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Score a slice of documents against a query. Returns one logit per
    /// document, in the same order as the input. Higher = more relevant.
    ///
    /// Empty `documents` returns an empty vec without running inference.
    pub fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Format each pair with the NVIDIA template.
        let texts: Vec<String> = documents
            .iter()
            .map(|doc| format!("question:{query} \n \n passage:{doc}"))
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(String::as_str).collect::<Vec<_>>(), true)
            .map_err(|e| Error::Asr(format!("tokenize reranker batch: {e}")))?;

        let batch_size = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        if seq_len == 0 {
            return Ok(vec![0.0; batch_size]);
        }

        // Flat-pack ids + mask into (batch, seq) arrays. Tokenizer's
        // BatchLongest padding guarantees uniform seq_len.
        let mut ids_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut mask_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        for enc in &encodings {
            for &id in enc.get_ids() {
                ids_flat.push(id as i64);
            }
            for &m in enc.get_attention_mask() {
                mask_flat.push(m as i64);
            }
        }

        let input_ids = Array2::from_shape_vec((batch_size, seq_len), ids_flat)
            .map_err(|e| Error::Asr(format!("input_ids reshape: {e}")))?;
        let attention_mask = Array2::from_shape_vec((batch_size, seq_len), mask_flat)
            .map_err(|e| Error::Asr(format!("attention_mask reshape: {e}")))?;

        let ids_val = ort::value::Value::from_array(input_ids)
            .map_err(|e| Error::Asr(format!("input_ids value: {e}")))?;
        let mask_val = ort::value::Value::from_array(attention_mask)
            .map_err(|e| Error::Asr(format!("attention_mask value: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Asr(format!("reranker session lock: {e}")))?;
        let outputs = session
            .run(ort::inputs!(
                "input_ids" => ids_val,
                "attention_mask" => mask_val
            ))
            .map_err(|e| Error::Asr(format!("reranker forward: {e}")))?;

        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Asr(format!("extract logits: {e}")))?;

        // Expected shape: [batch, 1]. Defensive check so a broken export
        // produces a clear error instead of silent nonsense.
        let dims = shape.as_ref();
        if dims.len() != 2 || dims[1] != 1 {
            return Err(Error::Asr(format!(
                "unexpected logits shape {dims:?}, expected [batch, 1]"
            )));
        }
        if dims[0] as usize != batch_size {
            return Err(Error::Asr(format!(
                "logits batch mismatch: got {}, expected {}",
                dims[0], batch_size
            )));
        }

        Ok(data.to_vec())
    }

    /// Convenience: score a list of documents and return the index of
    /// the highest-scoring one. Returns 0 for empty inputs (caller
    /// should check `documents.len()` first if that's not desired).
    pub fn argmax(&self, query: &str, documents: &[String]) -> Result<usize> {
        let scores = self.score(query, documents)?;
        Ok(scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0))
    }
}

/// Quantisation preference for `model.onnx` variants in `model_dir`.
/// Order: int4 (smallest) → int8 → fp32 fallback.
fn find_onnx(dir: &Path) -> Result<PathBuf> {
    for candidate in ["model_int4_full.onnx", "model_int8.onnx", "model.onnx"] {
        let p = dir.join(candidate);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(Error::Asr(format!(
        "no Nemotron ONNX model found in {}",
        dir.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: load the int4 model from the standard model dir and
    /// verify that a relevant document scores higher than an irrelevant
    /// one. Skipped when the model isn't present on disk, so CI that
    /// doesn't have the model doesn't fail.
    #[test]
    fn reranker_distinguishes_relevant_from_irrelevant() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/nemotron-rerank-1b");
        if !model_dir.join("tokenizer.json").exists() {
            eprintln!("[skip] model dir not found at {}", model_dir.display());
            return;
        }

        let reranker = NemotronReranker::new(NemotronRerankerConfig {
            model_dir,
            use_gpu: false, // CPU for CI portability; smoke test is tiny
        })
        .expect("load reranker");

        let query = "what is a panda?";
        let docs = vec![
            "A panda is a large black-and-white bear native to China.".to_string(),
            "The sky is blue and the grass is green.".to_string(),
        ];
        let scores = reranker.score(query, &docs).expect("score");
        assert_eq!(scores.len(), 2);
        // Relevant document must clearly outscore the irrelevant one.
        // Reference margin on the int4 export is ~31 (see Python sanity
        // check); any positive margin above 5 proves the model works.
        let margin = scores[0] - scores[1];
        assert!(
            margin > 5.0,
            "relevance margin too small: {} vs {} (delta={})",
            scores[0],
            scores[1],
            margin
        );

        // argmax convenience should agree.
        let best = reranker.argmax(query, &docs).expect("argmax");
        assert_eq!(best, 0);
    }
}
