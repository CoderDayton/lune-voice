//! TDT greedy decoder with confidence extraction.
//!
//! Ported from `parakeet-rs/src/model_tdt.rs` (greedy_decode) +
//! `decoder_tdt.rs` (token-to-text) + `timestamps.rs` (word grouping).
//! **Adds** softmax confidence scoring at each decode step.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, s};
use ort::session::Session;

use lune_core::error::{Error, Result};

use crate::ort_utils::build_ort_session;
use crate::vocab::Vocabulary;

// ── Public types ─────────────────────────────────────────────────────────────

/// A decoded token with timing and confidence.
#[derive(Debug, Clone)]
pub struct DecodedToken {
    pub token_id: usize,
    #[allow(dead_code)]
    pub duration: usize,
    /// Softmax probability of the chosen token.
    pub confidence: f32,
}

/// Final transcription result with aggregate confidence.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    /// Mean token confidence across the utterance.
    pub confidence: f32,
}

// ── TDT Model ────────────────────────────────────────────────────────────────

pub struct TdtModel {
    encoder: Session,
    decoder_joint: Session,
    vocab: Vocabulary,
}

impl TdtModel {
    /// Load encoder + decoder_joint ONNX sessions and vocabulary.
    pub fn new(model_dir: &Path, use_gpu: bool) -> Result<Self> {
        let encoder_path = find_file(
            model_dir,
            &[
                "encoder-model.onnx",
                "encoder.onnx",
                "encoder-model.int8.onnx",
            ],
            "encoder",
        )?;
        let decoder_path = find_file(
            model_dir,
            &[
                "decoder_joint-model.onnx",
                "decoder_joint-model.int8.onnx",
                "decoder_joint.onnx",
            ],
            "decoder_joint",
        )?;
        let vocab_path = model_dir.join("vocab.txt");

        if !vocab_path.exists() {
            return Err(Error::Asr(format!(
                "vocab.txt not found in {}",
                model_dir.display()
            )));
        }

        let vocab = Vocabulary::from_file(&vocab_path)?;

        let encoder = build_ort_session(&encoder_path, use_gpu)?;
        let decoder_joint = build_ort_session(&decoder_path, use_gpu)?;

        tracing::info!(
            vocab_size = vocab.size(),
            encoder = %encoder_path.display(),
            decoder = %decoder_path.display(),
            "TDT model loaded"
        );

        Ok(Self {
            encoder,
            decoder_joint,
            vocab,
        })
    }

    /// Run full transcription pipeline: encode → greedy decode → token-to-text.
    /// Returns the result plus raw decoded tokens for per-word confidence extraction.
    pub fn transcribe_with_tokens(
        &mut self,
        features: Array2<f32>,
    ) -> Result<(TranscriptionResult, Vec<DecodedToken>)> {
        let (encoder_out, _encoder_len) = self.run_encoder(&features)?;
        let decoded_tokens = self.greedy_decode(&encoder_out)?;
        let result = self.tokens_to_text(&decoded_tokens)?;
        Ok((result, decoded_tokens))
    }

    /// Access the vocabulary (needed for word confidence extraction).
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    // ── Encoder ──────────────────────────────────────────────────────────

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        // TDT encoder expects (batch=1, features, time)
        let input = features
            .t()
            .to_shape((1, feature_size, time_steps))
            .map_err(|e| Error::Asr(format!("reshape encoder input: {e}")))?
            .to_owned();

        let input_length = Array1::from_vec(vec![time_steps as i64]);

        let input_value = ort::value::Value::from_array(input)
            .map_err(|e| Error::Asr(format!("encoder input value: {e}")))?;
        let length_value = ort::value::Value::from_array(input_length)
            .map_err(|e| Error::Asr(format!("encoder length value: {e}")))?;

        let outputs = self
            .encoder
            .run(ort::inputs!(
                "audio_signal" => input_value,
                "length" => length_value
            ))
            .map_err(|e| Error::Asr(format!("encoder run: {e}")))?;

        let (shape, data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Asr(format!("extract encoder output: {e}")))?;

        let (_, lens_data) = outputs["encoded_lengths"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Asr(format!("extract encoder lengths: {e}")))?;

        let dims = shape.as_ref();
        if dims.len() != 3 {
            return Err(Error::Asr(format!(
                "expected 3D encoder output, got {dims:?}"
            )));
        }

        let (b, t, d) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
        let arr = Array3::from_shape_vec((b, t, d), data.to_vec())
            .map_err(|e| Error::Asr(format!("encoder array: {e}")))?;

        Ok((arr, lens_data[0]))
    }

    // ── Greedy TDT decode with confidence ────────────────────────────────

    fn greedy_decode(&mut self, encoder_out: &Array3<f32>) -> Result<Vec<DecodedToken>> {
        let encoder_dim = encoder_out.shape()[1];
        let time_steps = encoder_out.shape()[2];
        let vocab_size = self.vocab.size();
        let blank_id = self.vocab.blank_id;
        let max_tokens_per_step = 10;

        // LSTM states: (layers=2, batch=1, hidden=640)
        let mut state_h = Array3::<f32>::zeros((2, 1, 640));
        let mut state_c = Array3::<f32>::zeros((2, 1, 640));

        let mut decoded = Vec::new();
        let mut t = 0;
        let mut emitted_tokens = 0;
        let mut last_token = blank_id as i32;

        // Reusable buffers to avoid per-step allocation
        let mut vocab_logits = Vec::with_capacity(vocab_size);
        let mut duration_logits = Vec::with_capacity(8);

        while t < time_steps {
            // Single encoder frame: [0, :, t] → (1, enc_dim, 1)
            let frame = encoder_out.slice(s![0, .., t]).to_owned();
            let frame_3d = frame
                .to_shape((1, encoder_dim, 1))
                .map_err(|e| Error::Asr(format!("reshape frame: {e}")))?
                .to_owned();

            let targets = Array2::from_shape_vec((1, 1), vec![last_token])
                .map_err(|e| Error::Asr(format!("targets: {e}")))?;

            let frame_val = ort::value::Value::from_array(frame_3d)
                .map_err(|e| Error::Asr(format!("frame value: {e}")))?;
            let targets_val = ort::value::Value::from_array(targets)
                .map_err(|e| Error::Asr(format!("targets value: {e}")))?;
            let target_len_val = ort::value::Value::from_array(Array1::from_elem(1, 1i32))
                .map_err(|e| Error::Asr(format!("target_length value: {e}")))?;
            let state_h_val = ort::value::Value::from_array(state_h.clone())
                .map_err(|e| Error::Asr(format!("state_h value: {e}")))?;
            let state_c_val = ort::value::Value::from_array(state_c.clone())
                .map_err(|e| Error::Asr(format!("state_c value: {e}")))?;

            let outputs = self
                .decoder_joint
                .run(ort::inputs!(
                    "encoder_outputs" => frame_val,
                    "targets" => targets_val,
                    "target_length" => target_len_val,
                    "input_states_1" => state_h_val,
                    "input_states_2" => state_c_val
                ))
                .map_err(|e| Error::Asr(format!("decoder run: {e}")))?;

            // Extract logits: first vocab_size = token logits, last 5 = duration logits
            let (_, logits_data) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Asr(format!("extract logits: {e}")))?;

            // Single-pass: collect vocab logits + find argmax + track max simultaneously.
            // This eliminates a separate O(vocab_size) argmax scan after extend().
            vocab_logits.clear();
            let mut token_id = blank_id;
            let mut max_logit = f32::NEG_INFINITY;
            for (i, &v) in logits_data.iter().take(vocab_size).enumerate() {
                vocab_logits.push(v);
                // total_cmp: NaN-safe, branchless on x86 (ucomiss + cmov)
                if v.total_cmp(&max_logit).is_gt() {
                    max_logit = v;
                    token_id = i;
                }
            }

            // Duration argmax (small slice ~5 elements, single pass)
            duration_logits.clear();
            duration_logits.extend(logits_data.iter().copied().skip(vocab_size));
            let duration_step = duration_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if token_id != blank_id {
                // Softmax confidence: max_logit already known from single-pass above,
                // so we only need one more O(vocab_size) pass for exp_sum.
                let exp_sum: f32 = vocab_logits.iter().map(|&l| (l - max_logit).exp()).sum();
                let confidence = (vocab_logits[token_id] - max_logit).exp() / exp_sum;

                // Update LSTM states — copy into existing buffers to avoid
                // per-step Vec + Array3 allocations (~20KB/step saved).
                if let Ok((_shape, h_data)) = outputs["output_states_1"].try_extract_tensor::<f32>()
                {
                    state_h
                        .as_slice_mut()
                        .expect("state_h contiguous")
                        .copy_from_slice(h_data);
                }
                if let Ok((_shape, c_data)) = outputs["output_states_2"].try_extract_tensor::<f32>()
                {
                    state_c
                        .as_slice_mut()
                        .expect("state_c contiguous")
                        .copy_from_slice(c_data);
                }

                decoded.push(DecodedToken {
                    token_id,
                    duration: duration_step,
                    confidence,
                });
                last_token = token_id as i32;
                emitted_tokens += 1;
            }

            // Advance frame pointer
            if duration_step > 0 {
                t += duration_step;
                emitted_tokens = 0;
            } else if token_id == blank_id || emitted_tokens >= max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok(decoded)
    }

    // ── Token-to-text: CTC collapse → join → trim ────────────────────────
    //
    // The LLM handles punctuation, casing, and repetition correction.
    // Keep this path minimal — any preprocessing here risks confusing the LLM.

    fn tokens_to_text(&self, decoded: &[DecodedToken]) -> Result<TranscriptionResult> {
        if decoded.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                confidence: 1.0,
            });
        }

        let mut text = String::with_capacity(decoded.len() * 4);
        for dt in decoded {
            let Some(token_text) = self.vocab.get(dt.token_id) else {
                continue;
            };
            // Skip special tokens (blanks, padding, etc.) but keep <unk>
            if token_text.starts_with('<') && token_text.ends_with('>') && token_text != "<unk>" {
                continue;
            }
            text.push_str(&token_text.replace('▁', " "));
        }

        let confidence = decoded.iter().map(|d| d.confidence).sum::<f32>() / decoded.len() as f32;

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            confidence,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[cold]
#[allow(dead_code)]
fn find_file(dir: &Path, candidates: &[&str], kind: &str) -> Result<PathBuf> {
    for c in candidates {
        let p = dir.join(c);
        if p.exists() {
            return Ok(p);
        }
    }
    // Fallback: glob for `{kind}*.onnx`
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|s| s.to_str())
                && name.starts_with(kind)
                && name.ends_with(".onnx")
            {
                return Ok(p);
            }
        }
    }
    Err(Error::Asr(format!(
        "no {kind} model found in {}",
        dir.display()
    )))
}

#[cfg(test)]
mod tests {}
