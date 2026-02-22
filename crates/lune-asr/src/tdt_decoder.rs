//! TDT greedy decoder with confidence extraction.
//!
//! Ported from `parakeet-rs/src/model_tdt.rs` (greedy_decode) +
//! `decoder_tdt.rs` (token-to-text) + `timestamps.rs` (word grouping).
//! **Adds** softmax confidence scoring at each decode step.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, s};
use ort::session::Session;

use lune_core::error::{Error, Result};

use crate::vocab::Vocabulary;

// ── Public types ─────────────────────────────────────────────────────────────

/// A decoded token with timing and confidence.
#[derive(Debug, Clone)]
pub struct DecodedToken {
    pub token_id: usize,
    pub frame_index: usize,
    #[allow(dead_code)]
    pub duration: usize,
    /// Softmax probability of the chosen token.
    pub confidence: f32,
}

/// Final transcription result with per-token and aggregate confidence.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    #[allow(dead_code)]
    pub tokens: Vec<TimedToken>,
    /// Mean token confidence across the utterance.
    pub confidence: f32,
}

/// A token with start/end timestamps (seconds).
#[derive(Debug, Clone)]
pub struct TimedToken {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

// Precomputed: (encoder_stride=8 * hop=160) / sr=16000 = 0.08 seconds per encoder frame
const STRIDE_HOP_OVER_SR: f32 = (8 * 160) as f32 / 16000.0;

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

        let encoder = build_session(&encoder_path, use_gpu)?;
        let decoder_joint = build_session(&decoder_path, use_gpu)?;

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
    pub fn transcribe(&mut self, features: Array2<f32>) -> Result<TranscriptionResult> {
        let (encoder_out, _encoder_len) = self.run_encoder(&features)?;
        let decoded_tokens = self.greedy_decode(&encoder_out)?;
        self.tokens_to_text(&decoded_tokens)
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

                // Update LSTM states
                if let Ok((h_shape, h_data)) =
                    outputs["output_states_1"].try_extract_tensor::<f32>()
                {
                    let d = h_shape.as_ref();
                    state_h = Array3::from_shape_vec(
                        (d[0] as usize, d[1] as usize, d[2] as usize),
                        h_data.to_vec(),
                    )
                    .map_err(|e| Error::Asr(format!("update state_h: {e}")))?;
                }
                if let Ok((c_shape, c_data)) =
                    outputs["output_states_2"].try_extract_tensor::<f32>()
                {
                    let d = c_shape.as_ref();
                    state_c = Array3::from_shape_vec(
                        (d[0] as usize, d[1] as usize, d[2] as usize),
                        c_data.to_vec(),
                    )
                    .map_err(|e| Error::Asr(format!("update state_c: {e}")))?;
                }

                decoded.push(DecodedToken {
                    token_id,
                    frame_index: t,
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

    // ── Token-to-text (SentencePiece + digit spacing + word grouping) ────

    fn tokens_to_text(&self, decoded: &[DecodedToken]) -> Result<TranscriptionResult> {
        if decoded.is_empty() {
            // Silence = certainty (CognitivePipe convention: no tokens → high confidence)
            return Ok(TranscriptionResult {
                text: String::new(),
                tokens: Vec::new(),
                confidence: 1.0,
            });
        }

        let mut raw_tokens = Vec::with_capacity(decoded.len());
        let mut full_text = String::with_capacity(decoded.len() * 4);
        // Track last char separately to avoid O(n) `chars().last()` scans.
        let mut last_char: Option<char> = None;

        for (i, dt) in decoded.iter().enumerate() {
            let Some(token_text) = self.vocab.get(dt.token_id) else {
                continue;
            };

            // Precomputed constant eliminates per-token mul+div
            let start = dt.frame_index as f32 * STRIDE_HOP_OVER_SR;
            let end = if i + 1 < decoded.len() {
                decoded[i + 1].frame_index as f32 * STRIDE_HOP_OVER_SR
            } else {
                start + 0.01
            };

            // SentencePiece: ▁ → space
            let mut display = token_text.replace('▁', " ");

            // Digit spacing heuristic: insert space before digit tokens after 2+ letters
            // (except single uppercase like A4, B12).
            if !full_text.is_empty()
                && !display.starts_with(' ')
                && display.chars().all(|c| c.is_ascii_digit())
            {
                let trailing_letters: usize = full_text
                    .chars()
                    .rev()
                    .take_while(|c| c.is_alphabetic())
                    .count();
                let is_article_a = trailing_letters == 1 && last_char == Some('a');
                if trailing_letters > 1 || is_article_a {
                    display.insert(0, ' ');
                }
            }

            // Skip <special> tokens (keep <unk>)
            if token_text.starts_with('<') && token_text.ends_with('>') && token_text != "<unk>" {
                continue;
            }

            // Update last_char tracker from display (not token_text, which may have ▁)
            last_char = display.chars().last();
            full_text.push_str(&display);
            raw_tokens.push(TimedToken {
                text: display,
                start,
                end,
            });
        }

        // Word grouping: merge subword tokens by ▁/space boundaries
        let words = group_by_words(&raw_tokens);

        // Mean confidence (single pass, already O(n))
        let mean_confidence =
            decoded.iter().map(|d| d.confidence).sum::<f32>() / decoded.len() as f32;

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            tokens: words,
            confidence: mean_confidence,
        })
    }
}

// ── Word grouping ────────────────────────────────────────────────────────────

fn group_by_words(tokens: &[TimedToken]) -> Vec<TimedToken> {
    if tokens.is_empty() {
        return Vec::new();
    }

    // Estimate ~1 word per 2 tokens on average; avoids most reallocations.
    let mut words = Vec::with_capacity(tokens.len() / 2 + 1);
    let mut current_text = String::new();
    let mut current_start = 0.0f32;
    let mut last_word_lower = String::new();

    for (i, token) in tokens.iter().enumerate() {
        // Space-only tokens act as word separators
        if token.text.trim().is_empty() {
            if !current_text.is_empty() {
                // Avoid to_lowercase() allocation on cache-hit (common dedup path)
                if !current_text.eq_ignore_ascii_case(&last_word_lower) {
                    let lower = current_text.to_lowercase();
                    words.push(TimedToken {
                        text: current_text.clone(),
                        start: current_start,
                        end: if i > 0 { tokens[i - 1].end } else { token.end },
                    });
                    last_word_lower = lower;
                }
                current_text.clear();
            }
            continue;
        }

        // Compute trimmed once; reuse as `clean` to avoid double trim call below.
        let trimmed = token.text.trim_start_matches('▁').trim_start_matches(' ');
        let is_pure_punct =
            !token.text.is_empty() && token.text.chars().all(|c| c.is_ascii_punctuation());
        let is_contraction = trimmed.starts_with('\'');
        let is_hyphenation = trimmed.starts_with('-');

        let starts_word =
            (token.text.starts_with('▁') || token.text.starts_with(' ') || is_pure_punct)
                && !is_contraction
                && !is_hyphenation
                || i == 0;

        if starts_word && !current_text.is_empty() {
            if !current_text.eq_ignore_ascii_case(&last_word_lower) {
                let lower = current_text.to_lowercase();
                words.push(TimedToken {
                    text: current_text.clone(),
                    start: current_start,
                    end: tokens[i - 1].end,
                });
                last_word_lower = lower;
            }
            current_text.clear();
        }

        if current_text.is_empty() {
            current_start = token.start;
        }

        // Reuse `trimmed` (already computed above) — avoids a second trim scan.
        current_text.push_str(trimmed);
    }

    // Final word
    if !current_text.is_empty() && !current_text.eq_ignore_ascii_case(&last_word_lower) {
        words.push(TimedToken {
            text: current_text,
            start: current_start,
            end: tokens.last().unwrap().end,
        });
    }

    words
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[cold]
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

fn build_session(path: &Path, use_gpu: bool) -> Result<Session> {
    use ort::session::builder::GraphOptimizationLevel;

    let builder = Session::builder()
        .map_err(|e| Error::Asr(format!("session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Asr(format!("optimization level: {e}")))?
        .with_intra_threads(4)
        .map_err(|e| Error::Asr(format!("intra threads: {e}")))?;

    let builder = if use_gpu {
        #[cfg(feature = "cuda")]
        {
            builder
                .with_execution_providers([
                    ort::ep::CUDA::default().build(),
                    ort::ep::CPU::default().build().error_on_failure(),
                ])
                .map_err(|e| Error::Asr(format!("CUDA provider: {e}")))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::warn!("use_gpu=true but cuda feature not enabled, falling back to CPU");
            builder
        }
    } else {
        builder
    };

    builder
        .commit_from_file(path)
        .map_err(|e| Error::Asr(format!("load model {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_grouping_basic() {
        let tokens = vec![
            TimedToken {
                text: " Hello".into(),
                start: 0.0,
                end: 0.5,
            },
            TimedToken {
                text: " world".into(),
                start: 0.5,
                end: 1.0,
            },
        ];
        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn word_grouping_hyphenated() {
        let tokens = vec![
            TimedToken {
                text: " twenty".into(),
                start: 0.0,
                end: 0.3,
            },
            TimedToken {
                text: "-two".into(),
                start: 0.3,
                end: 0.6,
            },
            TimedToken {
                text: " apples".into(),
                start: 0.6,
                end: 1.0,
            },
        ];
        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "twenty-two");
        assert_eq!(words[1].text, "apples");
    }

    #[test]
    fn word_grouping_contraction() {
        let tokens = vec![
            TimedToken {
                text: " they".into(),
                start: 0.0,
                end: 0.3,
            },
            TimedToken {
                text: "'re".into(),
                start: 0.3,
                end: 0.5,
            },
            TimedToken {
                text: " here".into(),
                start: 0.5,
                end: 0.8,
            },
        ];
        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "they're");
        assert_eq!(words[1].text, "here");
    }

    #[test]
    fn word_deduplication() {
        let tokens = vec![
            TimedToken {
                text: " hello".into(),
                start: 0.0,
                end: 0.5,
            },
            TimedToken {
                text: " hello".into(),
                start: 0.5,
                end: 1.0,
            },
            TimedToken {
                text: " world".into(),
                start: 1.0,
                end: 1.5,
            },
        ];
        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn stride_hop_constant() {
        // Verify the precomputed constant matches the original arithmetic
        let expected = (8usize * 160) as f32 / 16000.0;
        assert!((STRIDE_HOP_OVER_SR - expected).abs() < f32::EPSILON);
    }
}
