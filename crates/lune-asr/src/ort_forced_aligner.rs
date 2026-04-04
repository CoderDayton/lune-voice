//! CTC-based forced aligner using wav2vec2 ONNX model.
//!
//! `OrtForcedAligner` owns a single wav2vec2 CTC ONNX session (360MB, ~94M params).
//! Input: raw 16kHz f32 PCM. Output: CTC log-prob emissions [1, T/320, 29].
//!
//! `align(pcm, text)` runs Viterbi forced alignment over the CTC trellis to
//! produce per-word timestamps, then returns words sorted by `start_ms`.

use std::sync::Mutex;

use ndarray::Array2;
use ort::session::Session;
use ort::value::{DynValue, Value};
use serde::{Deserialize, Serialize};

use lune_core::error::{Error, Result};

// ── Config ────────────────────────────────────────────────────────────────────

/// Serde-deserialisable counterpart to `fa_config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaConfig {
    /// Model type identifier (e.g. "wav2vec2_ctc").
    pub model_type: String,
    /// Expected sample rate (16000).
    pub sample_rate: u32,
    /// Ordered label list: index 0 = blank, index 1 = word delimiter '|', then characters.
    pub labels: Vec<String>,
    /// Index of the CTC blank token (typically 0).
    pub blank_id: usize,
    /// Character used as word delimiter in CTC vocabulary (typically "|").
    pub word_delimiter: String,
    /// Frame stride in milliseconds (typically 20).
    pub stride_ms: u32,
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-word timestamp output from `OrtForcedAligner::align`.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub text: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

type TokenIds = Vec<usize>;
type WordBoundaries = Vec<(usize, usize)>;

// ── Aligner struct ────────────────────────────────────────────────────────────

pub struct OrtForcedAligner {
    /// wav2vec2 CTC model: raw audio → CTC logits [1, T, num_labels].
    model: Mutex<Session>,
    fa_config: FaConfig,
}

impl OrtForcedAligner {
    pub fn new(model: Session, fa_config: FaConfig) -> Result<Self> {
        Ok(Self {
            model: Mutex::new(model),
            fa_config,
        })
    }

    /// Align `text` against `pcm` (16 kHz mono f32 PCM).
    ///
    /// Returns words sorted by `start_ms`.
    pub fn align(&self, pcm: &[f32], text: &str) -> Result<Vec<WordTimestamp>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Cap to 30s — wav2vec2 base handles up to ~30s well.
        const MAX_SAMPLES: usize = 30 * 16_000;
        let pcm = if pcm.len() > MAX_SAMPLES {
            &pcm[..MAX_SAMPLES]
        } else {
            pcm
        };

        // ── 1. Run wav2vec2 → CTC log-probs [1, T, C] ──────────────────────
        let log_probs = self.run_model(pcm)?;
        let num_frames = log_probs.nrows();
        let num_labels = log_probs.ncols();

        if num_frames == 0 {
            return Ok(vec![]);
        }

        // ── 2. Tokenize text into CTC label indices ─────────────────────────
        let (token_ids, word_boundaries) = self.tokenize_for_ctc(&words, num_labels)?;

        if token_ids.is_empty() {
            return Ok(vec![]);
        }

        // ── 3. Viterbi forced alignment ─────────────────────────────────────
        let frame_path = viterbi_align(&log_probs, &token_ids, self.fa_config.blank_id)?;

        // ── 4. Convert frame path → word timestamps ─────────────────────────
        let stride_ms = self.fa_config.stride_ms;
        let mut result: Vec<WordTimestamp> = Vec::with_capacity(words.len());

        for (word_idx, &(tok_start, tok_end)) in word_boundaries.iter().enumerate() {
            // Find first and last frame assigned to any token in this word.
            let mut first_frame: Option<usize> = None;
            let mut last_frame: Option<usize> = None;

            for (frame_idx, &assigned_tok) in frame_path.iter().enumerate() {
                if assigned_tok >= tok_start && assigned_tok < tok_end {
                    if first_frame.is_none() {
                        first_frame = Some(frame_idx);
                    }
                    last_frame = Some(frame_idx);
                }
            }

            let start_ms = first_frame.map(|f| f as u32 * stride_ms).unwrap_or(0);
            let end_ms = last_frame
                .map(|f| (f as u32 + 1) * stride_ms)
                .unwrap_or(start_ms);

            result.push(WordTimestamp {
                text: words[word_idx].to_string(),
                start_ms,
                end_ms,
            });
        }

        result.sort_by_key(|w| w.start_ms);
        Ok(result)
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Run the wav2vec2 model on raw PCM. Returns log-probs [T, C].
    fn run_model(&self, pcm: &[f32]) -> Result<Array2<f32>> {
        let input = Array2::from_shape_vec((1, pcm.len()), pcm.to_vec())
            .map_err(|e| Error::Asr(format!("fa input shape: {e}")))?;

        let input_val: DynValue = Value::from_array(input)
            .map_err(|e| Error::Asr(format!("fa input value: {e}")))?
            .into();

        let mut guard = self
            .model
            .lock()
            .map_err(|e| Error::Asr(format!("fa model lock: {e}")))?;
        let outputs = guard
            .run(vec![("input_values".to_string(), input_val)])
            .map_err(|e| Error::Asr(format!("fa model run: {e}")))?;

        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| Error::Asr(format!("fa output extract: {e}")))?;
        let logits = logits.view();
        // Shape: [1, T, C]
        if logits.ndim() != 3 {
            return Err(Error::Asr(format!(
                "expected 3D FA output, got {}D {:?}",
                logits.ndim(),
                logits.shape()
            )));
        }
        let t = logits.shape()[1];
        let c = logits.shape()[2];

        // Convert to log-softmax along C dimension → [T, C]
        let mut log_probs = Array2::<f32>::zeros((t, c));
        for frame in 0..t {
            let max_val = (0..c)
                .map(|j| logits[[0, frame, j]])
                .fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = (0..c)
                .map(|j| (logits[[0, frame, j]] - max_val).exp())
                .sum::<f32>()
                .ln()
                + max_val;
            for j in 0..c {
                log_probs[[frame, j]] = logits[[0, frame, j]] - log_sum_exp;
            }
        }

        Ok(log_probs)
    }

    /// Convert words to CTC token indices. Returns:
    /// - `token_ids`: flat list of label indices for the full text
    /// - `word_boundaries`: (start, end) indices into `token_ids` for each word
    ///
    /// Words are separated by the word delimiter token ('|').
    fn tokenize_for_ctc(
        &self,
        words: &[&str],
        num_labels: usize,
    ) -> Result<(TokenIds, WordBoundaries)> {
        let labels = &self.fa_config.labels;

        // Build char → label_id map.
        let mut char_to_id: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();
        for (idx, label) in labels.iter().enumerate() {
            if label.len() == 1 {
                let ch = label.chars().next().unwrap();
                char_to_id.insert(ch, idx);
            }
        }

        let delim_id = labels
            .iter()
            .position(|l| l == &self.fa_config.word_delimiter);

        let mut token_ids: TokenIds = Vec::new();
        let mut word_boundaries: WordBoundaries = Vec::new();

        for (i, word) in words.iter().enumerate() {
            if i > 0
                && let Some(d) = delim_id
            {
                token_ids.push(d);
            }

            let start = token_ids.len();
            for ch in word.to_uppercase().chars() {
                if let Some(&id) = char_to_id.get(&ch)
                    && id < num_labels
                {
                    token_ids.push(id);
                }
                // Skip characters not in vocab (punctuation, etc.)
            }
            let end = token_ids.len();

            if end > start {
                word_boundaries.push((start, end));
            } else {
                // Word had no mappable characters — assign empty range.
                word_boundaries.push((start, start));
            }
        }

        Ok((token_ids, word_boundaries))
    }
}

// ── Viterbi CTC Forced Alignment ─────────────────────────────────────────────

/// Viterbi forced alignment over CTC trellis.
///
/// Given `log_probs` [T, C] and `targets` (label indices), returns a
/// frame-level path: for each frame t, the index into `targets` that
/// was aligned to that frame.
///
/// Algorithm: blank-interleaved target sequence S = [blank, t0, blank, t1, ...].
/// DP recurrence with 3-way transitions (stay, step, skip).
fn viterbi_align(
    log_probs: &Array2<f32>,
    targets: &[usize],
    blank_id: usize,
) -> Result<Vec<usize>> {
    let num_frames = log_probs.nrows();
    let num_targets = targets.len();

    if num_targets == 0 {
        return Ok(vec![0; num_frames]);
    }

    // Build extended sequence S: interleave blanks.
    // S = [blank, targets[0], blank, targets[1], ..., targets[L-1], blank]
    let s_len = 2 * num_targets + 1;
    let s_tokens: Vec<usize> = (0..s_len)
        .map(|i| if i % 2 == 0 { blank_id } else { targets[i / 2] })
        .collect();

    // Count repeats (consecutive identical targets).
    let repeats: usize = targets.windows(2).filter(|w| w[0] == w[1]).count();

    // Must have enough frames.
    let min_frames = num_targets + repeats;
    if num_frames < min_frames {
        return Err(Error::Asr(format!(
            "FA: not enough frames ({num_frames}) for {num_targets} targets + {repeats} repeats"
        )));
    }

    // ── DP with rolling buffer ──────────────────────────────────────────

    // alphas: two rows (prev, cur) of size s_len.
    let mut alpha_prev = vec![f32::NEG_INFINITY; s_len];
    let mut alpha_cur = vec![f32::NEG_INFINITY; s_len];

    // Backpointers: backptr[t][s] = source s index at t-1.
    let mut backptr: Vec<Vec<usize>> = vec![vec![0; s_len]; num_frames];

    // Initialize t=0: can start at s=0 (blank) or s=1 (first token).
    alpha_prev[0] = log_probs[[0, s_tokens[0]]];
    if s_len > 1 {
        alpha_prev[1] = log_probs[[0, s_tokens[1]]];
    }

    // DP fill.
    for t in 1..num_frames {
        for value in alpha_cur.iter_mut().take(s_len) {
            *value = f32::NEG_INFINITY;
        }

        for s in 0..s_len {
            let emission = log_probs[[t, s_tokens[s]]];

            // Transition 0: stay (s, t-1) → (s, t).
            let mut best = alpha_prev[s];
            let mut best_src = s;

            // Transition 1: step (s-1, t-1) → (s, t).
            if s > 0 && alpha_prev[s - 1] > best {
                best = alpha_prev[s - 1];
                best_src = s - 1;
            }

            // Transition 2: skip (s-2, t-1) → (s, t).
            // Only if s is odd (actual token) and not a repeat of previous token.
            if s >= 2 && s % 2 == 1 {
                let cur_target = s / 2;
                let prev_target = cur_target.wrapping_sub(1);
                let is_repeat = cur_target > 0
                    && prev_target < num_targets
                    && targets[cur_target] == targets[prev_target];
                if !is_repeat && alpha_prev[s - 2] > best {
                    best = alpha_prev[s - 2];
                    best_src = s - 2;
                }
            }

            alpha_cur[s] = best + emission;
            backptr[t][s] = best_src;
        }

        std::mem::swap(&mut alpha_prev, &mut alpha_cur);
    }

    // ── Backtrack ───────────────────────────────────────────────────────

    // Best final state: last blank (s_len-1) or last token (s_len-2).
    let mut s = if s_len >= 2 && alpha_prev[s_len - 2] > alpha_prev[s_len - 1] {
        s_len - 2
    } else {
        s_len - 1
    };

    // Frame path: maps each frame to a target index (not s-index).
    let mut frame_path = vec![0usize; num_frames];

    for t in (0..num_frames).rev() {
        // Convert s-index to target index.
        // s even = blank → assign to nearest target.
        // s odd = target[s/2].
        frame_path[t] = if s % 2 == 1 {
            s / 2
        } else if s > 0 {
            (s - 1) / 2
        } else {
            0
        };

        if t > 0 {
            s = backptr[t][s];
        }
    }

    Ok(frame_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build simple log-probs where one token dominates per frame.
    fn make_log_probs(dominant: &[usize], num_classes: usize) -> Array2<f32> {
        let t = dominant.len();
        let mut lp = Array2::from_elem((t, num_classes), -10.0f32);
        for (frame, &cls) in dominant.iter().enumerate() {
            lp[[frame, cls]] = 0.0; // log(1) = 0, dominates
        }
        lp
    }

    #[test]
    fn viterbi_simple_cat() {
        // Labels: 0=blank, 1=C, 2=A, 3=T
        // Target: [1, 2, 3] ("CAT")
        // Extended S: [0, 1, 0, 2, 0, 3, 0]
        //              b  C  b  A  b  T  b
        // Frames: each token gets ~2 frames.
        let dominant = vec![1, 1, 2, 2, 3, 3]; // C C A A T T
        let log_probs = make_log_probs(&dominant, 4);
        let targets = vec![1, 2, 3];

        let path = viterbi_align(&log_probs, &targets, 0).unwrap();
        // All frames should map to valid target indices (0=C, 1=A, 2=T).
        assert_eq!(path.len(), 6);
        // First frames → C (target 0), middle → A (target 1), last → T (target 2).
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 0);
        assert_eq!(path[2], 1);
        assert_eq!(path[3], 1);
        assert_eq!(path[4], 2);
        assert_eq!(path[5], 2);
    }

    #[test]
    fn viterbi_with_blanks() {
        // Dominant: blank, C, blank, A, blank, T, blank
        let dominant = vec![0, 1, 0, 2, 0, 3, 0];
        let log_probs = make_log_probs(&dominant, 4);
        let targets = vec![1, 2, 3];

        let path = viterbi_align(&log_probs, &targets, 0).unwrap();
        assert_eq!(path.len(), 7);
        // Blank frames should map to nearest target.
        // Frame 0 (blank) → target 0 (C)
        // Frame 1 (C) → target 0
        // Frame 2 (blank) → target 0 or 1
        // Frame 3 (A) → target 1
        // Frame 4 (blank) → target 1 or 2
        // Frame 5 (T) → target 2
        // Frame 6 (blank) → target 2
        assert_eq!(path[1], 0); // C
        assert_eq!(path[3], 1); // A
        assert_eq!(path[5], 2); // T
    }

    #[test]
    fn viterbi_repeated_token() {
        // Target: [1, 1] ("CC") — must have blank separator.
        // Dominant: C, blank, C
        let dominant = vec![1, 0, 1];
        let log_probs = make_log_probs(&dominant, 4);
        let targets = vec![1, 1];

        let path = viterbi_align(&log_probs, &targets, 0).unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], 0); // first C
        // Frame 1 (blank) separates the two C's.
        assert_eq!(path[2], 1); // second C
    }

    #[test]
    fn viterbi_too_few_frames() {
        let log_probs = Array2::zeros((2, 4));
        let targets = vec![1, 2, 3]; // needs at least 3 frames
        let result = viterbi_align(&log_probs, &targets, 0);
        assert!(result.is_err());
    }

    #[test]
    fn tokenize_and_align_words() {
        // Test tokenization logic directly without needing a real ORT session.
        let labels = [
            "-".to_string(), // 0 = blank
            "|".to_string(), // 1 = word delimiter
            "A".to_string(), // 2
            "C".to_string(), // 3
            "T".to_string(), // 4
            "H".to_string(), // 5
            "E".to_string(), // 6
        ];

        // Build char → label_id map.
        let mut char_to_id: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();
        for (idx, label) in labels.iter().enumerate() {
            if label.len() == 1 {
                let ch = label.chars().next().unwrap();
                char_to_id.insert(ch, idx);
            }
        }
        let delim_id = Some(1usize); // "|"
        let num_labels = 7;

        let words = ["THE", "CAT"];
        let mut token_ids: Vec<usize> = Vec::new();
        let mut word_boundaries: Vec<(usize, usize)> = Vec::new();

        for (i, word) in words.iter().enumerate() {
            if i > 0
                && let Some(d) = delim_id
            {
                token_ids.push(d);
            }
            let start = token_ids.len();
            for ch in word.to_uppercase().chars() {
                if let Some(&id) = char_to_id.get(&ch)
                    && id < num_labels
                {
                    token_ids.push(id);
                }
            }
            let end = token_ids.len();
            word_boundaries.push((start, end));
        }

        // "THE" → [4, 5, 6], "|", "CAT" → [3, 2, 4]
        assert_eq!(token_ids, vec![4, 5, 6, 1, 3, 2, 4]);
        assert_eq!(word_boundaries, vec![(0, 3), (4, 7)]);
    }
}
