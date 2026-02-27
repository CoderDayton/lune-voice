//! ORT-based NAR forced aligner for post-utterance word-order correction.
//!
//! `OrtForcedAligner` shares the ASR encoder and embed_tokens sessions (via
//! `Arc<Mutex<Session>>`), loading only its own FA decoder separately.
//!
//! `align(pcm, text)` runs a single forward pass through the FA decoder —
//! no autoregressive loop — to obtain per-word timestamps, then returns the
//! words sorted by start_ms. The caller compares word order to detect drift.

use std::sync::{Arc, Mutex};

use ndarray::{Array2, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::{DynValue, Value};
use serde::{Deserialize, Serialize};

use lune_core::error::{Error, Result};

use crate::ort_qwen_stream::{HIDDEN_DIM, TOKEN_AUDIO_END, TOKEN_AUDIO_START, compute_qwen_mel};

// ── Config ────────────────────────────────────────────────────────────────────

/// Serde-deserialisable counterpart to `fa_config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaConfig {
    /// Special token ID appended after each word's tokens to mark a timestamp
    /// prediction slot (two per word: start and end).
    pub timestamp_token_id: i64,
    /// Duration in milliseconds of each timestamp class bucket.
    /// Typically 80 ms (one 1280-sample chunk at 16 kHz).
    pub timestamp_segment_ms: u32,
    /// Number of timestamp classes in the FA head output (e.g. 3750 for 300 s).
    pub timestamp_classes: usize,
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-word timestamp output from `OrtForcedAligner::align`.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub text: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

// ── Aligner struct ────────────────────────────────────────────────────────────

pub struct OrtForcedAligner {
    /// Shared ASR encoder (mel → audio features).
    encoder: Arc<Mutex<Session>>,
    /// Shared ASR embed_tokens (token IDs → embeddings).
    embed_tokens: Arc<Mutex<Session>>,
    /// FA-specific NAR decoder (no KV cache, timestamp head).
    decoder: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    fa_config: FaConfig,
}

impl OrtForcedAligner {
    pub fn new(
        encoder: Arc<Mutex<Session>>,
        embed_tokens: Arc<Mutex<Session>>,
        decoder: Session,
        tokenizer: tokenizers::Tokenizer,
        fa_config: FaConfig,
    ) -> Result<Self> {
        Ok(Self {
            encoder,
            embed_tokens,
            decoder: Mutex::new(decoder),
            tokenizer,
            fa_config,
        })
    }

    /// Align `text` against `pcm` (16 kHz mono f32 PCM).
    ///
    /// Returns words sorted by `start_ms`. If the returned order differs from
    /// the original word order in `text`, the caller should emit a correction.
    ///
    /// PCM is capped at `MAX_ALIGN_SAMPLES` (8 s) before mel extraction.
    /// The encoder was fine-tuned on ≤ 8 s windows; longer inputs produce
    /// garbage timestamps in the second half.
    pub fn align(&self, pcm: &[f32], text: &str) -> Result<Vec<WordTimestamp>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Cap to the encoder's training window — same constraint as the ASR decoder.
        const MAX_ALIGN_SAMPLES: usize = 8 * 16_000; // 8 s @ 16 kHz
        let pcm = if pcm.len() > MAX_ALIGN_SAMPLES {
            &pcm[..MAX_ALIGN_SAMPLES]
        } else {
            pcm
        };

        // ── 1. Mel → encoder → audio features [1, enc_len, HIDDEN_DIM] ──────
        let mel_f32 = compute_qwen_mel(pcm);
        let mel_f16: ndarray::Array3<half::f16> = mel_f32.mapv(half::f16::from_f32);

        let mel_dyn: DynValue = Value::from_array(mel_f16.into_dyn())
            .map_err(|e| Error::Asr(format!("fa mel value: {e}")))?
            .into();

        let mut enc_guard = self
            .encoder
            .lock()
            .map_err(|e| Error::Asr(format!("fa encoder lock: {e}")))?;
        let enc_out = enc_guard
            .run(vec![("mel".to_string(), mel_dyn)])
            .map_err(|e| Error::Asr(format!("fa encoder run: {e}")))?;

        let audio_features = enc_out[0]
            .try_extract_array::<half::f16>()
            .map_err(|e| Error::Asr(format!("fa encoder output: {e}")))?;
        let audio_features = audio_features.view();
        // Shape: [1, enc_len, HIDDEN_DIM]
        let enc_len = audio_features.shape()[1];

        // ── 2. Build per-word embeddings + track TS slot positions ────────────
        let ts_id = self.fa_config.timestamp_token_id;

        // embed [TOKEN_AUDIO_START] and [TOKEN_AUDIO_END] separately.
        let start_emb = self.embed_ids(&[TOKEN_AUDIO_START])?; // [1, 1, HIDDEN_DIM]
        let end_emb = self.embed_ids(&[TOKEN_AUDIO_END])?; // [1, 1, HIDDEN_DIM]

        // Compute total sequence length and record TS slot positions.
        // Sequence: [AUDIO_START | audio_features | AUDIO_END | word0_embs... | word1_embs... ]
        let prefix_len = 1 + enc_len + 1; // AUDIO_START + features + AUDIO_END
        let mut ts_slot_positions: Vec<(usize, usize)> = Vec::with_capacity(words.len());
        let mut word_embeds: Vec<ArrayD<half::f16>> = Vec::with_capacity(words.len());
        let mut cursor = prefix_len;

        for word in &words {
            let mut ids: Vec<i64> = self
                .tokenizer
                .encode(word.to_string(), false)
                .map_err(|e| Error::Asr(format!("fa tokenize: {e}")))?
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect();
            // Append two TS tokens (start slot, end slot).
            ids.push(ts_id);
            ids.push(ts_id);
            let n = ids.len();
            // TS positions are the last two token positions.
            let start_slot = cursor + n - 2;
            let end_slot = cursor + n - 1;
            ts_slot_positions.push((start_slot, end_slot));
            cursor += n;

            let emb = self.embed_ids(&ids)?;
            word_embeds.push(emb);
        }

        let total_len = cursor;

        // ── 3. Assemble inputs_embeds [1, total_len, HIDDEN_DIM] ─────────────
        let mut combined: Vec<half::f16> = Vec::with_capacity(total_len * HIDDEN_DIM);

        // AUDIO_START embedding.
        let start_data = start_emb.as_slice().unwrap_or(&[]);
        combined.extend_from_slice(&start_data[..HIDDEN_DIM]);

        // Audio features [enc_len, HIDDEN_DIM] — already f16.
        let enc_data = audio_features.as_slice().unwrap();
        combined.extend_from_slice(&enc_data[..enc_len * HIDDEN_DIM]);

        // AUDIO_END embedding.
        let end_data = end_emb.as_slice().unwrap_or(&[]);
        combined.extend_from_slice(&end_data[..HIDDEN_DIM]);

        // Per-word embeddings (each includes word tokens + 2 TS tokens).
        for emb in &word_embeds {
            let emb_data = emb.as_slice().unwrap_or(&[]);
            combined.extend_from_slice(emb_data);
        }

        let inputs_embeds = ArrayD::from_shape_vec(IxDyn(&[1, total_len, HIDDEN_DIM]), combined)
            .map_err(|e| Error::Asr(format!("fa embeds shape: {e}")))?;

        // ── 4. position_ids [1, total_len] i64 (standard 2D, not mrope) ──────
        let pos_data: Vec<i64> = (0..total_len as i64).collect();
        let position_ids = Array2::from_shape_vec((1, total_len), pos_data)
            .map_err(|e| Error::Asr(format!("fa pos_ids shape: {e}")))?;

        // ── 5. FA decoder forward pass ────────────────────────────────────────
        let emb_val: DynValue = Value::from_array(inputs_embeds)
            .map_err(|e| Error::Asr(format!("fa emb value: {e}")))?
            .into();
        let pos_val: DynValue = Value::from_array(position_ids)
            .map_err(|e| Error::Asr(format!("fa pos value: {e}")))?
            .into();

        let mut dec_guard = self
            .decoder
            .lock()
            .map_err(|e| Error::Asr(format!("fa decoder lock: {e}")))?;
        let dec_out = dec_guard
            .run(vec![
                ("inputs_embeds".to_string(), emb_val),
                ("position_ids".to_string(), pos_val),
            ])
            .map_err(|e| Error::Asr(format!("fa decoder run: {e}")))?;

        // logits: [1, total_len, timestamp_classes]
        let logits = dec_out[0]
            .try_extract_array::<half::f16>()
            .map_err(|e| Error::Asr(format!("fa logits output: {e}")))?;
        let logits = logits.view();
        let n_classes = self.fa_config.timestamp_classes;

        // ── 6. Argmax at each TS slot → raw timestamp index ───────────────────
        let mut raw_starts: Vec<u32> = Vec::with_capacity(words.len());
        let mut raw_ends: Vec<u32> = Vec::with_capacity(words.len());

        for &(start_slot, end_slot) in &ts_slot_positions {
            raw_starts.push(argmax_row(&logits, start_slot, n_classes));
            raw_ends.push(argmax_row(&logits, end_slot, n_classes));
        }

        // ── 7. LIS correction → monotone start indices ────────────────────────
        // We require start timestamps to be non-decreasing to detect word drift.
        let lis_idxs = lis_indices(&raw_starts);
        let lis_set: std::collections::HashSet<usize> = lis_idxs.iter().copied().collect();

        // ── 8. Build WordTimestamp entries, sorted by start_ms ────────────────
        let seg_ms = self.fa_config.timestamp_segment_ms;
        let mut result: Vec<WordTimestamp> = words
            .iter()
            .enumerate()
            .map(|(i, &word)| {
                let start_ms = if lis_set.contains(&i) {
                    raw_starts[i] * seg_ms
                } else {
                    // Word not in LIS — use carry-forward from last valid.
                    lis_idxs
                        .iter()
                        .rev()
                        .find(|&&j| j < i)
                        .map(|&j| raw_starts[j] * seg_ms)
                        .unwrap_or(raw_starts[i] * seg_ms)
                };
                let end_ms = raw_ends[i] * seg_ms;
                WordTimestamp {
                    text: word.to_string(),
                    start_ms,
                    end_ms,
                }
            })
            .collect();

        result.sort_by_key(|w| w.start_ms);
        Ok(result)
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Embed a slice of token IDs. Returns [1, n, HIDDEN_DIM] f16 tensor.
    fn embed_ids(&self, ids: &[i64]) -> Result<ArrayD<half::f16>> {
        let n = ids.len();
        let ids_arr = Array2::from_shape_vec((1, n), ids.to_vec())
            .map_err(|e| Error::Asr(format!("fa embed ids shape: {e}")))?;
        let ids_val: DynValue = Value::from_array(ids_arr)
            .map_err(|e| Error::Asr(format!("fa embed ids value: {e}")))?
            .into();

        let mut emb_guard = self
            .embed_tokens
            .lock()
            .map_err(|e| Error::Asr(format!("fa embed_tokens lock: {e}")))?;
        let out = emb_guard
            .run(vec![("input_ids".to_string(), ids_val)])
            .map_err(|e| Error::Asr(format!("fa embed_tokens run: {e}")))?;

        out[0]
            .try_extract_array::<half::f16>()
            .map(|t| t.view().to_owned())
            .map_err(|e| Error::Asr(format!("fa embed output: {e}")))
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Argmax over the timestamp-classes dimension at a specific sequence position.
/// `logits` is a view of shape [1, seq_len, n_classes].
fn argmax_row(logits: &ndarray::ArrayViewD<half::f16>, seq_pos: usize, n_classes: usize) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for cls in 0..n_classes {
        // Access element at [0, seq_pos, cls].
        let v = logits[[0, seq_pos, cls]].to_f32();
        if v > best_val {
            best_val = v;
            best_idx = cls;
        }
    }
    best_idx as u32
}

/// O(n log n) patience-sort LIS (Longest Increasing Subsequence).
/// Returns the indices (into `seq`) of a maximal non-decreasing subsequence.
fn lis_indices(seq: &[u32]) -> Vec<usize> {
    if seq.is_empty() {
        return vec![];
    }
    let n = seq.len();
    // `tails[i]` = smallest tail value of all LIS candidates of length i+1.
    let mut tails: Vec<u32> = Vec::new();
    // `tail_src[i]` = which index in `seq` is currently at tails[i].
    let mut tail_src: Vec<usize> = Vec::new();
    // `parent[i]` = predecessor index for seq[i] in the LIS ending at i.
    let mut parent: Vec<Option<usize>> = vec![None; n];

    for (i, &val) in seq.iter().enumerate() {
        // Binary search for the first position where tails[pos] >= val.
        let pos = tails.partition_point(|&t| t < val);
        if pos == tails.len() {
            tails.push(val);
            tail_src.push(i);
        } else {
            tails[pos] = val;
            tail_src[pos] = i;
        }
        parent[i] = if pos > 0 {
            Some(tail_src[pos - 1])
        } else {
            None
        };
    }

    // Reconstruct the LIS by following parent pointers from the last tail.
    let mut result = Vec::with_capacity(tails.len());
    let mut cur = Some(tail_src[tails.len() - 1]);
    while let Some(idx) = cur {
        result.push(idx);
        cur = parent[idx];
    }
    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lis_sorted() {
        assert_eq!(lis_indices(&[1, 2, 3, 4]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn lis_reversed() {
        // LIS of a reversed sequence has length 1 — any single element.
        let idx = lis_indices(&[4, 3, 2, 1]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn lis_with_inversion() {
        // [0, 5, 1, 2, 3] → LIS = [0, 1, 2, 3] (indices 0, 2, 3, 4)
        let idx = lis_indices(&[0, 5, 1, 2, 3]);
        assert_eq!(idx.len(), 4);
        // Ensure returned indices are in seq order and values are non-decreasing.
        let vals: Vec<u32> = idx.iter().map(|&i| [0u32, 5, 1, 2, 3][i]).collect();
        for w in vals.windows(2) {
            assert!(w[0] <= w[1], "LIS not non-decreasing: {w:?}");
        }
    }

    #[test]
    fn lis_empty() {
        assert!(lis_indices(&[]).is_empty());
    }
}
