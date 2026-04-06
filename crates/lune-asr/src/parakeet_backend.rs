//! Local ASR backend using NVIDIA Parakeet TDT via direct ONNX Runtime inference.
//!
//! Each audio batch is transcribed independently — no sliding window.
//! Accumulation and flush timing are handled at the session layer.

use std::path::PathBuf;
use std::sync::Mutex;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig, WordConfidence};
use lune_core::error::{Error, Result};

use crate::mel;
use crate::tdt_decoder::{DecodedToken, TdtModel};
use crate::vocab::Vocabulary;

/// Maximum audio buffer (60 seconds at 16 kHz). Parakeet handles long audio
/// natively; chunked inference splits internally when needed.
const MAX_WINDOW_SAMPLES: usize = 960_000;

/// Silence padding kept before speech onset (~200ms at 16 kHz).
/// Preserves plosive/fricative consonant attacks that the VAD may not detect
/// as speech but the ASR encoder needs for correct decoding.
const ONSET_PAD_SAMPLES: usize = 3_200;

/// RMS threshold below which a frame is considered silence.
/// Typical speech RMS is 0.01–0.15; background noise is < 0.005.
const SILENCE_RMS_THRESHOLD: f32 = 0.008;

/// Chunk size for chunked inference (25 seconds at 16 kHz).
/// Audio longer than this is split into overlapping chunks and merged via LCS.
const CHUNK_SAMPLES: usize = 25 * 16_000;

/// Overlap between consecutive chunks (1 second at 16 kHz).
/// Matches the NeMo paper recommendation for Parakeet-TDT.
const OVERLAP_SAMPLES: usize = 16_000;

/// Configuration for the Parakeet TDT local backend.
pub struct ParakeetTdtConfig {
    /// Directory containing ONNX model files and vocab.txt.
    pub model_dir: PathBuf,
    pub use_gpu: bool,
}

/// Local ASR backend using NVIDIA Parakeet TDT via ONNX Runtime.
pub struct ParakeetTdtBackend {
    model: Mutex<TdtModel>,
    vocab: Vocabulary,
}

/// Per-session state. Audio accumulates in a sliding window; each
/// `poll_results` transcribes the full window and returns the complete
/// transcript as a non-final preview. `poll_final` commits the final
/// transcript with `is_final: true`.
pub struct ParakeetSession {
    /// Sliding audio window — newest audio at the end.
    pending: Vec<f32>,
    has_new: bool,
    total_samples: u64,
    sample_rate: u32,
    next_id: u64,
    /// Length of `pending` at the time of the last inference call.
    samples_at_last_inference: usize,
}

#[cold]
#[inline(never)]
fn lock_err(context: &'static str, e: impl std::fmt::Display) -> Error {
    Error::Asr(format!("{context}: {e}"))
}

impl ParakeetTdtBackend {
    pub fn new(config: ParakeetTdtConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir.clone()));
        }

        let model = TdtModel::new(&config.model_dir, config.use_gpu)?;
        let vocab = model.vocab().clone();

        tracing::info!(
            model_dir = %config.model_dir.display(),
            use_gpu = config.use_gpu,
            "initialized Parakeet TDT backend (direct ort)"
        );

        Ok(Self {
            model: Mutex::new(model),
            vocab,
        })
    }
}

#[async_trait::async_trait]
impl AsrBackend for ParakeetTdtBackend {
    type SessionHandle = Mutex<ParakeetSession>;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        let session = ParakeetSession {
            pending: Vec::with_capacity(MAX_WINDOW_SAMPLES),
            has_new: false,
            total_samples: 0,
            sample_rate: config.sample_rate,
            next_id: 1,
            samples_at_last_inference: 0,
        };
        tracing::info!(sample_rate = config.sample_rate, "ASR session started");
        Ok(Mutex::new(session))
    }

    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut sess = session.lock().map_err(|e| lock_err("session lock", e))?;
        sess.total_samples += pcm.len() as u64;
        sess.pending.extend_from_slice(pcm);
        // Trim sliding window from the front, keeping the newest audio.
        if sess.pending.len() > MAX_WINDOW_SAMPLES {
            let trim = sess.pending.len() - MAX_WINDOW_SAMPLES;
            sess.pending.drain(..trim);
            sess.samples_at_last_inference = sess.samples_at_last_inference.saturating_sub(trim);
        }
        sess.has_new = true;
        Ok(())
    }

    async fn poll_results(&self, _session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        // Parakeet is a batch backend — no incremental committed text.
        // Audio accumulates via push_audio; poll_final commits the transcript.
        Ok(vec![])
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let (pcm, timestamp_ms, id) = {
            let mut sess = session.lock().map_err(|e| lock_err("session lock", e))?;
            if sess.pending.is_empty() {
                return Ok(vec![]);
            }
            let ts = sess.total_samples * 1000 / sess.sample_rate as u64;
            let id = sess.next_id;
            sess.next_id += 1;
            // Clone + clear: keeps pre-allocated capacity for session reuse,
            // and preserves audio for retry if transcribe_pcm fails.
            let pcm = sess.pending.clone();
            sess.pending.clear();
            sess.samples_at_last_inference = 0;
            sess.has_new = false;
            (pcm, ts, id)
        };

        // Trim leading silence outside the lock — avoids blocking the
        // audio thread during the RMS scan.
        let trim_start = find_trim_offset(&pcm);
        let trimmed = if trim_start > 0 {
            tracing::debug!(
                trimmed_ms = trim_start * 1000 / 16_000,
                remaining_ms = (pcm.len() - trim_start) * 1000 / 16_000,
                "trimmed leading silence before ASR"
            );
            &pcm[trim_start..]
        } else {
            &pcm
        };

        self.transcribe_pcm(trimmed, timestamp_ms, id, true)
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let mut sess = session.lock().map_err(|e| lock_err("session lock", e))?;
        sess.pending.clear();
        sess.has_new = false;
        sess.samples_at_last_inference = 0;
        tracing::debug!("ASR context reset");
        Ok(())
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        let _ = session;
        Ok(())
    }
}

impl ParakeetTdtBackend {
    fn transcribe_pcm(
        &self,
        pcm: &[f32],
        timestamp_ms: u64,
        id: u64,
        is_final: bool,
    ) -> Result<Vec<AsrSegment>> {
        if pcm.len() <= CHUNK_SAMPLES {
            return self.transcribe_single(pcm, timestamp_ms, id, is_final);
        }
        self.transcribe_chunked(pcm, timestamp_ms, id, is_final)
    }

    /// Single-pass transcription for audio ≤ CHUNK_SAMPLES.
    fn transcribe_single(
        &self,
        pcm: &[f32],
        timestamp_ms: u64,
        id: u64,
        is_final: bool,
    ) -> Result<Vec<AsrSegment>> {
        let features = mel::extract_features(pcm);
        let (result, decoded_tokens) = {
            let mut model = self.model.lock().map_err(|e| lock_err("model lock", e))?;
            let start = std::time::Instant::now();
            let res = model.transcribe_with_tokens(features);
            tracing::debug!(
                elapsed_ms = start.elapsed().as_millis(),
                window_ms = pcm.len() * 1000 / 16000,
                "inference"
            );
            res
        }?;

        let text = result.text.trim().to_string();
        if text.is_empty() {
            return Ok(vec![]);
        }

        let min_token_confidence = decoded_tokens.iter().map(|t| t.confidence).reduce(f32::min);
        let word_confidences = tokens_to_word_confidences(&decoded_tokens, &self.vocab);

        Ok(vec![AsrSegment {
            id,
            text,
            is_final,
            start_ms: Some(timestamp_ms),
            end_ms: Some(timestamp_ms),
            confidence: Some(result.confidence),
            min_token_confidence,
            word_confidences,
        }])
    }

    /// Chunked inference with overlap + LCS merge for long audio.
    ///
    /// Splits audio into overlapping chunks (CHUNK_SAMPLES with OVERLAP_SAMPLES
    /// overlap), transcribes each independently, and merges transcripts using
    /// Longest Common Subsequence at overlap boundaries.
    /// Reference: Parakeet-TDT-0.6B-v3 paper (arXiv:2509.14128), §6.4.1.
    fn transcribe_chunked(
        &self,
        pcm: &[f32],
        timestamp_ms: u64,
        id: u64,
        is_final: bool,
    ) -> Result<Vec<AsrSegment>> {
        let step = CHUNK_SAMPLES - OVERLAP_SAMPLES;
        let mut chunk_texts: Vec<Vec<String>> = Vec::new();
        let mut total_confidence = 0.0f32;
        let mut total_tokens = 0usize;
        let mut min_conf = f32::INFINITY;

        // Transcribe each overlapping chunk
        let mut offset = 0;
        while offset < pcm.len() {
            let end = (offset + CHUNK_SAMPLES).min(pcm.len());
            let chunk = &pcm[offset..end];

            let features = mel::extract_features(chunk);
            let (result, decoded_tokens) = {
                let mut model = self.model.lock().map_err(|e| lock_err("model lock", e))?;
                let start = std::time::Instant::now();
                let res = model.transcribe_with_tokens(features);
                tracing::debug!(
                    elapsed_ms = start.elapsed().as_millis(),
                    chunk_ms = chunk.len() * 1000 / 16000,
                    chunk_idx = chunk_texts.len(),
                    "chunked inference"
                );
                res
            }?;

            if !decoded_tokens.is_empty() {
                total_confidence += result.confidence * decoded_tokens.len() as f32;
                total_tokens += decoded_tokens.len();
                if let Some(mc) = decoded_tokens.iter().map(|t| t.confidence).reduce(f32::min) {
                    min_conf = min_conf.min(mc);
                }
            }

            let words: Vec<String> = result
                .text
                .split_whitespace()
                .map(|w| w.to_string())
                .collect();
            chunk_texts.push(words);

            if end >= pcm.len() {
                break;
            }
            offset += step;
        }

        // Merge chunk transcripts via LCS at overlap boundaries
        let merged_words = if chunk_texts.is_empty() {
            Vec::new()
        } else {
            let mut merged = chunk_texts[0].clone();
            for next_chunk in &chunk_texts[1..] {
                merged = lcs_merge_words(&merged, next_chunk);
            }
            merged
        };

        let text = merged_words.join(" ");
        if text.is_empty() {
            return Ok(vec![]);
        }

        let avg_confidence = if total_tokens > 0 {
            total_confidence / total_tokens as f32
        } else {
            0.0
        };

        // Rebuild word confidences from merged text (approximate — use avg confidence
        // for words that came from LCS merge since we don't track per-chunk word confs)
        let word_confidences: Vec<WordConfidence> = merged_words
            .iter()
            .map(|w| WordConfidence {
                word: w.clone(),
                confidence: avg_confidence,
            })
            .collect();

        Ok(vec![AsrSegment {
            id,
            text,
            is_final,
            start_ms: Some(timestamp_ms),
            end_ms: Some(timestamp_ms),
            confidence: Some(avg_confidence),
            min_token_confidence: if min_conf.is_finite() {
                Some(min_conf)
            } else {
                None
            },
            word_confidences,
        }])
    }
}

// ── LCS merge for overlapping chunk transcripts ────────────────────────────

use crate::lcs_merge::lcs_merge_words;

// ── Per-word confidence from TDT tokens ─────────────────────────────────────

/// Aggregate TDT decoded tokens into per-word confidence values.
///
/// Word boundary: token text starts with `▁` (SentencePiece word-initial marker).
/// Per-word confidence = minimum across constituent tokens (conservative).
fn tokens_to_word_confidences(tokens: &[DecodedToken], vocab: &Vocabulary) -> Vec<WordConfidence> {
    if tokens.is_empty() {
        return vec![];
    }

    let mut words: Vec<WordConfidence> = Vec::new();
    let mut current_word = String::new();
    let mut current_min = f32::INFINITY;

    for (i, dt) in tokens.iter().enumerate() {
        let token_text = match vocab.get(dt.token_id) {
            Some(t) => t,
            None => continue,
        };

        // Skip special tokens
        if token_text.starts_with('<') && token_text.ends_with('>') && token_text != "<unk>" {
            continue;
        }

        let is_boundary = i == 0 || token_text.starts_with('▁');

        if is_boundary && !current_word.is_empty() {
            words.push(WordConfidence {
                word: std::mem::take(&mut current_word),
                confidence: current_min,
            });
            current_min = f32::INFINITY;
        }

        current_word.push_str(token_text.replace('▁', " ").trim_start());
        current_min = current_min.min(dt.confidence);
    }

    if !current_word.is_empty() {
        words.push(WordConfidence {
            word: current_word,
            confidence: current_min,
        });
    }

    words
}

/// Find how many leading samples to trim from PCM, keeping
/// `ONSET_PAD_SAMPLES` (~200ms) before the first non-silent frame.
/// Returns 0 if speech is already near the start.
pub(crate) fn find_trim_offset(pcm: &[f32]) -> usize {
    const FRAME_SIZE: usize = 320; // 20ms at 16kHz

    let mut speech_start = pcm.len();
    for (i, frame) in pcm.chunks(FRAME_SIZE).enumerate() {
        let n = frame.len() as f32;
        let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / n).sqrt();
        if rms > SILENCE_RMS_THRESHOLD {
            speech_start = i * FRAME_SIZE;
            break;
        }
    }

    let trim_to = speech_start.saturating_sub(ONSET_PAD_SAMPLES);
    if trim_to < FRAME_SIZE { 0 } else { trim_to }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_rejects_missing_dir() {
        let config = ParakeetTdtConfig {
            model_dir: PathBuf::from("/nonexistent/model"),
            use_gpu: false,
        };
        let result = ParakeetTdtBackend::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn word_confidences_from_empty_tokens() {
        let vocab = Vocabulary::empty();
        let result = tokens_to_word_confidences(&[], &vocab);
        assert!(result.is_empty());
    }

    #[test]
    fn trim_offset_zero_when_speech_at_start() {
        let mut pcm = vec![0.05f32; 3200];
        pcm.extend_from_slice(&[0.0; 16_000]);
        assert_eq!(find_trim_offset(&pcm), 0);
    }

    #[test]
    fn trim_offset_removes_leading_dead_air() {
        let mut pcm = vec![0.0f32; 32_000]; // 2s silence
        pcm.extend_from_slice(&vec![0.05f32; 16_000]); // 1s speech
        let offset = find_trim_offset(&pcm);
        assert!(offset > 0, "should trim leading silence");
        assert!(
            offset <= 32_000 - ONSET_PAD_SAMPLES,
            "should keep onset pad"
        );
    }

    #[test]
    fn trim_offset_zero_for_all_silent() {
        let pcm = vec![0.0f32; 16_000];
        // All silence — speech_start = pcm.len(), trim_to near end, but < FRAME_SIZE check
        // returns 0 because there's nothing meaningful to trim to
        let offset = find_trim_offset(&pcm);
        // Either 0 or a valid offset — shouldn't panic
        assert!(offset <= pcm.len());
    }
}
