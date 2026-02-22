//! Local ASR backend using NVIDIA Parakeet TDT via direct ONNX Runtime inference.
//!
//! Replaces `parakeet-rs` wrapper to expose token-level confidence scores
//! needed for confidence-based routing.

use std::path::PathBuf;
use std::sync::Mutex;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::error::{Error, Result};

use crate::mel;
use crate::segment::SegmentTracker;
use crate::tdt_decoder::TdtModel;

/// Configuration for the Parakeet TDT local backend.
pub struct ParakeetTdtConfig {
    /// Directory containing ONNX model files and vocab.txt.
    pub model_dir: PathBuf,
    pub use_gpu: bool,
    /// Sliding context window size in seconds.
    pub context_secs: f32,
}

/// Local ASR backend using NVIDIA Parakeet TDT via ONNX Runtime.
///
/// `TdtModel` handles encoder, TDT decoder, and tokenizer directly.
/// We accumulate PCM in a sliding window and call `transcribe` on poll.
pub struct ParakeetTdtBackend {
    model: Mutex<TdtModel>,
    config: ParakeetTdtConfig,
}

/// Per-session state held behind a Mutex.
pub struct ParakeetSession {
    audio_buffer: Vec<f32>,
    /// Pre-allocated scratch for snapshot copies — reused across inference cycles
    /// to eliminate per-cycle heap allocation (~30ms @ 16kHz = ~480 KB/cycle).
    /// Owned by session; lent to poll_results via mem::take, returned in phase 4.
    audio_snapshot: Vec<f32>,
    max_context_samples: usize,
    sample_rate: u32,
    total_samples: u64,
    samples_since_inference: usize,
    /// Pre-computed: (sample_rate * 3) / 10 — avoids per-poll division.
    silence_threshold: usize,
    tracker: SegmentTracker,
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

        tracing::info!(
            model_dir = %config.model_dir.display(),
            use_gpu = config.use_gpu,
            "initialized Parakeet TDT backend (direct ort)"
        );

        Ok(Self {
            model: Mutex::new(model),
            config,
        })
    }
}

#[async_trait::async_trait]
impl AsrBackend for ParakeetTdtBackend {
    type SessionHandle = Mutex<ParakeetSession>;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        let sample_rate = config.sample_rate;
        let max_context_samples = (self.config.context_secs * sample_rate as f32) as usize;
        // 300ms threshold pre-computed once; avoids division on every poll_results call.
        let silence_threshold = (sample_rate as usize * 3) / 10;
        let session = ParakeetSession {
            audio_buffer: Vec::with_capacity(max_context_samples),
            // Same capacity as audio_buffer; amortizes alloc cost across all inference cycles.
            audio_snapshot: Vec::with_capacity(max_context_samples),
            max_context_samples,
            sample_rate,
            total_samples: 0,
            samples_since_inference: 0,
            silence_threshold,
            tracker: SegmentTracker::new(500),
        };
        tracing::info!(sample_rate, "ASR session started");
        Ok(Mutex::new(session))
    }

    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut session = session.lock().map_err(|e| lock_err("session lock", e))?;

        let pcm_len = pcm.len();
        let max = session.max_context_samples;

        if pcm_len >= max {
            // Fast path: large chunk replaces entire context window.
            session.audio_buffer.clear();
            session
                .audio_buffer
                .extend_from_slice(&pcm[pcm_len - max..]);
        } else {
            let buf_len = session.audio_buffer.len();
            let new_total = buf_len + pcm_len;
            if new_total > max {
                // Shift-then-append: discard oldest samples before extending.
                // Saves `pcm_len` element-copies vs. the extend-then-shift pattern
                // (extend-then-shift copies the new pcm_len elements twice).
                let discard = new_total - max;
                session.audio_buffer.copy_within(discard..buf_len, 0);
                session.audio_buffer.truncate(buf_len - discard);
            }
            session.audio_buffer.extend_from_slice(pcm);
        }

        session.total_samples += pcm_len as u64;
        session.samples_since_inference += pcm_len;

        Ok(())
    }

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        // Phase 1: brief lock — check readiness and take pre-allocated snapshot buffer.
        // Releasing the lock before mel extraction and inference allows push_audio
        // to make progress concurrently (critical for real-time streaming).
        let (mut audio_snapshot, timestamp_ms) = {
            let mut sess = session.lock().map_err(|e| lock_err("session lock", e))?;

            if sess.samples_since_inference < sess.silence_threshold {
                sess.tracker.check_silence_finalization();
                return Ok(sess.tracker.drain());
            }

            sess.samples_since_inference = 0;
            let ts = sess.total_samples * 1000 / sess.sample_rate as u64;

            // Take the pre-allocated snapshot buffer out of session (zero-allocation:
            // mem::take swaps in an empty Vec without touching the heap).
            // We refill it and return it in phase 4 for reuse next cycle.
            let mut snapshot = std::mem::take(&mut sess.audio_snapshot);
            snapshot.clear();
            snapshot.extend_from_slice(&sess.audio_buffer);
            (snapshot, ts)
        }; // session lock released — push_audio unblocked

        // Phase 2: mel extraction outside any lock (CPU-intensive STFT).
        let features = mel::extract_features(&audio_snapshot);
        // Keep audio_snapshot alive — its allocation is returned to the session in phase 4.

        // Phase 3: model inference under model lock only.
        let result = {
            let mut model = self.model.lock().map_err(|e| lock_err("model lock", e))?;

            let start = std::time::Instant::now();
            let res = model.transcribe(features);
            tracing::debug!(elapsed_ms = start.elapsed().as_millis(), "inference");
            res
        };

        // Phase 4: re-acquire session lock to update tracker and reclaim snapshot buffer.
        let mut sess = session.lock().map_err(|e| lock_err("session lock", e))?;

        // Return the snapshot allocation so next cycle reuses it (zero-alloc steady state).
        audio_snapshot.clear();
        sess.audio_snapshot = audio_snapshot;

        match result {
            Ok(transcription) => {
                let text = transcription.text.trim();
                if !text.is_empty() {
                    sess.tracker.push_partial(
                        text.to_string(),
                        Some(timestamp_ms),
                        Some(transcription.confidence),
                    );
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "TDT inference failed");
            }
        }

        sess.tracker.check_silence_finalization();
        // Avoid empty-Vec allocation on the common silence path.
        if sess.tracker.is_empty() {
            return Ok(vec![]);
        }
        Ok(sess.tracker.drain())
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let mut session = session.lock().map_err(|e| lock_err("session lock", e))?;
        session.audio_buffer.clear();
        session.samples_since_inference = 0;
        session.tracker.drain();
        tracing::debug!("ASR context reset");
        Ok(())
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        let mut session = session
            .into_inner()
            .map_err(|e| lock_err("session into_inner", e))?;
        session.tracker.finalize_all();
        tracing::info!(
            pending = session.tracker.len(),
            total_samples = session.total_samples,
            "ASR session ending"
        );
        session.tracker.drain();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_rejects_missing_dir() {
        let config = ParakeetTdtConfig {
            model_dir: PathBuf::from("/nonexistent/model"),
            use_gpu: false,
            context_secs: 30.0,
        };
        let result = ParakeetTdtBackend::new(config);
        assert!(result.is_err());
    }
}
