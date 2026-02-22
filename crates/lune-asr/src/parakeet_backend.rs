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
    max_context_samples: usize,
    sample_rate: u32,
    total_samples: u64,
    samples_since_inference: usize,
    tracker: SegmentTracker,
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
        let session = ParakeetSession {
            audio_buffer: Vec::with_capacity(max_context_samples),
            max_context_samples,
            sample_rate,
            total_samples: 0,
            samples_since_inference: 0,
            tracker: SegmentTracker::new(500),
        };
        tracing::info!(sample_rate, "ASR session started");
        Ok(Mutex::new(session))
    }

    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut session = session
            .lock()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;

        session.audio_buffer.extend_from_slice(pcm);
        session.total_samples += pcm.len() as u64;
        session.samples_since_inference += pcm.len();

        // Sliding window: drop oldest samples beyond context window.
        if session.audio_buffer.len() > session.max_context_samples {
            let excess = session.audio_buffer.len() - session.max_context_samples;
            session.audio_buffer.drain(..excess);
        }

        Ok(())
    }

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let mut session = session
            .lock()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;

        // Run inference after 300ms of new audio (4800 samples @ 16kHz).
        let threshold = (session.sample_rate as usize * 3) / 10;
        if session.samples_since_inference < threshold {
            session.tracker.check_silence_finalization();
            return Ok(session.tracker.drain());
        }

        session.samples_since_inference = 0;

        // Avoid cloning the full buffer: borrow for mel extraction.
        let features = mel::extract_features(&session.audio_buffer);
        let timestamp_ms = session.total_samples * 1000 / session.sample_rate as u64;

        // Run inference (release session lock, acquire model lock)
        let result = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| Error::Asr(format!("model lock: {e}")))?;

            let start = std::time::Instant::now();
            let res = model.transcribe(features);
            let elapsed = start.elapsed();
            tracing::debug!(elapsed_ms = elapsed.as_millis(), "inference");
            res
        };

        match result {
            Ok(transcription) => {
                let text = transcription.text.trim();
                if !text.is_empty() {
                    session.tracker.push_partial(
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

        session.tracker.check_silence_finalization();
        if session.tracker.is_empty() {
            return Ok(vec![]);
        }
        Ok(session.tracker.drain())
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let mut session = session
            .lock()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
        session.audio_buffer.clear();
        session.samples_since_inference = 0;
        session.tracker.drain();
        tracing::debug!("ASR context reset");
        Ok(())
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        let mut session = session
            .into_inner()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
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
