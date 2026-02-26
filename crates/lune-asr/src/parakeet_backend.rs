//! Local ASR backend using NVIDIA Parakeet TDT via direct ONNX Runtime inference.
//!
//! Each audio batch is transcribed independently — no sliding window.
//! Accumulation and flush timing are handled at the session layer.

use std::path::PathBuf;
use std::sync::Mutex;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::error::{Error, Result};

use crate::mel;
use crate::tdt_decoder::TdtModel;

/// Maximum sliding window length (10 seconds at 16 kHz).  Older audio is
/// trimmed from the front so the model always sees the most recent context.
const MAX_WINDOW_SAMPLES: usize = 160_000;

/// Configuration for the Parakeet TDT local backend.
pub struct ParakeetTdtConfig {
    /// Directory containing ONNX model files and vocab.txt.
    pub model_dir: PathBuf,
    pub use_gpu: bool,
}

/// Local ASR backend using NVIDIA Parakeet TDT via ONNX Runtime.
pub struct ParakeetTdtBackend {
    model: Mutex<TdtModel>,
}

/// Per-session state.  Audio accumulates in a sliding window; each
/// `poll_results` transcribes the full window and returns the complete
/// transcript.  The session layer treats each result as a preview that
/// overwrites the previous one — only committing text on silence or
/// word-count threshold (sherpa-onnx "simulate streaming" pattern).
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

        tracing::info!(
            model_dir = %config.model_dir.display(),
            use_gpu = config.use_gpu,
            "initialized Parakeet TDT backend (direct ort)"
        );

        Ok(Self {
            model: Mutex::new(model),
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

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        // Snapshot the sliding window under a brief lock.
        let (pcm, timestamp_ms, id) = {
            let mut sess = session.lock().map_err(|e| lock_err("session lock", e))?;
            if !sess.has_new {
                return Ok(vec![]);
            }
            sess.has_new = false;
            let ts = sess.total_samples * 1000 / sess.sample_rate as u64;
            let id = sess.next_id;
            sess.next_id += 1;
            sess.samples_at_last_inference = sess.pending.len();
            (sess.pending.clone(), ts, id)
        };

        // Mel extraction and inference on the full sliding window.
        let features = mel::extract_features(&pcm);
        let result = {
            let mut model = self.model.lock().map_err(|e| lock_err("model lock", e))?;
            let start = std::time::Instant::now();
            let res = model.transcribe(features);
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

        // Return the full transcript of the current window.  The session
        // layer treats this as a preview that replaces the previous partial.
        Ok(vec![AsrSegment {
            id,
            text,
            is_final: false,
            start_ms: Some(timestamp_ms),
            end_ms: Some(timestamp_ms),
            confidence: Some(result.confidence),
            min_token_confidence: None,
        }])
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
}
