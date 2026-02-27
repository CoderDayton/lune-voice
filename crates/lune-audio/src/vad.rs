//! Voice Activity Detection powered by TEN VAD ONNX model.
//!
//! Replaces the previous energy-based VAD with a neural network model
//! that provides superior speech boundary detection, especially for
//! end-of-speech transitions (critical for LLM pipeline latency).
//!
//! The 306KB ONNX model is embedded in the binary — no external files needed.

use ten_vad_rs::TenVad;
use tracing::warn;

/// Embedded TEN VAD ONNX model (~306KB).
const TEN_VAD_MODEL: &[u8] = include_bytes!("../models/ten-vad.onnx");

/// TEN VAD hop size: 256 samples = 16ms at 16kHz.
const HOP_SIZE: usize = 256;

/// Result of a single VAD chunk evaluation.
#[derive(Debug, Clone, Copy)]
pub struct VadResult {
    /// Whether the chunk is classified as speech (with hysteresis).
    pub is_speech: bool,
    /// Raw max-frame VAD score [0.0–1.0].
    pub vad_score: f32,
    /// RMS energy (diagnostic).
    pub energy: f32,
}

/// VAD configuration
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Speech-START threshold (0.0–1.0). Default: 0.7
    ///
    /// Higher values reduce false speech triggers.
    pub threshold: f32,
    /// Speech-END threshold (0.0–1.0). Default: 0.3
    ///
    /// Lower than start threshold to prevent premature speech-end
    /// detection during pauses within sentences.
    pub threshold_end: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            threshold_end: 0.3,
        }
    }
}

/// Voice Activity Detector using the TEN VAD ONNX model.
pub struct Vad {
    inner: TenVad,
    threshold_start: f32,
    threshold_end: f32,
    /// Hysteresis state: true while in a speech region.
    in_speech: bool,
    /// Reusable buffer for f32 → i16 conversion (avoids per-call allocation).
    i16_buf: Vec<i16>,
}

impl Vad {
    /// Create a new TEN VAD instance with the embedded ONNX model.
    ///
    /// # Panics
    /// Panics if the embedded ONNX model fails to load (should never happen
    /// unless the binary is corrupt).
    pub fn new(config: VadConfig, _sample_rate: u32) -> Self {
        let inner = TenVad::new_from_bytes(TEN_VAD_MODEL, 16000)
            .expect("failed to initialize TEN VAD from embedded model");
        Self {
            inner,
            threshold_start: config.threshold,
            threshold_end: config.threshold_end,
            in_speech: false,
            i16_buf: Vec::with_capacity(HOP_SIZE * 20), // ~320ms at 16kHz
        }
    }

    /// Process an audio chunk and return a [`VadResult`] with hysteresis.
    ///
    /// Internally splits the chunk into 256-sample (16ms) frames and runs each
    /// through the TEN VAD ONNX model. Uses hysteresis: a higher threshold to
    /// enter speech and a lower threshold to exit, preventing rapid toggling.
    ///
    /// Energy is computed as RMS of the f32 samples (diagnostic value, not used
    /// for the speech decision).
    pub fn process_chunk(&mut self, samples: &[f32]) -> VadResult {
        let energy = compute_rms(samples);

        // Convert f32 [-1.0, 1.0] → i16 for TEN VAD.
        self.i16_buf.clear();
        self.i16_buf.extend(samples.iter().map(|&s| {
            (s * i16::MAX as f32)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        }));

        // Process complete HOP_SIZE frames through the model.
        let mut max_score: f32 = 0.0;
        let mut offset = 0;
        while offset + HOP_SIZE <= self.i16_buf.len() {
            match self
                .inner
                .process_frame(&self.i16_buf[offset..offset + HOP_SIZE])
            {
                Ok(score) => max_score = max_score.max(score),
                Err(e) => {
                    warn!(error = %e, "TEN VAD inference error");
                    break;
                }
            }
            offset += HOP_SIZE;
        }

        // Hysteresis: higher threshold to enter speech, lower to exit.
        let is_speech = if self.in_speech {
            max_score >= self.threshold_end
        } else {
            max_score >= self.threshold_start
        };
        self.in_speech = is_speech;

        VadResult {
            is_speech,
            vad_score: max_score,
            energy,
        }
    }
}

/// Compute RMS (Root Mean Square) energy of f32 PCM samples.
#[inline]
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
