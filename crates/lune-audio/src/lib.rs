//! # lune-audio
//!
//! Professional audio capture for streaming ASR.
//!
//! ## Features
//! - Adaptive resampling (any mic rate → 16kHz)
//! - Zero-copy ring buffer (lock-free SPSC)
//! - Backpressure handling (drops stale audio if consumer lags)
//! - Works on X11, Wayland, headless servers
//!
//! ## Example
//! ```no_run
//! use lune_audio::{AudioStream, AudioConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = AudioConfig::default();
//!     let mut stream = AudioStream::new(config).unwrap();
//!     
//!     while let Some(chunk) = stream.next_chunk().await {
//!         println!("Audio: {} samples", chunk.samples.len());
//!     }
//! }
//! ```

pub mod calibrate;
mod capture;
pub mod pcen;
mod ring;
mod vad;

pub use calibrate::{CalibrationError, CalibrationInput, CalibrationResult};
pub use capture::{AudioConfig, AudioStream};
pub use pcen::{
    DEFAULT_NOISE_CEILING_RMS, EXPECTED_OUTPUT_RMS, PcenAgc, PcenAgcConfig, PcenFrameStats,
};
pub use vad::{Vad, VadConfig, VadResult};

/// A chunk of resampled audio ready for downstream processing.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// PCM samples at target sample rate (default 16kHz), mono, f32 in [-1.0, 1.0]
    pub samples: Vec<f32>,

    /// Monotonic timestamp (milliseconds since stream start)
    pub timestamp_ms: u64,
}

/// Errors that can occur during audio capture
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("No audio input device found")]
    NoDevice,

    #[error("Failed to build audio stream: {0}")]
    StreamBuild(String),

    #[error("Audio stream error: {0}")]
    StreamError(String),

    #[error("Resampler error: {0}")]
    Resampler(String),

    #[error("VAD initialization error: {0}")]
    Vad(String),
}
