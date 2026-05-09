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
pub use capture::{AudioConfig, AudioStream, AudioStreamHandle};
pub use pcen::{
    DEFAULT_NOISE_CEILING_RMS, EXPECTED_OUTPUT_RMS, PcenAgc, PcenAgcConfig, PcenFrameStats,
};
pub use vad::{Vad, VadConfig, VadResult};

/// Polling source for downstream consumers (sessions, tests).
///
/// Implemented by both [`AudioStream`] (owns the cpal capture) and
/// [`AudioStreamHandle`] (shares the ring buffer of an outer stream).
/// Lets consumers be agnostic about which kind of source they hold.
pub trait AudioPull: Send + Sync {
    /// Non-blocking pop of the next available chunk.
    fn try_next_chunk(&self) -> Option<AudioChunk>;
    /// True when this source can never produce more audio (file replay
    /// drained, no live capture). Live mics and shared handles return false.
    fn is_offline(&self) -> bool;
}

impl AudioPull for AudioStream {
    fn try_next_chunk(&self) -> Option<AudioChunk> {
        AudioStream::try_next_chunk(self)
    }
    fn is_offline(&self) -> bool {
        AudioStream::is_offline(self)
    }
}

impl AudioPull for AudioStreamHandle {
    fn try_next_chunk(&self) -> Option<AudioChunk> {
        AudioStreamHandle::try_next_chunk(self)
    }
    fn is_offline(&self) -> bool {
        false
    }
}

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
