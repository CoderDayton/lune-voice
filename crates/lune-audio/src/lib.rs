// SPDX-License-Identifier: GPL-3.0-or-later

//! # lune-audio
//!
//! Microphone capture for the dictation pipeline: cpal input → mono downmix →
//! resample to 16 kHz → a bounded ring buffer the session drains.
//!
//! The ring never blocks the audio callback; when it fills, the oldest chunk is
//! evicted and counted ([`AudioPull::dropped_chunks`]).
//!
//! Also here: the PCEN adaptive AGC ([`pcen`]) the capture loop applies, and an
//! Earshot-backed [`Vad`] used by the offline inspection examples.
//!
//! ## Example
//! ```no_run
//! use lune_audio::{AudioStream, AudioConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let stream = AudioStream::new(AudioConfig::default()).unwrap();
//!
//!     while let Some(chunk) = stream.next_chunk().await {
//!         println!("Audio: {} samples", chunk.samples.len());
//!     }
//! }
//! ```

mod capture;
mod devices;
pub mod pcen;
#[cfg(target_os = "linux")]
mod pulse;
mod ring;

pub use capture::{AudioConfig, AudioStream, AudioStreamHandle, ConsumerGuard};
pub use devices::{
    capture_buffer_size, capture_host, list_input_devices, migrated_input_device_name,
    playback_host,
};
pub use pcen::{
    DEFAULT_NOISE_CEILING_RMS, EXPECTED_OUTPUT_RMS, PCEN_HOP_SIZE, PcenAgc, PcenAgcConfig,
    PcenFrameStats,
};

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
    /// Ask the producer to emit any buffered sub-chunk audio as a final
    /// short chunk. Default no-op for sources without producer-side
    /// buffering (file replay already delivers its tail).
    fn request_flush(&self) {}
    /// Chunks the ring buffer discarded because the consumer fell behind.
    ///
    /// Non-zero means audio was lost from the *front* of the buffer, so the
    /// transcript is missing its opening. The only other symptom is a
    /// transcript that quietly starts mid-sentence, which is close to
    /// undiagnosable without this number.
    fn dropped_chunks(&self) -> u64;
    /// Announce that this source is being read, until the guard is dropped.
    ///
    /// [`Self::dropped_chunks`] only counts — and the ring only warns — while
    /// a guard is held. A pre-warmed stream outlives every session, so without
    /// this its idle ring would report a permanent stall.
    fn consume(&self) -> ConsumerGuard;
}

impl AudioPull for AudioStream {
    fn try_next_chunk(&self) -> Option<AudioChunk> {
        AudioStream::try_next_chunk(self)
    }
    fn is_offline(&self) -> bool {
        AudioStream::is_offline(self)
    }
    fn request_flush(&self) {
        AudioStream::request_flush(self)
    }
    fn dropped_chunks(&self) -> u64 {
        AudioStream::dropped_chunks(self)
    }
    fn consume(&self) -> ConsumerGuard {
        AudioStream::consume(self)
    }
}

impl AudioPull for AudioStreamHandle {
    fn try_next_chunk(&self) -> Option<AudioChunk> {
        AudioStreamHandle::try_next_chunk(self)
    }
    fn request_flush(&self) {
        AudioStreamHandle::request_flush(self)
    }
    fn is_offline(&self) -> bool {
        false
    }
    fn dropped_chunks(&self) -> u64 {
        AudioStreamHandle::dropped_chunks(self)
    }
    fn consume(&self) -> ConsumerGuard {
        AudioStreamHandle::consume(self)
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
