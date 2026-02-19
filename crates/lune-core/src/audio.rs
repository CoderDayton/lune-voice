use serde::{Deserialize, Serialize};

/// A chunk of audio data with metadata.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Mono PCM f32 samples.
    pub samples: Vec<f32>,
    /// Timestamp in milliseconds from session start.
    pub timestamp_ms: u64,
    /// Sample rate of the audio data.
    pub sample_rate: u32,
}

/// Configuration for audio capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCaptureConfig {
    /// Samples per chunk, aligned to ASR encoder expectations.
    pub frame_size: usize,
    /// Hop size between frames.
    pub hop_size: usize,
    /// Target sample rate.
    pub sample_rate: u32,
    /// Enable voice activity detection.
    pub vad_enabled: bool,
    /// VAD energy threshold (RMS).
    pub vad_threshold: f32,
}

impl Default for AudioCaptureConfig {
    fn default() -> Self {
        Self {
            frame_size: 480,
            hop_size: 480,
            sample_rate: 16000,
            vad_enabled: true,
            vad_threshold: 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_capture_config_round_trip() {
        let config = AudioCaptureConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: AudioCaptureConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.frame_size, 480);
        assert_eq!(parsed.sample_rate, 16000);
        assert!(parsed.vad_enabled);
    }

    #[test]
    fn audio_frame_creation() {
        let frame = AudioFrame {
            samples: vec![0.0; 480],
            timestamp_ms: 0,
            sample_rate: 16000,
        };
        assert_eq!(frame.samples.len(), 480);
    }
}
