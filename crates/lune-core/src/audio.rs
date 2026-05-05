use serde::{Deserialize, Serialize};

/// The canonical pipeline sample rate (Hz). All audio in Lune is processed at this rate.
pub const SAMPLE_RATE: usize = 16_000;

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

impl AudioFrame {
    /// Compute RMS energy inline — avoids allocations, SIMD-friendly reduction.
    #[inline(always)]
    pub fn rms(&self) -> f32 {
        let n = self.samples.len();
        if n == 0 {
            return 0.0;
        }
        // Manual reduction keeps LLVM auto-vectorizer happy on f32 slices.
        let sum_sq: f32 = self.samples.iter().map(|&s| s * s).sum();
        (sum_sq / n as f32).sqrt()
    }

    /// Returns true if the frame exceeds the given RMS threshold — zero-alloc VAD fast path.
    #[inline(always)]
    pub fn exceeds_energy(&self, threshold: f32) -> bool {
        // Early-exit: sum squared samples until we exceed threshold² * n.
        let limit = threshold * threshold * self.samples.len() as f32;
        let mut acc = 0.0f32;
        for &s in &self.samples {
            acc += s * s;
            if acc >= limit {
                return true;
            }
        }
        false
    }
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

    #[test]
    fn rms_silent_frame() {
        let frame = AudioFrame {
            samples: vec![0.0; 480],
            timestamp_ms: 0,
            sample_rate: 16000,
        };
        assert_eq!(frame.rms(), 0.0);
    }

    #[test]
    fn rms_constant_frame() {
        let frame = AudioFrame {
            samples: vec![1.0; 100],
            timestamp_ms: 0,
            sample_rate: 16000,
        };
        assert!((frame.rms() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn exceeds_energy_early_exit() {
        // Frame with one loud spike — should short-circuit.
        let mut samples = vec![0.0f32; 480];
        samples[0] = 1.0;
        let frame = AudioFrame {
            samples,
            timestamp_ms: 0,
            sample_rate: 16000,
        };
        assert!(frame.exceeds_energy(0.01));
    }

    #[test]
    fn exceeds_energy_silent() {
        let frame = AudioFrame {
            samples: vec![0.0; 480],
            timestamp_ms: 0,
            sample_rate: 16000,
        };
        assert!(!frame.exceeds_energy(0.01));
    }

    #[test]
    fn rms_empty_frame() {
        let frame = AudioFrame {
            samples: vec![],
            timestamp_ms: 0,
            sample_rate: 16000,
        };
        assert_eq!(frame.rms(), 0.0);
        assert!(!frame.exceeds_energy(0.01));
    }
}
