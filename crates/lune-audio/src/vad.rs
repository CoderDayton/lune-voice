//! Voice Activity Detection powered by Earshot (pure Rust).

use earshot::Detector;

/// Earshot hop size: 256 samples = 16ms at 16kHz.
const HOP_SIZE: usize = 256;

/// Result of single VAD chunk evaluation.
#[derive(Debug, Clone, Copy)]
pub struct VadResult {
    /// Whether chunk is classified as speech (with hysteresis).
    pub is_speech: bool,
    /// Raw max-frame VAD score [0.0–1.0].
    pub vad_score: f32,
    /// RMS energy (diagnostic).
    pub energy: f32,
}

/// VAD configuration
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Speech-START threshold (0.0–1.0). Default: 0.5
    pub threshold: f32,
    /// Speech-END threshold (0.0–1.0). Default: 0.3
    pub threshold_end: f32,
    /// Hangover duration in milliseconds. Default: 30
    pub hangover_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            threshold_end: 0.3,
            hangover_ms: 30,
        }
    }
}

/// Voice Activity Detector using Earshot (pure Rust, no ONNX).
pub struct Vad {
    inner: Detector,
    threshold_start: f32,
    threshold_end: f32,
    in_speech: bool,
    hangover_remaining: usize,
    hangover_total: usize,
    carry: Vec<f32>,
}

impl Vad {
    pub fn new(config: VadConfig, _sample_rate: u32) -> Result<Self, crate::AudioError> {
        if !(0.0..=1.0).contains(&config.threshold) || !config.threshold.is_finite() {
            return Err(crate::AudioError::Vad(format!(
                "threshold {} out of range [0.0, 1.0]",
                config.threshold
            )));
        }
        if !(0.0..=1.0).contains(&config.threshold_end) || !config.threshold_end.is_finite() {
            return Err(crate::AudioError::Vad(format!(
                "threshold_end {} out of range [0.0, 1.0]",
                config.threshold_end
            )));
        }
        if config.threshold_end > config.threshold {
            return Err(crate::AudioError::Vad(format!(
                "threshold_end ({}) must be <= threshold ({})",
                config.threshold_end, config.threshold
            )));
        }
        let hangover_samples = (config.hangover_ms as usize)
            .saturating_mul(16)
            .max(HOP_SIZE);
        Ok(Self {
            inner: Detector::default(),
            threshold_start: config.threshold,
            threshold_end: config.threshold_end,
            in_speech: false,
            hangover_remaining: 0,
            hangover_total: hangover_samples,
            carry: Vec::with_capacity(HOP_SIZE),
        })
    }

    pub fn process_chunk(&mut self, samples: &[f32]) -> VadResult {
        let energy = compute_rms(samples);

        let mut max_score: f32 = 0.0;
        let mut offset = 0;

        if !self.carry.is_empty() {
            let needed = HOP_SIZE - self.carry.len();
            if samples.len() >= needed {
                self.carry.extend_from_slice(&samples[..needed]);
                max_score = max_score.max(self.inner.predict_f32(&self.carry));
                self.carry.clear();
                offset = needed;
            } else {
                self.carry.extend_from_slice(samples);
            }
        }

        while offset + HOP_SIZE <= samples.len() {
            let frame = &samples[offset..offset + HOP_SIZE];
            let score = self.inner.predict_f32(frame);
            max_score = max_score.max(score);
            offset += HOP_SIZE;
        }

        if offset < samples.len() && self.carry.is_empty() {
            self.carry.extend_from_slice(&samples[offset..]);
        }

        let score_is_speech = if self.in_speech {
            max_score >= self.threshold_end
        } else {
            max_score >= self.threshold_start
        };

        let is_speech = if score_is_speech {
            self.hangover_remaining = self.hangover_total;
            true
        } else if self.hangover_remaining > 0 {
            self.hangover_remaining = self.hangover_remaining.saturating_sub(samples.len());
            true
        } else {
            false
        };
        self.in_speech = is_speech;

        VadResult {
            is_speech,
            vad_score: max_score,
            energy,
        }
    }
}

#[inline]
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples
        .iter()
        .map(|&s| if s.is_finite() { s * s } else { 0.0 })
        .sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    if rms.is_finite() { rms } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carry_over_retains_partial_frame() {
        let mut vad = Vad::new(VadConfig::default(), 16_000).unwrap();
        let _ = vad.process_chunk(&vec![0.0f32; HOP_SIZE - 1]);
        assert_eq!(vad.carry.len(), HOP_SIZE - 1);

        let _ = vad.process_chunk(&[0.0f32]);
        assert_eq!(vad.carry.len(), 0);
    }

    #[test]
    fn rejects_inverted_thresholds() {
        let r = Vad::new(
            VadConfig {
                threshold: 0.3,
                threshold_end: 0.5,
                hangover_ms: 30,
            },
            16_000,
        );
        assert!(matches!(r, Err(crate::AudioError::Vad(_))));
    }

    #[test]
    fn rejects_out_of_range_thresholds() {
        for (start, end) in [(-0.1, 0.0), (1.1, 0.5), (0.5, -0.01), (0.5, 1.01)] {
            let r = Vad::new(
                VadConfig {
                    threshold: start,
                    threshold_end: end,
                    hangover_ms: 30,
                },
                16_000,
            );
            assert!(
                matches!(r, Err(crate::AudioError::Vad(_))),
                "expected rejection for ({start}, {end})"
            );
        }
    }

    #[test]
    fn rejects_nan_thresholds() {
        let r = Vad::new(
            VadConfig {
                threshold: f32::NAN,
                threshold_end: 0.3,
                hangover_ms: 30,
            },
            16_000,
        );
        assert!(matches!(r, Err(crate::AudioError::Vad(_))));
    }

    #[test]
    fn hangover_ms_does_not_overflow() {
        let r = Vad::new(
            VadConfig {
                threshold: 0.5,
                threshold_end: 0.3,
                hangover_ms: u32::MAX,
            },
            16_000,
        );
        assert!(r.is_ok(), "huge hangover_ms must saturate, not panic/wrap");
    }

    #[test]
    fn rms_handles_non_finite_samples() {
        let samples = [0.5f32, f32::NAN, f32::INFINITY, -0.5];
        let rms = compute_rms(&samples);
        assert!(rms.is_finite(), "rms must be finite, got {rms}");
        assert!(rms > 0.0);
    }

    #[test]
    fn rms_all_nan_returns_zero() {
        let samples = [f32::NAN, f32::NAN, f32::NAN];
        assert_eq!(compute_rms(&samples), 0.0);
    }
}
