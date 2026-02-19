use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Mel spectrogram parameters matching typical ASR encoder expectations.
pub struct MelConfig {
    /// Number of mel bins.
    pub n_mels: usize,
    /// FFT window size in samples.
    pub n_fft: usize,
    /// Hop length in samples.
    pub hop_length: usize,
    /// Sample rate of input audio.
    pub sample_rate: u32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
        }
    }
}

/// Computes log-mel spectrograms from raw PCM audio.
pub struct MelSpectrogram {
    config: MelConfig,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
}

impl MelSpectrogram {
    pub fn new(config: MelConfig) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        let window = hann_window(config.n_fft);
        let mel_filters = mel_filterbank(config.n_mels, config.n_fft, config.sample_rate);

        Self {
            config,
            fft,
            window,
            mel_filters,
        }
    }

    /// Compute log-mel spectrogram from PCM f32 samples.
    ///
    /// Returns a 2D array: `[n_frames, n_mels]` flattened row-major.
    pub fn compute(&self, pcm: &[f32]) -> MelOutput {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_mels = self.config.n_mels;

        if pcm.len() < n_fft {
            return MelOutput {
                data: vec![],
                n_frames: 0,
                n_mels,
            };
        }

        let n_frames = (pcm.len() - n_fft) / hop + 1;
        let mut output = Vec::with_capacity(n_frames * n_mels);
        let n_freq = n_fft / 2 + 1;

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + n_fft;

            // Apply window.
            let mut windowed: Vec<Complex<f32>> = pcm[start..end]
                .iter()
                .zip(&self.window)
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            // FFT.
            self.fft.process(&mut windowed);

            // Power spectrum (magnitude squared).
            let power: Vec<f32> = windowed[..n_freq]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank and log.
            for filter in &self.mel_filters {
                let mel_energy: f32 = filter.iter().zip(&power).map(|(f, p)| f * p).sum();
                output.push((mel_energy.max(1e-10)).ln());
            }
        }

        MelOutput {
            data: output,
            n_frames,
            n_mels,
        }
    }
}

/// Output of mel spectrogram computation.
pub struct MelOutput {
    /// Flattened row-major `[n_frames, n_mels]`.
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

impl MelOutput {
    /// Get a single frame's mel values.
    pub fn frame(&self, idx: usize) -> &[f32] {
        let start = idx * self.n_mels;
        &self.data[start..start + self.n_mels]
    }
}

/// Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
        .collect()
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Build triangular mel filterbank.
///
/// Returns `n_mels` filters, each of length `n_fft/2 + 1`.
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f32>> {
    let n_freq = n_fft / 2 + 1;
    let f_max = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);

    // Equally spaced mel points.
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz then to FFT bin indices.
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&h| h * n_fft as f32 / sample_rate as f32)
        .collect();

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let mut filter = vec![0.0f32; n_freq];
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_freq {
            let kf = k as f32;
            if kf >= left && kf <= center && center > left {
                filter[k] = (kf - left) / (center - left);
            } else if kf > center && kf <= right && right > center {
                filter[k] = (right - kf) / (right - center);
            }
        }
        filters.push(filter);
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints() {
        let w = hann_window(400);
        assert_eq!(w.len(), 400);
        assert!(w[0].abs() < 1e-6, "first sample should be ~0");
        // Peak at center.
        assert!((w[200] - 1.0).abs() < 0.01);
    }

    #[test]
    fn mel_hz_round_trip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((hz - back).abs() < 0.1);
    }

    #[test]
    fn filterbank_shape() {
        let filters = mel_filterbank(80, 400, 16000);
        assert_eq!(filters.len(), 80);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn mel_spec_shape() {
        let mel = MelSpectrogram::new(MelConfig::default());
        // 1 second of 16kHz audio.
        let pcm = vec![0.0f32; 16000];
        let output = mel.compute(&pcm);

        // Expected frames: (16000 - 400) / 160 + 1 = 98
        assert_eq!(output.n_frames, 98);
        assert_eq!(output.n_mels, 80);
        assert_eq!(output.data.len(), 98 * 80);
    }

    #[test]
    fn mel_spec_short_audio() {
        let mel = MelSpectrogram::new(MelConfig::default());
        // Too short for even one frame.
        let pcm = vec![0.0f32; 100];
        let output = mel.compute(&pcm);
        assert_eq!(output.n_frames, 0);
        assert!(output.data.is_empty());
    }

    #[test]
    fn mel_spec_sine_wave() {
        let mel = MelSpectrogram::new(MelConfig::default());
        // 440Hz sine wave at 16kHz.
        let pcm: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let output = mel.compute(&pcm);
        assert!(output.n_frames > 0);
        // Mel energies should be finite.
        assert!(output.data.iter().all(|v| v.is_finite()));
        // Energy should concentrate in bins near 440Hz, not be uniform.
        let frame = output.frame(output.n_frames / 2);
        let max_bin = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // 440Hz should be in the lower mel bins (roughly bin 10-25).
        assert!(max_bin < 40, "440Hz energy should be in lower mel bins, got {max_bin}");
    }
}
