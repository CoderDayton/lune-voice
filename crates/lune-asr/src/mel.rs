//! Mel-spectrogram feature extraction for Parakeet TDT.
//!
//! Ported from `parakeet-rs/src/audio.rs`. Pipeline:
//! preemphasis → STFT (Hann, RustFFT) → mel filterbank → log(x + 2^-24) → per-feature normalize.

use std::f32::consts::PI;

use ndarray::Array2;
use rustfft::{FftPlanner, num_complex::Complex};

// ── Preprocessor constants (TDT 128-mel) ─────────────────────────────────────

const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const N_MELS: usize = 128;
const SAMPLE_RATE: usize = 16000;
const PREEMPHASIS: f32 = 0.97;

/// Extract mel-spectrogram features from 16 kHz mono PCM.
///
/// Returns `(time_steps, 128)` — the shape expected by the TDT encoder
/// (transposed to `(1, 128, T)` before inference).
pub fn extract_features(audio: &[f32]) -> Array2<f32> {
    if audio.is_empty() {
        return Array2::<f32>::zeros((0, N_MELS));
    }
    let pre = apply_preemphasis(audio, PREEMPHASIS);
    let spec = stft(&pre, N_FFT, HOP_LENGTH, WIN_LENGTH);
    let mel_fb = create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);
    let mel = mel_fb.dot(&spec);

    // log with additive guard (NeMo: log_zero_guard_type="add", value=2^-24)
    let log_guard: f32 = 2.0f32.powi(-24);
    let mel_log = mel.mapv(|x| (x + log_guard).ln());

    // Transpose to (time, features) for per-feature normalization.
    let mut features = mel_log.t().to_owned();
    normalize_per_feature(&mut features);
    features
}

// ── Pre-emphasis ─────────────────────────────────────────────────────────────

#[inline]
fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(audio.len());
    out.push(audio[0]);
    out.extend(
        audio[1..]
            .iter()
            .zip(audio[..audio.len() - 1].iter())
            .map(|(&cur, &prev)| cur - coef * prev),
    );
    out
}

// ── STFT ─────────────────────────────────────────────────────────────────────

#[inline]
fn hann_window(len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    out.extend((0..len).map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (len as f32 - 1.0)).cos()));
    out
}

/// Power spectrogram via RustFFT. Pad with `n_fft/2` on both sides.
fn stft(audio: &[f32], n_fft: usize, hop: usize, win: usize) -> Array2<f32> {
    let pad = n_fft / 2;
    let padded_len = pad + audio.len() + pad;
    let mut padded = Vec::with_capacity(padded_len);
    padded.resize(pad, 0.0f32);
    padded.extend_from_slice(audio);
    padded.resize(padded_len, 0.0f32);

    let window = hann_window(win);
    let num_frames = (padded.len() - n_fft) / hop + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut spec = Array2::<f32>::zeros((freq_bins, num_frames));

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut frame = vec![Complex::new(0.0f32, 0.0f32); n_fft];

    for f in 0..num_frames {
        let start = f * hop;
        // Zero out frame
        for c in frame.iter_mut() {
            *c = Complex::new(0.0, 0.0);
        }
        let copy_len = win.min(padded.len() - start);
        for (c, (&s, &w)) in frame[..copy_len].iter_mut().zip(
            padded[start..start + copy_len]
                .iter()
                .zip(window[..copy_len].iter()),
        ) {
            *c = Complex::new(s * w, 0.0);
        }
        fft.process(&mut frame);
        for k in 0..freq_bins {
            let mag = frame[k].norm();
            spec[[k, f]] = mag * mag;
        }
    }

    spec
}

// ── Mel filterbank (Slaney scale, librosa-compatible) ────────────────────────

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.068_751_777_420_949_12;

#[inline]
fn hz_to_mel(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

#[inline]
fn mel_to_hz(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

fn create_mel_filterbank(n_fft: usize, n_mels: usize, sr: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut fb = Array2::<f32>::zeros((n_mels, freq_bins));

    let fmax = sr as f64 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    mel_points.extend(
        (0..=n_mels + 1)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)),
    );

    let mut fft_freqs = Vec::with_capacity(freq_bins);
    fft_freqs.extend((0..freq_bins).map(|i| i as f64 * sr as f64 / n_fft as f64));

    let mut fdiff = Vec::with_capacity(mel_points.len() - 1);
    fdiff.extend(mel_points.windows(2).map(|w| w[1] - w[0]));

    for i in 0..n_mels {
        let enorm = (2.0 / (mel_points[i + 2] - mel_points[i])) as f32;
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            let val = 0.0f64.max(lower.min(upper)) as f32;
            fb[[i, k]] = val * enorm;
        }
    }

    fb
}

// ── Per-feature normalization (Bessel's correction) ──────────────────────────

fn normalize_per_feature(features: &mut Array2<f32>) {
    let num_frames = features.shape()[0];
    let num_features = features.shape()[1];
    if num_frames <= 1 {
        // Single frame or empty: zero-mean only, no variance normalization.
        for feat in 0..num_features {
            let mut col = features.column_mut(feat);
            let mean: f32 = col.iter().sum::<f32>() / num_frames.max(1) as f32;
            for val in col.iter_mut() {
                *val -= mean;
            }
        }
        return;
    }

    for feat in 0..num_features {
        let mut col = features.column_mut(feat);
        let mean: f32 = col.iter().sum::<f32>() / num_frames as f32;
        let var: f32 =
            col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (num_frames as f32 - 1.0);
        let std = var.sqrt() + 1e-5;
        for val in col.iter_mut() {
            *val = (*val - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_features_shape() {
        // 1 second of silence @ 16 kHz
        let audio = vec![0.0f32; 16000];
        let features = extract_features(&audio);
        assert_eq!(features.shape()[1], N_MELS); // 128 mel bins
        // Time steps: (16000 + 2*256 - 512) / 160 + 1 = 100
        assert!(features.shape()[0] > 0);
    }

    #[test]
    fn extract_features_empty_audio() {
        let features = extract_features(&[]);
        assert_eq!(features.shape(), &[0, N_MELS]);
    }

    #[test]
    fn extract_features_single_frame() {
        // Very short audio — should not panic on Bessel normalization
        let audio = vec![0.1f32; HOP_LENGTH + N_FFT];
        let features = extract_features(&audio);
        assert_eq!(features.shape()[1], N_MELS);
        assert!(features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preemphasis_preserves_length() {
        let audio = vec![1.0f32; 100];
        let pre = apply_preemphasis(&audio, 0.97);
        assert_eq!(pre.len(), 100);
    }

    #[test]
    fn mel_filterbank_shape() {
        let fb = create_mel_filterbank(512, 128, 16000);
        assert_eq!(fb.shape(), &[128, 257]);
    }
}
