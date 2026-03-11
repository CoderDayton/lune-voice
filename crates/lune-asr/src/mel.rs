//! Mel-spectrogram feature extraction for Parakeet TDT.
//!
//! Ported from `parakeet-rs/src/audio.rs`. Pipeline:
//! preemphasis → STFT (Hann, RustFFT) → mel filterbank → log(x + 2^-24) → per-feature normalize.

use std::f32::consts::PI;
use std::sync::{Arc, OnceLock};

use ndarray::Array2;
use rustfft::{Fft, FftPlanner, num_complex::Complex};

// ── Preprocessor constants (TDT 128-mel) ─────────────────────────────────────

const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const N_MELS: usize = 128;
const SAMPLE_RATE: usize = 16000;
const PREEMPHASIS: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24 precomputed

// ── Static caches for filterbank and window (computed once per process) ───────

static MEL_FILTERBANK: OnceLock<Array2<f32>> = OnceLock::new();
static HANN_WINDOW: OnceLock<Vec<f32>> = OnceLock::new();
static FFT_PLAN: OnceLock<Arc<dyn Fft<f32>>> = OnceLock::new();

#[inline(always)]
fn get_mel_filterbank() -> &'static Array2<f32> {
    MEL_FILTERBANK.get_or_init(|| create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE))
}

#[inline(always)]
fn get_hann_window() -> &'static [f32] {
    HANN_WINDOW.get_or_init(|| hann_window(WIN_LENGTH))
}

#[inline(always)]
fn get_fft_plan() -> Arc<dyn Fft<f32>> {
    Arc::clone(FFT_PLAN.get_or_init(|| {
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(N_FFT)
    }))
}

/// Extract mel-spectrogram features from 16 kHz mono PCM.
///
/// Returns `(time_steps, 128)` — the shape expected by the TDT encoder
/// (transposed to `(1, 128, T)` before inference).
pub fn extract_features(audio: &[f32]) -> Array2<f32> {
    if audio.is_empty() {
        return Array2::<f32>::zeros((0, N_MELS));
    }

    // Preemphasis in-place into owned buffer
    let pre = apply_preemphasis(audio, PREEMPHASIS);

    // STFT → power spectrogram (freq_bins × num_frames)
    let spec = stft(&pre, N_FFT, HOP_LENGTH, WIN_LENGTH);

    // mel filterbank dot product: (n_mels × freq_bins) · (freq_bins × T) = (n_mels × T)
    let mel_fb = get_mel_filterbank();
    let mut mel = mel_fb.dot(&spec);

    // log with additive guard (NeMo: log_zero_guard_type="add", value=2^-24)
    // Apply in-place to avoid extra allocation
    mel.mapv_inplace(|x| (x + LOG_GUARD).ln());

    // Transpose to (time, features) for per-feature normalization.
    let (n_mels, n_frames) = (mel.shape()[0], mel.shape()[1]);
    let mel_slice = mel.as_slice().expect("mel is contiguous");
    // zeros() initializes all elements; the loop below overwrites every one.
    // Sequential-write transpose: outer loop over t gives stride-1 writes to
    // feat_slice (cache-friendly), strided reads from mel_slice.
    let mut features = Array2::<f32>::zeros((n_frames, n_mels));
    {
        let feat_slice = features.as_slice_mut().expect("features is contiguous");
        // mel is (n_mels × n_frames) C-order: mel[i*n_frames + t]
        // features is (n_frames × n_mels) C-order: features[t*n_mels + i]
        for t in 0..n_frames {
            let feat_row = &mut feat_slice[t * n_mels..(t + 1) * n_mels];
            for i in 0..n_mels {
                feat_row[i] = mel_slice[i * n_frames + t];
            }
        }
    }

    normalize_per_feature(&mut features);
    features
}

// ── Pre-emphasis ─────────────────────────────────────────────────────────────

#[inline]
fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let n = audio.len();
    let mut out = vec![0.0f32; n];
    out[0] = audio[0];
    // Difference equation: y[i] = x[i] - coef*x[i-1]
    // Non-overlapping slices allow LLVM to auto-vectorize with FMA
    let src = audio;
    let dst = &mut out[1..];
    for (i, d) in dst.iter_mut().enumerate() {
        *d = src[i + 1] - coef * src[i];
    }
    out
}

// ── STFT ─────────────────────────────────────────────────────────────────────

#[inline]
fn hann_window(len: usize) -> Vec<f32> {
    let scale = 2.0 * PI / (len as f32 - 1.0);
    (0..len)
        .map(|i| 0.5 - 0.5 * (scale * i as f32).cos())
        .collect()
}

/// Power spectrogram via RustFFT. Pad with `n_fft/2` on both sides.
fn stft(audio: &[f32], n_fft: usize, hop: usize, win: usize) -> Array2<f32> {
    let pad = n_fft / 2;
    let padded_len = pad + audio.len() + pad;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + audio.len()].copy_from_slice(audio);

    let window = get_hann_window();
    let num_frames = (padded_len - n_fft) / hop + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut spec = Array2::<f32>::zeros((freq_bins, num_frames));

    let fft = get_fft_plan();

    let zero = Complex::new(0.0f32, 0.0f32);
    let mut frame = vec![zero; n_fft];
    let mut scratch = vec![zero; fft.get_inplace_scratch_len()];

    let spec_slice = spec.as_slice_mut().expect("spec is contiguous");

    for f in 0..num_frames {
        let start = f * hop;
        let copy_len = win.min(padded_len - start);

        // Zero the entire frame then apply window to the live portion.
        // Single fill replaces split_at_mut + two separate loops.
        frame.fill(zero);
        let src = &padded[start..start + copy_len];
        let win_slice = &window[..copy_len];
        for (c, (&s, &w)) in frame[..copy_len]
            .iter_mut()
            .zip(src.iter().zip(win_slice.iter()))
        {
            // im stays 0.0 from fill; only set real part
            c.re = s * w;
        }

        fft.process_with_scratch(&mut frame, &mut scratch);

        // OPT: norm_sqr() = re*re + im*im, eliminates a sqrt() vs norm()*norm()
        // This is the innermost loop: freq_bins × num_frames iterations.
        for k in 0..freq_bins {
            spec_slice[k * num_frames + f] = frame[k].norm_sqr();
        }
    }

    spec
}

// ── Mel filterbank (Slaney scale, librosa-compatible) ────────────────────────

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.068_751_777_420_949_12;

#[inline(always)]
fn hz_to_mel(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

#[inline(always)]
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
    let mel_range = mel_max - mel_min;
    let n_points = n_mels + 2;

    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_to_hz(mel_min + mel_range * i as f64 / (n_mels + 1) as f64))
        .collect();

    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sr as f64 / n_fft as f64)
        .collect();

    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    let fb_slice = fb.as_slice_mut().expect("fb is contiguous");

    for i in 0..n_mels {
        let enorm = (2.0 / (mel_points[i + 2] - mel_points[i])) as f32;
        let lo = mel_points[i];
        let hi = mel_points[i + 2];
        let di0 = fdiff[i];
        let di1 = fdiff[i + 1];

        for (k, &freq) in fft_freqs.iter().enumerate() {
            // Branchless triangle filter: max(0, min((freq-lo)/di0, (hi-freq)/di1))
            let lower = (freq - lo) / di0;
            let upper = (hi - freq) / di1;
            let val = lower.min(upper).max(0.0) as f32;
            fb_slice[i * freq_bins + k] = val * enorm;
        }
    }

    fb
}

// ── Per-feature normalization (Bessel's correction) ──────────────────────────

fn normalize_per_feature(features: &mut Array2<f32>) {
    let num_frames = features.shape()[0];
    let num_features = features.shape()[1];

    if num_frames == 0 {
        return;
    }

    let slice = features.as_slice_mut().expect("features is contiguous");

    if num_frames == 1 {
        // Zero-mean only; single frame → all values become 0
        slice.fill(0.0);
        return;
    }

    let inv_nm1 = 1.0 / (num_frames as f32 - 1.0);

    // OPT: Welford's online algorithm — one pass for mean+variance instead of two.
    // Numerically stable and eliminates one full strided traversal per feature.
    #[allow(clippy::needless_range_loop)] // `feat` seeds the strided index `k`, not just slice
    for feat in 0..num_features {
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        let mut k = feat;
        for count in 1u32..=num_frames as u32 {
            let x = slice[k];
            let delta = x - mean;
            mean += delta / count as f32;
            // delta2 = x - updated_mean; m2 accumulates sum of squared deviations
            m2 += delta * (x - mean);
            k += num_features;
        }
        // m2 / (n-1) = sample variance; recip to multiply instead of divide in normalize pass
        let std_inv = (m2 * inv_nm1).sqrt().max(1e-5).recip();

        // Normalize pass (second and final strided pass)
        k = feat;
        for _ in 0..num_frames {
            slice[k] = (slice[k] - mean) * std_inv;
            k += num_features;
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

    #[test]
    fn static_filterbank_is_consistent() {
        let fb1 = get_mel_filterbank();
        let fb2 = get_mel_filterbank();
        // Same pointer — truly static
        assert!(std::ptr::eq(fb1 as *const _, fb2 as *const _));
    }
}
