//! Whisper-compatible FP16 log-mel spectrogram for Qwen3-ASR.
//!
//! FFT runs in f32 for numerical stability; output is native f16.
//! Supports single and batch extraction.

use half::f16;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::{Arc, Mutex};

pub use lune_core::audio::SAMPLE_RATE;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;
const EPSILON: f32 = 1e-10;
const INV4_LOG10E: f32 = 0.25 * std::f32::consts::LOG10_E;

#[derive(Clone, Debug)]
pub struct MelSpecF16 {
    pub n_mels: usize,
    pub n_frames: usize,
    /// Mel-major layout: `[mel][frame]`.
    pub data: Vec<f16>,
}

#[derive(Clone, Debug)]
pub struct BatchMelF16 {
    pub batch: usize,
    pub n_mels: usize,
    pub max_frames: usize,
    pub lengths: Vec<i32>,
    /// `[B][M][T]` layout, zero-padded to `max_frames`.
    pub data: Vec<f16>,
}

/// Reusable per-call workspace. Capacity grows monotonically; the contents
/// are overwritten every call. Holding it behind a `Mutex` lets the public
/// API stay `&self` while still amortizing allocations across calls.
#[derive(Default)]
struct MelScratch {
    padded: Vec<f32>,
    fft_in: Vec<Complex<f32>>,
    power: Vec<f32>,
    mel_f32: Vec<f32>,
}

pub struct WhisperMelF16 {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    window_f32: Vec<f32>,
    mel_filters_f32: Vec<f32>,       // [n_mels][n_fft/2+1]
    mel_ranges: Vec<(usize, usize)>, // (start_bin, end_bin) per mel band
    fft: Arc<dyn Fft<f32>>,
    scratch: Mutex<MelScratch>,
}

impl WhisperMelF16 {
    pub fn new_qwen() -> Self {
        Self::new(SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, N_MELS)
    }

    pub fn new(
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
        n_mels: usize,
    ) -> Self {
        assert!(sample_rate > 0, "sample_rate must be positive");
        assert!(n_fft > 0, "n_fft must be positive");
        assert!(hop_length > 0, "hop_length must be positive");
        assert!(win_length > 0, "win_length must be positive");
        assert!(win_length <= n_fft);
        assert!(n_mels > 0);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let window_f32 = hann_window(win_length);
        let mel_filters_f32 = build_slaney_mel_filterbank(sample_rate, n_fft, n_mels);
        let n_bins = n_fft / 2 + 1;
        let mel_ranges = (0..n_mels)
            .map(|m| {
                let row = &mel_filters_f32[m * n_bins..(m + 1) * n_bins];
                let lo = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let hi = row
                    .iter()
                    .rposition(|&v| v > 0.0)
                    .map(|p| p + 1)
                    .unwrap_or(0);
                (lo, hi)
            })
            .collect();

        Self {
            n_fft,
            hop_length,
            win_length,
            n_mels,
            window_f32,
            mel_filters_f32,
            mel_ranges,
            fft,
            scratch: Mutex::new(MelScratch::default()),
        }
    }

    /// Extract mel spectrogram from f32 PCM. Returns FP16 mel-major.
    pub fn extract_from_f32(&self, pcm: &[f32]) -> MelSpecF16 {
        self.extract_from_f32_padded(pcm, 0)
    }

    /// Extract mel, then pad output to at least `min_frames` with the silence
    /// floor value. Avoids computing FFT on zero-padded silence.
    pub fn extract_from_f32_padded(&self, pcm: &[f32], min_frames: usize) -> MelSpecF16 {
        // Hold scratch for the whole call. Audio extraction is single-threaded
        // per session; uncontended Mutex overhead is nanoseconds vs ms of FFT.
        let mut guard = self.scratch.lock().unwrap_or_else(|p| p.into_inner());
        // Deref once into a plain &mut so the borrow checker can split fields.
        let scratch: &mut MelScratch = &mut guard;
        let MelScratch {
            padded,
            fft_in,
            power,
            mel_f32,
        } = scratch;

        reflect_pad_center_into(padded, pcm, self.n_fft / 2);
        let n_frames = num_frames(padded.len(), self.win_length, self.hop_length) - 1;
        let n_bins = self.n_fft / 2 + 1;

        power.clear();
        power.resize(n_bins * n_frames, 0.0);
        fft_in.clear();
        fft_in.resize(self.n_fft, Complex::new(0.0f32, 0.0));

        let full_cover = self.n_fft == self.win_length;

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let frame = &padded[start..start + self.win_length];

            if !full_cover {
                for x in fft_in.iter_mut() {
                    *x = Complex::new(0.0, 0.0);
                }
            }
            let win_offset = if full_cover {
                0
            } else {
                (self.n_fft - self.win_length) / 2
            };
            for i in 0..self.win_length {
                fft_in[win_offset + i] = Complex::new(frame[i] * self.window_f32[i], 0.0);
            }

            self.fft.process(fft_in);

            let dst = &mut power[frame_idx * n_bins..(frame_idx + 1) * n_bins];
            for k in 0..n_bins {
                let c = fft_in[k];
                dst[k] = c.re.mul_add(c.re, c.im * c.im);
            }
        }

        // Mel filterbank — iterate only non-zero filter bins per band.
        mel_f32.clear();
        mel_f32.resize(self.n_mels * n_frames, 0.0);
        for m in 0..self.n_mels {
            let (lo, hi) = self.mel_ranges[m];
            let filter = &self.mel_filters_f32[m * n_bins + lo..m * n_bins + hi];
            let mel_row = &mut mel_f32[m * n_frames..(m + 1) * n_frames];

            for t in 0..n_frames {
                let spec = &power[t * n_bins + lo..t * n_bins + hi];
                let mut acc = 0.0f32;
                for k in 0..filter.len() {
                    acc = filter[k].mul_add(spec[k], acc);
                }
                mel_row[t] = acc.max(EPSILON);
            }
        }

        whisper_log_normalize(mel_f32);

        let out_frames = n_frames.max(min_frames);
        if out_frames == n_frames {
            MelSpecF16 {
                n_mels: self.n_mels,
                n_frames,
                data: mel_f32.iter().map(|&v| f16::from_f32(v)).collect(),
            }
        } else {
            // Silence floor: minimum value after log normalization.
            let floor_val = mel_f32.iter().copied().fold(f32::INFINITY, f32::min);
            let floor_f16 = f16::from_f32(floor_val);
            let pad_frames = out_frames - n_frames;
            let mut data = Vec::with_capacity(self.n_mels * out_frames);
            for m in 0..self.n_mels {
                let row = &mel_f32[m * n_frames..(m + 1) * n_frames];
                data.extend(row.iter().map(|&v| f16::from_f32(v)));
                data.extend(std::iter::repeat_n(floor_f16, pad_frames));
            }
            MelSpecF16 {
                n_mels: self.n_mels,
                n_frames: out_frames,
                data,
            }
        }
    }

    /// Collate multiple clips into a padded batch tensor.
    pub fn collate_batch_from_f32(&self, clips: &[&[f32]]) -> BatchMelF16 {
        let specs: Vec<MelSpecF16> = clips.iter().map(|c| self.extract_from_f32(c)).collect();
        collate_specs(&specs)
    }
}

fn collate_specs(specs: &[MelSpecF16]) -> BatchMelF16 {
    assert!(!specs.is_empty());

    let batch = specs.len();
    let n_mels = specs[0].n_mels;
    assert!(
        specs.iter().all(|s| s.n_mels == n_mels),
        "collate_specs: all specs must share n_mels (got {n_mels} mixed with others)"
    );
    let max_frames = specs.iter().map(|s| s.n_frames).max().unwrap_or(0);

    let mut lengths = Vec::with_capacity(batch);
    let mut data = vec![f16::from_f32(0.0); batch * n_mels * max_frames];

    for (b, spec) in specs.iter().enumerate() {
        lengths.push(spec.n_frames as i32);
        for m in 0..n_mels {
            let src = &spec.data[m * spec.n_frames..(m + 1) * spec.n_frames];
            let dst_off = b * n_mels * max_frames + m * max_frames;
            data[dst_off..dst_off + spec.n_frames].copy_from_slice(src);
        }
    }

    BatchMelF16 {
        batch,
        n_mels,
        max_frames,
        lengths,
        data,
    }
}

fn num_frames(total_len: usize, win_length: usize, hop_length: usize) -> usize {
    if total_len <= win_length {
        1
    } else {
        1 + (total_len - win_length) / hop_length
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0];
    }
    let two_pi = core::f32::consts::PI * 2.0;
    (0..n)
        .map(|i| 0.5 - 0.5 * (two_pi * i as f32 / n as f32).cos())
        .collect()
}

fn reflect_pad_center_into(out: &mut Vec<f32>, x: &[f32], pad: usize) {
    out.clear();
    if x.is_empty() {
        out.resize(pad * 2 + WIN_LENGTH, 0.0);
        return;
    }
    let start = -(pad as isize);
    let end = x.len() as isize + pad as isize;
    out.reserve((end - start) as usize);
    for i in start..end {
        out.push(reflect_get(x, i));
    }
}

fn reflect_get(x: &[f32], idx: isize) -> f32 {
    if x.len() == 1 {
        return x[0];
    }
    let n = x.len() as isize;
    let mut i = idx;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - i - 2;
        } else {
            return x[i as usize];
        }
    }
}

fn whisper_log_normalize(mel: &mut [f32]) {
    // Whisper-spec floor when no real signal: log10(EPSILON)/4 + 1
    // = ln(EPSILON) * INV4_LOG10E + 1
    let epsilon_floor = EPSILON.ln().mul_add(INV4_LOG10E, 1.0);

    // Fuse log10 + peak-find into one pass. Substitute the silence floor
    // for any non-finite value so a stray NaN from upstream cannot poison
    // peak (NaN comparisons all return false → peak stays NEG_INFINITY).
    let mut peak = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        let ln_v = v.max(EPSILON).ln();
        let scaled = ln_v.mul_add(INV4_LOG10E, 1.0);
        let scaled = if scaled.is_finite() {
            scaled
        } else {
            epsilon_floor
        };
        *v = scaled;
        if scaled > peak {
            peak = scaled;
        }
    }
    // If every value was non-finite, peak fell back through epsilon_floor
    // path. If mel was empty, leave it untouched.
    if !peak.is_finite() {
        return;
    }
    // floor = (peak_log10 - 8) / 4 + 1 = peak_scaled - 2
    let floor = peak - 2.0;
    for v in mel.iter_mut() {
        if *v < floor {
            *v = floor;
        }
    }
}

fn build_slaney_mel_filterbank(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fmin = 0.0f32;
    let fmax = sample_rate as f32 / 2.0;

    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);

    let mel_points = linspace(mel_min, mel_max, n_mels + 2);
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_slaney).collect();

    let fft_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut fb = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let enorm = 2.0 / (right - left).max(EPSILON);

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let up = (freq - left) / (center - left).max(EPSILON);
            let down = (right - freq) / (right - center).max(EPSILON);
            fb[m * n_bins + k] = up.min(down).max(0.0) * enorm;
        }
    }
    fb
}

fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    match n {
        0 => vec![],
        1 => vec![start],
        _ => {
            let step = (end - start) / (n as f32 - 1.0);
            (0..n).map(|i| start + step * i as f32).collect()
        }
    }
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_log_normalize_finite_with_nan() {
        let mut mel = vec![0.5_f32, f32::NAN, 1e-12, f32::INFINITY, 0.25];
        whisper_log_normalize(&mut mel);
        assert!(
            mel.iter().all(|v| v.is_finite()),
            "log normalize must produce all-finite output"
        );
    }

    #[test]
    fn whisper_log_normalize_empty_is_noop() {
        let mut mel: Vec<f32> = vec![];
        whisper_log_normalize(&mut mel);
        assert!(mel.is_empty());
    }

    #[test]
    fn extract_from_f32_with_nan_audio_is_finite() {
        let mel = WhisperMelF16::new_qwen();
        let mut pcm = vec![0.1f32; SAMPLE_RATE]; // 1s
        pcm[100] = f32::NAN;
        pcm[200] = f32::INFINITY;
        let spec = mel.extract_from_f32(&pcm);
        // The pre-FFT path doesn't filter NaN currently; whisper_log_normalize
        // is the safety net. Output should be all finite f16 (no NaN tile).
        assert!(spec.data.iter().all(|v| v.to_f32().is_finite()));
    }

    #[test]
    #[should_panic(expected = "n_fft must be positive")]
    fn rejects_zero_n_fft() {
        let _ = WhisperMelF16::new(16_000, 0, 160, 400, 128);
    }

    #[test]
    #[should_panic(expected = "all specs must share n_mels")]
    fn collate_specs_rejects_mismatched_n_mels() {
        let a = MelSpecF16 {
            n_mels: 80,
            n_frames: 1,
            data: vec![f16::from_f32(0.0); 80],
        };
        let b = MelSpecF16 {
            n_mels: 128,
            n_frames: 1,
            data: vec![f16::from_f32(0.0); 128],
        };
        let _ = collate_specs(&[a, b]);
    }
}
