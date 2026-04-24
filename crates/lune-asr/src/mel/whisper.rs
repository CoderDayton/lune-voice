//! Whisper-compatible FP16 log-mel spectrogram for Qwen3-ASR.
//!
//! FFT runs in f32 for numerical stability; output is native f16.
//! Supports single and batch extraction.

use half::f16;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;
const EPSILON: f32 = 1e-10;

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

pub struct WhisperMelF16 {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    window_f32: Vec<f32>,
    mel_filters_f32: Vec<f32>, // [n_mels][n_fft/2+1]
    fft: Arc<dyn Fft<f32>>,
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
        assert!(win_length <= n_fft);
        assert!(n_mels > 0);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let window_f32 = hann_window(win_length);
        let mel_filters_f32 = build_slaney_mel_filterbank(sample_rate, n_fft, n_mels);

        Self {
            n_fft,
            hop_length,
            win_length,
            n_mels,
            window_f32,
            mel_filters_f32,
            fft,
        }
    }

    /// Extract mel spectrogram from f32 PCM. Returns FP16 mel-major.
    pub fn extract_from_f32(&self, pcm: &[f32]) -> MelSpecF16 {
        let padded = reflect_pad_center(pcm, self.n_fft / 2);
        // Whisper drops the last STFT frame: `log_spec[:, :-1]`
        let n_frames = num_frames(padded.len(), self.win_length, self.hop_length) - 1;
        let n_bins = self.n_fft / 2 + 1;

        let mut power = vec![0.0f32; n_bins * n_frames];
        let mut fft_in = vec![Complex::new(0.0f32, 0.0); self.n_fft];
        let win_offset = (self.n_fft - self.win_length) / 2;

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let frame = &padded[start..start + self.win_length];

            for x in &mut fft_in {
                *x = Complex::new(0.0, 0.0);
            }
            for i in 0..self.win_length {
                fft_in[win_offset + i].re = frame[i] * self.window_f32[i];
            }

            self.fft.process(&mut fft_in);

            let dst = &mut power[frame_idx * n_bins..(frame_idx + 1) * n_bins];
            for k in 0..n_bins {
                let c = fft_in[k];
                dst[k] = c.re.mul_add(c.re, c.im * c.im);
            }
        }

        // Mel filterbank
        let mut mel_f32 = vec![0.0f32; self.n_mels * n_frames];
        for m in 0..self.n_mels {
            let filter = &self.mel_filters_f32[m * n_bins..(m + 1) * n_bins];
            let mel_row = &mut mel_f32[m * n_frames..(m + 1) * n_frames];

            for t in 0..n_frames {
                let spec = &power[t * n_bins..(t + 1) * n_bins];
                let mut acc = 0.0f32;
                for k in 0..n_bins {
                    acc += filter[k] * spec[k];
                }
                mel_row[t] = acc.max(EPSILON);
            }
        }

        whisper_log_normalize(&mut mel_f32);

        MelSpecF16 {
            n_mels: self.n_mels,
            n_frames,
            data: mel_f32.into_iter().map(f16::from_f32).collect(),
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

fn reflect_pad_center(x: &[f32], pad: usize) -> Vec<f32> {
    if x.is_empty() {
        return vec![0.0; pad * 2 + WIN_LENGTH];
    }
    let start = -(pad as isize);
    let end = x.len() as isize + pad as isize;
    let mut out = Vec::with_capacity((end - start) as usize);
    for i in start..end {
        out.push(reflect_get(x, i));
    }
    out
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
    for v in mel.iter_mut() {
        *v = v.max(EPSILON).log10();
    }
    let peak = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = peak - 8.0;
    for v in mel.iter_mut() {
        if *v < floor {
            *v = floor;
        }
        *v = (*v + 4.0) / 4.0;
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
