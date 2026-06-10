//! PCEN-style adaptive AGC for the VAD frontend.
//!
//! Operates on f32 mono audio in `[-1, 1]`. Tracks signal and noise envelopes
//! independently and applies one scalar gain per hop. The gain is bounded
//! both by a desired output level (`target_rms`) and by a noise ceiling so
//! that amplified background noise never exceeds `noise_ceiling_rms`.
//!
//! Math (per hop, where `x` is the frame and `α ∈ (0, 1]`):
//! ```text
//! frame_pow      = mean(x²)
//! sig_env_pow    = (1 − s_sig)·sig_env_pow   + s_sig·frame_pow
//! noise_env_pow  = asymmetric IIR (fast down, slow up)
//! desired_gain   = target_rms / (ε + sig_env_pow^(α/2))
//! noise_cap      = noise_ceiling_rms / (ε + sqrt(noise_env_pow))
//! raw_gain       = clamp(min(desired_gain, noise_cap), min_gain, max_gain)
//! last_gain      = (1 − s_gain)·last_gain + s_gain·raw_gain
//! y              = clamp(x · last_gain, −1, 1)
//! ```

use crate::AudioError;

/// Default target output RMS — the "expected output loudness".
pub const EXPECTED_OUTPUT_RMS: f32 = 0.10;

/// Default noise ceiling: amplified background noise will not exceed this RMS.
pub const DEFAULT_NOISE_CEILING_RMS: f32 = 0.02;

/// Tunables for [`PcenAgc`].
#[derive(Debug, Clone)]
pub struct PcenAgcConfig {
    /// Desired output speech RMS (the "expected gain" target).
    pub target_rms: f32,
    /// Cap on amplified-noise RMS. Limits boost in quiet rooms.
    pub noise_ceiling_rms: f32,
    /// Hard ceiling on per-frame gain.
    pub max_gain: f32,
    /// Hard floor on per-frame gain (allows mild attenuation of loud input).
    pub min_gain: f32,
    /// PCEN exponent on the signal envelope. `1.0` = full AGC.
    pub alpha: f32,
    /// Time constant of the signal envelope (ms).
    pub tau_signal_ms: f32,
    /// Time constant when the noise envelope rises (ms). Should be slow.
    pub tau_noise_up_ms: f32,
    /// Time constant when the noise envelope falls (ms). Tracks minima.
    pub tau_noise_down_ms: f32,
    /// Time constant of the gain smoother (ms). Suppresses pumping.
    pub tau_gain_ms: f32,
    /// Number of leading frames during which the AGC tracks envelopes but
    /// holds `gain = 1.0`. Prevents misapplied gain on hotkey-onset speech
    /// before the noise tracker has had time to drain to the room floor.
    pub warmup_frames: u32,
    /// Numerical floor for divisions and envelope clamps.
    pub eps: f32,
}

impl Default for PcenAgcConfig {
    fn default() -> Self {
        Self {
            target_rms: EXPECTED_OUTPUT_RMS,
            noise_ceiling_rms: DEFAULT_NOISE_CEILING_RMS,
            max_gain: 12.0,
            min_gain: 0.25,
            alpha: 0.85,
            tau_signal_ms: 250.0,
            tau_noise_up_ms: 5_000.0,
            tau_noise_down_ms: 200.0,
            tau_gain_ms: 80.0,
            warmup_frames: 16,
            eps: 1.0e-8,
        }
    }
}

/// Per-frame diagnostics returned by [`PcenAgc::process_frame`].
#[derive(Debug, Clone, Copy)]
pub struct PcenFrameStats {
    /// Smoothed gain actually applied to this frame.
    pub gain_applied: f32,
    /// √signal_envelope (post-update).
    pub signal_rms: f32,
    /// √noise_envelope (post-update).
    pub noise_rms: f32,
    /// 10·log10(signal/noise). Saturates at ±120 dB on degenerate input.
    pub snr_db: f32,
}

/// Noise-aware adaptive AGC with per-hop scalar gain.
pub struct PcenAgc {
    cfg: PcenAgcConfig,
    hop_samples: usize,
    s_signal: f32,
    s_noise_up: f32,
    s_noise_down: f32,
    s_gain: f32,
    sig_env_pow: f32,
    noise_env_pow: f32,
    last_gain: f32,
    warmup_remaining: u32,
    initialized: bool,
    half_alpha: f32,
}

impl PcenAgc {
    /// Build an AGC for the given hop size at `sample_rate` Hz.
    pub fn new(
        cfg: PcenAgcConfig,
        sample_rate: u32,
        hop_samples: usize,
    ) -> Result<Self, AudioError> {
        if sample_rate == 0 {
            return Err(AudioError::Vad("pcen: sample_rate must be > 0".into()));
        }
        if hop_samples == 0 {
            return Err(AudioError::Vad("pcen: hop_samples must be > 0".into()));
        }
        validate(&cfg)?;
        let hop_ms = (hop_samples as f32) * 1_000.0 / (sample_rate as f32);
        let warmup_remaining = cfg.warmup_frames;
        let half_alpha = 0.5 * cfg.alpha;
        Ok(Self {
            s_signal: coeff(hop_ms, cfg.tau_signal_ms),
            s_noise_up: coeff(hop_ms, cfg.tau_noise_up_ms),
            s_noise_down: coeff(hop_ms, cfg.tau_noise_down_ms),
            s_gain: coeff(hop_ms, cfg.tau_gain_ms),
            cfg,
            hop_samples,
            sig_env_pow: 0.0,
            noise_env_pow: 0.0,
            last_gain: 1.0,
            warmup_remaining,
            initialized: false,
            half_alpha,
        })
    }

    /// Reset internal state. Next frame re-seeds the envelopes.
    pub fn reset(&mut self) {
        self.sig_env_pow = 0.0;
        self.noise_env_pow = 0.0;
        self.last_gain = 1.0;
        self.warmup_remaining = self.cfg.warmup_frames;
        self.initialized = false;
    }

    /// Process one frame in place. `frame.len()` must equal `hop_samples`.
    pub fn process_frame(&mut self, frame: &mut [f32]) -> PcenFrameStats {
        debug_assert_eq!(
            frame.len(),
            self.hop_samples,
            "pcen: frame length must equal hop_samples"
        );

        let frame_pow = mean_square_finite(frame).max(self.cfg.eps);

        if !self.initialized {
            self.sig_env_pow = frame_pow;
            self.noise_env_pow = frame_pow;
            self.last_gain = 1.0;
            self.initialized = true;
        } else {
            self.sig_env_pow = lerp(self.sig_env_pow, frame_pow, self.s_signal);
            let s_noise = if frame_pow < self.noise_env_pow {
                self.s_noise_down
            } else {
                self.s_noise_up
            };
            self.noise_env_pow = lerp(self.noise_env_pow, frame_pow, s_noise);
        }
        self.sig_env_pow = self.sig_env_pow.max(self.cfg.eps);
        self.noise_env_pow = self.noise_env_pow.max(self.cfg.eps);

        if self.warmup_remaining > 0 {
            // Track envelopes but hold gain at unity until trackers settle.
            self.warmup_remaining -= 1;
            self.last_gain = 1.0;
        } else {
            // desired_gain: how much we'd amplify to hit target signal RMS.
            let sig_norm = self.sig_env_pow.powf(self.half_alpha).max(self.cfg.eps);
            let desired_gain = self.cfg.target_rms / sig_norm;

            // noise_cap: how much we *can* amplify before noise exceeds its ceiling.
            let noise_rms_now = self.noise_env_pow.sqrt().max(self.cfg.eps);
            let noise_cap = self.cfg.noise_ceiling_rms / noise_rms_now;

            let mut raw_gain = desired_gain.min(noise_cap).min(self.cfg.max_gain);
            raw_gain = raw_gain.max(self.cfg.min_gain);
            if !raw_gain.is_finite() {
                raw_gain = 1.0;
            }

            self.last_gain = lerp(self.last_gain, raw_gain, self.s_gain);
            self.last_gain = self.last_gain.clamp(self.cfg.min_gain, self.cfg.max_gain);
            if !self.last_gain.is_finite() {
                self.last_gain = 1.0;
            }
        }

        let g = self.last_gain;
        for x in frame.iter_mut() {
            let v = if x.is_finite() { *x * g } else { 0.0 };
            *x = v.clamp(-1.0, 1.0);
        }

        let sig_rms = self.sig_env_pow.sqrt();
        let noise_rms = self.noise_env_pow.sqrt();
        let ratio = (self.sig_env_pow / self.noise_env_pow).max(self.cfg.eps);
        let snr_db = 10.0 * ratio.log10();
        PcenFrameStats {
            gain_applied: g,
            signal_rms: sig_rms,
            noise_rms,
            snr_db: snr_db.clamp(-120.0, 120.0),
        }
    }

    pub fn signal_rms(&self) -> f32 {
        self.sig_env_pow.sqrt()
    }
    pub fn noise_rms(&self) -> f32 {
        self.noise_env_pow.sqrt()
    }
    pub fn last_gain(&self) -> f32 {
        self.last_gain
    }
    pub fn hop_samples(&self) -> usize {
        self.hop_samples
    }
    pub fn warmup_remaining(&self) -> u32 {
        self.warmup_remaining
    }
}

fn validate(cfg: &PcenAgcConfig) -> Result<(), AudioError> {
    let bad = |k: &str| Err(AudioError::Vad(format!("pcen: invalid {k}")));
    if !cfg.target_rms.is_finite() || cfg.target_rms <= 0.0 {
        return bad("target_rms");
    }
    if !cfg.noise_ceiling_rms.is_finite() || cfg.noise_ceiling_rms <= 0.0 {
        return bad("noise_ceiling_rms");
    }
    if !cfg.max_gain.is_finite() || cfg.max_gain <= 0.0 {
        return bad("max_gain");
    }
    if !cfg.min_gain.is_finite() || cfg.min_gain <= 0.0 || cfg.min_gain > cfg.max_gain {
        return bad("min_gain");
    }
    if !cfg.alpha.is_finite() || cfg.alpha <= 0.0 || cfg.alpha > 1.0 {
        return bad("alpha");
    }
    for (k, v) in [
        ("tau_signal_ms", cfg.tau_signal_ms),
        ("tau_noise_up_ms", cfg.tau_noise_up_ms),
        ("tau_noise_down_ms", cfg.tau_noise_down_ms),
        ("tau_gain_ms", cfg.tau_gain_ms),
    ] {
        if !v.is_finite() || v <= 0.0 {
            return bad(k);
        }
    }
    if !cfg.eps.is_finite() || cfg.eps <= 0.0 {
        return bad("eps");
    }
    Ok(())
}

#[inline]
fn coeff(hop_ms: f32, tau_ms: f32) -> f32 {
    let s = 1.0 - (-hop_ms / tau_ms).exp();
    s.clamp(0.0, 1.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn mean_square_finite(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    // Per-sample finite gate: a single non-finite sample is dropped from the
    // sum rather than collapsing the whole frame's energy to zero. Division is
    // by the full frame length, matching the original contract (and the
    // identical guard in `vad::compute_rms`). The branch costs some SIMD width,
    // but PCEN runs at audio rate where it's negligible.
    let sum_sq: f32 = frame
        .iter()
        .map(|&x| if x.is_finite() { x * x } else { 0.0 })
        .sum();
    let m = sum_sq / frame.len() as f32;
    if m.is_finite() { m } else { 0.0 }
}

#[cfg(test)]
mod tests;
