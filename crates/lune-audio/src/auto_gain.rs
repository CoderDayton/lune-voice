// SPDX-License-Identifier: GPL-3.0-or-later

//! Auto mode for the PCEN AGC: measure the whole clip, then decide how much
//! gain control it gets.
//!
//! PCEN's gain moves frame by frame, which helps a quiet or noisy mic but
//! changes clean audio already at its target. The ASR backends decode the clip
//! in one batch at stop, so the decision can look at all of it first:
//!
//! - further than [`BYPASS_WINDOW_DB`] off the target level: full PCEN
//! - near the target: untouched while clean, then PCEN whose strength ramps
//!   from 0 at [`CLEAN_SNR_DB`] to 1 at [`FULL_STRENGTH_SNR_DB`], so a small
//!   change in noise never steps the gain
//! - clipped, or no speech: untouched
//!
//! A single fixed gain for off-target clips was tried and dropped: on real
//! recordings it scored like no gain at all, while PCEN cut word errors.

use crate::AudioError;
use crate::pcen::{PCEN_HOP_SIZE, PcenAgc, PcenAgcConfig, apply_gain};

/// Within this many dB of the target level, a clean clip is left alone.
pub const BYPASS_WINDOW_DB: f32 = 3.0;

/// At or above this SNR a clip counts as clean.
pub const CLEAN_SNR_DB: f32 = 20.0;

/// At or below this SNR a near-target clip gets full PCEN.
pub const FULL_STRENGTH_SNR_DB: f32 = 10.0;

/// Speech level below this means the clip holds nothing worth boosting.
const NO_SPEECH_DB: f32 = -70.0;

const NOISE_PERCENTILE: f32 = 0.10;
const SPEECH_PERCENTILE: f32 = 0.90;
const CLIP_LEVEL: f32 = 0.99;
const CLIP_FRACTION: f32 = 0.001;
const POWER_FLOOR: f32 = 1.0e-12;

/// Level statistics of one clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelProfile {
    /// 90th percentile of 250 ms window levels, dBFS.
    pub speech_db: f32,
    /// 10th percentile of 16 ms frame levels, dBFS.
    pub noise_db: f32,
    pub clipped: bool,
}

/// What auto mode does to a clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GainPlan {
    Bypass,
    Adaptive { strength: f32 },
}

impl LevelProfile {
    pub fn measure(samples: &[f32], sample_rate: u32) -> Self {
        let window = (sample_rate as usize / 4).max(1);
        let mut windows = block_db(samples, window);
        if windows.is_empty() {
            windows = block_db(samples, samples.len().max(1));
        }
        let mut frames = block_db(samples, PCEN_HOP_SIZE);
        if frames.is_empty() {
            frames.clone_from(&windows);
        }
        let clipped = samples
            .iter()
            .filter(|s| s.is_finite() && s.abs() >= CLIP_LEVEL)
            .count();
        Self {
            speech_db: percentile(&mut windows, SPEECH_PERCENTILE),
            noise_db: percentile(&mut frames, NOISE_PERCENTILE),
            clipped: clipped as f32 > CLIP_FRACTION * samples.len() as f32,
        }
    }

    pub fn snr_db(&self) -> f32 {
        self.speech_db - self.noise_db
    }
}

pub fn plan(profile: &LevelProfile, cfg: &PcenAgcConfig) -> GainPlan {
    if profile.clipped || profile.speech_db < NO_SPEECH_DB {
        return GainPlan::Bypass;
    }
    let off_target_db = profile.speech_db - 20.0 * cfg.target_rms.log10();
    if off_target_db.abs() > BYPASS_WINDOW_DB {
        return GainPlan::Adaptive { strength: 1.0 };
    }
    let strength =
        ((CLEAN_SNR_DB - profile.snr_db()) / (CLEAN_SNR_DB - FULL_STRENGTH_SNR_DB)).clamp(0.0, 1.0);
    if strength > 0.0 {
        GainPlan::Adaptive { strength }
    } else {
        GainPlan::Bypass
    }
}

/// Measure `samples`, plan, and apply the plan in place.
pub fn apply_auto(
    samples: &mut [f32],
    sample_rate: u32,
    cfg: &PcenAgcConfig,
) -> Result<GainPlan, AudioError> {
    let plan = plan(&LevelProfile::measure(samples, sample_rate), cfg);
    match plan {
        GainPlan::Bypass => {}
        GainPlan::Adaptive { strength } => {
            let cfg = PcenAgcConfig {
                strength,
                ..cfg.clone()
            };
            let mut agc = PcenAgc::new(cfg, sample_rate, PCEN_HOP_SIZE)?;
            let mut frames = samples.chunks_exact_mut(PCEN_HOP_SIZE);
            for frame in &mut frames {
                agc.process_frame(frame);
            }
            apply_gain(frames.into_remainder(), agc.last_gain());
        }
    }
    Ok(plan)
}

/// Level of each whole `block`-sample block, dBFS; a trailing partial block
/// is dropped. Non-finite samples count as silence.
fn block_db(samples: &[f32], block: usize) -> Vec<f32> {
    samples
        .chunks_exact(block)
        .map(|b| {
            let power = b
                .iter()
                .map(|&x| if x.is_finite() { x * x } else { 0.0 })
                .sum::<f32>()
                / block as f32;
            10.0 * power.max(POWER_FLOOR).log10()
        })
        .collect()
}

/// Nearest-rank percentile; the floor level for an empty set.
fn percentile(values: &mut [f32], p: f32) -> f32 {
    if values.is_empty() {
        return 10.0 * POWER_FLOOR.log10();
    }
    values.sort_unstable_by(f32::total_cmp);
    let rank = ((p * values.len() as f32).ceil() as usize).clamp(1, values.len());
    values[rank - 1]
}

#[cfg(test)]
mod tests;
