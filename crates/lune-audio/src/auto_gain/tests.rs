// SPDX-License-Identifier: GPL-3.0-or-later

use super::*;

const SR: u32 = 16_000;

/// Peak amplitude of a sine whose RMS is the default target (0.1 = −20 dBFS).
const AT_TARGET: f32 = 0.1414;

/// Speech-like tone bursts: 200 ms on, 100 ms off, for `secs`.
fn speech(amp: f32, secs: f32) -> Vec<f32> {
    let n = (SR as f32 * secs) as usize;
    let on = SR as usize / 5;
    let period = on + SR as usize / 10;
    (0..n)
        .map(|i| {
            if i % period < on {
                amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SR as f32).sin()
            } else {
                0.0
            }
        })
        .collect()
}

fn noise(rms: f32, n: usize) -> Vec<f32> {
    let mut state: u32 = 0x5EED;
    // Uniform in [-1, 1] has RMS 1/√3.
    let scale = rms * 3f32.sqrt();
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn scaled(x: &[f32], g: f32) -> Vec<f32> {
    x.iter().map(|s| s * g).collect()
}

fn cfg() -> PcenAgcConfig {
    PcenAgcConfig::default()
}

fn target_db() -> f32 {
    20.0 * cfg().target_rms.log10()
}

#[test]
fn measures_speech_level_and_digital_silence_floor() {
    let p = LevelProfile::measure(&speech(AT_TARGET, 3.0), SR);
    assert!((p.speech_db - target_db()).abs() < 1.0, "{p:?}");
    assert!(p.noise_db < -100.0, "{p:?}");
    assert!(!p.clipped);
}

#[test]
fn a_clean_clip_near_the_target_is_left_untouched() {
    let clip = speech(0.12, 3.0);
    let mut out = clip.clone();
    assert_eq!(apply_auto(&mut out, SR, &cfg()).unwrap(), GainPlan::Bypass);
    assert_eq!(out, clip, "bypass must not change a sample");
}

#[test]
fn a_clean_clip_within_3_db_of_target_is_still_untouched() {
    for g in [0.85, 1.25] {
        let p = LevelProfile::measure(&speech(AT_TARGET * g, 3.0), SR);
        assert_eq!(plan(&p, &cfg()), GainPlan::Bypass, "gain {g}: {p:?}");
    }
}

#[test]
fn a_clean_clip_further_off_target_gets_full_pcen() {
    for g in [0.6, 0.1, 5.0] {
        let p = LevelProfile::measure(&speech(AT_TARGET * g, 3.0), SR);
        assert_eq!(
            plan(&p, &cfg()),
            GainPlan::Adaptive { strength: 1.0 },
            "gain {g}: {p:?}"
        );
    }
}

#[test]
fn full_pcen_brings_a_quiet_clean_clip_up() {
    let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
    let clip = speech(AT_TARGET * 0.1, 3.0);
    let mut out = clip.clone();
    assert_eq!(
        apply_auto(&mut out, SR, &cfg()).unwrap(),
        GainPlan::Adaptive { strength: 1.0 }
    );
    assert!(rms(&out) > 2.0 * rms(&clip));
}

#[test]
fn a_noisy_clip_gets_pcen_and_more_of_it_as_snr_falls() {
    let n = 3 * SR as usize;
    let voice = speech(AT_TARGET, 3.0);
    let strength = |noise_rms: f32| {
        let p = LevelProfile::measure(&mix(&voice, &noise(noise_rms, n)), SR);
        match plan(&p, &cfg()) {
            GainPlan::Adaptive { strength } => (strength, p),
            other => panic!("expected PCEN for {p:?}, got {other:?}"),
        }
    };
    let (mild, p_mild) = strength(0.025);
    let (harsh, p_harsh) = strength(0.06);
    assert!(p_mild.snr_db() > p_harsh.snr_db());
    assert!(mild > 0.0 && mild < 1.0, "{mild}");
    assert!(harsh > mild, "harsh {harsh} vs mild {mild}");
    assert!(harsh <= 1.0);
}

fn profile(speech_db: f32, noise_db: f32) -> LevelProfile {
    LevelProfile {
        speech_db,
        noise_db,
        clipped: false,
    }
}

fn strength_of(p: LevelProfile) -> f32 {
    match plan(&p, &cfg()) {
        GainPlan::Bypass => 0.0,
        GainPlan::Adaptive { strength } => strength,
    }
}

/// Crossing the clean SNR threshold must not step the gain: a near-target
/// clip ramps up from untouched, and an off-target one stays at full PCEN.
#[test]
fn strength_is_continuous_across_the_clean_snr_threshold() {
    let t = target_db();
    let near_below = strength_of(profile(t, t - CLEAN_SNR_DB + 0.1));
    let near_above = strength_of(profile(t, t - CLEAN_SNR_DB - 0.1));
    assert!(near_below < 0.02, "{near_below}");
    assert_eq!(near_above, 0.0);

    let quiet = t - 20.0;
    for snr in [CLEAN_SNR_DB - 0.1, CLEAN_SNR_DB + 0.1, 5.0, 60.0] {
        assert_eq!(strength_of(profile(quiet, quiet - snr)), 1.0, "snr {snr}");
    }
}

#[test]
fn a_very_noisy_near_target_clip_gets_full_pcen() {
    let t = target_db();
    assert_eq!(strength_of(profile(t, t - 5.0)), 1.0);
}

#[test]
fn a_clipped_clip_is_never_boosted() {
    let mut clip = speech(0.014, 3.0);
    for s in clip.iter_mut().step_by(200) {
        *s = 1.0;
    }
    let p = LevelProfile::measure(&clip, SR);
    assert!(p.clipped, "{p:?}");
    assert_eq!(plan(&p, &cfg()), GainPlan::Bypass);
}

#[test]
fn silence_and_empty_input_are_left_alone() {
    let mut silent = vec![0.0f32; SR as usize];
    assert_eq!(
        apply_auto(&mut silent, SR, &cfg()).unwrap(),
        GainPlan::Bypass
    );
    let mut empty: Vec<f32> = Vec::new();
    assert_eq!(
        apply_auto(&mut empty, SR, &cfg()).unwrap(),
        GainPlan::Bypass
    );
}

#[test]
fn a_clip_shorter_than_one_window_is_still_measured() {
    let p = LevelProfile::measure(&scaled(&speech(AT_TARGET, 0.2), 0.1), SR);
    assert!((p.speech_db - (target_db() - 20.0)).abs() < 2.0, "{p:?}");
}

#[test]
fn adaptive_output_stays_finite_and_in_range() {
    let n = 3 * SR as usize;
    let mut clip = mix(&speech(0.02, 3.0), &noise(0.01, n));
    clip[17] = f32::NAN;
    let plan = apply_auto(&mut clip, SR, &cfg()).unwrap();
    assert!(matches!(plan, GainPlan::Adaptive { .. }), "{plan:?}");
    assert!(
        clip.iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s))
    );
}
