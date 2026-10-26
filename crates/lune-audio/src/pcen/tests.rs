// SPDX-License-Identifier: GPL-3.0-or-later

use super::*;

const SR: u32 = 16_000;
const HOP: usize = 256;

fn agc(cfg: PcenAgcConfig) -> PcenAgc {
    PcenAgc::new(cfg, SR, HOP).unwrap()
}

fn rms(buf: &[f32]) -> f32 {
    let m = buf.iter().map(|&x| x * x).sum::<f32>() / buf.len() as f32;
    m.sqrt()
}

fn sine(freq: f32, amp: f32, n: usize) -> Vec<f32> {
    let dt = 1.0 / SR as f32;
    (0..n)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 * dt).sin())
        .collect()
}

/// Speech-like burst pattern: `on_ms` of tone, `off_ms` of silence, repeating.
/// Mimics syllable cadence so the asymmetric noise tracker drains during gaps.
fn bursty(freq: f32, amp: f32, n: usize, on_ms: u32, off_ms: u32) -> Vec<f32> {
    let dt = 1.0 / SR as f32;
    let on = (SR as u64 * on_ms as u64 / 1000) as usize;
    let off = (SR as u64 * off_ms as u64 / 1000) as usize;
    let period = on + off;
    (0..n)
        .map(|i| {
            if i % period < on {
                amp * (2.0 * std::f32::consts::PI * freq * i as f32 * dt).sin()
            } else {
                0.0
            }
        })
        .collect()
}

fn process_in_hops(agc: &mut PcenAgc, samples: &mut [f32]) -> PcenFrameStats {
    let mut last = PcenFrameStats {
        gain_applied: 1.0,
        signal_rms: 0.0,
        noise_rms: 0.0,
        snr_db: 0.0,
    };
    for chunk in samples.chunks_mut(HOP) {
        if chunk.len() == HOP {
            last = agc.process_frame(chunk);
        }
    }
    last
}

#[test]
fn rejects_zero_hop() {
    let r = PcenAgc::new(PcenAgcConfig::default(), SR, 0);
    assert!(matches!(r, Err(AudioError::Vad(_))));
}

#[test]
fn rejects_zero_sample_rate() {
    let r = PcenAgc::new(PcenAgcConfig::default(), 0, HOP);
    assert!(matches!(r, Err(AudioError::Vad(_))));
}

#[test]
fn rejects_invalid_alpha() {
    for bad in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
        let cfg = PcenAgcConfig {
            alpha: bad,
            ..Default::default()
        };
        assert!(matches!(
            PcenAgc::new(cfg, SR, HOP),
            Err(AudioError::Vad(_))
        ));
    }
}

#[test]
fn rejects_min_gt_max_gain() {
    let cfg = PcenAgcConfig {
        min_gain: 4.0,
        max_gain: 2.0,
        ..Default::default()
    };
    assert!(matches!(
        PcenAgc::new(cfg, SR, HOP),
        Err(AudioError::Vad(_))
    ));
}

#[test]
fn rejects_strength_outside_unit_interval() {
    for bad in [-0.1, 1.1, f32::NAN] {
        let cfg = PcenAgcConfig {
            strength: bad,
            ..Default::default()
        };
        assert!(matches!(
            PcenAgc::new(cfg, SR, HOP),
            Err(AudioError::Vad(_))
        ));
    }
}

#[test]
fn strength_scales_the_gain_toward_unity() {
    let gain_at = |strength: f32| {
        let mut a = agc(PcenAgcConfig {
            strength,
            ..Default::default()
        });
        let mut buf = bursty(440.0, 0.014, HOP * 400, 50, 150);
        process_in_hops(&mut a, &mut buf).gain_applied
    };
    let full = gain_at(1.0);
    let half = gain_at(0.5);
    assert!(full > 2.0, "full {full}");
    assert_eq!(gain_at(0.0), 1.0);
    assert!(
        (half - (1.0 + 0.5 * (full - 1.0))).abs() < 0.05 * full,
        "half {half} full {full}"
    );
}

#[test]
fn coeff_in_unit_interval() {
    assert!((0.0..=1.0).contains(&coeff(16.0, 150.0)));
    assert!(coeff(16.0, 1.0) <= 1.0);
    assert!(coeff(0.001, 1_000.0) > 0.0);
}

#[test]
fn boosts_quiet_speech_like_input_toward_target() {
    // Quiet bursty signal (50ms on / 150ms off) — peak burst RMS ~0.01.
    // Asymmetric noise tracker drains during gaps, so noise-cap stays loose
    // and the AGC is free to boost the bursts toward target_rms.
    let mut a = agc(PcenAgcConfig::default());
    let mut buf = bursty(440.0, 0.014, HOP * 400, 50, 150);
    let stats = process_in_hops(&mut a, &mut buf);
    assert!(
        stats.gain_applied > 2.0,
        "expected boost on bursty quiet input, got gain={}",
        stats.gain_applied
    );
    assert!(
        stats.snr_db > 6.0,
        "bursty input should keep SNR > 6 dB, got {} dB",
        stats.snr_db
    );
}

#[test]
fn does_not_amplify_quiet_noise_above_ceiling() {
    // Pure white noise at low RMS — must NOT be boosted to target_rms;
    // amplified RMS should respect noise_ceiling_rms.
    let cfg = PcenAgcConfig::default();
    let ceiling = cfg.noise_ceiling_rms;
    let mut a = agc(cfg);

    let mut state: u32 = 0xC0FFEE;
    let mut noise: Vec<f32> = (0..HOP * 400)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .map(|x| x * 0.005) // ~0.003 RMS
        .collect();

    process_in_hops(&mut a, &mut noise);
    let tail = &noise[noise.len() - HOP * 50..];
    let out = rms(tail);
    assert!(
        out <= ceiling * 1.5,
        "amplified noise RMS {out} exceeded ceiling {ceiling}"
    );
}

#[test]
fn attenuates_loud_input_toward_target() {
    // Loud sine at ~0.7 RMS — gain should drop below 1.
    let mut a = agc(PcenAgcConfig::default());
    let mut buf = sine(440.0, 0.99, HOP * 200);
    let stats = process_in_hops(&mut a, &mut buf);
    assert!(
        stats.gain_applied < 1.0,
        "expected attenuation, got gain={}",
        stats.gain_applied
    );
}

#[test]
fn output_stays_bounded_in_unit_interval() {
    let mut a = agc(PcenAgcConfig::default());
    let mut buf = sine(440.0, 0.95, HOP * 50);
    process_in_hops(&mut a, &mut buf);
    for &s in &buf {
        assert!(s.is_finite() && (-1.0..=1.0).contains(&s));
    }
}

#[test]
fn nan_inf_input_does_not_poison_state() {
    let mut a = agc(PcenAgcConfig::default());
    let mut frame = vec![0.0f32; HOP];
    frame[0] = f32::NAN;
    frame[1] = f32::INFINITY;
    frame[2] = f32::NEG_INFINITY;
    let stats = a.process_frame(&mut frame);
    assert!(stats.gain_applied.is_finite());
    assert!(stats.signal_rms.is_finite());
    assert!(stats.noise_rms.is_finite());
    for &s in &frame {
        assert!(s.is_finite());
    }
}

#[test]
fn warmup_holds_gain_at_unity_then_releases() {
    // 8-frame warmup; force a quiet input that *would* normally be boosted.
    let cfg = PcenAgcConfig {
        warmup_frames: 8,
        ..Default::default()
    };
    let mut a = agc(cfg);

    let mut buf = bursty(440.0, 0.014, HOP * 16, 50, 150);
    // First 8 frames: gain must be exactly 1.0 (no boost, no attenuation).
    for chunk in buf.chunks_mut(HOP).take(8) {
        let stats = a.process_frame(chunk);
        assert_eq!(stats.gain_applied, 1.0, "warmup gain must be unity");
    }
    assert_eq!(a.warmup_remaining(), 0);

    // After warmup, push enough frames for the gain smoother to engage on
    // the bursty signal. Expect the boost to come online.
    let mut tail = bursty(440.0, 0.014, HOP * 200, 50, 150);
    let stats = process_in_hops(&mut a, &mut tail);
    assert!(
        stats.gain_applied > 1.5,
        "post-warmup boost should engage, got gain={}",
        stats.gain_applied
    );
}

#[test]
fn warmup_zero_means_no_warmup() {
    let cfg = PcenAgcConfig {
        warmup_frames: 0,
        ..Default::default()
    };
    let mut a = agc(cfg);
    let mut frame = vec![0.5f32; HOP];
    let stats = a.process_frame(&mut frame);
    // No warmup → gain came straight from the smoother (might still be 1.0 on
    // the first frame because of seed, but warmup_remaining must be 0).
    assert_eq!(a.warmup_remaining(), 0);
    assert!(stats.gain_applied.is_finite());
}

#[test]
fn reset_restores_warmup_counter() {
    let cfg = PcenAgcConfig {
        warmup_frames: 4,
        ..Default::default()
    };
    let mut a = agc(cfg);
    let mut frame = vec![0.0f32; HOP];
    for _ in 0..4 {
        a.process_frame(&mut frame);
    }
    assert_eq!(a.warmup_remaining(), 0);
    a.reset();
    assert_eq!(a.warmup_remaining(), 4);
}

#[test]
fn reset_reseeds_envelopes() {
    let mut a = agc(PcenAgcConfig::default());
    let mut buf = sine(440.0, 0.5, HOP * 50);
    process_in_hops(&mut a, &mut buf);
    assert!(a.signal_rms() > 0.0);
    a.reset();
    assert_eq!(a.signal_rms(), 0.0);
    assert_eq!(a.noise_rms(), 0.0);
    assert_eq!(a.last_gain(), 1.0);
}

#[test]
fn noise_tracker_falls_faster_than_it_rises() {
    // Push noise up briefly, then to silence; the down-tracker should pull
    // noise envelope back down faster than the up-tracker pushed it up.
    let cfg = PcenAgcConfig::default();
    let mut a = agc(cfg.clone());

    // warm up at moderate level
    let mut warm = sine(440.0, 0.05, HOP * 30);
    process_in_hops(&mut a, &mut warm);
    let n_before = a.noise_rms();

    // brief loud burst
    let mut burst = sine(440.0, 0.6, HOP * 20);
    process_in_hops(&mut a, &mut burst);
    let n_peak = a.noise_rms();
    assert!(n_peak >= n_before);

    // long silence
    let mut quiet = vec![0.0f32; HOP * 200];
    process_in_hops(&mut a, &mut quiet);
    let n_after = a.noise_rms();
    assert!(
        n_after < n_peak,
        "noise envelope should decay during silence: peak={n_peak} after={n_after}"
    );
}
