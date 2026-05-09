//! Microphone calibration: derive VAD thresholds from short audio samples.
//!
//! The caller collects two vectors of per-frame VAD scores (0.0–1.0):
//! one with the user staying quiet (room noise) and one with the user
//! reading a prompt that contains intra-sentence pauses. [`recommend`]
//! returns a pair of thresholds tuned to that mic + voice + room.

const MIN_NOISE_FRAMES: usize = 60;
const MIN_SPEECH_FRAMES: usize = 250;
const MIN_SEPARATION: f32 = 0.20;

const ACTIVE_WINDOW: usize = 10;
const ACTIVE_THRESHOLD: f32 = 0.4;
const DIP_MIN_FRAMES: usize = 5;
const DIP_MAX_FRAMES: usize = 60;

const FALLBACK_PAUSE_OFFSET: f32 = 0.10;
const START_NOISE_MARGIN: f32 = 0.10;
const START_SPEECH_MARGIN: f32 = 0.05;
const END_NOISE_MARGIN: f32 = 0.05;
const END_PAUSE_MARGIN: f32 = 0.05;
const HYSTERESIS_GAP: f32 = 0.05;

const START_FLOOR: f32 = 0.30;
const START_CEILING: f32 = 0.80;
const END_FLOOR: f32 = 0.10;

#[derive(Debug, Clone)]
pub struct CalibrationInput {
    pub noise_scores: Vec<f32>,
    pub speech_scores: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationResult {
    pub speech_threshold: f32,
    pub speech_threshold_end: f32,
    pub noise_p95: f32,
    pub speech_p10: f32,
    pub pause_p20: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("noise sample too short ({0} frames, need ≥ {1})")]
    NoiseTooShort(usize, usize),
    #[error("speech sample too short ({0} frames, need ≥ {1})")]
    SpeechTooShort(usize, usize),
    #[error("insufficient signal: speech_p10 ({0:.2}) - noise_p95 ({1:.2}) < {2:.2}")]
    InsufficientSignal(f32, f32, f32),
}

/// Compute recommended VAD thresholds from per-frame score vectors.
pub fn recommend(input: &CalibrationInput) -> Result<CalibrationResult, CalibrationError> {
    if input.noise_scores.len() < MIN_NOISE_FRAMES {
        return Err(CalibrationError::NoiseTooShort(
            input.noise_scores.len(),
            MIN_NOISE_FRAMES,
        ));
    }
    if input.speech_scores.len() < MIN_SPEECH_FRAMES {
        return Err(CalibrationError::SpeechTooShort(
            input.speech_scores.len(),
            MIN_SPEECH_FRAMES,
        ));
    }

    let noise_p95 = percentile(&input.noise_scores, 0.95);
    let active = active_speech_scores(&input.speech_scores);
    let speech_p10 = if active.is_empty() {
        percentile(&input.speech_scores, 0.10)
    } else {
        percentile(&active, 0.10)
    };

    if speech_p10 - noise_p95 < MIN_SEPARATION {
        return Err(CalibrationError::InsufficientSignal(
            speech_p10,
            noise_p95,
            MIN_SEPARATION,
        ));
    }

    let pause_p20 = pause_scores(&input.speech_scores)
        .map(|s| percentile(&s, 0.20))
        .unwrap_or(noise_p95 + FALLBACK_PAUSE_OFFSET);

    let start_upper = (speech_p10 - START_SPEECH_MARGIN).min(START_CEILING);
    let start = clamp(noise_p95 + START_NOISE_MARGIN, START_FLOOR, start_upper);
    let end_floor_candidate = (noise_p95 + END_NOISE_MARGIN).max(pause_p20 - END_PAUSE_MARGIN);
    let end = clamp(end_floor_candidate, END_FLOOR, start - HYSTERESIS_GAP);

    Ok(CalibrationResult {
        speech_threshold: round2(start),
        speech_threshold_end: round2(end),
        noise_p95: round2(noise_p95),
        speech_p10: round2(speech_p10),
        pause_p20: round2(pause_p20),
    })
}

fn percentile(values: &[f32], p: f32) -> f32 {
    debug_assert!(!values.is_empty());
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn rolling_mean(values: &[f32], window: usize) -> Vec<f32> {
    if values.is_empty() || window == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut sum = 0.0f32;
    for (i, &v) in values.iter().enumerate() {
        sum += v;
        if i >= window {
            sum -= values[i - window];
        }
        let len = (i + 1).min(window) as f32;
        out.push(sum / len);
    }
    out
}

fn active_speech_scores(scores: &[f32]) -> Vec<f32> {
    let smoothed = rolling_mean(scores, ACTIVE_WINDOW);
    scores
        .iter()
        .zip(smoothed.iter())
        .filter_map(|(&s, &m)| if m > ACTIVE_THRESHOLD { Some(s) } else { None })
        .collect()
}

/// Collect VAD scores from "intra-utterance dips": short low-score regions
/// bounded on both sides by active-speech regions. Returns None if the
/// speech sample contains no qualifying dip (caller falls back to a
/// noise-derived estimate).
fn pause_scores(scores: &[f32]) -> Option<Vec<f32>> {
    let smoothed = rolling_mean(scores, ACTIVE_WINDOW);
    let active: Vec<bool> = smoothed.iter().map(|&m| m > ACTIVE_THRESHOLD).collect();

    let mut out = Vec::new();
    let mut i = 0;
    while i < active.len() {
        if active[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < active.len() && !active[i] {
            i += 1;
        }
        let end = i;
        let len = end - start;
        let bounded_left = start > 0 && active[start - 1];
        let bounded_right = end < active.len() && active[end];
        if bounded_left && bounded_right && (DIP_MIN_FRAMES..=DIP_MAX_FRAMES).contains(&len) {
            out.extend_from_slice(&scores[start..end]);
        }
    }

    if out.is_empty() { None } else { Some(out) }
}

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if hi < lo { lo } else { v.clamp(lo, hi) }
}

fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(value: f32, n: usize) -> Vec<f32> {
        vec![value; n]
    }

    #[test]
    fn noise_too_short_rejected() {
        let r = recommend(&CalibrationInput {
            noise_scores: vec_of(0.05, 10),
            speech_scores: vec_of(0.8, 300),
        });
        assert!(matches!(r, Err(CalibrationError::NoiseTooShort(..))));
    }

    #[test]
    fn speech_too_short_rejected() {
        let r = recommend(&CalibrationInput {
            noise_scores: vec_of(0.05, 100),
            speech_scores: vec_of(0.8, 100),
        });
        assert!(matches!(r, Err(CalibrationError::SpeechTooShort(..))));
    }

    #[test]
    fn insufficient_signal_rejected() {
        let r = recommend(&CalibrationInput {
            noise_scores: vec_of(0.55, 100),
            speech_scores: vec_of(0.60, 300),
        });
        assert!(matches!(r, Err(CalibrationError::InsufficientSignal(..))));
    }

    #[test]
    fn quiet_room_clear_voice_recommends_low_thresholds() {
        let mut speech = vec_of(0.85, 600);
        for s in speech.iter_mut().take(230).skip(200) {
            *s = 0.20;
        }
        let r = recommend(&CalibrationInput {
            noise_scores: vec_of(0.05, 100),
            speech_scores: speech,
        })
        .unwrap();
        assert!(r.speech_threshold >= 0.30 && r.speech_threshold <= 0.55);
        assert!(r.speech_threshold_end < r.speech_threshold);
        assert!(r.speech_threshold_end >= END_FLOOR);
    }

    #[test]
    fn noisy_room_clear_voice_raises_start_threshold() {
        let mut speech = vec_of(0.90, 600);
        for s in speech.iter_mut().take(225).skip(200) {
            *s = 0.30;
        }
        let r = recommend(&CalibrationInput {
            noise_scores: vec_of(0.30, 100),
            speech_scores: speech,
        })
        .unwrap();
        assert!(r.speech_threshold > 0.35);
        assert!(r.speech_threshold_end >= 0.30);
        assert!(r.speech_threshold_end < r.speech_threshold);
    }

    #[test]
    fn no_intra_speech_dip_uses_noise_fallback() {
        let r = recommend(&CalibrationInput {
            noise_scores: vec_of(0.05, 100),
            speech_scores: vec_of(0.85, 400),
        })
        .unwrap();
        let expected_pause = (0.05_f32 + FALLBACK_PAUSE_OFFSET).max(END_FLOOR);
        assert!((r.pause_p20 - expected_pause).abs() < 0.05);
        assert!(r.speech_threshold_end < r.speech_threshold);
    }

    #[test]
    fn hysteresis_gap_always_present() {
        let r = recommend(&CalibrationInput {
            noise_scores: vec_of(0.05, 100),
            speech_scores: vec_of(0.95, 300),
        })
        .unwrap();
        assert!((r.speech_threshold - r.speech_threshold_end) >= HYSTERESIS_GAP - 0.01);
    }

    #[test]
    fn percentile_handles_unsorted() {
        let v = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        assert!((percentile(&v, 0.0) - 0.1).abs() < 1e-6);
        assert!((percentile(&v, 1.0) - 0.9).abs() < 1e-6);
        assert!((percentile(&v, 0.5) - 0.5).abs() < 1e-6);
    }
}
