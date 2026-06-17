//! Open-vocabulary LLM-edit calibration (CRC).
//!
//! Gates the arbitrary edits the LLM correction proposes, keyed on the
//! decode-confidence margin, via the CRC kernel in [`crate::crc`] — so the
//! over-correction rate is conformally bounded.
//!
//! Observations are collected at dictation time (the LLM's per-word decode
//! margin, with the delivered text as truth) and persisted; calibration is
//! recomputed cheaply at startup from the stored observations.

use crate::crc;

/// The LLM-edit gate's source of truth. Two knobs fully determine the gate, and
/// every consumer — calibration, the live `EditGate`, the seed generator — reads
/// them from HERE, so there is no scattered magic number to drift. To change the
/// gate's behaviour, change a value in this block.
///
/// `ALPHA`: target per-edit over-correction rate (the conformal budget).
/// `TAU_FLOOR`: minimum decode margin (nats) the gate will ever trust. Floors
/// the conformal τ before enablement is judged, so the channel cannot enable at
/// the degenerate τ=0 that CRC's "no binding constraint" branch returns when
/// over-correction samples are sparse (a synthetic seed, or a careful user).
/// Below ~2 nats the margin is not decisive (good fixes score ≥4,
/// hallucinations ≤0.5).
pub const ALPHA: f64 = crc::DEFAULT_ALPHA;
pub const TAU_FLOOR: f64 = 2.0;

/// One LLM-edit observation for the single global edit channel (no
/// confusion-set index).
#[derive(Debug, Clone, Copy)]
pub struct EditObs {
    /// The raw word already equalled the delivered (truth) word — editing it
    /// risks over-correction.
    pub observed_correct: bool,
    /// Decode margin of the LLM's proposed word; `None` when the LLM proposed
    /// no edit at this slot (still counts toward the conformal denominator).
    pub margin: Option<f64>,
    /// The LLM's proposed word equals the delivered word (a beneficial flip).
    pub flip_fixes: bool,
}

/// Calibration for the single open-vocab LLM-edit channel.
#[derive(Debug, Clone)]
pub struct EditCalibration {
    pub enabled: bool,
    pub tau: f64,
    pub net_benefit: f64,
    pub n_correct: usize,
    pub n_error: usize,
}

/// Calibrate the LLM-edit channel from `(raw → delivered)` edit observations:
/// a conformal margin threshold that bounds over-correction at `alpha`, enabled
/// only under a Wilson net-benefit lower bound. Returns `None` when no data.
///
/// `min_tau` floors the threshold before enablement is judged: CRC's "no binding
/// constraint" branch returns τ=0 when over-correction samples are sparse, which
/// would admit every edit. Flooring first means the channel only enables if it
/// is net-beneficial at the threshold it will actually operate at. Pass `0.0`
/// for no floor.
pub fn calibrate_edits(obs: &[EditObs], alpha: f64, min_tau: f64) -> Option<EditCalibration> {
    if obs.is_empty() {
        return None;
    }
    let n_correct = obs.iter().filter(|o| o.observed_correct).count();
    let n_error = obs.len() - n_correct;

    // Correct-case margins are the only ones that can cause over-correction.
    let correct_margins: Vec<f64> = obs
        .iter()
        .filter(|o| o.observed_correct)
        .filter_map(|o| o.margin)
        .collect();
    let tau = crc::conformal_threshold(&correct_margins, n_correct, alpha).max(min_tau);
    // No finite threshold can certify the budget yet (cold start) → no
    // calibration, identical to the no-data path; the gate stays off.
    if !tau.is_finite() {
        return None;
    }

    let over = correct_margins.iter().filter(|&&m| m > tau).count();
    let fixed = obs
        .iter()
        .filter(|o| !o.observed_correct && o.flip_fixes)
        .filter(|o| o.margin.is_some_and(|m| m > tau))
        .count();
    let net = fixed as f64 - over as f64;
    let decisive = fixed + over;
    let enabled = crc::is_enabled(fixed, decisive, net);

    Some(EditCalibration {
        enabled,
        tau,
        net_benefit: net,
        n_correct,
        n_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(observed_correct: bool, margin: Option<f64>, flip_fixes: bool) -> EditObs {
        EditObs {
            observed_correct,
            margin,
            flip_fixes,
        }
    }

    #[test]
    fn empty_history_is_none() {
        assert!(calibrate_edits(&[], 0.05, 0.0).is_none());
    }

    #[test]
    fn cold_start_no_calibration_fail_safe() {
        // Too few observations to certify at alpha → tau infinite → no
        // calibration (gate stays off), same as the no-data path.
        let data = vec![obs(true, Some(1.0), false), obs(false, Some(5.0), true)];
        assert!(calibrate_edits(&data, 0.01, 0.0).is_none());
    }

    #[test]
    fn separable_channel_enables_and_bounds() {
        // Good fixes carry high margins, over-corrections low — separable.
        let mut data = Vec::new();
        for _ in 0..80 {
            data.push(obs(false, Some(5.0), true)); // error fixed, high margin
            data.push(obs(true, Some(0.2), false)); // correct word, low margin
            data.push(obs(true, None, false)); // correct word, no edit proposed
        }
        let c = calibrate_edits(&data, 0.05, 0.0).unwrap();
        assert!(
            c.enabled,
            "separable net-beneficial channel must enable: {c:?}"
        );
        assert!(c.tau.is_finite());
        // tau sits at/above the corruption margins, below the fix margins.
        assert!(c.tau >= 0.2 && c.tau < 5.0, "tau={}", c.tau);
    }

    #[test]
    fn tau_never_below_floor() {
        // Any non-empty, non-cold-start calibration operates at >= TAU_FLOOR.
        let mut data = Vec::new();
        for _ in 0..300 {
            data.push(obs(true, None, false));
            data.push(obs(true, Some(0.1), false)); // low-margin over-risk
        }
        for _ in 0..20 {
            data.push(obs(false, Some(6.0), true));
        }
        let c = calibrate_edits(&data, ALPHA, TAU_FLOOR).unwrap();
        assert!(c.tau.is_finite());
        assert!(c.tau >= TAU_FLOOR, "tau {} below floor {TAU_FLOOR}", c.tau);
    }

    #[test]
    fn enablement_judged_at_floored_tau_not_degenerate_zero() {
        // Regression for the production bug: many already-correct words plus
        // sparse low-margin over-corrections collapse the raw conformal τ to 0.
        // At τ=0 the over-corrections outnumber fixes → net ≤ 0 → would disable.
        // Judged at the floor, the low-margin over-corrections drop out → net > 0
        // → enables. This dataset mirrors the real 845-observation DB.
        let mut data = Vec::new();
        for _ in 0..779 {
            data.push(obs(true, None, false)); // correct, no edit proposed
        }
        for _ in 0..30 {
            data.push(obs(true, Some(0.5), false)); // correct, low-margin over-risk
        }
        for _ in 0..17 {
            data.push(obs(false, Some(5.0), true)); // confident genuine fix
        }
        for _ in 0..11 {
            data.push(obs(false, Some(1.0), true)); // fix below the floor
        }
        for _ in 0..8 {
            data.push(obs(false, Some(0.3), false)); // unfixed error
        }

        // No floor → judged at τ=0 → net ≤ 0 → disabled.
        let raw = calibrate_edits(&data, 0.05, 0.0).unwrap();
        assert_eq!(raw.tau, 0.0, "expected degenerate tau=0, got {}", raw.tau);
        assert!(
            !raw.enabled,
            "at tau=0 the channel must look net-negative: {raw:?}"
        );

        // Floored → judged at τ=TAU_FLOOR → net > 0 → enabled.
        let floored = calibrate_edits(&data, 0.05, TAU_FLOOR).unwrap();
        assert_eq!(floored.tau, TAU_FLOOR);
        assert!(floored.enabled, "floored channel must enable: {floored:?}");
        assert!(floored.net_benefit > 0.0);
    }

    #[test]
    fn golden_calibration_is_stable() {
        // Fixed inputs → fixed outputs. Any change to the calibration math breaks
        // this; update the expected values deliberately, never by accident.
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(obs(true, None, false));
        }
        for _ in 0..40 {
            data.push(obs(true, Some(1.0), false)); // over-risk at 1.0
        }
        for _ in 0..25 {
            data.push(obs(false, Some(4.0), true)); // fixes at 4.0
        }
        let c = calibrate_edits(&data, 0.05, TAU_FLOOR).unwrap();
        assert_eq!(c.n_correct, 140);
        assert_eq!(c.n_error, 25);
        assert_eq!(c.tau, 2.0);
        assert_eq!(c.net_benefit, 25.0); // 25 fixes >2, 0 over >2
        assert!(c.enabled);
    }
}
