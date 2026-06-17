//! Conformal-Risk-Control kernel for the LLM-edit calibration.
//!
//! These primitives back the open-vocabulary LLM-edit calibration
//! ([`crate::edit_calib`]):
//!
//! - over-correction is a monotone loss in the threshold, so split-conformal
//!   risk control applies directly (Angelopoulos, Bates, Fisch, Lei, Schuster —
//!   arXiv:2208.02814);
//! - a channel is *enabled* only when it nets a benefit under a Wilson lower
//!   bound — the selective-prediction / reject-option decision (Chow 1970;
//!   Geifman & El-Yaniv, NeurIPS 2017).

/// Default over-correction budget — target P[over-correction] for the LLM-edit
/// channel.
///
/// Also the cold-start floor: `conformal_threshold` returns a finite τ only once
/// the channel has `n_correct ≥ ⌈1/α⌉ − 1` correct observations (at α=0.05, ≥19).
/// 0.05 trades a looser 5% over-correction bound for a reachable activation point
/// on a single user's history; cold start stays fail-safe (disabled).
pub const DEFAULT_ALPHA: f64 = 0.05;
/// z for the 95% Wilson lower bound used in the enablement test.
pub const WILSON_Z: f64 = 1.96;
/// Minimum decisive flips (fixes + breaks) before a channel may be enabled.
pub const MIN_DECISIVE: usize = 5;

/// Conformal Risk Control threshold for a monotone over-correction loss.
///
/// With loss ℓ_i(τ)=1[margin_i>τ] over the `n_correct` correct cases (bounded
/// by 1), CRC picks the smallest τ with `(n/(n+1))·R̂(τ)+1/(n+1) ≤ α`, which
/// reduces to allowing at most `k = ⌊α(n+1)−1⌋` correct-case margins above τ.
/// That guarantees `E[over-correction rate] ≤ α` on a fresh occurrence.
pub fn conformal_threshold(correct_margins: &[f64], n_correct: usize, alpha: f64) -> f64 {
    let k_max = (alpha * (n_correct as f64 + 1.0) - 1.0).floor();
    if k_max < 0.0 {
        // Too few samples to certify α at any finite τ → never flip.
        return f64::INFINITY;
    }
    let k = k_max as usize;
    let mut m = correct_margins.to_vec();
    m.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // desc
    if m.len() <= k {
        // Even flipping every candidate stays within budget → no constraint.
        return 0.0;
    }
    m[k] // (k+1)-th largest: exactly k margins lie strictly above it
}

/// Lower bound of the Wilson score interval for `s`/`n` at confidence `z`.
pub fn wilson_lower(s: usize, n: usize, z: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n = n as f64;
    let phat = s as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = phat + z2 / (2.0 * n);
    let half = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    ((centre - half) / denom).max(0.0)
}

/// Selective-enablement decision: positive net benefit AND a Wilson lower bound
/// on the per-flip win-rate above 0.5, so a few lucky samples can't enable a
/// channel. `fixed` = beneficial flips above τ, `decisive` = fixed + over.
pub fn is_enabled(fixed: usize, decisive: usize, net: f64) -> bool {
    net > 0.0 && decisive >= MIN_DECISIVE && wilson_lower(fixed, decisive, WILSON_Z) > 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformal_infinite_when_data_too_small() {
        // n=10, alpha=0.01 → k_max = 0.01*11 - 1 < 0 → never flip (cold-start safe).
        assert_eq!(conformal_threshold(&[1.0; 10], 10, 0.01), f64::INFINITY);
    }

    #[test]
    fn conformal_bounds_overcorrection() {
        // 100 correct margins 1..=100, alpha=0.05 → k = floor(0.05*101-1)=4 →
        // exactly 4 strictly above tau.
        let margins: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let tau = conformal_threshold(&margins, 100, 0.05);
        let over = margins.iter().filter(|&&m| m > tau).count();
        assert!(over <= 5, "over={over} should be <= floor(alpha*(n+1))");
    }

    #[test]
    fn wilson_is_conservative_on_small_n() {
        // 3/3 successes: point estimate 1.0, but the lower bound stays well below.
        assert!(wilson_lower(3, 3, WILSON_Z) < 0.5);
        // 50/50: lower bound clears 0.5.
        assert!(wilson_lower(50, 50, WILSON_Z) > 0.5);
    }

    #[test]
    fn is_enabled_requires_decisive_and_winrate() {
        assert!(!is_enabled(4, 4, 4.0)); // < MIN_DECISIVE
        assert!(!is_enabled(3, 6, 0.0)); // net not positive
        assert!(is_enabled(20, 20, 20.0)); // strong, certified
    }
}
