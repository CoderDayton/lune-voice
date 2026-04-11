//! Isotonic calibration of ASR per-word confidences.
//!
//! Raw token confidences from the ASR decoder are *not* probabilities of
//! correctness — they are softmax outputs with a model- and dataset-specific
//! calibration curve. A fixed threshold like `c < 0.65` treats the raw
//! confidence as if it were calibrated, which is Bayes-suboptimal.
//!
//! This module provides a non-parametric calibrator fit via the
//! **Pool Adjacent Violators (PAV)** algorithm. Given labelled calibration
//! samples `(c, is_correct)`, it learns a monotone non-decreasing map
//! `g: [0,1] → [0,1]` such that `g(c) ≈ P(word is correct | raw confidence = c)`.
//!
//! ## Mathematical guarantee
//!
//! PAV is the maximum-likelihood monotone estimator under the
//! exchangeability assumption. Specifically, it minimises the weighted
//! squared error
//!
//! ```text
//!     argmin_{g non-decreasing} Σᵢ wᵢ (yᵢ - g(cᵢ))²
//! ```
//!
//! and converges uniformly to the true conditional probability as the
//! sample size → ∞ (Barlow et al. 1972, *Statistical Inference under
//! Order Restrictions*).
//!
//! ## Integration
//!
//! The calibrator is stored serialised and loaded at session startup.
//! Existing contextual gating heuristics in `AsrSegment::mark_uncertain_contextual`
//! continue to apply unchanged — they operate on *calibrated* confidences
//! instead of raw ones, composing cleanly with PAV.

use serde::{Deserialize, Serialize};

/// Monotone non-decreasing calibrator: raw confidence → P(word correct | c).
///
/// Stored as a sorted list of breakpoints `(c_i, g_i)` with `c_0 < c_1 < …`
/// and `g_0 ≤ g_1 ≤ …`. Lookup uses linear interpolation between breakpoints
/// and flat extrapolation outside the data range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    /// Sorted (raw_confidence, P(correct | raw)) pairs.
    /// Length ≥ 1. For the identity calibrator, holds `[(0.0, 0.0), (1.0, 1.0)]`.
    breakpoints: Vec<(f32, f32)>,
}

impl Default for IsotonicCalibrator {
    fn default() -> Self {
        Self::identity()
    }
}

impl IsotonicCalibrator {
    /// Identity calibrator: `calibrate(c) = c`. Use when no calibration data
    /// is available; downstream gate behaviour is unchanged from raw input.
    pub fn identity() -> Self {
        Self {
            breakpoints: vec![(0.0, 0.0), (1.0, 1.0)],
        }
    }

    /// Fit from labelled samples using the Pool Adjacent Violators algorithm.
    ///
    /// Inputs: `(raw_confidence, is_correct)` pairs where `is_correct = true`
    /// when the word was correctly transcribed. Empty input returns the
    /// identity calibrator. Output is a monotone non-decreasing map from
    /// raw confidence to `P(correct | raw)`.
    pub fn fit(samples: &[(f32, bool)]) -> Self {
        if samples.is_empty() {
            return Self::identity();
        }

        // Stage 1: sort by confidence ascending. Duplicate confidences are
        // tolerated — PAV merges them in the forward pass.
        let mut pairs: Vec<(f32, f32)> = samples
            .iter()
            .map(|&(c, ok)| (c.clamp(0.0, 1.0), if ok { 1.0 } else { 0.0 }))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Stage 2: PAV forward pass. Each block tracks (count, x, y_mean).
        // Merge with predecessor either when the previous block's y exceeds
        // the current block's y (monotone violation) or when they share an
        // x value (duplicate consolidation).
        #[derive(Clone, Copy)]
        struct Block {
            count: f32,
            x: f32,
            y: f32,
        }
        let mut blocks: Vec<Block> = Vec::with_capacity(pairs.len());
        for (x, y) in pairs {
            let mut cur = Block { count: 1.0, x, y };
            while let Some(&prev) = blocks.last() {
                let monotone_violation = prev.y > cur.y;
                let duplicate_x = (prev.x - cur.x).abs() < f32::EPSILON;
                if monotone_violation || duplicate_x {
                    let total = prev.count + cur.count;
                    cur = Block {
                        count: total,
                        x: cur.x,
                        y: (prev.y * prev.count + cur.y * cur.count) / total,
                    };
                    blocks.pop();
                } else {
                    break;
                }
            }
            blocks.push(cur);
        }

        // Stage 3: extract breakpoints. Pad with `(0.0, first_y)` and
        // `(1.0, last_y)` so lookups outside the training range extrapolate
        // flat rather than falling off a cliff.
        let mut breakpoints: Vec<(f32, f32)> = Vec::with_capacity(blocks.len() + 2);
        if let Some(first) = blocks.first()
            && first.x > 0.0
        {
            breakpoints.push((0.0, first.y));
        }
        for b in &blocks {
            breakpoints.push((b.x, b.y));
        }
        if let Some(last) = blocks.last()
            && last.x < 1.0
        {
            breakpoints.push((1.0, last.y));
        }

        Self { breakpoints }
    }

    /// Map a raw confidence to a calibrated `P(correct | raw)`.
    ///
    /// Linear interpolation between breakpoints; flat extrapolation outside
    /// `[breakpoints[0].0, breakpoints.last().0]`. NaN input returns `0.5`
    /// (maximally uncertain fallback) rather than propagating NaN downstream.
    pub fn calibrate(&self, raw: f32) -> f32 {
        if raw.is_nan() {
            return 0.5;
        }
        let raw = raw.clamp(0.0, 1.0);
        debug_assert!(
            !self.breakpoints.is_empty(),
            "calibrator must have ≥1 breakpoint"
        );

        if raw <= self.breakpoints[0].0 {
            return self.breakpoints[0].1;
        }
        let last = self.breakpoints[self.breakpoints.len() - 1];
        if raw >= last.0 {
            return last.1;
        }

        // Binary search for the enclosing breakpoints. `partition_point`
        // returns the first index where `x > raw`, so `idx - 1` is the last
        // breakpoint with `x ≤ raw`. The `max(1)` is defensive — the edge
        // cases above already guarantee `idx ≥ 1` here.
        let idx = self.breakpoints.partition_point(|&(x, _)| x <= raw).max(1);
        let (x0, y0) = self.breakpoints[idx - 1];
        let (x1, y1) = self.breakpoints[idx];
        if (x1 - x0).abs() < f32::EPSILON {
            return y1;
        }
        let t = (raw - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }

    /// Number of stored breakpoints — useful for test assertions and
    /// diagnostic logging at calibrator-load time.
    pub fn breakpoint_count(&self) -> usize {
        self.breakpoints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_passthrough() {
        let cal = IsotonicCalibrator::identity();
        for c in [0.0, 0.1, 0.5, 0.65, 0.9, 1.0] {
            assert!((cal.calibrate(c) - c).abs() < 1e-5, "c={c}");
        }
    }

    #[test]
    fn empty_fit_is_identity() {
        let cal = IsotonicCalibrator::fit(&[]);
        assert!((cal.calibrate(0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn trivial_monotone_data_preserved() {
        let samples = vec![
            (0.1, false),
            (0.3, false),
            (0.5, true),
            (0.7, true),
            (0.9, true),
        ];
        let cal = IsotonicCalibrator::fit(&samples);
        assert!(cal.calibrate(0.1) < 0.1);
        assert!(cal.calibrate(0.9) > 0.9);
        let vals: Vec<f32> = (0..=10).map(|i| cal.calibrate(i as f32 / 10.0)).collect();
        for w in vals.windows(2) {
            assert!(w[0] <= w[1] + 1e-5, "non-monotone: {} → {}", w[0], w[1]);
        }
    }

    #[test]
    fn pav_corrects_violators() {
        // Confidence 0.3 more often correct than 0.7 — a classic violator.
        // PAV should merge both into one block with mean 0.5.
        let samples = vec![
            (0.3, true),
            (0.3, true),
            (0.3, true),
            (0.7, false),
            (0.7, false),
            (0.7, false),
        ];
        let cal = IsotonicCalibrator::fit(&samples);
        let c03 = cal.calibrate(0.3);
        let c07 = cal.calibrate(0.7);
        assert!(
            (c03 - c07).abs() < 1e-3,
            "PAV did not merge violators: {c03} vs {c07}"
        );
        assert!((c03 - 0.5).abs() < 0.05, "merged block value wrong: {c03}");
    }

    #[test]
    fn overconfident_model_corrected_downward() {
        // Typical ASR pathology: raw 0.85 → actually 60% correct.
        let mut samples = Vec::new();
        for _ in 0..60 {
            samples.push((0.85, true));
        }
        for _ in 0..40 {
            samples.push((0.85, false));
        }
        for _ in 0..95 {
            samples.push((0.95, true));
        }
        for _ in 0..5 {
            samples.push((0.95, false));
        }
        let cal = IsotonicCalibrator::fit(&samples);
        assert!((cal.calibrate(0.85) - 0.60).abs() < 0.05);
        assert!((cal.calibrate(0.95) - 0.95).abs() < 0.05);
        let mid = cal.calibrate(0.90);
        assert!(
            mid > 0.60 && mid < 0.95,
            "interpolation did not land between: {mid}"
        );
    }

    #[test]
    fn all_correct_monotone_to_one() {
        let samples: Vec<(f32, bool)> = (0..50).map(|i| (i as f32 / 50.0, true)).collect();
        let cal = IsotonicCalibrator::fit(&samples);
        assert!(cal.calibrate(0.0) > 0.95);
        assert!(cal.calibrate(1.0) > 0.95);
    }

    #[test]
    fn all_wrong_monotone_to_zero() {
        let samples: Vec<(f32, bool)> = (0..50).map(|i| (i as f32 / 50.0, false)).collect();
        let cal = IsotonicCalibrator::fit(&samples);
        assert!(cal.calibrate(0.0) < 0.05);
        assert!(cal.calibrate(1.0) < 0.05);
    }

    #[test]
    fn serde_roundtrip() {
        let samples = vec![(0.2, false), (0.5, true), (0.8, true)];
        let cal = IsotonicCalibrator::fit(&samples);
        let json = serde_json::to_string(&cal).unwrap();
        let decoded: IsotonicCalibrator = serde_json::from_str(&json).unwrap();
        for c in [0.1, 0.3, 0.6, 0.9] {
            let a = cal.calibrate(c);
            let b = decoded.calibrate(c);
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn boundary_clamping() {
        let samples = vec![(0.3, true), (0.7, false)];
        let cal = IsotonicCalibrator::fit(&samples);
        let a = cal.calibrate(-0.5);
        let b = cal.calibrate(1.5);
        assert!((0.0..=1.0).contains(&a));
        assert!((0.0..=1.0).contains(&b));
    }

    #[test]
    fn nan_input_returns_neutral() {
        let cal = IsotonicCalibrator::identity();
        let v = cal.calibrate(f32::NAN);
        assert!(
            (v - 0.5).abs() < 1e-5,
            "NaN should map to neutral 0.5, got {v}"
        );
    }

    #[test]
    fn monotonicity_invariant_random() {
        let samples = vec![
            (0.05, false),
            (0.1, true),
            (0.15, false),
            (0.2, true),
            (0.3, false),
            (0.5, true),
            (0.5, true),
            (0.7, true),
            (0.85, false),
            (0.9, true),
            (0.95, true),
        ];
        let cal = IsotonicCalibrator::fit(&samples);
        let xs: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        for w in xs.windows(2) {
            let a = cal.calibrate(w[0]);
            let b = cal.calibrate(w[1]);
            assert!(
                a <= b + 1e-5,
                "non-monotone at {}→{}: {a} vs {b}",
                w[0],
                w[1]
            );
        }
    }
}
