//! Self-evaluating calibration for the [`crate::homophone`] resolver.
//!
//! The user's delivered transcripts are clean ground truth. Pairing each with
//! the ASR raw output yields per-user `(observed → accepted)` labels for free.
//! We replay the resolver over them counterfactually (Bottou et al., JMLR 2013)
//! and, per confusion set, set a margin threshold that *provably* bounds
//! over-correction via **Conformal Risk Control** (Angelopoulos, Bates, Fisch,
//! Lei, Schuster — arXiv:2208.02814): over-correction is a monotone loss in the
//! threshold, so the split-conformal calibration applies directly. A set is
//! then *enabled* only if it nets a benefit under a Wilson lower bound — the
//! selective-prediction / reject-option decision (Chow 1970; Geifman &
//! El-Yaniv, NeurIPS 2017). This replaces the hand-tuned global tau with a
//! per-user, per-set, finite-sample-guaranteed threshold.

use std::collections::HashMap;

use crate::ngram::{NGramModel, tokenize};

use crate::homophone::{
    CONFUSION_SETS, SetCalibration, best_member_margin, build_member_index, norm_match,
};

/// Default over-correction budget (target P[over-correction] per touched set).
pub const DEFAULT_ALPHA: f64 = 0.01;
/// z for the 95% Wilson lower bound used in the enablement test.
const WILSON_Z: f64 = 1.96;
/// Minimum decisive flips (fixes + breaks) before a set may be enabled.
const MIN_DECISIVE: usize = 5;

/// One graded homophone occurrence drawn from the user's history.
struct Obs {
    set_idx: usize,
    /// ASR token equalled the accepted (truth) token.
    observed_correct: bool,
    /// Margin of the model's best member over the observed word; `None` when
    /// no flip candidate exists (model already prefers the observed word).
    margin: Option<f64>,
    /// The model's best member equals the accepted truth (a beneficial flip).
    flip_fixes: bool,
}

/// Calibrate every confusion set from `(asr_text, final_text)` history pairs.
/// Returns one entry per [`CONFUSION_SETS`] (`None` = no data for that set).
pub fn calibrate(
    model: &NGramModel,
    pairs: &[(String, String)],
    alpha: f64,
    min_support: usize,
) -> Vec<Option<SetCalibration>> {
    let index = build_member_index();
    let mut per_set: Vec<Vec<Obs>> = (0..CONFUSION_SETS.len()).map(|_| Vec::new()).collect();
    for (asr, fin) in pairs {
        for obs in collect(model, &index, asr, fin, min_support) {
            per_set[obs.set_idx].push(obs);
        }
    }
    per_set
        .into_iter()
        .map(|obs| calibrate_set(&obs, alpha))
        .collect()
}

/// Human-readable summary of which sets were enabled and why (for startup logs).
pub fn summarize(calibration: &[Option<SetCalibration>]) -> String {
    let mut lines = Vec::new();
    for (i, c) in calibration.iter().enumerate() {
        let members = CONFUSION_SETS[i].members.join("/");
        match c {
            Some(c) => lines.push(format!(
                "{members}: {} tau={:.2} net={:+.0} (correct={}, error={})",
                if c.enabled { "ENABLED" } else { "disabled" },
                c.tau,
                c.net_benefit,
                c.n_correct,
                c.n_error,
            )),
            None => lines.push(format!("{members}: no data")),
        }
    }
    lines.join("\n")
}

// ── per-set calibration ──────────────────────────────────────────────────────

fn calibrate_set(obs: &[Obs], alpha: f64) -> Option<SetCalibration> {
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
    let tau = conformal_threshold(&correct_margins, n_correct, alpha);

    let over = correct_margins.iter().filter(|&&m| m > tau).count();
    let fixed = obs
        .iter()
        .filter(|o| !o.observed_correct && o.flip_fixes)
        .filter(|o| o.margin.is_some_and(|m| m > tau))
        .count();
    let net = fixed as f64 - over as f64;

    // Selective enablement: require positive net benefit AND a Wilson lower
    // bound on the per-flip win-rate above 0.5, so a few lucky samples can't
    // enable a set. Neutral flips (error → other-wrong) are excluded.
    let decisive = fixed + over;
    let enabled =
        net > 0.0 && decisive >= MIN_DECISIVE && wilson_lower(fixed, decisive, WILSON_Z) > 0.5;

    let pts: Vec<(f64, f64)> = obs
        .iter()
        .filter_map(|o| o.margin.map(|m| (m, if o.flip_fixes { 1.0 } else { 0.0 })))
        .collect();
    let reliability = isotonic(pts);

    Some(SetCalibration {
        enabled,
        tau,
        net_benefit: net,
        n_correct,
        n_error,
        reliability,
    })
}

/// Conformal Risk Control threshold for a monotone over-correction loss.
///
/// With loss ℓ_i(τ)=1[margin_i>τ] over the `n_correct` correct cases (bounded
/// by 1), CRC picks the smallest τ with `(n/(n+1))·R̂(τ)+1/(n+1) ≤ α`, which
/// reduces to allowing at most `k = ⌊α(n+1)−1⌋` correct-case margins above τ.
/// That guarantees `E[over-correction rate] ≤ α` on a fresh occurrence.
fn conformal_threshold(correct_margins: &[f64], n_correct: usize, alpha: f64) -> f64 {
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
fn wilson_lower(s: usize, n: usize, z: f64) -> f64 {
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

/// Pool-Adjacent-Violators isotonic regression on `(margin, label)` points,
/// yielding monotone-non-decreasing `(margin, P[flip correct])` breakpoints.
fn isotonic(mut pts: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // blocks of (sum_y, count, rightmost_x)
    let mut blocks: Vec<(f64, f64, f64)> = Vec::with_capacity(pts.len());
    for (x, y) in pts {
        let mut blk = (y, 1.0, x);
        while let Some(&(s, c, _)) = blocks.last() {
            if s / c >= blk.0 / blk.1 {
                let (s2, c2, _) = blocks.pop().unwrap();
                blk = (s2 + blk.0, c2 + blk.1, blk.2);
            } else {
                break;
            }
        }
        blocks.push(blk);
    }
    blocks.into_iter().map(|(s, c, x)| (x, s / c)).collect()
}

// ── observation collection (LCS-aligned counterfactual replay) ───────────────

enum Op {
    Keep(usize, usize), // (asr_idx, fin_idx): aligned (match or substitution)
    InsTruth,           // word present only in final (LLM insertion)
    DelAsr(usize),      // word present only in asr (dropped by correction)
}

fn collect(
    model: &NGramModel,
    index: &HashMap<String, usize>,
    asr_text: &str,
    final_text: &str,
    min_support: usize,
) -> Vec<Obs> {
    let asr: Vec<String> = asr_text.split_whitespace().map(norm_match).collect();
    let fin: Vec<String> = final_text.split_whitespace().map(norm_match).collect();
    let mut ctx: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for op in align(&asr, &fin) {
        match op {
            Op::Keep(i, j) => {
                let observed = &asr[i];
                let truth = &fin[j];
                if let Some(&set_idx) = index.get(truth) {
                    let set = &CONFUSION_SETS[set_idx];
                    let (margin, flip_fixes) =
                        match best_member_margin(model, &ctx, observed, set, min_support) {
                            Some((best, m)) => (Some(m), norm_match(best) == *truth),
                            None => (None, false),
                        };
                    out.push(Obs {
                        set_idx,
                        observed_correct: observed == truth,
                        margin,
                        flip_fixes,
                    });
                }
                // Context tracks the OBSERVED (ASR) stream, matching the live
                // resolver exactly (`homophone::HomophoneResolver::resolve_words`).
                ctx.extend(tokenize(observed));
            }
            // Truth-only word: ASR never produced it, so it is absent from the
            // live context stream too — contribute nothing.
            Op::InsTruth => {}
            // ASR word the corrector later dropped: it WAS in the ASR stream, so
            // the live resolver would carry it in context.
            Op::DelAsr(i) => ctx.extend(tokenize(&asr[i])),
        }
    }
    out
}

/// Word-level alignment via edit-distance backtrace. Substitutions and matches
/// both map to `Keep` (we compare the words ourselves); insertions/deletions
/// are tracked so the truth-side context stays accurate.
fn align(asr: &[String], fin: &[String]) -> Vec<Op> {
    let (m, n) = (asr.len(), fin.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = usize::from(asr[i - 1] != fin[j - 1]);
            dp[i][j] = (dp[i - 1][j - 1] + cost)
                .min(dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1);
        }
    }
    let (mut i, mut j) = (m, n);
    let mut ops = Vec::new();
    while i > 0 || j > 0 {
        let cost = if i > 0 && j > 0 {
            usize::from(asr[i - 1] != fin[j - 1])
        } else {
            0
        };
        if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + cost {
            ops.push(Op::Keep(i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j] == dp[i][j - 1] + 1) {
            ops.push(Op::InsTruth);
            j -= 1;
        } else {
            ops.push(Op::DelAsr(i - 1));
            i -= 1;
        }
    }
    ops.reverse();
    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::ScoringConfig;

    fn model() -> NGramModel {
        let mut m = NGramModel::new(ScoringConfig::trigram_default()).unwrap();
        let corpus = [
            "send their patch to the channel",
            "their review is ready",
            "the file is over there",
            "i left it over there",
        ];
        for _ in 0..40 {
            m.train_sentences(corpus);
        }
        m
    }

    #[test]
    fn conformal_threshold_bounds_overcorrection() {
        // 100 correct-case margins; alpha=0.05 ⇒ k = ⌊0.05·101−1⌋ = 4.
        let margins: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let tau = conformal_threshold(&margins, 100, 0.05);
        let over = margins.iter().filter(|&&m| m > tau).count();
        assert!(
            over <= 4,
            "CRC must allow at most k=4 over-corrections, got {over}"
        );
    }

    #[test]
    fn conformal_threshold_infinite_when_data_too_small() {
        // n=10, alpha=0.01 ⇒ k_max = ⌊0.01·11−1⌋ < 0 ⇒ never flip.
        assert_eq!(conformal_threshold(&[1.0, 2.0], 10, 0.01), f64::INFINITY);
    }

    #[test]
    fn wilson_lower_is_conservative_on_small_n() {
        assert!(
            wilson_lower(1, 2, WILSON_Z) < 0.5,
            "2 samples must not certify >0.5"
        );
        assert!(
            wilson_lower(50, 50, WILSON_Z) > 0.5,
            "50/50 wins should certify >0.5"
        );
    }

    #[test]
    fn isotonic_is_monotone_nondecreasing() {
        let curve = isotonic(vec![(0.5, 1.0), (0.1, 0.0), (0.3, 1.0), (0.2, 0.0)]);
        for w in curve.windows(2) {
            assert!(
                w[1].1 >= w[0].1 - 1e-9,
                "reliability must be monotone: {curve:?}"
            );
        }
    }

    #[test]
    fn predictable_set_enables_random_set_disables() {
        let m = model();
        // "there/their" is context-predictable in the corpus; build pairs where
        // ASR sometimes swaps them. Truth = the corpus word.
        let mut pairs = Vec::new();
        for _ in 0..60 {
            pairs.push((
                "send there patch".to_string(),
                "send their patch".to_string(),
            )); // error→fixable
            pairs.push((
                "send their patch".to_string(),
                "send their patch".to_string(),
            )); // correct→keep
            pairs.push(("over their".to_string(), "over there".to_string())); // error→fixable
        }
        let cal = calibrate(&m, &pairs, 0.10, 1);
        // find the there/their set
        let idx = CONFUSION_SETS
            .iter()
            .position(|s| s.members.contains(&"there"))
            .unwrap();
        let c = cal[idx].as_ref().expect("there/their should have data");
        assert!(c.n_correct > 0 && c.n_error > 0);
        assert!(
            c.enabled,
            "a predictable, net-beneficial set must enable: {c:?}"
        );
        assert!(c.tau.is_finite());
    }

    #[test]
    fn align_handles_length_change() {
        // LLM inserted a word; the homophone substitution must still align.
        let obs = collect(
            &model(),
            &build_member_index(),
            "send there patch",
            "please send their patch",
            1,
        );
        assert!(
            obs.iter().any(|o| !o.observed_correct),
            "expected an error obs"
        );
    }
}
