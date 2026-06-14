//! Homophone resolver tuning harness.
//!
//! Run:   cargo run --release --example homophone_tune -p lune-app
//!
//! Trains a REAL `lune_ngram::NGramModel` on a synthetic, context-predictable
//! "user" corpus, applies a homophone ASR-error model, and measures the REAL
//! `lune_dict::homophone::HomophoneResolver` under each policy. Synthetic
//! ground truth is the only way to score homophone correction (no labelled
//! corpus exists) — but the model and resolver code paths are the production
//! ones, so the numbers transfer. Use it to tune the margin `tau`, the context
//! support floor, and which confusion sets are safe to enable.
//!
//! Metrics (over every homophone-set word, slots corrupted at `P_ERR`):
//!   baseline    = error if we keep the ASR output
//!   resolved    = error after the resolver
//!   over_corr   = fraction where ASR was RIGHT but the resolver broke it
//!   net         = baseline - resolved  (higher = better)

use lune_dict::homophone::{HomophoneResolver, ResolverConfig, ResolverPolicy};
use lune_dict::homophone_calib;
use lune_ngram::{NGramModel, ScoringConfig};

const P_ERR: f64 = 0.18; // ASR homophone substitution rate
const N_TEST: usize = 2500; // test sentences per seed
const SEEDS: u64 = 8; // independent train/test draws averaged per cell

// ─── deterministic RNG (xorshift64*, no external deps) ───────────────────────
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ─── confusion sets + generic (ASR) priors, used only for corruption ─────────
struct Confusion {
    members: &'static [&'static str],
    prior: &'static [f64],
}
const CORRUPT_SETS: &[Confusion] = &[
    Confusion {
        members: &["there", "their", "they're"],
        prior: &[0.55, 0.33, 0.12],
    },
    Confusion {
        members: &["to", "too", "two"],
        prior: &[0.80, 0.13, 0.07],
    },
    Confusion {
        members: &["your", "you're"],
        prior: &[0.62, 0.38],
    },
    Confusion {
        members: &["no", "know"],
        prior: &[0.60, 0.40],
    },
    Confusion {
        members: &["than", "then"],
        prior: &[0.45, 0.55],
    },
];
fn confusion_for(word: &str) -> Option<&'static Confusion> {
    CORRUPT_SETS.iter().find(|c| c.members.contains(&word))
}

// ─── context-predictable user language (fixed templates) ─────────────────────
const TEMPLATES: &[&str] = &[
    "i left my keys over there",
    "their code review is ready",
    "they're going to ship today",
    "we need to merge the branch",
    "this build is too slow again",
    "send your patch to the channel",
    "you're breaking the pipeline",
    "i know the root cause now",
    "there is no regression here",
    "the latency is worse than before",
    "we fixed the bug and then deployed",
    "their server is over there still",
    "i think their approach is cleaner",
    "make sure your tokens are fresh",
    "we shipped it to production",
];

/// One sentence of the user's "true" speech. `eps` injects idiolect noise:
/// with probability `eps` a homophone is swapped for a random set member,
/// decorrelating context from word so the n-gram has nothing to learn.
fn gen_sentence(rng: &mut Rng, eps: f64) -> Vec<String> {
    let t = TEMPLATES[rng.below(TEMPLATES.len())];
    t.split_whitespace()
        .map(|w| match confusion_for(w) {
            Some(c) if rng.unit() < eps => c.members[rng.below(c.members.len())].to_owned(),
            _ => w.to_owned(),
        })
        .collect()
}

fn weighted_pick(rng: &mut Rng, c: &Confusion, exclude: &str) -> &'static str {
    let idxs: Vec<usize> = (0..c.members.len())
        .filter(|&i| c.members[i] != exclude)
        .collect();
    let wsum: f64 = idxs.iter().map(|&i| c.prior[i]).sum();
    let mut t = rng.unit() * wsum;
    for &i in &idxs {
        t -= c.prior[i];
        if t <= 0.0 {
            return c.members[i];
        }
    }
    c.members[*idxs.last().expect("confusion set has >= 2 members")]
}

/// ASR pass: each homophone is swapped for another set member (by generic
/// prior) with probability `P_ERR`; non-homophones pass through.
fn corrupt(rng: &mut Rng, truth: &[String]) -> Vec<String> {
    truth
        .iter()
        .map(|w| match confusion_for(w) {
            Some(c) if rng.unit() < P_ERR => weighted_pick(rng, c, w).to_owned(),
            _ => w.clone(),
        })
        .collect()
}

struct Metrics {
    baseline: f64,
    resolved: f64,
    over_corr: f64,
}

/// Average metrics over `SEEDS` independent train/test draws. Single-seed
/// cells carry enough training-draw variance to invert conclusions (an early
/// run showed a spurious error spike at one N that vanished under averaging).
fn evaluate(n_train: usize, eps: f64, policy: ResolverPolicy, min_context_gain: f64) -> Metrics {
    let (mut slots, mut base_err, mut res_err, mut over) = (0usize, 0usize, 0usize, 0usize);
    for seed in 0..SEEDS {
        let mut rng = Rng::new(1000 + seed);
        let mut model = NGramModel::new(ScoringConfig::trigram_default()).expect("model");
        for _ in 0..n_train {
            model.train_sentence(&gen_sentence(&mut rng, eps).join(" "));
        }
        let resolver = HomophoneResolver::with_config(ResolverConfig {
            policy,
            min_context_gain,
            ..Default::default()
        });
        for _ in 0..N_TEST {
            let truth = gen_sentence(&mut rng, eps);
            let observed = corrupt(&mut rng, &truth);
            let owords: Vec<&str> = observed.iter().map(String::as_str).collect();
            let mut finalw = observed.clone();
            for r in resolver.resolve_words(&model, &owords, &[]) {
                finalw[r.index] = r.to;
            }
            for (i, tw) in truth.iter().enumerate() {
                if confusion_for(tw).is_none() {
                    continue;
                }
                slots += 1;
                let ok_base = observed[i] == *tw;
                let ok_final = finalw[i] == *tw;
                base_err += usize::from(!ok_base);
                res_err += usize::from(!ok_final);
                over += usize::from(ok_base && !ok_final);
            }
        }
    }
    let s = slots.max(1) as f64;
    Metrics {
        baseline: base_err as f64 / s,
        resolved: res_err as f64 / s,
        over_corr: over as f64 / s,
    }
}

/// Conformal Risk Control validation: calibrate per-set tau on a held-out
/// "history", then measure on FRESH data whether the empirical over-correction
/// rate honours the target `alpha`. Returns (over-correction, coverage,
/// sets-enabled) averaged over seeds.
fn conformal_validation(alpha: f64, n_calib: usize, n_test: usize) -> (f64, f64, usize) {
    let (mut over, mut correct, mut fixed, mut errors, mut enabled) = (0usize, 0, 0, 0, 0);
    for seed in 0..SEEDS {
        let mut rng = Rng::new(2000 + seed);
        let mut model = NGramModel::new(ScoringConfig::trigram_default()).expect("model");
        let mut pairs = Vec::with_capacity(n_calib);
        for _ in 0..n_calib {
            let truth = gen_sentence(&mut rng, 0.05);
            model.train_sentence(&truth.join(" "));
            let observed = corrupt(&mut rng, &truth);
            pairs.push((observed.join(" "), truth.join(" ")));
        }
        let cal = homophone_calib::calibrate(&model, &pairs, alpha, 1);
        enabled += cal
            .iter()
            .filter(|c| c.as_ref().is_some_and(|c| c.enabled))
            .count();
        let resolver = HomophoneResolver::with_calibration(ResolverConfig::default(), cal);
        for _ in 0..n_test {
            let truth = gen_sentence(&mut rng, 0.05);
            let observed = corrupt(&mut rng, &truth);
            let owords: Vec<&str> = observed.iter().map(String::as_str).collect();
            let mut finalw = observed.clone();
            for r in resolver.resolve_words(&model, &owords, &[]) {
                finalw[r.index] = r.to;
            }
            for (i, tw) in truth.iter().enumerate() {
                if confusion_for(tw).is_none() {
                    continue;
                }
                if observed[i] == *tw {
                    correct += 1;
                    over += usize::from(finalw[i] != *tw);
                } else {
                    errors += 1;
                    fixed += usize::from(finalw[i] == *tw);
                }
            }
        }
    }
    (
        over as f64 / correct.max(1) as f64,
        fixed as f64 / errors.max(1) as f64,
        enabled / SEEDS as usize,
    )
}

fn main() {
    println!("Homophone resolver tuning — synthetic ground truth, REAL model + resolver");
    println!("P_ERR={P_ERR}  N_TEST={N_TEST}  seeds={SEEDS} (averaged)\n");

    println!("== EXP 1 — history scaling (RerankArgmax vs MarginGate tau=2.0) ==");
    println!(
        "{:>6} | {:>9} {:>11} {:>11} | {:>10} {:>10}",
        "N", "baseline", "rerank", "gate2.0", "rr_over", "gate_over"
    );
    println!("{}", "-".repeat(72));
    for n in [25usize, 50, 100, 250, 500, 1000, 2000] {
        let rr = evaluate(n, 0.05, ResolverPolicy::RerankArgmax, 0.0);
        let g = evaluate(n, 0.05, ResolverPolicy::MarginGate { tau: 2.0 }, 0.0);
        println!(
            "{n:>6} | {:>9.4} {:>11.4} {:>11.4} | {:>10.4} {:>10.4}",
            rr.baseline, rr.resolved, g.resolved, rr.over_corr, g.over_corr
        );
    }

    println!("\n== EXP 2 — margin tau sweep at N=500 ==");
    println!(
        "{:>6} | {:>9} {:>9} {:>10} | {:>9}",
        "tau", "baseline", "resolved", "over_corr", "net"
    );
    println!("{}", "-".repeat(60));
    for tau in [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0] {
        let m = evaluate(500, 0.05, ResolverPolicy::MarginGate { tau }, 0.0);
        println!(
            "{tau:>6.2} | {:>9.4} {:>9.4} {:>10.4} | {:>+9.4}",
            m.baseline,
            m.resolved,
            m.over_corr,
            m.baseline - m.resolved
        );
    }

    println!("\n== EXP 3 — learnable signal + contextual-gain gate (tau=2.0, N=500) ==");
    println!(
        "{:>6} | {:>9} {:>10} {:>10} | {:>9} {:>9}",
        "eps", "baseline", "gate g=0", "gate g=1", "ov g=0", "ov g=1"
    );
    println!("{}", "-".repeat(64));
    for eps in [0.0_f64, 0.05, 0.2, 0.5] {
        let g0 = evaluate(500, eps, ResolverPolicy::MarginGate { tau: 2.0 }, 0.0);
        let g1 = evaluate(500, eps, ResolverPolicy::MarginGate { tau: 2.0 }, 1.0);
        println!(
            "{eps:>6.2} | {:>9.4} {:>10.4} {:>10.4} | {:>9.4} {:>9.4}",
            g0.baseline, g0.resolved, g1.resolved, g0.over_corr, g1.over_corr
        );
    }

    println!("\n== EXP 4 — conformal over-correction guarantee (per-set tau via CRC) ==");
    println!(
        "{:>7} | {:>12} {:>9} | {:>9}",
        "alpha", "emp_overcor", "coverage", "n_enabled"
    );
    println!("{}", "-".repeat(50));
    for alpha in [0.005_f64, 0.01, 0.02, 0.05] {
        let (over, cov, n_en) = conformal_validation(alpha, 500, 2000);
        let flag = if over <= alpha + 0.01 { "ok" } else { "HIGH" };
        println!("{alpha:>7.3} | {over:>12.4} {cov:>9.4} | {n_en:>9}  {flag}");
    }
}
