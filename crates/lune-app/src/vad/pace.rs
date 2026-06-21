//! PACE — Pause-Adaptive Censored Endpointing.
//!
//! A self-supervised, online (O(1)/event) estimator of the per-speaker endpoint
//! threshold: the silence duration after which an utterance is treated as
//! finished. It learns a high quantile of the speaker's internal-pause
//! distribution under endogenous censoring (the threshold being learned is
//! itself the level at which observations are censored).
//!
//! Direct port of the verified Python reference (`pace.py`). The engine is pure
//! arithmetic — `erf` and the exploration RNG are inlined to avoid new
//! dependencies. Wiring into the span gate lives in `super::config`/`super::mod`.

// Interim: the engine API is consumed by the `from_pipeline`/span-gate wiring
// (Stage B). Removed once that wiring lands.
#![allow(dead_code)]

// ── standard-normal helpers ────────────────────────────────────────────────
const SQRT2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/sqrt(2*pi)

/// Abramowitz & Stegun 7.1.26 — max abs error 1.5e-7, ample for this use.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT2))
}

fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Overflow-safe logistic.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Recover the latent `N(mu, sigma^2)` parameters from the mean/variance of its
/// upper-truncated (`X <= h`) law, plus the latent survival `s_H = P(X > h)`.
///
/// Fixed-point inversion of the truncated-normal moment map
/// `m = mu - sigma*r`, `v = sigma^2 (1 - b*r - r^2)`, `r = phi(b)/Phi(b)`,
/// `b = (h-mu)/sigma`. Reduces to the identity as `h -> +inf`.
fn latent_from_truncated(m_obs: f64, v_obs: f64, h: f64, iters: usize) -> (f64, f64, f64) {
    let mut mu = m_obs;
    let mut sigma = v_obs.max(1e-12).sqrt();
    for _ in 0..iters {
        let b = (h - mu) / sigma;
        let phi = norm_cdf(b);
        if phi < 1e-6 {
            break;
        }
        let r = norm_pdf(b) / phi;
        let denom = 1.0 - b * r - r * r;
        if denom < 1e-3 {
            break;
        }
        let sigma_new = (v_obs / denom).sqrt();
        let mu_new = m_obs + sigma_new * r;
        if !(mu_new.is_finite() && sigma_new.is_finite()) {
            break;
        }
        mu = mu_new;
        sigma = sigma_new;
    }
    let s_h = 1.0 - norm_cdf((h - mu) / sigma);
    (mu, sigma, s_h)
}

// ── seedable RNG (SplitMix64) — only used for the exploration coin ──────────
#[derive(Clone)]
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn seed(s: u64) -> Self {
        Self {
            state: s ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ── parameters ─────────────────────────────────────────────────────────────
#[derive(Clone, Copy)]
pub(crate) struct Params {
    pub(crate) z_target: f64,
    pub(crate) z99: f64,
    pub(crate) d_star: f64,
    pub(crate) s_blend: f64,
    pub(crate) p_min: f64,
    pub(crate) p_max: f64,
    pub(crate) gamma: f64,
    pub(crate) gamma_drift: f64,
    pub(crate) gamma_drift_slow: f64,
    pub(crate) drift_rise: f64,
    pub(crate) c_hor: f64,
    pub(crate) tau_min: f64,
    pub(crate) tau_max: f64,
    pub(crate) h_min: f64,
    /// Ceiling for the *observation* horizon, decoupled from `tau_max` (the
    /// *acting* ceiling). `tau_act` never delays an endpoint past `tau_max`, but
    /// a rare exploration event may wait up to `h_max` to observe a heavy tail,
    /// so the nonparametric P95 is fed un-censored samples even when
    /// `Q99 > tau_max`.
    pub(crate) h_max: f64,
    pub(crate) var_floor: f64,
    pub(crate) n_warm: u64,
    pub(crate) warm_samples: u64,
    pub(crate) target_q: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            z_target: 1.6448536,
            z99: 2.3263479,
            d_star: 0.08,
            s_blend: 0.03,
            p_min: 0.04,
            p_max: 0.40,
            gamma: 0.999,
            gamma_drift: 0.997,
            gamma_drift_slow: 0.9997,
            drift_rise: 0.05,
            c_hor: 1.3,
            tau_min: 0.25,
            tau_max: 2.00,
            h_min: 1.50,
            h_max: 3.50,
            var_floor: 1e-4,
            n_warm: 10_000,
            warm_samples: 200,
            target_q: 0.95,
        }
    }
}

impl Params {
    pub(crate) fn alpha(&self) -> f64 {
        1.0 - self.target_q
    }
}

// ── P^2 online quantile (Jain–Chlamtac) ────────────────────────────────────
#[derive(Clone)]
pub(crate) struct P2 {
    p: f64,
    dn: [f64; 5],
    q: [f64; 5],
    n: [f64; 5],
    np: [f64; 5],
    cnt: usize,
}

impl P2 {
    pub(crate) fn new(p: f64) -> Self {
        Self {
            p,
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            q: [0.0; 5],
            n: [0.0; 5],
            np: [0.0; 5],
            cnt: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.q = [0.0; 5];
        self.n = [0.0; 5];
        self.np = [0.0; 5];
        self.cnt = 0;
    }

    pub(crate) fn add(&mut self, x: f64) {
        if self.cnt < 5 {
            // insertion-sort x into the filled prefix q[0..cnt]
            let mut i = self.cnt;
            while i > 0 && self.q[i - 1] > x {
                self.q[i] = self.q[i - 1];
                i -= 1;
            }
            self.q[i] = x;
            self.cnt += 1;
            if self.cnt == 5 {
                let p = self.p;
                self.n = [1.0, 2.0, 3.0, 4.0, 5.0];
                self.np = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0];
            }
            return;
        }

        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x < self.q[1] {
            0
        } else if x < self.q[2] {
            1
        } else if x < self.q[3] {
            2
        } else if x <= self.q[4] {
            3
        } else {
            self.q[4] = x;
            3
        };

        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }

        for i in 1..4 {
            let d = self.np[i] - self.n[i];
            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1.0)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1.0)
            {
                let s: i64 = if d >= 0.0 { 1 } else { -1 };
                let sf = s as f64;
                let qp = self.q[i]
                    + sf / (self.n[i + 1] - self.n[i - 1])
                        * ((self.n[i] - self.n[i - 1] + sf) * (self.q[i + 1] - self.q[i])
                            / (self.n[i + 1] - self.n[i])
                            + (self.n[i + 1] - self.n[i] - sf) * (self.q[i] - self.q[i - 1])
                                / (self.n[i] - self.n[i - 1]));
                if self.q[i - 1] < qp && qp < self.q[i + 1] {
                    self.q[i] = qp;
                } else {
                    let j = (i as i64 + s) as usize;
                    self.q[i] += sf * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i]);
                }
                self.n[i] += sf;
            }
        }
    }

    /// Tail quantile estimate, or `default` until 5 markers exist (no
    /// small-sample median masquerading as a 0.95/0.99 quantile).
    pub(crate) fn value(&self, default: f64) -> f64 {
        if self.cnt >= 5 { self.q[2] } else { default }
    }
}

// ── branch-coverage instrumentation (cheap; exercised by tests) ────────────
#[derive(Clone, Default)]
pub(crate) struct Branches {
    pub(crate) warming: u64,
    pub(crate) steady: u64,
    pub(crate) ingest: u64,
    pub(crate) resume_no_ingest: u64,
    pub(crate) endpoint: u64,
    pub(crate) drift_reset: u64,
}

// ── per-event outcome and context ──────────────────────────────────────────
#[derive(Clone, Copy)]
pub(crate) struct StepResult {
    pub(crate) tau_act: f64,
    pub(crate) h: f64,
    pub(crate) explore: bool,
    pub(crate) resumed: bool,
    pub(crate) q_par: f64,
    pub(crate) q_np: f64,
    pub(crate) s_h: f64,
    pub(crate) w_eff: f64,
    pub(crate) did_reset: bool,
}

/// Decision carried from `begin` to `resolve` for one silence event.
#[derive(Clone, Copy)]
pub(crate) struct EventCtx {
    pub(crate) wait_limit: f64,
    explore: bool,
    warming: bool,
    tau_act: f64,
    h: f64,
    q_par: f64,
    q_np: f64,
    s_h: f64,
    w_eff: f64,
}

// ── engine ─────────────────────────────────────────────────────────────────
pub(crate) struct Pace {
    p: Params,
    rng: Rng,
    mu: f64,
    var: f64,
    dbar_fast: f64,
    dbar_slow: f64,
    n_samples: u64,
    ev: u64,
    h_prev: f64,
    p95: P2,
    p99: P2,
    pub(crate) branches: Branches,
}

impl Pace {
    pub(crate) fn new(params: Params, rng: Rng, mu0: f64, var0: f64) -> Self {
        Self {
            p: params,
            rng,
            mu: mu0,
            var: var0,
            dbar_fast: 0.0,
            dbar_slow: 0.0,
            n_samples: 0,
            ev: 0,
            h_prev: params.tau_max,
            p95: P2::new(params.target_q),
            p99: P2::new(0.99),
            branches: Branches::default(),
        }
    }

    /// Default warm start (~600 ms median guess, sigma ~0.5).
    pub(crate) fn with_defaults(seed: u64) -> Self {
        Self::new(Params::default(), Rng::seed(seed), (0.6f64).ln(), 0.25)
    }

    fn ingest(&mut self, g: f64) {
        let lg = g.max(1e-9).ln();
        let diff = lg - self.mu;
        let incr = (1.0 - self.p.gamma) * diff;
        self.mu += incr;
        self.var = self.p.gamma * (self.var + diff * incr); // West EWMA variance
        self.p95.add(g);
        self.p99.add(g);
        self.n_samples += 1;
    }

    /// Open a silence event: choose explore/exploit and return the wait limit
    /// (the silence budget for this span) plus the context to `resolve` with.
    pub(crate) fn begin(&mut self) -> EventCtx {
        let p = self.p;
        self.ev += 1;

        let h = self.h_prev.ln();
        let (mu_lat, sig_lat, s_h) = latent_from_truncated(self.mu, self.var, h, 12);
        let q_par = (mu_lat + sig_lat * p.z_target).exp();
        let q99_par = (mu_lat + sig_lat * p.z99).exp();
        let q_np = self.p95.value(q_par);
        let q99 = self.p99.value(q99_par);

        let d = (q_np / q_par).ln().abs();
        self.dbar_fast = p.gamma_drift * self.dbar_fast + (1.0 - p.gamma_drift) * d;
        self.dbar_slow = p.gamma_drift_slow * self.dbar_slow + (1.0 - p.gamma_drift_slow) * d;
        let w = sigmoid(-(d - p.d_star) / p.s_blend);

        let cens = clip(s_h / p.alpha(), 0.0, 1.0);
        let mut w_eff = clip(w + w * (1.0 - w) * cens, 0.0, 1.0);

        let warming = self.ev < p.n_warm || self.n_samples < p.warm_samples;
        if warming {
            w_eff = w_eff.min(0.5);
        }

        let tau_act = clip(w_eff * q_par + (1.0 - w_eff) * q_np, p.tau_min, p.tau_max);
        // Observation horizon clips to h_max (decoupled from the acting ceiling
        // tau_max) so heavy tails (Q99 > tau_max) feed P95 un-censored samples.
        let h_new = clip(p.c_hor * q99.max(q99_par), p.h_min, p.h_max);

        let p_explore = if warming {
            self.branches.warming += 1;
            p.p_max
        } else {
            self.branches.steady += 1;
            p.p_min + (p.p_max - p.p_min) * (1.0 - w_eff)
        };
        let explore = self.rng.next_f64() < p_explore;
        let wait_limit = if explore { h_new } else { tau_act };

        EventCtx {
            wait_limit,
            explore,
            warming,
            tau_act,
            h: h_new,
            q_par,
            q_np,
            s_h,
            w_eff,
        }
    }

    /// Close a silence event: `resumed` = speech returned before `wait_limit`
    /// at the given `gap`; otherwise we endpointed (a cut). Updates estimators
    /// (only on an exploration resume) and the drift detector.
    pub(crate) fn resolve(&mut self, ctx: &EventCtx, resumed: bool, gap: f64) -> StepResult {
        if resumed {
            if ctx.explore {
                self.ingest(gap);
                self.branches.ingest += 1;
            } else {
                self.branches.resume_no_ingest += 1;
            }
        } else {
            self.branches.endpoint += 1;
        }

        let mut did_reset = false;
        if !ctx.warming && (self.dbar_fast - self.dbar_slow) > self.p.drift_rise {
            self.p95.reset();
            self.p99.reset();
            self.var = self.var.max(0.25);
            self.dbar_slow = self.dbar_fast;
            self.branches.drift_reset += 1;
            did_reset = true;
        }

        self.h_prev = ctx.h;
        StepResult {
            tau_act: ctx.tau_act,
            h: ctx.h,
            explore: ctx.explore,
            resumed,
            q_par: ctx.q_par,
            q_np: ctx.q_np,
            s_h: ctx.s_h,
            w_eff: ctx.w_eff,
            did_reset,
        }
    }

    /// Synchronous convenience used by tests/sims: `draw_gap(wait_limit)`
    /// returns `(gap, resumed)`.
    pub(crate) fn step(&mut self, mut draw_gap: impl FnMut(f64) -> (f64, bool)) -> StepResult {
        let ctx = self.begin();
        let (g, resumed) = draw_gap(ctx.wait_limit);
        self.resolve(&ctx, resumed, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn std_normal(rng: &mut Rng) -> f64 {
        // Box–Muller from two uniforms.
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn erf_matches_known_values() {
        assert!((erf(0.5) - 0.520_499_9).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-6);
        assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-6);
        assert!((erf(2.0) - 0.995_322_3).abs() < 1e-6);
    }

    /// Fix 1: invert the truncated-normal moment map analytically (no MC), so
    /// the recovery is exact up to erf error and the fixed-point tolerance.
    #[test]
    fn latent_inversion_round_trips() {
        for &(mu, sig, h) in &[(0.0, 1.0, 1.0), (-0.5, 0.5, 0.69), (0.3, 0.6, 0.69)] {
            let b = (h - mu) / sig;
            let r = norm_pdf(b) / norm_cdf(b);
            let m = mu - sig * r; // truncated mean
            let v = sig * sig * (1.0 - b * r - r * r); // truncated variance
            let (mu_r, sig_r, s_h) = latent_from_truncated(m, v, h, 12);
            let true_sh = 1.0 - norm_cdf(b);
            // 12-iter fixed point recovers the latent params to ~1% even under
            // heavy truncation (s_H up to 0.25) — negligible for the quantile
            // Q_par = exp(mu+sig*z) that consumes them.
            assert!((mu_r - mu).abs() < 1.5e-2, "mu {mu} -> {mu_r}");
            assert!((sig_r - sig).abs() < 1.5e-2, "sig {sig} -> {sig_r}");
            assert!((s_h - true_sh).abs() < 1.5e-2, "s_H {true_sh} -> {s_h}");
        }
    }

    #[test]
    fn latent_extreme_truncation_is_finite() {
        let (mu, sig, s_h) = latent_from_truncated(-5.0, 0.01, 0.0, 12);
        assert!(mu.is_finite() && sig.is_finite() && s_h.is_finite());
    }

    #[test]
    fn p2_returns_default_before_five_samples() {
        let mut p2 = P2::new(0.95);
        assert_eq!(p2.value(1.234), 1.234);
        for i in 0..3 {
            p2.add(0.1 * (i as f64 + 1.0));
        }
        assert_eq!(p2.value(1.234), 1.234);
    }

    #[test]
    fn p2_tracks_high_quantile() {
        let mut rng = Rng::seed(11);
        let mut p2 = P2::new(0.95);
        let mut xs = Vec::with_capacity(200_000);
        for _ in 0..200_000 {
            let g = (0.4f64.ln() + 0.55 * std_normal(&mut rng)).exp(); // lognormal
            p2.add(g);
            xs.push(g);
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let emp = xs[(0.95 * xs.len() as f64) as usize];
        let err = p2.value(0.0) / emp - 1.0;
        assert!(err.abs() < 0.02, "P2 0.95 err {:.3}%", err * 100.0);
    }

    #[test]
    fn west_ewma_variance_is_unbiased() {
        let g = 0.999;
        let mut acc = 0.0;
        let seeds = 4;
        for s in 0..seeds {
            let mut rng = Rng::seed(100 + s);
            let (mut mu, mut var) = (0.0, 1.0);
            for _ in 0..800_000 {
                let v = std_normal(&mut rng);
                let diff = v - mu;
                let incr = (1.0 - g) * diff;
                mu += incr;
                var = g * (var + diff * incr);
            }
            acc += var;
        }
        let mean_var = acc / seeds as f64;
        assert!((mean_var - 1.0).abs() < 0.03, "mean EWMA var {mean_var:.4}");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn sigmoid_is_overflow_safe() {
        assert_eq!(sigmoid(1e6), 1.0);
        assert_eq!(sigmoid(-1e6), 0.0);
    }

    #[test]
    fn branch_coverage_all_paths_hit() {
        let p = Params {
            n_warm: 500,
            ..Params::default()
        };
        let mut eng = Pace::new(p, Rng::seed(0), 0.6f64.ln(), 0.25);
        let mut rng = Rng::seed(7);
        for _ in 0..20_000 {
            eng.step(|limit| {
                let g = (0.5f64.ln() + 0.6 * std_normal(&mut rng)).exp();
                if g <= limit {
                    (g, true)
                } else {
                    (limit, false)
                }
            });
        }
        let b = &eng.branches;
        assert!(b.warming > 0, "warming");
        assert!(b.steady > 0, "steady");
        assert!(b.ingest > 0, "ingest");
        assert!(b.resume_no_ingest > 0, "resume_no_ingest");
        assert!(b.endpoint > 0, "endpoint");
    }

    #[test]
    fn terminal_events_do_not_drag_tau_act() {
        // Half the events are TERMINAL (speaker genuinely stopped — never
        // resumes, at any wait limit); the rest are internal pauses
        // ~Lognormal(ln 0.4, 0.55), true Q95 ~= 0.99. Terminals never ingest
        // (they produce no resume), so they must not pull tau_act toward
        // TAU_MAX=2.0 the way a naive "no-resume => long pause" learner would.
        let p = Params {
            n_warm: 2000,
            ..Params::default()
        };
        let mut eng = Pace::new(p, Rng::seed(1), 0.6f64.ln(), 0.25);
        let mut rng = Rng::seed(7);
        let mut term = Rng::seed(99);
        let (mut sum, mut cnt) = (0.0, 0u64);
        for ev in 0..40_000 {
            let r = eng.step(|limit| {
                if term.next_f64() < 0.5 {
                    (limit, false) // terminal: never resumes
                } else {
                    let g = (0.4f64.ln() + 0.55 * std_normal(&mut rng)).exp();
                    if g <= limit {
                        (g, true)
                    } else {
                        (limit, false)
                    }
                }
            });
            if ev >= 25_000 {
                sum += r.tau_act;
                cnt += 1;
            }
        }
        let mean_tau = sum / cnt as f64;
        // Stays near the internal Q95 (~0.99), far below TAU_MAX=2.0.
        assert!(
            (mean_tau - 0.99).abs() < 0.15,
            "tau_act {mean_tau} dragged by 50% terminal events"
        );
    }

    #[test]
    fn drift_reset_fires_on_regime_change() {
        let p = Params {
            n_warm: 500,
            ..Params::default()
        };
        let mut eng = Pace::new(p, Rng::seed(0), 0.6f64.ln(), 0.25);
        let mut rng = Rng::seed(3);
        let mut first_reset: Option<usize> = None;
        for ev in 0..16_000 {
            // regime A (short pauses) then regime B (long pauses) at ev=8000
            let (m, s) = if ev < 8_000 {
                (0.35f64.ln(), 0.5)
            } else {
                (1.10f64.ln(), 0.5)
            };
            let r = eng.step(|limit| {
                let g = (m + s * std_normal(&mut rng)).exp();
                if g <= limit {
                    (g, true)
                } else {
                    (limit, false)
                }
            });
            if r.did_reset && first_reset.is_none() && ev > 8_000 {
                first_reset = Some(ev);
            }
        }
        assert!(eng.branches.drift_reset > 0, "drift_reset never fired");
        assert!(first_reset.is_some(), "no reset after regime change");
    }
}
