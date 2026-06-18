//! Conformal LLM-edit gate: OFF vs ON.
//!
//! Run:   cargo run --release --example llm_edit_gate -p lune-app
//!
//! Measures the benefit of gating the open-vocabulary LLM correction: each LLM
//! edit carries a decode-confidence **margin** (exposed by `generate_scored` —
//! `logit(emitted) − max logit(other)`, ≥ 0 under greedy decode), and we apply
//! the edit only when the margin clears a threshold `tau` calibrated from
//! `(raw, final)` history via **Conformal Risk Control** (Angelopoulos et al.,
//! arXiv:2208.02814), enabled only under a Wilson net-benefit lower bound
//! (Chow 1970 reject option).
//!
//! No labelled open-vocab correction corpus exists, so ground truth is
//! synthetic, but the calibration math runs the production
//! `edit_calib::calibrate_edits` / `crc::conformal_threshold` / `crc::wilson_lower`
//! (same n_correct denominator, same `0.0` no-constraint branch, same rule).
//!
//!   OFF = trust every LLM edit (apply the rewrite wholesale once the LLM ran —
//!         note this is *post* the `gate.rs` run/skip decision, not `gate.rs`)
//!   ON  = apply an edit only if the channel is enabled AND margin > tau
//!
//! Metrics (over edit slots):
//!   base     = error if we keep raw ASR (no LLM)
//!   off/on   = error after applying edits under each policy
//!   over     = fraction of ALREADY-CORRECT words the policy broke
//!   net      = base - err  (higher = better; negative = the policy hurt)

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
    /// Standard normal via Box–Muller.
    fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ─── scenario constants ──────────────────────────────────────────────────────
const P_ERROR: f64 = 0.18; // fraction of slots the ASR gets wrong
const PROPOSE_ERR: f64 = 0.90; // LLM proposes an edit on a real error
const PROPOSE_COR: f64 = 0.20; // LLM over-eagerly edits an already-correct word
const FIX_ACC: f64 = 0.70; // a proposed edit on a real error lands on the truth
const SIGMA: f64 = 1.0; // margin noise
const N_CALIB: usize = 4_000;
const N_TEST: usize = 20_000;
const SEEDS: u64 = 8;

/// One slot observation — the synthetic analogue of `edit_calib::EditObs`.
/// Recorded for EVERY slot (not just proposed edits): a correct word the LLM
/// left alone is `margin: None`, observed_correct: true — and it still counts
/// toward the conformal denominator, exactly as the real calibrator does.
type Obs = lune_dict::edit_calib::EditObs;

/// Draw one slot. Margins are non-negative (greedy decode `top1−top2 ≥ 0`):
/// corruptions ~ |N(0, σ)| (low confidence), good fixes ~ max(0, N(sep, σ)).
/// `sep` is the separation between good edits and corruptions — the signal
/// strength of the decode confidence.
fn draw_slot(rng: &mut Rng, sep: f64) -> Obs {
    let observed_correct = rng.unit() >= P_ERROR;
    if observed_correct {
        if rng.unit() < PROPOSE_COR {
            // Over-correction: editing a correct word always breaks it.
            Obs {
                observed_correct: true,
                margin: Some(rng.gauss(0.0, SIGMA).abs()),
                flip_fixes: false,
            }
        } else {
            Obs {
                observed_correct: true,
                margin: None,
                flip_fixes: false,
            }
        }
    } else if rng.unit() < PROPOSE_ERR {
        let fixes = rng.unit() < FIX_ACC;
        let mu = if fixes { sep } else { sep / 2.0 };
        Obs {
            observed_correct: false,
            margin: Some(rng.gauss(mu, SIGMA).max(0.0)),
            flip_fixes: fixes,
        }
    } else {
        Obs {
            observed_correct: false,
            margin: None,
            flip_fixes: false,
        } // LLM left a real error
    }
}

// ─── calibration via the production CRC kernel ──────────────────────────────

/// Calibrate the LLM-edit channel using the REAL production calibrator
/// (`lune_dict::edit_calib::calibrate_edits` → `lune_dict::crc`), so this
/// benchmark exercises shipped code rather than a copy. Returns the gate
/// `(enabled, tau)`; an empty/insufficient history disables the gate.
fn calibrate(history: &[Obs], alpha: f64) -> (bool, f64) {
    match lune_dict::edit_calib::calibrate_edits(history, alpha, 0.0) {
        Some(c) => (c.enabled, c.tau),
        None => (false, f64::INFINITY),
    }
}

// ─── evaluation ──────────────────────────────────────────────────────────────
#[derive(Default, Clone, Copy)]
struct Metrics {
    base_err: f64,
    off_err: f64,
    on_err: f64,
    off_over: f64,
    on_over: f64,
    enabled: f64,
}

/// Average over `SEEDS` independent calibrate/test draws.
fn evaluate(alpha: f64, sep: f64) -> Metrics {
    let mut m = Metrics::default();
    for seed in 0..SEEDS {
        let mut rng = Rng::new(7_000 + seed);

        // History → calibrate (records every slot, mirroring the real collector).
        let history: Vec<Obs> = (0..N_CALIB).map(|_| draw_slot(&mut rng, sep)).collect();
        let (enabled, tau) = calibrate(&history, alpha);
        m.enabled += f64::from(enabled);

        // Fresh test set.
        let (mut base, mut off, mut on) = (0usize, 0usize, 0usize);
        let (mut n_correct, mut off_over, mut on_over) = (0usize, 0usize, 0usize);
        for _ in 0..N_TEST {
            let o = draw_slot(&mut rng, sep);
            base += usize::from(!o.observed_correct);
            if o.observed_correct {
                n_correct += 1;
            }

            // Apply condition mirrors the over-count: strict margin > tau.
            let off_apply = o.margin.is_some();
            let on_apply = enabled && o.margin.is_some_and(|x| x > tau);
            let off_final = if off_apply {
                o.flip_fixes
            } else {
                o.observed_correct
            };
            let on_final = if on_apply {
                o.flip_fixes
            } else {
                o.observed_correct
            };

            off += usize::from(!off_final);
            on += usize::from(!on_final);
            if o.observed_correct {
                off_over += usize::from(!off_final); // a correct word the policy broke
                on_over += usize::from(!on_final);
            }
        }
        let s = N_TEST as f64;
        let c = n_correct.max(1) as f64;
        m.base_err += base as f64 / s;
        m.off_err += off as f64 / s;
        m.on_err += on as f64 / s;
        m.off_over += off_over as f64 / c;
        m.on_over += on_over as f64 / c;
    }
    let d = SEEDS as f64;
    Metrics {
        base_err: m.base_err / d,
        off_err: m.off_err / d,
        on_err: m.on_err / d,
        off_over: m.off_over / d,
        on_over: m.on_over / d,
        enabled: m.enabled / d,
    }
}

fn row(label: String, m: Metrics, alpha: Option<f64>) {
    let flag = match alpha {
        Some(a) if m.on_over <= a + 0.005 => "ok",
        Some(_) => "HIGH",
        None => "",
    };
    println!(
        "{label:>7} | {:>6.4} {:>8.4} {:>8.4} | {:>8.4} {:>8.4} | {:>+7.4} {:>+7.4} | {:>4.2} {}",
        m.base_err,
        m.off_err,
        m.on_err,
        m.off_over,
        m.on_over,
        m.base_err - m.off_err,
        m.base_err - m.on_err,
        m.enabled,
        flag,
    );
}

fn header(first: &str) {
    println!(
        "{first:>7} | {:>6} {:>8} {:>8} | {:>8} {:>8} | {:>7} {:>7} | {:>4}",
        "base", "off_err", "on_err", "off_over", "on_over", "net_off", "net_on", "enab"
    );
    println!("{}", "-".repeat(82));
}

fn main() {
    println!("Conformal LLM-edit gate — OFF (trust every edit) vs ON (calibrated)");
    println!(
        "P_ERROR={P_ERROR}  over-correct rate={PROPOSE_COR}  fix_acc={FIX_ACC}  seeds={SEEDS}\n"
    );

    println!("== EXP 1 — over-correction guarantee (vary alpha, margin sep=2.0) ==");
    header("alpha");
    for alpha in [0.005_f64, 0.01, 0.02, 0.05, 0.10] {
        let m = evaluate(alpha, 2.0);
        row(format!("{alpha:.3}"), m, Some(alpha));
    }

    println!("\n== EXP 2 — signal strength (vary margin sep, alpha=0.02) ==");
    println!("sep=0 → margin carries no information (gate must fail safe)");
    header("sep");
    for sep in [0.0_f64, 0.5, 1.0, 2.0, 3.0, 4.0] {
        let m = evaluate(0.02, sep);
        row(format!("{sep:.1}"), m, Some(0.02));
    }
}
