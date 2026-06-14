//! Real-data pipeline-integration benchmark for the homophone resolver.
//!
//! Unlike `homophone_tune` (fully synthetic), this runs on the user's ACTUAL
//! dictation history from the sqlite DB:
//!   - trains the n-gram on real `sessions.transcript` text,
//!   - replays the REAL `correction_pairs` (raw ASR → accepted final) if any,
//!   - and, for volume + known ground truth, injects ASR-style homophone
//!     confusion into real transcripts (real language, simulated error).
//!
//! It then evaluates SEVERAL distinct ways the resolver could plug into the
//! real pipeline, side by side, so the integration decision is data-driven:
//!
//!   PreLLM  · Static global     — drop-in `MarginGate(tau=2.0)`, every set on
//!   PreLLM  · Conformal per-set — self-evaluating guard (CRC α-bounded tau)
//!   PreLLM  · Conformal + cgain — adds the contextual-gain gate
//!   PreLLM  · Rerank (ungated)  — the anti-pattern, for contrast
//!   PostLLM · Safety-net        — run on the accepted output to catch misses
//!   GateSig · Confidence signal — non-editing: flag utterances for the LLM
//!
//! Run:   cargo run --release --example homophone_pipeline -p lune-app
//! DB:    LUNE_DB=/path/to/lune.db cargo run --release --example homophone_pipeline -p lune-app

use std::path::PathBuf;
use std::time::Instant;

use lune_dict::homophone::{
    CONFUSION_SETS, HomophoneResolver, ResolverConfig, ResolverPolicy, SetCalibration,
};
use lune_dict::homophone_calib;
use lune_dict::ngram::{NGramModel, ScoringConfig};
use lune_persist::{AppPaths, Database};

const ALPHA: f64 = 0.01; // conformal over-correction budget
const P_ERR: f64 = 0.15; // injected ASR homophone substitution rate
const MAX_SESSIONS: u32 = 8000;

// ─── deterministic RNG ───────────────────────────────────────────────────────
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

/// Lowercase + trim edge punctuation (interior apostrophe kept) — mirrors the
/// resolver's internal normalization so set membership matches.
fn norm(w: &str) -> String {
    w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .to_ascii_lowercase()
}

/// The confusion set (member surface forms) containing `word`, if any.
fn members_of(word: &str) -> Option<&'static [&'static str]> {
    let n = norm(word);
    CONFUSION_SETS
        .iter()
        .find(|s| s.members.iter().any(|m| norm(m) == n))
        .map(|s| s.members)
}

/// Split a transcript into lowercase word tokens for evaluation.
fn words_of(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(norm)
        .filter(|w| !w.is_empty())
        .collect()
}

/// Inject ASR-style homophone confusion into a real transcript: each homophone
/// is swapped for a random other set member with probability `P_ERR`.
fn corrupt(rng: &mut Rng, truth: &[String]) -> Vec<String> {
    truth
        .iter()
        .map(|w| match members_of(w) {
            Some(mem) if rng.unit() < P_ERR => {
                let alts: Vec<&str> = mem.iter().copied().filter(|m| norm(m) != *w).collect();
                if alts.is_empty() {
                    w.clone()
                } else {
                    norm(alts[rng.below(alts.len())])
                }
            }
            _ => w.clone(),
        })
        .collect()
}

#[derive(Default)]
struct Metrics {
    utterances: usize,
    touched: usize,
    fixes: usize,
    breaks: usize,
    misses: usize,
    correct_slots: usize,
    error_slots: usize,
    durations_ns: Vec<u128>,
}

impl Metrics {
    fn net(&self) -> i64 {
        self.fixes as i64 - self.breaks as i64
    }
    fn over_rate(&self) -> f64 {
        self.breaks as f64 / self.correct_slots.max(1) as f64
    }
    fn pct_touched(&self) -> f64 {
        100.0 * self.touched as f64 / self.utterances.max(1) as f64
    }
    fn pctl(&self, p: f64) -> f64 {
        if self.durations_ns.is_empty() {
            return 0.0;
        }
        let mut v = self.durations_ns.clone();
        v.sort_unstable();
        let idx = ((v.len() - 1) as f64 * p).round() as usize;
        v[idx] as f64 / 1000.0 // µs
    }
}

/// Evaluate a PreLLM editing strategy: resolve the (corrupted) raw, compare the
/// result to the accepted truth at every homophone slot.
fn eval_prellm(
    model: &NGramModel,
    resolver: &HomophoneResolver,
    test: &[(Vec<String>, Vec<String>)],
) -> Metrics {
    let mut m = Metrics::default();
    for (raw, truth) in test {
        m.utterances += 1;
        let words: Vec<&str> = raw.iter().map(String::as_str).collect();
        let t0 = Instant::now();
        let res = resolver.resolve_words(model, &words, &[]);
        m.durations_ns.push(t0.elapsed().as_nanos());

        let mut out = raw.clone();
        for r in &res {
            out[r.index] = r.to.clone();
        }
        if !res.is_empty() {
            m.touched += 1;
        }
        for (i, tw) in truth.iter().enumerate() {
            if members_of(tw).is_none() {
                continue;
            }
            let observed_correct = raw[i] == *tw;
            if observed_correct {
                m.correct_slots += 1;
                m.breaks += usize::from(out[i] != *tw);
            } else {
                m.error_slots += 1;
                if out[i] == *tw {
                    m.fixes += 1;
                } else {
                    m.misses += 1;
                }
            }
        }
    }
    m
}

fn build_resolver(kind: &str, cal: &[Option<SetCalibration>]) -> HomophoneResolver {
    match kind {
        "static" => HomophoneResolver::with_config(ResolverConfig::default()),
        "conformal" => HomophoneResolver::with_calibration(ResolverConfig::default(), cal.to_vec()),
        "conformal_cgain" => HomophoneResolver::with_calibration(
            ResolverConfig {
                min_context_gain: 1.0,
                ..ResolverConfig::default()
            },
            cal.to_vec(),
        ),
        "rerank" => HomophoneResolver::with_config(ResolverConfig {
            policy: ResolverPolicy::RerankArgmax,
            ..ResolverConfig::default()
        }),
        _ => unreachable!(),
    }
}

fn main() {
    // ── locate + open the real DB ───────────────────────────────────────────
    let db_path = std::env::var("LUNE_DB")
        .ok()
        .map(PathBuf::from)
        .or_else(|| AppPaths::resolve().map(|p| p.database_file()));
    let Some(db_path) = db_path else {
        eprintln!("Could not resolve an app data dir. Set LUNE_DB=/path/to/lune.db");
        return;
    };
    if !db_path.exists() {
        eprintln!("No DB at {db_path:?}. Dictate a few sessions first, or set LUNE_DB.");
        return;
    }
    let mut db = match Database::open(&db_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to open {db_path:?}: {e}");
            return;
        }
    };

    // ── load real history ───────────────────────────────────────────────────
    let sessions = db.list_sessions(MAX_SESSIONS).unwrap_or_default();
    // list_sessions is newest-first; reverse to oldest-first for a temporal split.
    let mut transcripts: Vec<String> = sessions
        .iter()
        .rev()
        .filter(|s| matches!(s.status, lune_core::session::SessionStatus::Completed))
        .filter_map(|s| s.transcript.clone())
        .filter(|t| !t.trim().is_empty())
        .collect();
    let real_pairs = db.list_correction_pairs(MAX_SESSIONS).unwrap_or_default();

    println!("== Homophone pipeline benchmark (REAL data) ==");
    println!("DB:           {}", db_path.display());
    println!("transcripts:  {}", transcripts.len());
    println!("real pairs:   {}", real_pairs.len());

    if transcripts.len() < 20 {
        eprintln!(
            "\nOnly {} usable transcripts — too few for a meaningful run.\n\
             Dictate more, then re-run. (Need ~20+.)",
            transcripts.len()
        );
        return;
    }

    // ── temporal split: train/calibrate on the older 70%, test on newer 30% ──
    let split = transcripts.len() * 7 / 10;
    let test_transcripts = transcripts.split_off(split);
    let calib_transcripts = transcripts;

    let mut model = NGramModel::new(ScoringConfig::trigram_default()).expect("model");
    for t in &calib_transcripts {
        model.train_sentence(t);
    }

    let mut rng = Rng::new(0xA5A5);
    let calib_pairs: Vec<(String, String)> = calib_transcripts
        .iter()
        .map(|t| {
            let truth = words_of(t);
            (corrupt(&mut rng, &truth).join(" "), truth.join(" "))
        })
        .collect();
    let test: Vec<(Vec<String>, Vec<String>)> = test_transcripts
        .iter()
        .map(|t| {
            let truth = words_of(t);
            (corrupt(&mut rng, &truth), truth)
        })
        .collect();

    let homophone_slots: usize = test
        .iter()
        .flat_map(|(_, t)| t.iter())
        .filter(|w| members_of(w).is_some())
        .count();
    println!(
        "split:        {} calib / {} test transcripts, {} homophone slots in test\n",
        calib_transcripts.len(),
        test.len(),
        homophone_slots
    );

    // ── conformal calibration on the real-language calib split ───────────────
    let cal = homophone_calib::calibrate(&model, &calib_pairs, ALPHA, 1);
    let cal_loose = homophone_calib::calibrate(&model, &calib_pairs, 0.05, 1);
    let n_strict = cal
        .iter()
        .filter(|c| c.as_ref().is_some_and(|c| c.enabled))
        .count();
    let n_loose = cal_loose
        .iter()
        .filter(|c| c.as_ref().is_some_and(|c| c.enabled))
        .count();
    println!("-- conformal calibration (alpha={ALPHA}) --");
    println!("{}", homophone_calib::summarize(&cal));
    println!("(alpha=0.05 would enable {n_loose} sets vs {n_strict} at alpha={ALPHA})\n");

    // ── strategy comparison (PreLLM editing) ─────────────────────────────────
    println!(
        "== Strategy comparison — PreLLM editing (on {} test utterances) ==",
        test.len()
    );
    println!(
        "{:<26} | {:>7} {:>5} {:>6} {:>6} {:>5} | {:>8} | {:>7} {:>7}",
        "strategy", "touch%", "fix", "break", "miss", "net", "overrate", "p50us", "p95us"
    );
    println!("{}", "-".repeat(92));
    let rows = [
        ("PreLLM · Static global", build_resolver("static", &cal)),
        (
            "PreLLM · Conformal a=0.01",
            build_resolver("conformal", &cal),
        ),
        (
            "PreLLM · Conformal a=0.05",
            build_resolver("conformal", &cal_loose),
        ),
        (
            "PreLLM · Conformal + cgain",
            build_resolver("conformal_cgain", &cal),
        ),
        ("PreLLM · Rerank (ungated)", build_resolver("rerank", &cal)),
    ];
    for (label, r) in &rows {
        let m = eval_prellm(&model, r, &test);
        println!(
            "{label:<26} | {:>6.1} {:>5} {:>6} {:>6} {:>5} | {:>8.4} | {:>7.1} {:>7.1}",
            m.pct_touched(),
            m.fixes,
            m.breaks,
            m.misses,
            m.net(),
            m.over_rate(),
            m.pctl(0.50),
            m.pctl(0.95),
        );
    }

    // baseline floor: raw vs truth, no resolver
    let base_errors: usize = test
        .iter()
        .flat_map(|(raw, t)| raw.iter().zip(t).filter(|(_, w)| members_of(w).is_some()))
        .filter(|(r, t)| r != t)
        .count();
    println!(
        "\nbaseline homophone errors (no resolver): {base_errors} / {homophone_slots} slots\n"
    );

    // ── PostLLM safety-net: run on the ACCEPTED output, count second-guesses ──
    let conformal = build_resolver("conformal", &cal);
    let (mut post_touched, mut post_changes) = (0usize, 0usize);
    for (_, truth) in &test {
        let words: Vec<&str> = truth.iter().map(String::as_str).collect();
        let res = conformal.resolve_words(&model, &words, &[]);
        if !res.is_empty() {
            post_touched += 1;
            post_changes += res.len();
        }
    }
    println!("== PostLLM · Safety-net (resolver run on the accepted output) ==");
    println!(
        "would touch {}/{} accepted utterances ({} slot changes) — each is a proposed catch \
         OR a disagreement with the accepted text; inspect before enabling.\n",
        post_touched,
        test.len(),
        post_changes
    );

    // ── GateSignal: non-editing flag for the bypass gate ─────────────────────
    let (mut flagged, mut flagged_with_err, mut utt_with_err) = (0usize, 0usize, 0usize);
    for (raw, truth) in &test {
        let has_err = raw
            .iter()
            .zip(truth)
            .any(|(r, t)| members_of(t).is_some() && r != t);
        let words: Vec<&str> = raw.iter().map(String::as_str).collect();
        let flag = !conformal.resolve_words(&model, &words, &[]).is_empty();
        utt_with_err += usize::from(has_err);
        flagged += usize::from(flag);
        flagged_with_err += usize::from(flag && has_err);
    }
    let precision = 100.0 * flagged_with_err as f64 / flagged.max(1) as f64;
    let recall = 100.0 * flagged_with_err as f64 / utt_with_err.max(1) as f64;
    println!("== GateSignal · Confidence signal (non-editing) ==");
    println!(
        "flags {}/{} utterances; of flagged, {:.1}% truly contained a homophone error \
         (precision); catches {:.1}% of utterances that had one (recall).\n",
        flagged,
        test.len(),
        precision,
        recall
    );

    // ── real correction pairs (gold, if the user has dictated post-feature) ──
    if !real_pairs.is_empty() {
        // Only equal-length pairs admit a positional homophone comparison;
        // length changes come from LLM insert/delete and are out of scope here.
        let aligned: Vec<(Vec<String>, Vec<String>)> = real_pairs
            .iter()
            .map(|(r, f)| (words_of(r), words_of(f)))
            .filter(|(r, f)| r.len() == f.len())
            .collect();
        println!("== REAL correction pairs (gold: raw ASR → accepted final) ==");
        println!(
            "{} pairs total, {} usable (equal length); conformal resolver on them:",
            real_pairs.len(),
            aligned.len()
        );
        if !aligned.is_empty() {
            let m = eval_prellm(&model, &conformal, &aligned);
            println!(
                "  fix={} break={} miss={} net={} overrate={:.4}\n",
                m.fixes,
                m.breaks,
                m.misses,
                m.net(),
                m.over_rate()
            );
        } else {
            println!();
        }
    }

    // ── concrete examples: real utterances where strategies diverge ──────────
    println!("== Example real utterances (raw → conformal → truth) ==");
    let mut shown = 0;
    for (raw, truth) in &test {
        if shown >= 5 {
            break;
        }
        let words: Vec<&str> = raw.iter().map(String::as_str).collect();
        let res = conformal.resolve_words(&model, &words, &[]);
        if res.is_empty() {
            continue;
        }
        let mut out = raw.clone();
        for r in &res {
            out[r.index] = r.to.clone();
        }
        if out != *truth || raw != truth {
            println!("  raw:   {}", raw.join(" "));
            println!("  fixed: {}", out.join(" "));
            println!("  truth: {}\n", truth.join(" "));
            shown += 1;
        }
    }
    if shown == 0 {
        println!(
            "  (resolver made no edits on the test split — likely all sets disabled or low signal)"
        );
    }
}
