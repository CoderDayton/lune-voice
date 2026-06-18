//! Seed the CSC (LLM-edit gate) calibration from realistic synthetic errors.
//!
//! The conformal gate stays disabled until enough real `(raw → final)` edit
//! observations accumulate. This bootstraps it: inject ASR-style errors into
//! clean reference text, run the REAL correction pipeline to harvest decode
//! margins (NOT fabricated — fabricated margins would calibrate τ to fiction
//! and never transfer to a real decode), label each observation exactly as the
//! runtime does, then calibrate.
//!
//! Diagnostics only by default. With `--write` it persists the harvested
//! observations into the user's `edit_observations` table so the next launch
//! enables CSC from this warm-start prior (which real observations then dilute,
//! the table trimming oldest-first).
//!
//! Run:   cargo run --release --features cuda --example seed_edit_calib -p lune-app
//!        cargo run --release --features cuda --example seed_edit_calib -p lune-app -- --write

use std::path::PathBuf;
use std::sync::Arc;

use lune_core::config::StylePreset;
use lune_core::llm::{LlmService, PostProcessInput, TextPostProcessor};
use lune_dict::edit_calib::{self, EditObs};
use lune_llm::CorrectionPipeline;
use lune_llm::backend::ort_llm::{OrtStyleLlm, OrtStyleLlmConfig};

fn model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = PathBuf::from(home).join(".local/share/lune-voice/models/qwen3-0.6b");
    dir.exists().then_some(dir)
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Deterministic PRNG — `Math.random`/time are intentionally unavailable; the
/// seed makes each run reproducible so the harvested calibration is stable.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

/// Homophone confusions the validator's `ALLOWED_SUBS` can repair (so the LLM's
/// fix is rule-accepted → labelled as a beneficial flip with a real margin).
const HOMOPHONE_SWAPS: &[(&str, &str)] = &[
    ("to", "too"),
    ("too", "to"),
    ("there", "their"),
    ("their", "there"),
    ("know", "no"),
    ("no", "know"),
    ("write", "right"),
    ("right", "write"),
    ("hear", "here"),
    ("here", "hear"),
    ("by", "buy"),
    ("buy", "by"),
    ("for", "four"),
    ("one", "won"),
    ("meet", "meat"),
    ("brake", "break"),
    ("break", "brake"),
    ("weather", "whether"),
];

fn strip_word(w: &str) -> (&str, &str) {
    // split trailing punctuation so swaps match on the bare alpha token
    let end = w
        .char_indices()
        .rev()
        .take_while(|(_, c)| !c.is_alphanumeric())
        .last()
        .map(|(i, _)| i)
        .unwrap_or(w.len());
    w.split_at(end)
}

/// Introduce a 1-char spelling slip (validator `looks_like_spelling_repair`
/// fixes Levenshtein ≤ 1 on words of length ≥ 4).
fn typo(word: &str, rng: &mut Rng) -> String {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 5 {
        return word.to_string();
    }
    let i = 1 + rng.below(chars.len() - 2); // keep first/last stable
    let mut out: Vec<char> = chars.clone();
    match rng.below(3) {
        0 => {
            out.remove(i);
        } // drop
        1 => {
            out.insert(i, chars[i]);
        } // duplicate
        _ => {
            // adjacent-key-ish substitution: shift one letter
            let c = chars[i];
            if c.is_ascii_lowercase() {
                let shifted = (((c as u8 - b'a' + 1) % 26) + b'a') as char;
                out[i] = shifted;
            }
        }
    }
    out.into_iter().collect()
}

/// Corrupt a clean sentence with a few rule-fixable errors. Returns the
/// corrupted sentence; the clean original is the ground truth.
fn corrupt(clean: &str, rng: &mut Rng) -> String {
    let words: Vec<&str> = clean.split_whitespace().collect();
    if words.is_empty() {
        return clean.to_string();
    }
    let mut out: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    let mut hits = 0;
    let budget = 1 + words.len() / 8; // ~1 error per 8 words
    for (idx, w) in words.iter().enumerate() {
        if hits >= budget {
            break;
        }
        let (bare, punct) = strip_word(w);
        let lower = bare.to_ascii_lowercase();
        // 1) homophone swap
        if let Some((_, to)) = HOMOPHONE_SWAPS.iter().find(|(from, _)| *from == lower) {
            if rng.chance(60) {
                let cased = if bare.chars().next().is_some_and(|c| c.is_uppercase()) {
                    let mut cs: Vec<char> = to.chars().collect();
                    if let Some(first) = cs.first_mut() {
                        *first = first.to_ascii_uppercase();
                    }
                    cs.into_iter().collect::<String>()
                } else {
                    (*to).to_string()
                };
                out[idx] = format!("{cased}{punct}");
                hits += 1;
                continue;
            }
        }
        // 2) spelling slip on a longer word
        if bare.len() >= 5 && rng.chance(25) {
            let slipped = typo(bare, rng);
            if slipped != bare {
                out[idx] = format!("{slipped}{punct}");
                hits += 1;
            }
        }
    }
    out.join(" ")
}

/// Split a passage into rough sentences for more, shorter samples.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let s = cur.trim().to_string();
            if s.split_whitespace().count() >= 3 {
                out.push(s);
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if tail.split_whitespace().count() >= 3 {
        out.push(tail.to_string());
    }
    out
}

fn main() {
    let write = std::env::args().any(|a| a == "--write");

    // `--check`: read the existing edit_observations and report what the runtime
    // calibration (with the τ floor) would compute. No model, no generation.
    if std::env::args().any(|a| a == "--check") {
        match lune_persist::AppPaths::resolve() {
            Some(paths) => match lune_persist::Database::open(&paths.database_file()) {
                Ok(mut db) => {
                    let rows = db
                        .list_edit_observations(lune_persist::EDIT_OBSERVATION_CAP)
                        .unwrap_or_default();
                    let obs: Vec<EditObs> = rows
                        .into_iter()
                        .map(|(oc, m, ff)| EditObs {
                            observed_correct: oc,
                            margin: m.map(f64::from),
                            flip_fixes: ff,
                        })
                        .collect();
                    println!("DB edit_observations: {}", obs.len());
                    match edit_calib::calibrate_edits(
                        &obs,
                        edit_calib::ALPHA,
                        edit_calib::TAU_FLOOR,
                    ) {
                        Some(c) => println!(
                            "calibration: enabled={} tau={:.3} net={:+.0} n_correct={} n_error={}",
                            c.enabled, c.tau, c.net_benefit, c.n_correct, c.n_error
                        ),
                        None => println!("calibration: None (no observations)"),
                    }
                }
                Err(e) => eprintln!("open DB failed: {e}"),
            },
            None => eprintln!("could not resolve app data dir"),
        }
        return;
    }

    let Some(dir) = model_dir() else {
        eprintln!("[skip] model not found at ~/.local/share/lune-voice/models/qwen3-0.6b");
        return;
    };

    // ── clean English corpus from fixture references ─────────────────────────
    let fixtures = workspace_root().join("fixtures/audio");
    let mut corpus: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(&fixtures) {
        for e in entries.flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if p.extension().and_then(|s| s.to_str()) != Some("txt") {
                continue;
            }
            if name.starts_with("mls_") {
                continue; // non-English; the gate is English-only
            }
            if let Ok(text) = std::fs::read_to_string(&p) {
                for s in sentences(&text) {
                    if seen.insert(s.clone()) {
                        corpus.push(s);
                    }
                }
            }
        }
    }
    println!("corpus: {} unique English sentences", corpus.len());
    if corpus.is_empty() {
        eprintln!("[skip] no English fixture references found");
        return;
    }

    // ── real correction pipeline ─────────────────────────────────────────────
    let llm = match OrtStyleLlm::new(OrtStyleLlmConfig {
        model_dir: dir,
        use_gpu: true,
        cuda_required: false,
        max_new_tokens: 512,
    }) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[skip] LLM load failed: {e}");
            return;
        }
    };
    let pipeline = CorrectionPipeline::new(Arc::new(llm) as Arc<dyn LlmService>);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    let mut obs: Vec<EditObs> = Vec::new();
    let mut corrupted_n = 0usize;

    for (i, clean) in corpus.iter().enumerate() {
        let bad = corrupt(clean, &mut rng);
        if bad != *clean {
            corrupted_n += 1;
        }
        let input = PostProcessInput {
            raw_text: bad.clone(),
            style_preset: Arc::new(StylePreset::minimal()),
            app_category: None,
            dictionary: Vec::new(),
            language: "en".into(),
            correction_needed: true,
        };
        match rt.block_on(pipeline.transform(&input)) {
            Ok(out) => {
                for (observed_correct, margin, flip_fixes) in out.edit_observations {
                    obs.push(EditObs {
                        observed_correct,
                        margin: margin.map(f64::from),
                        flip_fixes,
                    });
                }
            }
            Err(e) => eprintln!("  [{i}] transform failed: {e}"),
        }
    }

    // ── harvest summary ──────────────────────────────────────────────────────
    let n_correct = obs.iter().filter(|o| o.observed_correct).count();
    let n_error = obs.len() - n_correct;
    let with_margin = obs.iter().filter(|o| o.margin.is_some()).count();
    let correct_with_margin = obs
        .iter()
        .filter(|o| o.observed_correct && o.margin.is_some())
        .count();
    let fixes_with_margin = obs
        .iter()
        .filter(|o| !o.observed_correct && o.flip_fixes && o.margin.is_some())
        .count();
    println!(
        "\nharvested {} observations from {corrupted_n} corrupted sentences",
        obs.len()
    );
    println!("  n_correct={n_correct}  n_error={n_error}  with_margin={with_margin}");
    println!("  correct_with_margin (over-risk pool)={correct_with_margin}");
    println!("  error+flip_fixes+margin (fix pool)={fixes_with_margin}");

    let mut fix_margins: Vec<f64> = obs
        .iter()
        .filter(|o| !o.observed_correct && o.flip_fixes)
        .filter_map(|o| o.margin)
        .collect();
    fix_margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  fix margins: {fix_margins:?}");

    // ── calibrate ────────────────────────────────────────────────────────────
    let alpha = edit_calib::ALPHA;
    match edit_calib::calibrate_edits(&obs, alpha, edit_calib::TAU_FLOOR) {
        Some(c) => {
            println!(
                "\ncalibration @ α={alpha} (τ floored at {}): enabled={} tau={:.3} net={:+.0} (n_correct={}, n_error={})",
                edit_calib::TAU_FLOOR,
                c.enabled,
                c.tau,
                c.net_benefit,
                c.n_correct,
                c.n_error
            );
            if !c.enabled {
                println!("  NOT enabled — need net>0, decisive≥5, Wilson win-rate>0.5 above tau.");
            }
        }
        None => println!("\ncalibration: None (no observations)"),
    }

    if write {
        println!(
            "\n[--write] persisting {} observations to the user DB…",
            obs.len()
        );
        let triples: Vec<(bool, Option<f32>, bool)> = obs
            .iter()
            .map(|o| (o.observed_correct, o.margin.map(|m| m as f32), o.flip_fixes))
            .collect();
        match lune_persist::AppPaths::resolve() {
            Some(paths) => match lune_persist::Database::open(&paths.database_file()) {
                Ok(mut db) => {
                    let sentinel = uuid::Uuid::nil();
                    match db.insert_edit_observations(sentinel, &triples) {
                        Ok(()) => {
                            println!(
                                "  wrote {} observations (session_id=nil seed).",
                                triples.len()
                            )
                        }
                        Err(e) => eprintln!("  write failed: {e}"),
                    }
                }
                Err(e) => eprintln!("  open DB failed: {e}"),
            },
            None => eprintln!("  could not resolve app data dir"),
        }
    } else {
        println!("\n(diagnostics only — re-run with `-- --write` to persist)");
    }
}
