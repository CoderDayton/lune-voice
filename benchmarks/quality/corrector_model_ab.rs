//! corrector_model_ab — side-by-side A/B of the shipped style/correction LLM
//! against a candidate fine-tune, across several system-prompt variants.
//!
//! Why this exists: the production correction prompt
//! (`lune_core::config::BASE_SYSTEM_PROMPT`) tells the model to *preserve any
//! punctuation already present in the input*. For dictation that is often
//! wrong — speech has no real punctuation, so the ASR's commas, periods, and
//! sentence breaks are unreliable. This bench runs the baseline model and the
//! candidate fine-tune across four prompts so we can see which model × prompt
//! pair cleans punctuation best without mangling the words:
//!   base      — the shipped prompt (control).
//!   oneliner  — a single-sentence minimal instruction.
//!   flow      — a surgical edit of base: only the "preserve punctuation" rule
//!               is swapped for "join one continuous thought with commas, not
//!               periods; merge stray fragments".
//!   tuned     — a best-judgment rewrite of base that re-punctuates but keeps
//!               capitalization, terminal marks, and every content word.
//!
//! Inputs are curated raw→reference text fixtures (no ASR needed): `raw`
//! mimics lune-voice output (wrong/missing punctuation, fragments, run-ons)
//! and `reference` is the ideal minimal-edit correction (same words, fixed
//! punctuation/casing). Metrics vs reference: WER (word correctness,
//! case/punct-insensitive), cerCASE (case+punct-sensitive char error), and two
//! punctuation F1s — pF1pos (placement-aware: a mark must land on the same
//! aligned word) and pF1cnt (count-only). pF1pos is the headline; pF1cnt is
//! kept alongside to expose where a prompt gets the mark *count* right but the
//! *placement* wrong.
//!
//! Run (GPU): cargo run --release --features cuda --example corrector_model_ab -p lune-app
//!     (CPU): cargo run --release --example corrector_model_ab -p lune-app
//!   MODELS=old|new|both                 which model(s) to load   (default both)
//!   PROMPTS=base,oneliner,flow,tuned    comma list               (default all)
//!   LUNE_BENCH_PRINT=0      hide the per-fixture side-by-side dump
//!   FORCE_CPU=1             ignore CUDA even when built with it

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::time::Instant;

use bench_utils::{cased_cer, init_ort, models_dir, percentile, word_error_rate, wrap_print};
use lune_core::llm::LlmService;
use lune_llm::{OrtStyleLlm, OrtStyleLlmConfig};

/// The shipped Qwen3-0.6B style LLM lives here. `DIR_STYLE_LLM` now points at
/// the candidate, so the baseline directory is named explicitly.
const BASELINE_DIR: &str = "qwen3-0.6b";

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// One correction case: `raw` is what the corrector receives, `reference` is
/// the ideal minimal-edit output (identical words, corrected punctuation).
struct Fixture {
    name: &'static str,
    raw: &'static str,
    reference: &'static str,
}

/// Punctuation / sentence-boundary stress set. Every reference keeps the exact
/// words of its `raw` (modulo obvious typo/contraction fixes) and only repairs
/// punctuation, casing, and sentence breaks — so a high score means the model
/// re-punctuated correctly without paraphrasing.
const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "lune_voice_paragraph",
        raw: "like on this sentence. this paragraph right here. this is using lune voice. \
              right. and punctuation is all wrong the sentence breaks are all wrong fragments \
              and everything",
        reference: "Like on this sentence, this paragraph right here, this is using Lune Voice, \
                    right? And punctuation is all wrong, the sentence breaks are all wrong, \
                    fragments and everything.",
    },
    Fixture {
        name: "run_on",
        raw: "i finished the report send it over when you get a sec ill be at lunch till one",
        reference: "I finished the report. Send it over when you get a sec. \
                    I'll be at lunch till one.",
    },
    Fixture {
        name: "fragments",
        raw: "can you tell me. what time it is. i need to leave soon",
        reference: "Can you tell me what time it is? I need to leave soon.",
    },
    Fixture {
        name: "missing_terminal",
        raw: "hey just checking in on the deploy did the migration finish if not we should \
              roll back before the demo at three",
        reference: "Hey, just checking in on the deploy. Did the migration finish? \
                    If not, we should roll back before the demo at three.",
    },
    Fixture {
        name: "comma_splice",
        raw: "we should meet at noon, then grab lunch, the place near the office is good",
        reference: "We should meet at noon, then grab lunch. The place near the office is good.",
    },
    Fixture {
        name: "questions",
        raw: "did you see john. where is he.",
        reference: "Did you see John? Where is he?",
    },
    Fixture {
        name: "typos_plus_punct",
        raw: "i dont no why the computor isnt working its been like this all day",
        reference: "I don't know why the computer isn't working. It's been like this all day.",
    },
    Fixture {
        name: "list",
        raw: "pick up bread eggs and milk on the way home also grab coffee",
        reference: "Pick up bread, eggs, and milk on the way home. Also grab coffee.",
    },
];

// ─── System-prompt variants ──────────────────────────────────────────────────

struct PromptVariant {
    name: &'static str,
    system: String,
}

/// `base` is the shipped prompt verbatim (control). `oneliner` is the minimal
/// one-sentence instruction. `flow` is a surgical edit of the base prompt that
/// only touches the comma/period-flow rule (the `lune_voice_paragraph`
/// failure). `tuned` is a best-judgment rewrite of the base prompt that learns
/// from the earlier harmful variants — it still re-punctuates, but explicitly
/// keeps capitalization, terminal marks, and every content word.
fn prompt_variants() -> Vec<PromptVariant> {
    let base = lune_core::config::BASE_SYSTEM_PROMPT.replace("{lang}", "en");
    // Surgical: replace ONLY the "preserve existing punctuation" rule. Every
    // other base rule (comma-within-sentence, split-on-thought-shift,
    // capitalization, the do-not list, the examples) is untouched. If the
    // anchor ever changes, `flow` degrades gracefully to `base`.
    let flow = base.replace(
        "- Preserve any punctuation already present in the input.",
        "- The input punctuation comes from speech pauses, so it is often wrong: \
         keep it only where it is correct.\n\
         - When consecutive short sentences are really one continuous thought, \
         join them with commas instead of periods, and merge stray fragments \
         into the sentence they belong to.",
    );
    vec![
        PromptVariant {
            name: "base",
            system: base,
        },
        PromptVariant {
            name: "oneliner",
            system: ONELINER_SYSTEM.to_string(),
        },
        PromptVariant {
            name: "flow",
            system: flow,
        },
        PromptVariant {
            name: "tuned",
            system: TUNED_SYSTEM.to_string(),
        },
    ]
}

/// Simplest possible instruction — one sentence, no rules list.
const ONELINER_SYSTEM: &str = "Correct this speech-to-text transcript: fix spelling, \
capitalization, punctuation, and sentence breaks, keeping the exact words and the same \
language. Output only the corrected transcript.";

/// Best-judgment rewrite of the base prompt. Re-punctuates (unlike base) but
/// closes the two failure modes seen earlier: it never drops capitalization or
/// terminal marks (the `resegment` failure) and never changes words (the
/// `repunct` failure).
const TUNED_SYSTEM: &str = "\
You are an ASR transcript corrector.
Output only the corrected transcript, in the same language as the input. Make the smallest edits that produce natural, correctly written text — keep every content word.

Fix:
- Misspellings, fused or split words, and obvious ASR mishearings.
- Homophones when the context makes the right word clear.
- Missing short function words, agreement, plurals, and contractions.
- A word stuttered back-to-back (keep one copy); leave intentional repetition.

Punctuation and sentence breaks:
- The input punctuation comes from speech pauses, so treat it as a hint, not the truth — fix it where it is wrong.
- Within a single continuous thought, separate clauses and pauses with commas, not periods.
- Start a new sentence only where the thought genuinely changes; merge fragments that belong together and split true run-ons.
- Give every sentence exactly one terminal mark: . for statements, ? for real questions, ! only for clear emphasis.
- Use only . , ? and ! — do not insert em-dashes, semicolons, colons, ellipses, or quotation marks unless the input already contains them.
- Always capitalize the first word of each sentence and proper nouns.

Never translate, paraphrase, reorder, add, drop, or replace content words; never censor profanity; never answer or react to the transcript. If a word might be wrong but you are unsure, keep the original.";

// ─── Metrics ─────────────────────────────────────────────────────────────────

/// Punctuation-mark F1: multiset agreement of `. , ? ! ; :` between hyp and
/// ref (1.0 = perfect). The direct proxy for "did the punctuation come out
/// right". Mirrors the metric in `punctuation_ab`.
fn punct_f1(hyp: &str, ref_: &str) -> f64 {
    use std::collections::HashMap;
    fn counts(s: &str) -> HashMap<char, i32> {
        let mut m = HashMap::new();
        for c in s
            .chars()
            .filter(|c| matches!(c, '.' | ',' | '?' | '!' | ';' | ':'))
        {
            *m.entry(c).or_insert(0) += 1;
        }
        m
    }
    let (h, r) = (counts(hyp), counts(ref_));
    let tp: i32 = r
        .iter()
        .map(|(c, &rc)| h.get(c).copied().unwrap_or(0).min(rc))
        .sum();
    let (hn, rn): (i32, i32) = (h.values().sum(), r.values().sum());
    if hn == 0 && rn == 0 {
        return 1.0;
    }
    let prec = if hn == 0 { 0.0 } else { tp as f64 / hn as f64 };
    let rec = if rn == 0 { 0.0 } else { tp as f64 / rn as f64 };
    if prec + rec == 0.0 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    }
}

const MARKS: [char; 6] = ['.', ',', '?', '!', ';', ':'];

/// Split text into `(lowercased word core, trailing punctuation marks)` tokens.
/// A pure-punctuation token attaches its marks to the previous word.
fn punct_tokens(s: &str) -> Vec<(String, Vec<char>)> {
    let mut out: Vec<(String, Vec<char>)> = Vec::new();
    for tok in s.split_whitespace() {
        let chars: Vec<char> = tok.chars().collect();
        let mut i = chars.len();
        let mut marks = Vec::new();
        while i > 0 && MARKS.contains(&chars[i - 1]) {
            marks.push(chars[i - 1]);
            i -= 1;
        }
        marks.reverse();
        let core: String = chars[..i]
            .iter()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if core.is_empty() {
            if let Some(last) = out.last_mut() {
                last.1.extend(marks);
            }
            continue;
        }
        out.push((core, marks));
    }
    out
}

/// Needleman–Wunsch alignment of two word sequences. Returns one entry per
/// alignment column: `(Some(i), Some(j))` = match/substitution, `(Some, None)`
/// = hyp-only, `(None, Some)` = ref-only.
fn align_words(a: &[String], b: &[String]) -> Vec<(Option<usize>, Option<usize>)> {
    let (n, m) = (a.len(), b.len());
    let sub = |i: usize, j: usize| usize::from(a[i] != b[j]);
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = (dp[i - 1][j - 1] + sub(i - 1, j - 1))
                .min(dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1);
        }
    }
    let (mut i, mut j) = (n, m);
    let mut res = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + sub(i - 1, j - 1) {
            res.push((Some(i - 1), Some(j - 1)));
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            res.push((Some(i - 1), None));
            i -= 1;
        } else {
            res.push((None, Some(j - 1)));
            j -= 1;
        }
    }
    res.reverse();
    res
}

/// Placement-aware punctuation F1. A mark scores as a true positive only when
/// it is attached to the *same aligned word* in both hyp and ref — so a comma
/// in the wrong slot is both a false positive and a false negative, unlike the
/// count-only [`punct_f1`]. This is the metric that sees re-segmentation.
fn punct_place_f1(hyp: &str, ref_: &str) -> f64 {
    use std::collections::HashMap;
    let (ht, rt) = (punct_tokens(hyp), punct_tokens(ref_));
    let hw: Vec<String> = ht.iter().map(|(w, _)| w.clone()).collect();
    let rw: Vec<String> = rt.iter().map(|(w, _)| w.clone()).collect();
    let counts = |marks: &[char]| {
        let mut m: HashMap<char, i32> = HashMap::new();
        for &c in marks {
            *m.entry(c).or_insert(0) += 1;
        }
        m
    };
    let (mut tp, mut fp, mut fne) = (0i32, 0i32, 0i32);
    for col in align_words(&hw, &rw) {
        match col {
            (Some(i), Some(j)) => {
                let (hc, rc) = (counts(&ht[i].1), counts(&rt[j].1));
                for &k in &MARKS {
                    let (h, r) = (*hc.get(&k).unwrap_or(&0), *rc.get(&k).unwrap_or(&0));
                    tp += h.min(r);
                    fp += (h - r).max(0);
                    fne += (r - h).max(0);
                }
            }
            (Some(i), None) => fp += ht[i].1.len() as i32,
            (None, Some(j)) => fne += rt[j].1.len() as i32,
            (None, None) => {}
        }
    }
    if tp + fp + fne == 0 {
        return 1.0;
    }
    let prec = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let rec = if tp + fne == 0 {
        0.0
    } else {
        tp as f64 / (tp + fne) as f64
    };
    if prec + rec == 0.0 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

// ─── Result rows ─────────────────────────────────────────────────────────────

/// One (model, prompt, fixture) run.
struct Run {
    model: &'static str,
    prompt: &'static str,
    fixture: &'static str,
    output: String,
    wer: f64,
    cer: f64,
    /// Placement-aware punctuation F1 (headline — sees re-segmentation).
    pf1_pos: f64,
    /// Count-only punctuation F1 (mark multiset; blind to placement).
    pf1_cnt: f64,
    ms: f64,
}

struct ModelEntry {
    /// Short key for the `MODELS` filter (`old` / `new`).
    key: &'static str,
    label: &'static str,
    dir_name: &'static str,
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          CORRECTOR MODEL × PROMPT A/B (punctuation)              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();
    let dir = models_dir();

    let all_models = [
        ModelEntry {
            key: "old",
            label: "baseline Qwen3-0.6B",
            dir_name: BASELINE_DIR,
        },
        ModelEntry {
            key: "new",
            label: "candidate corrector",
            dir_name: lune_core::models::DIR_STYLE_LLM,
        },
    ];
    let models_filter = std::env::var("MODELS").unwrap_or_else(|_| "both".into());
    let models: Vec<&ModelEntry> = all_models
        .iter()
        .filter(|m| {
            models_filter == "both" || models_filter.split(',').any(|tok| tok.trim() == m.key)
        })
        .collect();

    let prompts_filter = std::env::var("PROMPTS").ok();
    let prompts: Vec<PromptVariant> = prompt_variants()
        .into_iter()
        .filter(|p| {
            prompts_filter
                .as_deref()
                .is_none_or(|f| f.split(',').any(|tok| tok.trim() == p.name))
        })
        .collect();

    let print = std::env::var("LUNE_BENCH_PRINT").map_or(true, |v| v != "0");

    println!(
        "  models: {} · prompts: {} · fixtures: {} · gpu: {}",
        models.iter().map(|m| m.key).collect::<Vec<_>>().join(","),
        prompts.iter().map(|p| p.name).collect::<Vec<_>>().join(","),
        FIXTURES.len(),
        has_gpu,
    );
    if prompts.is_empty() || models.is_empty() {
        println!("\n[skip] no model or prompt selected (check MODELS=/PROMPTS=)");
        return;
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut runs: Vec<Run> = Vec::new();

    // Model-outer so only one LLM is resident at a time (the two models plus
    // ORT's CUDA arena would otherwise compete for VRAM).
    for m in &models {
        let model_dir = dir.join(m.dir_name);
        if !model_dir.join("tokenizer.json").exists() {
            println!(
                "\n[skip] {} — not found at {}",
                m.label,
                model_dir.display()
            );
            continue;
        }
        println!("\n── loading {} ({}) ──", m.label, m.dir_name);
        let llm = match OrtStyleLlm::new(OrtStyleLlmConfig {
            model_dir,
            use_gpu: has_gpu,
            cuda_required: false,
            max_new_tokens: 512,
        }) {
            Ok(l) => l,
            Err(e) => {
                println!("[skip] {} — load failed: {e}", m.label);
                continue;
            }
        };

        for p in &prompts {
            for fx in FIXTURES {
                // Production user framing — exactly what `apply_correction`
                // builds — so only the system prompt varies between arms.
                let user = OrtStyleLlm::user_prompt_with_examples(fx.raw, "en", &[]);
                let t0 = Instant::now();
                let output = rt
                    .block_on(llm.complete(&p.system, &user))
                    .unwrap_or_else(|e| {
                        eprintln!("  [warn] {}/{}/{}: {e}", m.label, p.name, fx.name);
                        fx.raw.to_string()
                    });
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                runs.push(Run {
                    model: m.label,
                    prompt: p.name,
                    fixture: fx.name,
                    wer: word_error_rate(&output, fx.reference),
                    cer: cased_cer(&output, fx.reference),
                    pf1_pos: punct_place_f1(&output, fx.reference),
                    pf1_cnt: punct_f1(&output, fx.reference),
                    ms,
                    output,
                });
            }
        }
        // `llm` drops here, freeing the model before the next one loads.
    }

    if runs.is_empty() {
        println!("\n[skip] no runs completed — are the models on disk?");
        return;
    }

    if print {
        print_per_fixture(&runs);
    }
    print_summary(&runs);
}

/// Per-fixture side-by-side: raw, reference, then every arm's output. This is
/// the qualitative view — read it to judge which output actually reads best.
fn print_per_fixture(runs: &[Run]) {
    println!("\n═══ Per-fixture outputs ═══");
    for fx in FIXTURES {
        let rows: Vec<&Run> = runs.iter().filter(|r| r.fixture == fx.name).collect();
        if rows.is_empty() {
            continue;
        }
        println!("\n── {} ──", fx.name);
        println!("  raw:");
        wrap_print("    ", fx.raw, 80);
        println!("  reference:");
        wrap_print("    ", fx.reference, 80);
        for r in rows {
            println!(
                "  [{} · {}]  WER {:.0}%  cerCASE {:.0}%  punctF1 pos {:.2}/cnt {:.2}  {:.0}ms",
                r.model,
                r.prompt,
                r.wer * 100.0,
                r.cer * 100.0,
                r.pf1_pos,
                r.pf1_cnt,
                r.ms,
            );
            wrap_print("    ", &r.output, 80);
        }
    }
}

/// Aggregate table over all fixtures, one row per (model, prompt) arm, plus a
/// verdict picking the best arm by punctF1 (WER as tiebreak).
fn print_summary(runs: &[Run]) {
    println!("\n═══ Summary (mean over {} fixtures) ═══", FIXTURES.len());
    println!(
        "  (lower WER / cerCASE = better; higher punctF1 = better; \
         pF1pos is placement-aware, pF1cnt is count-only)\n"
    );
    println!(
        "  {:<22} {:<10} {:>7} {:>9} {:>7} {:>7} {:>8}",
        "model", "prompt", "WER%", "cerCASE%", "pF1pos", "pF1cnt", "p50 ms"
    );
    println!(
        "  {:<22} {:<10} {:>7} {:>9} {:>7} {:>7} {:>8}",
        "──────", "──────", "────", "───────", "──────", "──────", "──────"
    );

    // Stable arm order: order of first appearance in `runs`.
    let mut arms: Vec<(&str, &str)> = Vec::new();
    for r in runs {
        if !arms.iter().any(|&(m, p)| m == r.model && p == r.prompt) {
            arms.push((r.model, r.prompt));
        }
    }

    let mut best: Option<(f64, f64, String)> = None; // (pF1pos, -wer, "model · prompt")
    for (m, p) in &arms {
        let rows: Vec<&Run> = runs
            .iter()
            .filter(|r| r.model == *m && r.prompt == *p)
            .collect();
        let wer = mean(&rows.iter().map(|r| r.wer).collect::<Vec<_>>());
        let cer = mean(&rows.iter().map(|r| r.cer).collect::<Vec<_>>());
        let pos = mean(&rows.iter().map(|r| r.pf1_pos).collect::<Vec<_>>());
        let cnt = mean(&rows.iter().map(|r| r.pf1_cnt).collect::<Vec<_>>());
        let mut lat: Vec<f64> = rows.iter().map(|r| r.ms).collect();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&lat, 50.0);
        println!(
            "  {:<22} {:<10} {:>6.1} {:>8.1} {:>7.2} {:>7.2} {:>7.0}",
            m,
            p,
            wer * 100.0,
            cer * 100.0,
            pos,
            cnt,
            p50,
        );
        let key = (pos, -wer, format!("{m} · {p}"));
        if best.as_ref().is_none_or(|b| (key.0, key.1) > (b.0, b.1)) {
            best = Some(key);
        }
    }

    if let Some((pos, neg_wer, label)) = best {
        println!(
            "\n  → best by punctuation placement: {label}  (pF1pos {:.2}, WER {:.1}%)",
            pos,
            -neg_wer * 100.0,
        );
        println!("  (Small fixture set — read the per-fixture outputs before trusting the table.)");
    }
}
