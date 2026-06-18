//! sampling_tune — grid search over the correction LLM's sampling parameters to
//! answer one question rigorously: does ANY non-greedy config produce a *more
//! correct* correction than deterministic greedy?
//!
//! Method (grid, not Bayesian — the goal is "don't miss anything": a grid is
//! exhaustive within its discretized ranges, whereas Bayesian optimization
//! trades coverage for sample-efficiency and can step over a narrow optimum):
//!   1. A set of (raw, gold) cases — realistic ASR transcripts paired with the
//!      ideal correction. Several are chosen so greedy is known to slip (tricky
//!      homophones), giving sampling a fair chance to win.
//!   2. Score every output against gold on two axes (similarity =
//!      1 - Levenshtein / max_len, in [0, 1]):
//!        * content — word-level similarity, lowercased and punctuation/casing
//!          insensitive (measures *correctness*: homophones fixed, no dropped
//!          or invented words). THIS IS THE PRIMARY metric.
//!        * char — char-level similarity on lightly normalized text (also
//!          reflects punctuation/style churn).
//!   3. Greedy (top_k 1, identity penalties) is the deterministic baseline.
//!   4. Sweep the grid over known-good sampling ranges; for each config run
//!      `SAMPLES` generations per case. A case is "won" by non-greedy iff some
//!      sampled output scores strictly higher on content than greedy.
//!
//! Output: greedy baseline, the top configs by mean content score, and — the
//! point of the whole thing — every case where non-greedy beat greedy (with the
//! texts), or a plain statement that none did.
//!
//! Tunables (env): SAMPLES (default 3) generations per config per case.
//!
//! Run: cargo run --release --features cuda --example sampling_tune -p lune-app

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use bench_utils::{init_ort, models_dir};
use lune_core::config::BASE_SYSTEM_PROMPT;
use lune_core::llm::LlmService;
use lune_llm::backend::sampler::SamplingParams;
use lune_llm::{OrtStyleLlm, OrtStyleLlmConfig};

/// (raw ASR transcript, ideal correction). A few are deliberately tricky
/// homophones where greedy may not make the fix — so sampling can win if it
/// ever helps.
const CASES: &[(&str, &str)] = &[
    (
        "i think we should meat at noon to go over the new feature",
        "I think we should meet at noon to go over the new feature.",
    ),
    (
        "the whether is nice today so lets go outside for a bit",
        "The weather is nice today, so let's go outside for a bit.",
    ),
    (
        "can you send me the report when you get a chance ill be at lunch till one",
        "Can you send me the report when you get a chance? I'll be at lunch till one.",
    ),
    (
        "she said the deploy finished but the migration didnt run we should roll back",
        "She said the deploy finished, but the migration didn't run. We should roll back.",
    ),
    (
        "we use github and kubernetes for most of the backend infrastructure",
        "We use GitHub and Kubernetes for most of the backend infrastructure.",
    ),
    (
        "did you see john where is he i needed to ask him about the budget",
        "Did you see John? Where is he? I needed to ask him about the budget.",
    ),
    (
        "he asked weather the new build passed all the tests on the ci pipeline",
        "He asked whether the new build passed all the tests on the CI pipeline.",
    ),
    (
        "i could of finished it sooner if i had the right tools",
        "I could have finished it sooner if I had the right tools.",
    ),
    (
        "their going to deploy the changes after they review the code",
        "They're going to deploy the changes after they review the code.",
    ),
    (
        "we need to right a test for the new endpoint before we merge",
        "We need to write a test for the new endpoint before we merge.",
    ),
    (
        "tell him to stop by my desk after the stand up i have a few questions",
        "Tell him to stop by my desk after the stand-up. I have a few questions.",
    ),
    (
        "its been awhile since we reviewed the on boarding docs for new hires",
        "It's been a while since we reviewed the onboarding docs for new hires.",
    ),
];

/// Word tokens for the content metric: lowercase, hyphen→boundary, keep
/// apostrophes (so contractions stay intact), everything else splits.
fn content_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '\'' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Char sequence for the char metric: straighten smart quotes/dashes, collapse
/// runs of whitespace, trim.
fn norm_chars(s: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::new();
    let mut prev_space = false;
    for ch in s.trim().chars() {
        let c = match ch {
            '\u{2019}' | '\u{2018}' => '\'',
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2014}' | '\u{2013}' => '-',
            other => other,
        };
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out
}

/// Levenshtein distance over any comparable slice (chars or word tokens).
fn lev<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Similarity in [0, 1]: 1 - edit_distance / max_len.
fn sim<T: PartialEq>(a: &[T], b: &[T]) -> f32 {
    let maxlen = a.len().max(b.len());
    if maxlen == 0 {
        return 1.0;
    }
    1.0 - lev(a, b) as f32 / maxlen as f32
}

struct Prepared {
    user: String,
    gold_tokens: Vec<String>,
    gold_chars: Vec<char>,
    greedy_out: String,
    greedy_cs: f32,
    greedy_chs: f32,
}

fn main() {
    let samples: usize = std::env::var("SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(3);

    let has_gpu = init_ort();
    let llm_dir = models_dir().join(lune_core::models::DIR_STYLE_LLM);
    if !llm_dir.join("tokenizer.json").exists() {
        println!("  [skip] no style LLM model at {llm_dir:?}");
        return;
    }
    let mut llm = match OrtStyleLlm::new(OrtStyleLlmConfig {
        model_dir: llm_dir,
        use_gpu: has_gpu,
        ..Default::default()
    }) {
        Ok(l) => l,
        Err(e) => {
            println!("  [skip] LLM load failed: {e}");
            return;
        }
    };
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let system = BASE_SYSTEM_PROMPT.replace("{lang}", "English");
    let base = llm.sampling_params().clone();

    let greedy = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 1,
        min_p: 0.0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        eos_token_id: base.eos_token_id,
        stop_token_ids: base.stop_token_ids.clone(),
    };

    // --- Greedy baseline (deterministic): one run per case. ---
    let mut cases: Vec<Prepared> = Vec::with_capacity(CASES.len());
    llm.set_sampling_params(greedy.clone());
    for (raw, gold) in CASES {
        let user = OrtStyleLlm::user_prompt_with_examples(raw, "English", &[]);
        let out = rt
            .block_on(llm.complete(&system, &user))
            .unwrap_or_default();
        let gold_tokens = content_tokens(gold);
        let gold_chars = norm_chars(gold);
        let greedy_cs = sim(&content_tokens(&out), &gold_tokens);
        let greedy_chs = sim(&norm_chars(&out), &gold_chars);
        cases.push(Prepared {
            user,
            gold_tokens,
            gold_chars,
            greedy_out: out,
            greedy_cs,
            greedy_chs,
        });
    }

    // --- Grid over known-good sampling ranges. ---
    let temps = [0.3f32, 0.6, 0.9];
    let top_ps = [0.8f32, 0.95];
    let top_ks = [20usize, 40];
    let reps = [1.0f32, 1.1, 1.3];
    let press = [0.0f32, 0.5];
    let mut grid: Vec<(String, SamplingParams)> = Vec::new();
    for &t in &temps {
        for &tp in &top_ps {
            for &tk in &top_ks {
                for &rp in &reps {
                    for &pp in &press {
                        let label = format!("t{t:.1} p{tp:.2} k{tk} r{rp:.1} pp{pp:.1}");
                        grid.push((
                            label,
                            SamplingParams {
                                temperature: t,
                                top_p: tp,
                                top_k: tk,
                                min_p: 0.0,
                                repetition_penalty: rp,
                                frequency_penalty: 0.0,
                                presence_penalty: pp,
                                eos_token_id: base.eos_token_id,
                                stop_token_ids: base.stop_token_ids.clone(),
                            },
                        ));
                    }
                }
            }
        }
    }

    let total_gens = grid.len() * cases.len() * samples;
    println!();
    println!("sampling_tune: grid search for non-greedy > greedy on correction quality");
    println!(
        "  {} configs × {} cases × {samples} samples = {total_gens} sampled generations (+{} greedy)",
        grid.len(),
        cases.len(),
        cases.len()
    );
    println!("  grid: temp{temps:?} top_p{top_ps:?} top_k{top_ks:?} rep{reps:?} presence{press:?}");
    println!(
        "  metric: content = word-level sim to gold (case/punct-insensitive); char = char-level sim\n"
    );

    // --- Greedy baseline report. ---
    let g_cs: f32 = cases.iter().map(|c| c.greedy_cs).sum::<f32>() / cases.len() as f32;
    let g_chs: f32 = cases.iter().map(|c| c.greedy_chs).sum::<f32>() / cases.len() as f32;
    println!("greedy baseline:  mean content {g_cs:.4}   mean char {g_chs:.4}");
    for (i, c) in cases.iter().enumerate() {
        let flag = if c.greedy_cs < 0.999 {
            "  <-- greedy imperfect"
        } else {
            ""
        };
        println!(
            "  [{:>2}] content {:.3}  {}{flag}",
            i + 1,
            c.greedy_cs,
            c.greedy_out
        );
    }
    println!();

    // --- Sweep. Track per-config means + per-case global best sampled output. ---
    #[derive(Clone)]
    struct CaseBest {
        cs: f32,
        text: String,
        label: String,
    }
    let mut config_means: Vec<(String, f32, f32)> = Vec::with_capacity(grid.len());
    let mut best: Vec<CaseBest> = cases
        .iter()
        .map(|_| CaseBest {
            cs: f32::MIN,
            text: String::new(),
            label: String::new(),
        })
        .collect();
    // Regression accounting over every sampled generation.
    let (mut n_better, mut n_equal, mut n_worse) = (0usize, 0usize, 0usize);
    const EPS: f32 = 1e-4;

    for (gi, (label, params)) in grid.iter().enumerate() {
        llm.set_sampling_params(params.clone());
        let (mut sum_cs, mut sum_chs, mut cnt) = (0.0f32, 0.0f32, 0usize);
        for (ci, c) in cases.iter().enumerate() {
            for _ in 0..samples {
                let out = rt
                    .block_on(llm.complete(&system, &c.user))
                    .unwrap_or_default();
                let cs = sim(&content_tokens(&out), &c.gold_tokens);
                let chs = sim(&norm_chars(&out), &c.gold_chars);
                sum_cs += cs;
                sum_chs += chs;
                cnt += 1;
                if cs > c.greedy_cs + EPS {
                    n_better += 1;
                } else if cs < c.greedy_cs - EPS {
                    n_worse += 1;
                } else {
                    n_equal += 1;
                }
                if cs > best[ci].cs {
                    best[ci] = CaseBest {
                        cs,
                        text: out,
                        label: label.clone(),
                    };
                }
            }
        }
        let mean_cs = sum_cs / cnt as f32;
        let mean_chs = sum_chs / cnt as f32;
        config_means.push((label.clone(), mean_cs, mean_chs));
        println!(
            "  [config {:>2}/{}] {label}  ->  mean content {mean_cs:.4}  char {mean_chs:.4}",
            gi + 1,
            grid.len()
        );
    }
    println!();

    // --- Top configs by mean content score. ---
    config_means.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("top configs by mean content score (greedy mean = {g_cs:.4}):");
    for (label, mean_cs, mean_chs) in config_means.iter().take(8) {
        let delta = mean_cs - g_cs;
        println!("  {label}   content {mean_cs:.4} ({delta:+.4} vs greedy)   char {mean_chs:.4}");
    }
    let best_cfg = config_means.first().map(|c| c.1).unwrap_or(0.0);
    println!();

    // --- The verdict: did non-greedy ever beat greedy on content? ---
    let mut wins = 0usize;
    println!("=== did non-greedy ever produce a MORE CORRECT output than greedy? ===");
    for (i, (c, b)) in cases.iter().zip(best.iter()).enumerate() {
        if b.cs > c.greedy_cs + EPS {
            wins += 1;
            println!("  [{}] WIN by {:+.3} content", i + 1, b.cs - c.greedy_cs);
            println!("      greedy ({:.3}): {}", c.greedy_cs, c.greedy_out);
            println!("      best   ({:.3}): {}   [{}]", b.cs, b.text, b.label);
        }
    }
    if wins == 0 {
        println!(
            "  NO. Across {} cases and {total_gens} sampled generations, no non-greedy",
            cases.len()
        );
        println!("  config produced a more correct correction than greedy on any case.");
    }
    println!();

    // --- Aggregate verdict. ---
    let tot = (n_better + n_equal + n_worse).max(1) as f32;
    println!("summary:");
    println!(
        "  best config mean content {best_cfg:.4} vs greedy {g_cs:.4}  ({:+.4})",
        best_cfg - g_cs
    );
    println!(
        "  per-generation vs greedy: better {n_better} ({:.1}%)  equal {n_equal} ({:.1}%)  worse {n_worse} ({:.1}%)",
        100.0 * n_better as f32 / tot,
        100.0 * n_equal as f32 / tot,
        100.0 * n_worse as f32 / tot,
    );
    println!(
        "  cases where non-greedy beat greedy: {wins}/{}",
        cases.len()
    );
}
