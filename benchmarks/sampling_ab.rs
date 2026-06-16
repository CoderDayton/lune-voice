//! sampling_ab — the correction LLM's output under greedy vs sampling, on the
//! same raw transcripts, with many samples per arm so run-to-run variance is
//! visible.
//!
//! Three arms per input:
//!   * greedy   — deterministic baseline (top_k 1), run once.
//!   * k20      — the sampling defaults but top_k 20, run `SAMPLES` times.
//!   * k40      — the live sampling defaults (top_k 40), run `SAMPLES` times.
//!
//! For each sampling arm we tally distinct outputs and mark whether each
//! matches greedy, so "does top_k change anything" and "how much does it drift"
//! are answered directly. The output is the RAW model correction (pre
//! `validate_and_merge`) so the model's behavior is exposed without smoothing.
//!
//! Run: cargo run --release --features cuda --example sampling_ab -p lune-app

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use bench_utils::{init_ort, models_dir};
use lune_core::config::BASE_SYSTEM_PROMPT;
use lune_core::llm::LlmService;
use lune_llm::backend::sampler::SamplingParams;
use lune_llm::{OrtStyleLlm, OrtStyleLlmConfig};

/// Sampling runs per sampling arm per input. Bumped up to gather variance data.
const SAMPLES: usize = 8;

/// Realistic raw ASR transcripts (lowercase, no punctuation, an error or two) —
/// the kind of input the correction pass actually sees. A mix of homophones,
/// run-ons, and filler so drift would be visible.
const INPUTS: &[&str] = &[
    "i think we should meat at noon to go over the new feature",
    "the whether is nice today so lets go outside for a bit",
    "can you send me the report when you get a chance ill be at lunch till one",
    "she said the deploy finished but the migration didnt run we should roll back",
    "tell him to stop by my desk after the stand up i have a few questions",
    "we use github and kubernetes for most of the backend infrastructure",
    "did you see john where is he i needed to ask him about the budget",
    "basically the idea is that we just need to rethink how we do testing right now",
    "um so like i was thinking maybe we could push the release to friday instead",
    "the api returned a five hundred error when i tried to fetch the user profile",
    "lets sync up tomorrow morning before the client call to align on the messaging",
    "he asked weather the new build passed all the tests on the ci pipeline",
    "i cant find the file it was in the downloads folder yesterday but its gone now",
    "remind me to email sarah about the contract she needs it signed by end of day",
];

/// Run one config `n` times and tally distinct outputs in first-seen order.
fn run_n(
    llm: &mut OrtStyleLlm,
    rt: &tokio::runtime::Runtime,
    params: &SamplingParams,
    system: &str,
    user: &str,
    n: usize,
) -> Vec<(String, usize)> {
    llm.set_sampling_params(params.clone());
    let mut tally: Vec<(String, usize)> = Vec::new();
    for _ in 0..n {
        let out = rt.block_on(llm.complete(system, user)).unwrap_or_default();
        match tally.iter_mut().find(|(t, _)| *t == out) {
            Some(entry) => entry.1 += 1,
            None => tally.push((out, 1)),
        }
    }
    tally
}

/// Print each distinct variant with its share and whether it equals greedy.
fn print_tally(label: &str, tally: &[(String, usize)], greedy: &str) {
    let n: usize = tally.iter().map(|(_, c)| c).sum();
    for (text, count) in tally {
        let mark = if text == greedy { "=greedy" } else { " DIFF  " };
        println!("    {label} {count}/{n} [{mark}]: {text}");
    }
}

fn main() {
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

    // Build explicit sampling arms — the instance default is penalized greedy
    // (temperature 0), so we can't inherit a usable sampling temperature from
    // it; only EOS/stop tokens come from the probed instance.
    let base = llm.sampling_params().clone();
    let k40 = SamplingParams {
        temperature: 0.7,
        top_p: 0.8,
        top_k: 40,
        min_p: 0.0,
        repetition_penalty: 1.1,
        frequency_penalty: 0.0,
        presence_penalty: 0.5,
        eos_token_id: base.eos_token_id,
        stop_token_ids: base.stop_token_ids.clone(),
    };
    let k20 = SamplingParams {
        top_k: 20,
        ..k40.clone()
    };
    let greedy = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 1,
        min_p: 0.0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        eos_token_id: k40.eos_token_id,
        stop_token_ids: k40.stop_token_ids.clone(),
    };

    println!();
    println!(
        "sampling_ab: {} inputs × {SAMPLES} samples/arm — greedy vs k20 vs k40",
        INPUTS.len()
    );
    println!(
        "  defaults: temp {:.1}  top_p {:.1}  rep {:.1}  presence {:.1}  (top_k 20 vs 40)\n",
        k40.temperature, k40.top_p, k40.repetition_penalty, k40.presence_penalty,
    );

    // Aggregate variance stats.
    let mut k20_varied = 0usize; // inputs with >1 distinct sampled output
    let mut k40_varied = 0usize;
    let mut k20_diff = 0usize; // inputs where any sample != greedy
    let mut k40_diff = 0usize;

    for (i, raw) in INPUTS.iter().enumerate() {
        let user = OrtStyleLlm::user_prompt_with_examples(raw, "English", &[]);

        let g = run_n(&mut llm, &rt, &greedy, &system, &user, 1);
        let g_out = g[0].0.clone();
        let t20 = run_n(&mut llm, &rt, &k20, &system, &user, SAMPLES);
        let t40 = run_n(&mut llm, &rt, &k40, &system, &user, SAMPLES);

        if t20.len() > 1 {
            k20_varied += 1;
        }
        if t40.len() > 1 {
            k40_varied += 1;
        }
        if t20.iter().any(|(t, _)| *t != g_out) {
            k20_diff += 1;
        }
        if t40.iter().any(|(t, _)| *t != g_out) {
            k40_diff += 1;
        }

        println!("[{}] raw:    {raw}", i + 1);
        println!("    greedy: {g_out}");
        print_tally("k20", &t20, &g_out);
        print_tally("k40", &t40, &g_out);
        println!();
    }

    let n = INPUTS.len();
    println!("summary ({SAMPLES} samples per arm per input, {n} inputs):");
    println!(
        "  k20: {k20_varied}/{n} inputs varied run-to-run, {k20_diff}/{n} differed from greedy"
    );
    println!(
        "  k40: {k40_varied}/{n} inputs varied run-to-run, {k40_diff}/{n} differed from greedy"
    );
}
