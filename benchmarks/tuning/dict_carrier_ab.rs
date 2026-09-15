// SPDX-License-Identifier: GPL-3.0-or-later

//! dict_carrier_ab — A/B: do dictionary terms survive the correction pass
//! WITHOUT the user-prompt dictionary-carrier examples?
//!
//! The carriers ("I said {phrase} in the meeting." pairs) are the MIDDLE of a
//! three-layer dictionary-protection stack: `apply_dictionary_seed` already
//! writes the replacement into the text before the LLM sees it, and
//! `validate_and_merge` rejects divergent edits afterward. This benchmark
//! isolates the carriers' marginal value: it runs the configured remote LLM on
//! pre-seeded transcripts twice — arm A WITH carriers, arm B WITHOUT — straight
//! through `complete()` (no `validate_and_merge`), and reports whether the
//! seeded dictionary term survives verbatim in each arm.
//!
//! Because production adds `validate_and_merge` on top, "survives in arm B
//! here" is a CONSERVATIVE lower bound on production safety.
//!
//! Run: LUNE_BENCH_LLM_BASE_URL=... LUNE_BENCH_LLM_MODEL=... \
//!      cargo run --release --example dict_carrier_ab -p lune-voice
//!      (LUNE_BENCH_PRINT=1 dumps each arm's full output.)

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use lune_core::config::BASE_SYSTEM_PROMPT;
use lune_core::dict::DictionaryEntry;
use lune_core::llm::LlmService;
use lune_llm::OpenAiLlmBackend;

/// Build a few-shot prompt: each `(input, output)` carrier pair shown as an
/// example, then the transcript to rewrite. Reconstructs the shape of the
/// removed `OrtStyleLlm::user_prompt_with_examples` so the A/B still isolates
/// the dictionary carriers' marginal value against the real local LLM.
fn user_prompt_with_examples(
    transcript: &str,
    _lang: &str,
    examples: &[(String, String)],
) -> String {
    let mut p = String::new();
    for (inp, out) in examples {
        p.push_str("Input: ");
        p.push_str(inp);
        p.push_str("\nOutput: ");
        p.push_str(out);
        p.push('\n');
    }
    p.push_str("Input: ");
    p.push_str(transcript);
    p.push_str("\nOutput:");
    p
}

/// A pre-seeded transcript (the replacement is already present, as it would be
/// after `apply_dictionary_seed`) plus the term we expect to survive.
struct Case {
    term: &'static str,
    transcript: &'static str,
}

/// Normal, real-world dictionary terms — brands, products, and places with
/// specific casing. No invented non-words this time.
fn dictionary() -> Vec<DictionaryEntry> {
    [
        ("kubernetes", "Kubernetes"),
        ("github", "GitHub"),
        ("postgres", "PostgreSQL"),
        ("youtube", "YouTube"),
        ("dayton", "Dayton"),
        ("stripe", "Stripe"),
    ]
    .into_iter()
    .map(|(p, r)| DictionaryEntry {
        phrase: p.to_string(),
        replacement: r.to_string(),
        priority: 5,
    })
    .collect()
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            term: "Kubernetes",
            transcript: "i deployed the service to Kubernetes this morning",
        },
        Case {
            term: "GitHub",
            transcript: "we should sync the GitHub repo before the demo",
        },
        Case {
            term: "PostgreSQL",
            transcript: "the data lives in PostgreSQL on the staging box",
        },
        Case {
            term: "YouTube",
            transcript: "she posted the recap on YouTube yesterday",
        },
        Case {
            term: "Dayton",
            transcript: "Dayton said he would join the call at three",
        },
        Case {
            term: "Stripe",
            transcript: "we switched billing over to Stripe in march",
        },
    ]
}

/// Mirror `dictionary_examples()`: English carrier sentences, take(6).
fn carriers(dict: &[DictionaryEntry]) -> Vec<(String, String)> {
    dict.iter()
        .take(6)
        .map(|e| {
            (
                format!("I said {} in the meeting.", e.phrase),
                format!("I said {} in the meeting.", e.replacement),
            )
        })
        .collect()
}

fn yn(b: bool) -> &'static str {
    if b { "keep" } else { "LOST" }
}

fn main() {
    let Some(config) = bench_utils::remote_llm_config_from_env() else {
        println!(
            "  [skip] {}/{} not set — point them at Ollama, llama-server, LM Studio, or a hosted API",
            bench_utils::LUNE_BENCH_LLM_BASE_URL,
            bench_utils::LUNE_BENCH_LLM_MODEL,
        );
        return;
    };
    let llm = OpenAiLlmBackend::new(config).expect("bench LLM config must be valid");
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let print = std::env::var("LUNE_BENCH_PRINT").is_ok_and(|v| v != "0");

    let dict = dictionary();
    let carriers = carriers(&dict);
    let system = BASE_SYSTEM_PROMPT.to_owned();
    let cases = cases();

    println!();
    println!(
        "dict_carrier_ab: {} cases, real local LLM (raw complete(), no validate_and_merge)\n",
        cases.len()
    );
    println!("  {:12}  {:>10}  {:>9}", "term", "A(carrier)", "B(none)");

    let (mut keep_a, mut keep_b) = (0usize, 0usize);
    for c in &cases {
        let user_a = user_prompt_with_examples(c.transcript, "English", &carriers);
        let user_b = user_prompt_with_examples(c.transcript, "English", &[]);
        let out_a = rt
            .block_on(llm.complete(&system, &user_a))
            .unwrap_or_default();
        let out_b = rt
            .block_on(llm.complete(&system, &user_b))
            .unwrap_or_default();
        let (a, b) = (out_a.contains(c.term), out_b.contains(c.term));
        keep_a += a as usize;
        keep_b += b as usize;
        println!("  {:12}  {:>10}  {:>9}", c.term, yn(a), yn(b));
        if print {
            println!("     A: {out_a:?}");
            println!("     B: {out_b:?}");
        }
    }

    let n = cases.len();
    println!();
    println!("  preserved WITH carriers (A):    {keep_a}/{n}");
    println!("  preserved WITHOUT carriers (B): {keep_b}/{n}");
    if keep_b >= keep_a {
        println!(
            "\n  → SAFE to drop carriers: arm B preserves >= arm A, and production still adds\n    validate_and_merge on top of this."
        );
    } else {
        println!(
            "\n  → carriers HELP: arm B lost {} term(s) that A kept — inspect with LUNE_BENCH_PRINT=1\n    and check whether validate_and_merge would recover them before dropping.",
            keep_a - keep_b
        );
    }
}
