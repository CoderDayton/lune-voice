// SPDX-License-Identifier: GPL-3.0-or-later

//! Styling-LLM capability benchmark.
//!
//! Answers one question: **is the configured remote styling LLM good enough
//! to drive the styling presets we ship?**
//!
//! It exercises the exact production path — `StylePreset::build_styling_prompt`
//! → `LlmService::complete("", prompt)` → `fix_all_number_spaces` /
//! `contains_forbidden_output`, the same three calls `StylePipeline::transform`
//! makes — across three axes:
//!
//!   1. every built-in preset with **prompt only** (few-shot pairs stripped),
//!   2. the same presets **with their built-in few-shot pairs**, and
//!   3. the user's **"Enhance Prompt"** custom style loaded live from
//!      `settings.toml`, both prompt-only and with model-**generated** few-shot
//!      pairs (`few_shot::generate_few_shot_examples`, the styles-page path).
//!
//! For each (style, arm, input) it prints raw → styled, per-call latency, and
//! flags automatic failure modes (empty / meta-commentary / `<think>` leak /
//! echoed example / no-op) plus a per-style heuristic "did it obey the rule"
//! check. A summary table and an overall verdict close it out.
//!
//! Run (from repo root; needs LUNE_BENCH_LLM_BASE_URL / _MODEL set, pointing
//! at Ollama, llama-server, LM Studio, or a hosted API):
//!
//!   cgs bench --env TYPE=styling
//!
//! Env knobs:
//!   STYLES=casual,enhance   only run these styles (comma list of ids/keys)
//!   MAX_TOKENS=512          per-call generation cap (production uses 512)
//!   STYLING_VERBOSE=1       print the full prompt sent for the first case
//!   GEN_FEWSHOT=0           skip generating few-shot pairs for Enhance Prompt

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::time::Instant;

use lune_core::config::StylePreset;
use lune_core::llm::LlmService;
use lune_llm::OpenAiLlmBackend;
use lune_llm::few_shot::generate_few_shot_examples;
use lune_llm::prompt::{contains_forbidden_output, fix_all_number_spaces};
use lune_persist::AppPaths;

// ─── Test inputs (raw ASR-style: lowercase, no punctuation, fillers) ─────────

/// General dictation, fed to casual / professional / formal / writing.
const GENERAL_INPUTS: &[&str] = &[
    "so um i was thinking we could meet tomorrow maybe around noon if that works for everybody",
    "hey just checking in on the deploy did the migration finish if not we should roll back before the demo at three",
    "the report is basically done i just need to like double check the numbers and then ill send it over",
    "she said i'll be there but honestly i don't think she meant it we'll see i guess",
];

/// Developer speech, fed to the technical style.
const TECHNICAL_INPUTS: &[&str] = &[
    "lets add an async function called fetch user that takes a user id and returns an optional user from the database",
    "the retry helper should catch the timeout error and back off exponentially before it calls send again",
];

/// Spoken prompts, fed to the Enhance Prompt custom style.
const PROMPT_INPUTS: &[&str] = &[
    "write a function that sorts a list of numbers",
    "make me a landing page for my bakery with a hero section and a menu",
    "explain how photosynthesis works to a five year old",
];

/// Casual *strips* structure, so it needs already-formatted input (capitals,
/// quotes, semicolons, exclamation) to have anything to remove — raw ASR text
/// is already lowercase/unpunctuated and would make every arm a no-op.
const CASUAL_INPUTS: &[&str] = &[
    "She said: \"I will be there\"; however, I do not think she meant it.",
    "The plan was simple: cut costs; raise prices; hire fewer people!",
    "Are you coming to the demo? We cannot start without you.",
    "I finished the report. Send it over when you get a second!",
];

fn inputs_for(kind: &str) -> &'static [&'static str] {
    match kind {
        "casual" => CASUAL_INPUTS,
        "technical" => TECHNICAL_INPUTS,
        "enhance" => PROMPT_INPUTS,
        _ => GENERAL_INPUTS,
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")))
        .unwrap_or(default)
}

/// Automatic red flags — the failure modes production silently discards
/// (empty / meta / think-leak) plus benchmark-only tells (echoed example,
/// no-op). Empty vec = the output is at least usable.
fn red_flags(out: &str, input: &str, pairs: &[(String, String)]) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if out.trim().is_empty() {
        flags.push("EMPTY");
        return flags; // nothing else is meaningful
    }
    if contains_forbidden_output(out) {
        flags.push("META");
    }
    if out.contains("<think") || out.contains("</think") {
        flags.push("THINK");
    }
    // Verbatim passthrough — the model returned the input untouched. Casing
    // and punctuation changes (e.g. what Casual does) count as real edits, so
    // this is an exact trimmed compare, not a normalized one.
    if out.trim() == input.trim() {
        flags.push("NO-OP");
    }
    // Task-execution instead of restyle: the model emitted code the input
    // never asked for (a fenced block, or a bare def/function/class start).
    // This is the tiny model's dominant miss, and a plain backtick check
    // would otherwise score these code dumps as passing.
    let codey = |t: &str| {
        let h = t.trim_start();
        t.contains("```")
            || h.starts_with("def ")
            || h.starts_with("function ")
            || h.starts_with("async function")
            || h.starts_with("class ")
    };
    if codey(out) && !codey(input) {
        flags.push("CODEGEN");
    }
    // Copied one of the illustrative examples verbatim instead of styling the
    // real transcript — a small-model failure the prompt explicitly forbids.
    let echoed = pairs.iter().any(|(pi, po)| {
        (po.len() > 4 && out.contains(po.as_str())) || (pi.len() > 8 && out.contains(pi.as_str()))
    });
    if echoed {
        flags.push("ECHO");
    }
    flags
}

/// Per-style heuristic: did the output actually obey the headline rule?
/// Advisory only (regex-free, deliberately loose). `None` = no cheap check.
fn green_check(kind: &str, input: &str, out: &str) -> Option<bool> {
    let contractions = [
        "don't",
        "won't",
        "can't",
        "i'll",
        "we'll",
        "they're",
        "it's",
        "i'm",
        "isn't",
        "didn't",
        "that's",
        "let's",
        "you're",
        "he's",
        "she's",
        "wouldn't",
        "couldn't",
        "shouldn't",
    ];
    let has_contraction = |t: &str| contractions.iter().any(|c| t.to_lowercase().contains(c));
    match kind {
        // Casual: no quotes / semicolons / colons / exclamation, lowercase start.
        "casual" => {
            let clean = !out.contains('"')
                && !out.contains(';')
                && !out.contains(':')
                && !out.contains('!');
            let lc_start = out
                .trim()
                .chars()
                .find(|c| c.is_alphabetic())
                .map(|c| c.is_lowercase())
                .unwrap_or(true);
            Some(clean && lc_start)
        }
        // Professional / Formal: contractions expanded away.
        "professional" | "formal" => Some(!has_contraction(out)),
        // Technical: at least one code span backticked.
        "technical" => Some(out.contains('`') && !out.contains("```")),
        // Writing: fewer fillers than the raw input.
        "writing" => {
            let fillers = [
                "so ",
                "just ",
                "basically",
                " like ",
                "um ",
                "uh ",
                "kind of",
            ];
            let count = |t: &str| {
                let t = format!(" {} ", t.to_lowercase());
                fillers.iter().map(|f| t.matches(f).count()).sum::<usize>()
            };
            Some(count(out) <= count(input))
        }
        // Enhance: instruction says preserve length — flag runaway expansion.
        "enhance" => {
            let wi = input.split_whitespace().count().max(1) as f64;
            let wo = out.split_whitespace().count() as f64;
            let ratio = wo / wi;
            Some((0.6..=2.0).contains(&ratio))
        }
        _ => None,
    }
}

/// One styling call on the production path. Returns the raw model output (so
/// the caller can flag empty/meta/etc.) plus the accepted, number-space-fixed
/// text and the wall-clock latency.
fn style_once(
    rt: &tokio::runtime::Runtime,
    llm: &OpenAiLlmBackend,
    preset: &StylePreset,
    input: &str,
) -> (String, u128) {
    let (system, user) = preset.build_styling_prompt(input);
    let t = Instant::now();
    let out = rt
        .block_on(llm.complete(&system, &user))
        .unwrap_or_else(|e| format!("<error: {e}>"));
    (out, t.elapsed().as_millis())
}

// ─── Report accumulation ─────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct ArmStat {
    n: usize,
    clean: usize, // no red flags
    green: usize, // heuristic passed (only counts checks that ran)
    green_n: usize,
    ms_total: u128,
}

fn run_arm(
    rt: &tokio::runtime::Runtime,
    llm: &OpenAiLlmBackend,
    kind: &str,
    arm: &str,
    preset: &StylePreset,
    verbose_first: &mut bool,
) -> ArmStat {
    let mut stat = ArmStat::default();
    println!("\n── {} · {arm} ──", preset.name);
    if *verbose_first && env_flag("STYLING_VERBOSE", false) {
        let (sys, usr) = preset.build_styling_prompt(inputs_for(kind)[0]);
        println!("  [system]\n{sys}\n  [user]\n{usr}\n  [end prompt]");
        *verbose_first = false;
    }
    for input in inputs_for(kind) {
        let (out, ms) = style_once(rt, llm, preset, input);
        let flags = red_flags(&out, input, &preset.few_shot_pairs);
        let display = fix_all_number_spaces(out.trim());
        let green = green_check(kind, input, &out);

        stat.n += 1;
        stat.ms_total += ms;
        if flags.is_empty() {
            stat.clean += 1;
        }
        if let Some(pass) = green {
            stat.green_n += 1;
            if pass {
                stat.green += 1;
            }
        }

        let flag_str = if flags.is_empty() {
            "ok".to_string()
        } else {
            flags.join(",")
        };
        let green_str = match green {
            Some(true) => " ✓rule",
            Some(false) => " ✗rule",
            None => "",
        };
        println!("  raw : {input}");
        println!("  out : {display}");
        println!("        ({ms} ms · {flag_str}{green_str})");
    }
    stat
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let paths = AppPaths::resolve().expect("resolve XDG paths");
    let Some(mut config) = bench_utils::remote_llm_config_from_env() else {
        println!(
            "  [skip] {}/{} not set — point them at Ollama, llama-server, LM Studio, or a hosted API",
            bench_utils::LUNE_BENCH_LLM_BASE_URL,
            bench_utils::LUNE_BENCH_LLM_MODEL,
        );
        return;
    };

    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    config.max_tokens = max_tokens as u32;
    let style_filter: Option<Vec<String>> = std::env::var("STYLES").ok().map(|v| {
        v.split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    });

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Styling-LLM capability benchmark                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("endpoint   : {}", config.base_url);
    println!(
        "model      : {}   max_new_tokens: {max_tokens}",
        config.model
    );

    let load = Instant::now();
    let llm = OpenAiLlmBackend::new(config).expect("bench LLM config must be valid");
    println!("load time  : {} ms", load.elapsed().as_millis());

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    // Assemble the styles under test: (kind key, display preset).
    // `kind` drives input selection + heuristic; the built-in id doubles as it.
    let builtins: Vec<StylePreset> = StylePreset::all_builtin()
        .iter()
        .map(|a| (**a).clone())
        .collect();

    // Enhance Prompt custom style, loaded live from settings.toml.
    let enhance = load_enhance(&paths);

    let want = |k: &str| {
        style_filter
            .as_ref()
            .map(|f| f.iter().any(|s| s == k))
            .unwrap_or(true)
    };

    // (kind, base preset)
    let mut plan: Vec<(String, StylePreset)> = Vec::new();
    for p in &builtins {
        if want(&p.id) {
            plan.push((p.id.clone(), p.clone()));
        }
    }
    let run_enhance = want("enhance") && enhance.is_some();

    let mut verbose_first = true;
    // Rows: (kind, arm, ArmStat)
    let mut rows: Vec<(String, &'static str, ArmStat)> = Vec::new();

    for (kind, preset) in &plan {
        println!(
            "\n════════════ {} ({}) ════════════",
            preset.name, preset.id
        );
        println!("rules: {}", preset.style_instructions);

        // Arm A — prompt only (few-shot pairs stripped).
        let mut bare = preset.clone();
        bare.few_shot_pairs.clear();
        let a = run_arm(&rt, &llm, kind, "prompt-only", &bare, &mut verbose_first);
        rows.push((kind.clone(), "prompt-only", a));

        // Arm B — with built-in few-shot pairs (skip if the preset ships none).
        if preset.few_shot_pairs.is_empty() {
            println!("\n── {} · few-shot ── (none defined, skipped)", preset.name);
        } else {
            let b = run_arm(&rt, &llm, kind, "few-shot", preset, &mut verbose_first);
            rows.push((kind.clone(), "few-shot", b));
        }
    }

    if run_enhance {
        let preset = enhance.unwrap();
        println!(
            "\n════════════ {} ({}) ════════════",
            preset.name, preset.id
        );
        println!("rules: {}", preset.style_instructions);

        // Arm A — prompt only (this style ships no pairs anyway).
        let mut bare = preset.clone();
        bare.few_shot_pairs.clear();
        let a = run_arm(
            &rt,
            &llm,
            "enhance",
            "prompt-only",
            &bare,
            &mut verbose_first,
        );
        rows.push(("enhance".into(), "prompt-only", a));

        // Arm B — generate few-shot pairs with the model (styles-page path),
        // then style with them. Tests the generation capability too.
        if env_flag("GEN_FEWSHOT", true) {
            println!("\n── generating few-shot pairs (few_shot::generate_few_shot_examples) ──");
            match rt.block_on(generate_few_shot_examples(
                &llm,
                &preset.name,
                &preset.style_instructions,
            )) {
                Ok(pairs) => {
                    for (i, o) in &pairs {
                        println!("  gen: {i:?} -> {o:?}");
                    }
                    let mut with = preset.clone();
                    with.few_shot_pairs = pairs;
                    let b = run_arm(&rt, &llm, "enhance", "few-shot", &with, &mut verbose_first);
                    rows.push(("enhance".into(), "few-shot(gen)", b));
                }
                Err(e) => println!("  few-shot generation FAILED: {e}"),
            }
        }
    } else if want("enhance") {
        println!("\n[note] no Enhance Prompt custom style found in settings.toml — skipped");
    }

    // ─── Summary ─────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "{:<14} {:<14} {:>7} {:>8} {:>10}",
        "STYLE", "ARM", "CLEAN", "RULE", "AVG ms"
    );
    let mut clean_total = 0usize;
    let mut clean_denom = 0usize;
    for (kind, arm, s) in &rows {
        let avg = if s.n > 0 { s.ms_total / s.n as u128 } else { 0 };
        let rule = if s.green_n > 0 {
            format!("{}/{}", s.green, s.green_n)
        } else {
            "—".to_string()
        };
        println!(
            "{:<14} {:<14} {:>5}/{:<1} {:>8} {:>10}",
            kind, arm, s.clean, s.n, rule, avg
        );
        clean_total += s.clean;
        clean_denom += s.n;
    }
    println!("\n{:─<62}", "");
    let pct = if clean_denom > 0 {
        100.0 * clean_total as f64 / clean_denom as f64
    } else {
        0.0
    };
    println!("Overall clean (no red flags): {clean_total}/{clean_denom}  ({pct:.0}%)");
    println!(
        "CLEAN = no empty/meta/think/echo/no-op.  RULE = passed the per-style\n\
         heuristic (advisory).  Read the raw→out pairs above for the real\n\
         verdict — the numbers only flag obvious breakage."
    );
}

/// Load the user's "Enhance Prompt" custom style from `settings.toml`.
///
/// Matches by id/name containing "enhance" (case-insensitive). Falls back to
/// the first custom style if none matches but some exist. Returns `None` when
/// there are no custom styles (or settings can't be read).
fn load_enhance(paths: &AppPaths) -> Option<StylePreset> {
    let mut db = lune_persist::Database::open(&paths.database_file()).ok()?;
    let styles = db.list_custom_styles().ok()?;
    styles
        .iter()
        .find(|s| {
            s.preset.name.to_lowercase().contains("enhance")
                || s.preset.id.to_lowercase().contains("enhance")
        })
        .or_else(|| styles.first())
        .map(|s| s.preset.clone())
}
