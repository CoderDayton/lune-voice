// SPDX-License-Identifier: GPL-3.0-or-later

//! Prompt-strategy A/B for the configured remote styling LLM.
//!
//! Tests the folk claim that tiny models do better with **explicit numbered
//! rules** than with **few-shot examples** (examples "distract" a 0.8B model
//! with their content). It isolates two variables, holding everything else
//! byte-identical to the production framing so the comparison is fair:
//!
//!   • rule FORMAT   — prose sentence vs a numbered `1. 2. 3.` list
//!   • rule PLACEMENT — all-in-user (production, empty system) vs a system role
//!
//! Four arms per (style, input):
//!   1. prose-user: production path: empty system, prose rules in the user
//!      message (== `build_styling_prompt` with no examples).
//!   2. numbered-user: same framing + text, rules as a numbered list, empty
//!      system. Isolates FORMAT vs arm 1.
//!   3. numbered-system: same content as arm 2, but framing + rules moved
//!      into the system role. Isolates PLACEMENT vs arm 2.
//!   4. few-shot: production few-shot path (`build_styling_prompt` with the
//!      preset's built-in pairs). The baseline the claim says should lose.
//!      Skipped for styles with no pairs.
//!
//! Run:  cgs bench --env TYPE=rulecmp (needs LUNE_BENCH_LLM_BASE_URL / _MODEL)
//! Env:  STYLES=casual,technical   MAX_TOKENS=512   STYLING_VERBOSE=1

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::fmt::Write as _;
use std::time::Instant;

use lune_core::config::StylePreset;
use lune_core::llm::LlmService;
use lune_llm::OpenAiLlmBackend;
use lune_llm::prompt::{contains_forbidden_output, fix_all_number_spaces};
use lune_persist::AppPaths;

// ─── Inputs (shared with styling_capability) ─────────────────────────────────

const GENERAL_INPUTS: &[&str] = &[
    "so um i was thinking we could meet tomorrow maybe around noon if that works for everybody",
    "hey just checking in on the deploy did the migration finish if not we should roll back before the demo at three",
    "the report is basically done i just need to like double check the numbers and then ill send it over",
    "she said i'll be there but honestly i don't think she meant it we'll see i guess",
];
const TECHNICAL_INPUTS: &[&str] = &[
    "lets add an async function called fetch user that takes a user id and returns an optional user from the database",
    "the retry helper should catch the timeout error and back off exponentially before it calls send again",
];
const PROMPT_INPUTS: &[&str] = &[
    "write a function that sorts a list of numbers",
    "make me a landing page for my bakery with a hero section and a menu",
    "explain how photosynthesis works to a five year old",
];
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

// ─── Prompt builders ─────────────────────────────────────────────────────────

/// Framing shared by every no-example arm — the exact wording
/// `build_styling_prompt` uses, minus its "examples" sentence. Keeping this
/// identical is what makes the prose-vs-numbered comparison honest.
const FRAMING: &str = "Apply the style below to the transcript. Change only what the style requires; keep the original wording, content, and meaning everywhere else. Output only the transcript with the style applied.";

const TAIL: &str = "Now rewrite only this transcript in the style above:";

/// Split prose instructions into atomic rules on sentence / clause breaks.
fn numbered_rules(instructions: &str) -> Vec<String> {
    instructions
        .split(['.', ';'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn rules_block(preset: &StylePreset) -> String {
    let mut b = String::new();
    for (i, r) in numbered_rules(&preset.style_instructions)
        .iter()
        .enumerate()
    {
        let _ = writeln!(b, "{}. {r}.", i + 1);
    }
    b
}

/// (system, user) for a given arm.
fn build(arm: &str, preset: &StylePreset, input: &str) -> (String, String) {
    match arm {
        // Production path, examples stripped. NOTE: since the builder moved to a
        // (system, user) split, "prose-user" now places rules in the system role
        // too — this experiment already served its purpose (rules ≯ few-shot);
        // the arm names are kept for continuity.
        "prose-user" => {
            let mut bare = preset.clone();
            bare.few_shot_pairs.clear();
            bare.build_styling_prompt(input)
        }
        // Production path with the built-in few-shot pairs.
        "few-shot" => preset.build_styling_prompt(input),
        // Same framing + text as prose-user, rules as a numbered list. Empty system.
        "numbered-user" => {
            let user = format!(
                "{FRAMING}\n\nStyle: {}\nRules:\n{}\n{TAIL}\n{input}\n",
                preset.name,
                rules_block(preset),
            );
            (String::new(), user)
        }
        // Same content as numbered-user, framing + rules moved into the system role.
        "numbered-system" => {
            let system = format!(
                "{FRAMING}\n\nStyle: {}\nRules:\n{}",
                preset.name,
                rules_block(preset),
            );
            (system, format!("{TAIL}\n{input}\n"))
        }
        other => panic!("unknown arm {other}"),
    }
}

const ARMS: &[&str] = &["prose-user", "numbered-user", "numbered-system", "few-shot"];

// ─── Scoring (mirrors styling_capability) ────────────────────────────────────

fn red_flags(out: &str, input: &str, pairs: &[(String, String)]) -> Vec<&'static str> {
    let mut f = Vec::new();
    if out.trim().is_empty() {
        f.push("EMPTY");
        return f;
    }
    if contains_forbidden_output(out) {
        f.push("META");
    }
    if out.contains("<think") || out.contains("</think") {
        f.push("THINK");
    }
    if out.trim() == input.trim() {
        f.push("NO-OP");
    }
    // Task-execution instead of restyle: emitted code the input didn't ask
    // for (fenced block or a bare def/function/class definition).
    let codey = |t: &str| {
        let h = t.trim_start();
        t.contains("```")
            || h.starts_with("def ")
            || h.starts_with("function ")
            || h.starts_with("async function")
            || h.starts_with("class ")
    };
    if codey(out) && !codey(input) {
        f.push("CODEGEN");
    }
    let echoed = pairs.iter().any(|(pi, po)| {
        (po.len() > 4 && out.contains(po.as_str())) || (pi.len() > 8 && out.contains(pi.as_str()))
    });
    if echoed {
        f.push("ECHO");
    }
    f
}

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
        "casual" => {
            let clean = !out.contains('"')
                && !out.contains(';')
                && !out.contains(':')
                && !out.contains('!');
            let lc_start = out
                .trim()
                .chars()
                .find(|c| c.is_alphabetic())
                .map(char::is_lowercase)
                .unwrap_or(true);
            Some(clean && lc_start)
        }
        "professional" | "formal" => Some(!has_contraction(out)),
        "technical" => Some(out.contains('`') && !out.contains("```")),
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
        "enhance" => {
            let wi = input.split_whitespace().count().max(1) as f64;
            let wo = out.split_whitespace().count() as f64;
            Some((0.6..=2.0).contains(&(wo / wi)))
        }
        _ => None,
    }
}

#[derive(Default, Clone)]
struct Stat {
    n: usize,
    clean: usize,
    green: usize,
    green_n: usize,
    ms: u128,
}

// ─── Runner ──────────────────────────────────────────────────────────────────

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
    let filter: Option<Vec<String>> = std::env::var("STYLES").ok().map(|v| {
        v.split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    });
    let verbose = std::env::var("STYLING_VERBOSE").is_ok();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Prompt-strategy A/B: numbered rules vs few-shot            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("model : {}   max_new_tokens: {max_tokens}", config.model);
    println!("endpoint : {}", config.base_url);
    let load = Instant::now();
    let llm = OpenAiLlmBackend::new(config).expect("bench LLM config must be valid");
    println!("load  : {} ms", load.elapsed().as_millis());
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    // Styles under test: built-ins + Enhance Prompt.
    let mut styles: Vec<(String, StylePreset)> = StylePreset::all_builtin()
        .iter()
        .map(|a| (a.id.clone(), (**a).clone()))
        .collect();
    if let Some(e) = load_enhance(&paths) {
        styles.push(("enhance".into(), e));
    }
    let want = |k: &str| {
        filter
            .as_ref()
            .map(|f| f.iter().any(|s| s == k))
            .unwrap_or(true)
    };

    // stats[kind][arm]
    let mut stats: Vec<(String, Vec<(&'static str, Stat)>)> = Vec::new();
    let mut printed_sample = false;

    for (kind, preset) in &styles {
        if !want(kind) {
            continue;
        }
        println!("\n════════════ {} ({kind}) ════════════", preset.name);
        println!("rules(prose)   : {}", preset.style_instructions);
        println!("rules(numbered):");
        print!("{}", rules_block(preset));

        let mut per_arm: Vec<(&'static str, Stat)> = Vec::new();
        for &arm in ARMS {
            // few-shot arm is only meaningful when the preset ships pairs.
            if arm == "few-shot" && preset.few_shot_pairs.is_empty() {
                continue;
            }
            let mut st = Stat::default();
            println!("\n── {} · {arm} ──", preset.name);
            if verbose && !printed_sample {
                let (s, u) = build(arm, preset, inputs_for(kind)[0]);
                println!(
                    "  [system]{}\n  [user]\n{u}\n  [/prompt]",
                    if s.is_empty() { " (empty)" } else { "" }
                );
                if !s.is_empty() {
                    println!("  [system body]\n{s}");
                }
            }
            for input in inputs_for(kind) {
                let (system, user) = build(arm, preset, input);
                let t = Instant::now();
                let out = rt
                    .block_on(llm.complete(&system, &user))
                    .unwrap_or_else(|e| format!("<error: {e}>"));
                let ms = t.elapsed().as_millis();
                let flags = red_flags(&out, input, &preset.few_shot_pairs);
                let green = green_check(kind, input, &out);

                st.n += 1;
                st.ms += ms;
                if flags.is_empty() {
                    st.clean += 1;
                }
                if let Some(p) = green {
                    st.green_n += 1;
                    st.green += p as usize;
                }
                let fs = if flags.is_empty() {
                    "ok".into()
                } else {
                    flags.join(",")
                };
                let gs = match green {
                    Some(true) => " ✓rule",
                    Some(false) => " ✗rule",
                    None => "",
                };
                println!("  raw : {input}");
                println!("  out : {}", fix_all_number_spaces(out.trim()));
                println!("        ({ms} ms · {fs}{gs})");
            }
            per_arm.push((arm, st));
        }
        printed_sample = printed_sample || verbose;
        stats.push((kind.clone(), per_arm));
    }

    // ─── Summary ────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary — clean/N · rule-pass/N · avg ms, best arm per style ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "{:<13} {:<16} {:>8} {:>8} {:>8}",
        "STYLE", "ARM", "CLEAN", "RULE", "AVG ms"
    );
    for (kind, per_arm) in &stats {
        // Rank by rule-pass rate, then clean rate.
        let score = |s: &Stat| {
            let rule = if s.green_n > 0 {
                s.green as f64 / s.green_n as f64
            } else {
                -1.0
            };
            let clean = if s.n > 0 {
                s.clean as f64 / s.n as f64
            } else {
                0.0
            };
            (rule, clean)
        };
        let best = per_arm
            .iter()
            .max_by(|a, b| score(&a.1).partial_cmp(&score(&b.1)).unwrap())
            .map(|(arm, _)| *arm)
            .unwrap_or("");
        for (arm, s) in per_arm {
            let avg = if s.n > 0 { s.ms / s.n as u128 } else { 0 };
            let rule = if s.green_n > 0 {
                format!("{}/{}", s.green, s.green_n)
            } else {
                "—".into()
            };
            let mark = if *arm == best { " ◀ best" } else { "" };
            println!(
                "{:<13} {:<16} {:>6}/{:<1} {:>8} {:>8}{mark}",
                kind, arm, s.clean, s.n, rule, avg
            );
        }
        println!("{:─<55}", "");
    }
    println!(
        "Best arm ranked by rule-pass rate, then clean rate. `numbered-*`\n\
         winning over `few-shot`/`prose-user` supports the tiny-model claim;\n\
         read the raw→out pairs for the real story."
    );
}
