// SPDX-License-Identifier: GPL-3.0-or-later

//! Confusion mining — decode fixtures, word-align against references, extract
//! substitution pairs where model picked wrong BPE path.
//!
//! Run:   cargo run --release --example confusion_mine -p lune-voice
//! GPU:   cargo run --release --features whisper-vulkan --example confusion_mine -p lune-voice
//! Filter: LUNE_BENCH_FIXTURE=librispeech_40744 cargo run...
//!
//! Output: prints JSON report to stdout and writes `confusion_bank.json`
//! to workspace root (gitignored).

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::collections::HashMap;

use bench_utils::{BenchmarkHarness, normalize_text, workspace_root};
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use serde_json::json;

// ─── Alignment types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum EditOp {
    Match,
    Substitute,
    Insert,
    Delete,
}

/// Word-level Levenshtein alignment with backtrace.
/// Returns vec of (op, ref_word_or_none, hyp_word_or_none).
fn align_words<'a>(
    ref_words: &'a [&str],
    hyp_words: &'a [&str],
) -> Vec<(EditOp, Option<&'a str>, Option<&'a str>)> {
    let m = ref_words.len();
    let n = hyp_words.len();

    // Full DP matrix for backtrace
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1) {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            if ref_words[i - 1] == hyp_words[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1]);
            }
        }
    }

    // Backtrace
    let mut ops = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && ref_words[i - 1] == hyp_words[j - 1] {
            ops.push((
                EditOp::Match,
                Some(ref_words[i - 1]),
                Some(hyp_words[j - 1]),
            ));
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            ops.push((
                EditOp::Substitute,
                Some(ref_words[i - 1]),
                Some(hyp_words[j - 1]),
            ));
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            ops.push((EditOp::Delete, Some(ref_words[i - 1]), None));
            i -= 1;
        } else {
            ops.push((EditOp::Insert, None, Some(hyp_words[j - 1])));
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

// ─── Output types ───────────────────────────────────────────────────────────

struct ConfusionEntry {
    reference: String,
    hypothesis: String,
    count: usize,
    fixtures: Vec<String>,
}

struct ErrorInstance {
    op: String,
    reference: Option<String>,
    hypothesis: Option<String>,
    context: String,
}

struct FixtureDetail {
    name: String,
    ref_words: usize,
    substitutions: usize,
    insertions: usize,
    deletions: usize,
    wer_percent: f64,
    errors: Vec<ErrorInstance>,
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      CONFUSION MINING                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let harness =
        match std::panic::catch_unwind(|| BenchmarkHarness::builder().with_tracing().build()) {
            Ok(h) => h,
            Err(_) => {
                eprintln!("[skip] Could not build harness (missing models?)");
                return;
            }
        };

    if harness.fixtures.is_empty() {
        eprintln!("[skip] No fixtures");
        return;
    }

    // Display name for the JSON report — resolve from bench_utils helper.
    let backend_name = match bench_utils::benchmark_asr_entry() {
        Ok(e) => e.display_name,
        Err(msg) => {
            eprintln!("[skip] {msg}");
            return;
        }
    };
    println!("  Backend: {backend_name}");

    let root = workspace_root();

    let mut all_fixture_details: Vec<FixtureDetail> = Vec::new();
    // Key: (normalized_ref, normalized_hyp) -> (count, fixture_names)
    let mut pair_counts: HashMap<(String, String), (usize, Vec<String>)> = HashMap::new();
    let mut total_ref_words = 0usize;
    let mut total_subs = 0usize;
    let mut total_ins = 0usize;
    let mut total_del = 0usize;

    for fixture in &harness.fixtures {
        println!();
        println!("── {} ({:.1}s) ──", fixture.name, fixture.audio_secs);

        // Raw ASR decode — no VAD, no LLM, model output.
        // Detect language from fixture name for proper multilingual support.
        let lang = language_from_fixture(fixture.name);
        let hyp = harness
            .rt
            .block_on(decode_raw(&harness.asr, &fixture.pcm, lang));
        let hyp = match hyp {
            Some(t) => t,
            None => {
                println!("  [error] no output");
                continue;
            }
        };

        println!("  Hyp: {}", truncate(&hyp, 100));
        println!("  Ref: {}", truncate(&fixture.reference, 100));

        // Normalize and align
        let ref_norm = normalize_text(&fixture.reference);
        let hyp_norm = normalize_text(&hyp);
        let ref_words: Vec<&str> = ref_norm.split_whitespace().collect();
        let hyp_words: Vec<&str> = hyp_norm.split_whitespace().collect();

        let ops = align_words(&ref_words, &hyp_words);

        let mut fx_subs = 0;
        let mut fx_ins = 0;
        let mut fx_del = 0;
        let mut errors = Vec::new();

        for (idx, (op, rw, hw)) in ops.iter().enumerate() {
            match op {
                EditOp::Match => {}
                EditOp::Substitute => {
                    fx_subs += 1;
                    let rw = rw.unwrap();
                    let hw = hw.unwrap();
                    let ctx = build_context(&ops, idx);
                    errors.push(ErrorInstance {
                        op: "sub".into(),
                        reference: Some(rw.to_string()),
                        hypothesis: Some(hw.to_string()),
                        context: ctx,
                    });
                    let key = (rw.to_string(), hw.to_string());
                    let entry = pair_counts.entry(key).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    if !entry.1.contains(&fixture.name.to_string()) {
                        entry.1.push(fixture.name.to_string());
                    }
                    println!("  SUB: {rw} → {hw}");
                }
                EditOp::Insert => {
                    fx_ins += 1;
                    let hw = hw.unwrap();
                    let ctx = build_context(&ops, idx);
                    errors.push(ErrorInstance {
                        op: "ins".into(),
                        reference: None,
                        hypothesis: Some(hw.to_string()),
                        context: ctx,
                    });
                    println!("  INS: ∅ → {hw}");
                }
                EditOp::Delete => {
                    fx_del += 1;
                    let rw = rw.unwrap();
                    let ctx = build_context(&ops, idx);
                    errors.push(ErrorInstance {
                        op: "del".into(),
                        reference: Some(rw.to_string()),
                        hypothesis: None,
                        context: ctx,
                    });
                    println!("  DEL: {rw} → ∅");
                }
            }
        }

        let n_ref = ref_words.len();
        let wer = if n_ref > 0 {
            (fx_subs + fx_ins + fx_del) as f64 / n_ref as f64 * 100.0
        } else {
            0.0
        };
        println!("  Words: {n_ref}  S:{fx_subs}  I:{fx_ins}  D:{fx_del}  WER:{wer:.1}%");

        total_ref_words += n_ref;
        total_subs += fx_subs;
        total_ins += fx_ins;
        total_del += fx_del;

        all_fixture_details.push(FixtureDetail {
            name: fixture.name.to_string(),
            ref_words: n_ref,
            substitutions: fx_subs,
            insertions: fx_ins,
            deletions: fx_del,
            wer_percent: wer,
            errors,
        });
    }

    // Build sorted confusion pairs
    let mut confusion_pairs: Vec<ConfusionEntry> = pair_counts
        .into_iter()
        .map(|((r, h), (count, fixtures))| ConfusionEntry {
            reference: r,
            hypothesis: h,
            count,
            fixtures,
        })
        .collect();
    confusion_pairs.sort_by(|a, b| b.count.cmp(&a.count).then(a.reference.cmp(&b.reference)));

    let overall_wer = if total_ref_words > 0 {
        (total_subs + total_ins + total_del) as f64 / total_ref_words as f64 * 100.0
    } else {
        0.0
    };

    let pairs_json: Vec<serde_json::Value> = confusion_pairs
        .iter()
        .map(|p| {
            json!({
                "reference": p.reference,
                "hypothesis": p.hypothesis,
                "count": p.count,
                "fixtures": p.fixtures,
            })
        })
        .collect();

    let details_json: Vec<serde_json::Value> = all_fixture_details
        .iter()
        .map(|d| {
            json!({
                "name": d.name,
                "ref_words": d.ref_words,
                "substitutions": d.substitutions,
                "insertions": d.insertions,
                "deletions": d.deletions,
                "wer_percent": d.wer_percent,
                "errors": d.errors.iter().map(|e| json!({
                    "op": e.op,
                    "reference": e.reference,
                    "hypothesis": e.hypothesis,
                    "context": e.context,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    let report = json!({
        "backend": backend_name,
        "total_fixtures": all_fixture_details.len(),
        "total_ref_words": total_ref_words,
        "total_substitutions": total_subs,
        "total_insertions": total_ins,
        "total_deletions": total_del,
        "wer_percent": overall_wer,
        "confusion_pairs": pairs_json,
        "fixture_details": details_json,
    });

    // Write JSON report
    let json_str = serde_json::to_string_pretty(&report).expect("serialize");
    let out_path = root.join("benchmarks").join("confusion_bank.json");
    std::fs::write(&out_path, &json_str).expect("write benchmarks/confusion_bank.json");

    let n_pairs = confusion_pairs.len();
    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(
        "  Total: {} words, S:{} I:{} D:{}, WER:{:.1}%",
        total_ref_words, total_subs, total_ins, total_del, overall_wer
    );
    println!("  Confusion pairs: {n_pairs}");
    println!("  Written to: {}", out_path.display());
    println!("════════════════════════════════════════════════════════════════════");
}

// ─── Helpers ────────────────────────────────────────────────────────────────

async fn decode_raw(
    asr: &lune_asr::AsrRegistry,
    pcm: &[f32],
    language: Option<&str>,
) -> Option<String> {
    let config = AsrSessionConfig {
        language: language.map(|s| s.to_string()),
        ..Default::default()
    };
    let session = asr.start_session(config).await.ok()?;
    asr.push_audio(&session, pcm).await.ok()?;
    let segments = asr.poll_final(&session).await.ok()?;
    let _ = asr.end_session(session).await;

    let text: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let text = text.trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// Detect language hint from fixture name. Returns BCP-47 style code
/// that maps to Qwen3's canonical language names in start_session.
/// Returns None for auto-detect if no language can be inferred.
fn language_from_fixture(name: &str) -> Option<&str> {
    // LUNE_AUTO_LANG=1 → None for all fixtures (test auto-detect)
    if std::env::var("LUNE_AUTO_LANG")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return None;
    }
    if name.contains("french") {
        Some("fr")
    } else if name.contains("german") {
        Some("de")
    } else if name.contains("italian") {
        Some("it")
    } else if name.contains("spanish") {
        Some("es")
    } else if name.contains("chinese") || name.contains("mandarin") {
        Some("zh")
    } else if name.contains("japanese") {
        Some("ja")
    } else if name.contains("korean") {
        Some("ko")
    } else if name.contains("portuguese") {
        Some("pt")
    } else if name.contains("russian") {
        Some("ru")
    } else {
        Some("en-US")
    } // Default English for unlabeled fixtures
}

fn build_context(ops: &[(EditOp, Option<&str>, Option<&str>)], center: usize) -> String {
    let start = center.saturating_sub(3);
    let end = (center + 4).min(ops.len());
    let mut parts = Vec::new();
    for (i, (op, rw, hw)) in ops[start..end].iter().enumerate() {
        let idx = start + i;
        let word = match op {
            EditOp::Match => rw.unwrap_or("?").to_string(),
            EditOp::Substitute if idx == center => {
                format!("[{}→{}]", rw.unwrap_or("?"), hw.unwrap_or("?"))
            }
            EditOp::Substitute => rw.unwrap_or("?").to_string(),
            EditOp::Insert if idx == center => format!("[+{}]", hw.unwrap_or("?")),
            EditOp::Insert => hw.unwrap_or("?").to_string(),
            EditOp::Delete if idx == center => format!("[-{}]", rw.unwrap_or("?")),
            EditOp::Delete => String::new(),
        };
        if !word.is_empty() {
            parts.push(word);
        }
    }
    parts.join(" ")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}
