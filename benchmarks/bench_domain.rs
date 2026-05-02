//! Domain benchmark — runs the full pipeline against categorized fixtures
//! and reports per-category WER, command exact-match, proper-noun accuracy,
//! and long-form truncation.
//!
//! cargo run --release --features cuda --example bench_domain -p lune-app

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, char_error_rate, load_wav_f32, process_memory_usage, word_error_rate,
    wrap_print,
};
use lune_core::command::VoiceCommand;
use lune_core::session::SessionEvent;
use tokio::sync::mpsc;

#[derive(Clone)]
struct ManifestEntry {
    id: String,
    category: String,
    expect_bypass: Option<bool>,
}

fn parse_manifest(text: &str) -> Option<Vec<ManifestEntry>> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let arr = v.get("fixtures")?.as_array()?;
    Some(
        arr.iter()
            .filter_map(|e| {
                Some(ManifestEntry {
                    id: e.get("id")?.as_str()?.to_string(),
                    category: e.get("category")?.as_str()?.to_string(),
                    expect_bypass: e.get("expect_bypass").and_then(|v| v.as_bool()),
                })
            })
            .collect(),
    )
}

struct DomainFixture {
    entry: ManifestEntry,
    pcm: Vec<f32>,
    reference: String,
    audio_secs: f64,
}

#[allow(dead_code)]
struct FixtureResult {
    id: String,
    category: String,
    wer: f64,
    cer: f64,
    exact_match: bool,
    audio_secs: f64,
    wall_secs: f64,
    hyp_words: usize,
    ref_words: usize,
    used_llm: bool,
    expect_bypass: Option<bool>,
    unresolved: Vec<String>,
}

fn category_dir(cat: &str) -> &str {
    match cat {
        "quiet_dictation" => "quiet_dictation",
        "long_form" => "long_form",
        "commands" => "commands",
        "punctuation" => "punctuation",
        "proper_nouns" => "proper_nouns",
        other => other,
    }
}

fn load_domain_fixtures(root: &Path) -> Vec<DomainFixture> {
    let bench_dir = root.join("fixtures").join("bench");
    let manifest_path = bench_dir.join("manifest.json");
    let manifest_text = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  [skip] Cannot read {}: {e}", manifest_path.display());
            return Vec::new();
        }
    };
    let entries = match parse_manifest(&manifest_text) {
        Some(e) => e,
        None => {
            eprintln!("  [skip] Invalid manifest");
            return Vec::new();
        }
    };

    let mut fixtures = Vec::new();
    for entry in entries {
        let dir = bench_dir.join(category_dir(&entry.category));
        let wav_path = dir.join(format!("{}.wav", entry.id));
        let txt_path = dir.join(format!("{}.txt", entry.id));

        if !wav_path.exists() || !txt_path.exists() {
            continue;
        }

        let pcm = load_wav_f32(&wav_path);
        let reference = std::fs::read_to_string(&txt_path)
            .expect("read reference")
            .trim()
            .to_string();
        let audio_secs = pcm.len() as f64 / 16_000.0;

        fixtures.push(DomainFixture {
            entry,
            pcm,
            reference,
            audio_secs,
        });
    }
    fixtures
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║             DOMAIN BENCHMARK (categorized fixtures)            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .with_llm()
        .skip_fixtures()
        .build();

    if let Some(mem) = process_memory_usage() {
        println!("  Memory after load: {}", mem.display());
    }

    let root = bench_utils::workspace_root();
    let fixtures = load_domain_fixtures(&root);
    if fixtures.is_empty() {
        println!("  [skip] No domain fixtures found in fixtures/bench/");
        println!("         Record WAVs and add .txt references per manifest.json");
        println!();
        return;
    }

    println!("  Loaded {} fixtures", fixtures.len());
    println!();

    let mut results: Vec<FixtureResult> = Vec::new();

    for fix in &fixtures {
        print!("  {:8} {:20}", fix.entry.category, fix.entry.id);

        if fix.entry.category == "commands" {
            harness
                .asr
                .set_boost_phrases(std::slice::from_ref(&fix.reference), 4.0);
        } else {
            harness.asr.set_boost_phrases(&[], 0.0);
        }

        let start = Instant::now();
        let outcome = harness
            .rt
            .block_on(run_pipeline(fix, &harness))
            .unwrap_or_default();
        let used_llm = outcome.used_llm;
        let unresolved = outcome.unresolved.clone();
        let text = if fix.entry.category == "commands" {
            outcome.command_text.unwrap_or(outcome.final_text)
        } else {
            outcome.final_text
        };
        let wall = start.elapsed().as_secs_f64();

        let wer = word_error_rate(&text, &fix.reference) * 100.0;
        let cer = char_error_rate(&text, &fix.reference) * 100.0;
        let exact =
            bench_utils::normalize_text(&text) == bench_utils::normalize_text(&fix.reference);

        let ref_words = fix.reference.split_whitespace().count();
        let hyp_words = text.split_whitespace().count();
        let rtf = if fix.audio_secs > 0.0 {
            wall / fix.audio_secs
        } else {
            0.0
        };

        let gate_tag = if used_llm { "LLM" } else { "BYPASS" };
        let gate_ok = match fix.entry.expect_bypass {
            Some(true) => !used_llm,
            Some(false) => used_llm,
            None => true,
        };
        let gate_mark = if !gate_ok { " ⚠" } else { "" };
        println!(
            "  WER {:5.1}%  CER {:5.1}%  {:.1}s/{:.1}s  RTF {:.3}  [{}{}]  {}",
            wer,
            cer,
            wall,
            fix.audio_secs,
            rtf,
            gate_tag,
            gate_mark,
            if exact { "✓" } else { "" }
        );

        if !unresolved.is_empty() {
            println!("    unresolved: {}", unresolved.join(", "));
        }

        if wer > 0.0 {
            wrap_print("    hyp: ", &text, 72);
            wrap_print("    ref: ", &fix.reference, 72);
        }

        results.push(FixtureResult {
            id: fix.entry.id.clone(),
            category: fix.entry.category.clone(),
            wer,
            cer,
            exact_match: exact,
            audio_secs: fix.audio_secs,
            wall_secs: wall,
            hyp_words,
            ref_words,
            used_llm,
            expect_bypass: fix.entry.expect_bypass,
            unresolved,
        });
    }

    // ── Per-category summary ─────────────────────────────────────────
    println!();
    println!("═══ Per-Category Summary ═══");
    println!();
    println!(
        "  {:20}  {:>5}  {:>7}  {:>7}  {:>7}  {:>9}",
        "Category", "N", "WER", "CER", "RTF", "Exact"
    );
    println!(
        "  {:20}  {:>5}  {:>7}  {:>7}  {:>7}  {:>9}",
        "────────", "─", "───", "───", "───", "─────"
    );

    let mut by_cat: BTreeMap<&str, Vec<&FixtureResult>> = BTreeMap::new();
    for r in &results {
        by_cat.entry(&r.category).or_default().push(r);
    }

    for (cat, items) in &by_cat {
        let n = items.len() as f64;
        let avg_wer = items.iter().map(|r| r.wer).sum::<f64>() / n;
        let avg_cer = items.iter().map(|r| r.cer).sum::<f64>() / n;
        let total_audio = items.iter().map(|r| r.audio_secs).sum::<f64>();
        let total_wall = items.iter().map(|r| r.wall_secs).sum::<f64>();
        let rtf = if total_audio > 0.0 {
            total_wall / total_audio
        } else {
            0.0
        };
        let exact_count = items.iter().filter(|r| r.exact_match).count();
        println!(
            "  {:20}  {:>5}  {:>6.1}%  {:>6.1}%  {:>6.3}  {:>4}/{:<4}",
            cat,
            items.len(),
            avg_wer,
            avg_cer,
            rtf,
            exact_count,
            items.len()
        );
    }

    // ── Overall ──────────────────────────────────────────────────────
    let n = results.len() as f64;
    let avg_wer = results.iter().map(|r| r.wer).sum::<f64>() / n;
    let avg_cer = results.iter().map(|r| r.cer).sum::<f64>() / n;
    let total_audio = results.iter().map(|r| r.audio_secs).sum::<f64>();
    let total_wall = results.iter().map(|r| r.wall_secs).sum::<f64>();
    let overall_rtf = if total_audio > 0.0 {
        total_wall / total_audio
    } else {
        0.0
    };
    let total_exact = results.iter().filter(|r| r.exact_match).count();

    println!();
    let llm_count = results.iter().filter(|r| r.used_llm).count();
    let gate_total = results.iter().filter(|r| r.expect_bypass.is_some()).count();
    let gate_correct = results
        .iter()
        .filter(|r| match r.expect_bypass {
            Some(true) => !r.used_llm,
            Some(false) => r.used_llm,
            None => false,
        })
        .count();
    println!(
        "  Overall: WER {:.1}%  CER {:.1}%  RTF {:.3}  Exact {}/{}  LLM {}/{}",
        avg_wer,
        avg_cer,
        overall_rtf,
        total_exact,
        results.len(),
        llm_count,
        results.len()
    );
    if gate_total > 0 {
        println!(
            "  Gate accuracy: {}/{} ({:.0}%)",
            gate_correct,
            gate_total,
            gate_correct as f64 / gate_total as f64 * 100.0
        );
    }

    // ── Command exact-match ──────────────────────────────────────────
    if let Some(cmds) = by_cat.get("commands") {
        let em = cmds.iter().filter(|r| r.exact_match).count();
        println!(
            "  Command exact-match: {}/{} ({:.0}%)",
            em,
            cmds.len(),
            em as f64 / cmds.len() as f64 * 100.0
        );
    }

    // ── Proper-noun exact-match ──────────────────────────────────────
    if let Some(pn) = by_cat.get("proper_nouns") {
        let em = pn.iter().filter(|r| r.exact_match).count();
        println!(
            "  Proper-noun exact-match: {}/{} ({:.0}%)",
            em,
            pn.len(),
            em as f64 / pn.len() as f64 * 100.0
        );
    }

    // ── Long-form truncation ─────────────────────────────────────────
    if let Some(lf) = by_cat.get("long_form") {
        let truncated = lf
            .iter()
            .filter(|r| (r.hyp_words as f64) < r.ref_words as f64 * 0.7)
            .count();
        println!("  Long-form truncation: {}/{} clips", truncated, lf.len());
    }

    // ── Slowest fixtures ───────────────────────────────────────────────
    let mut slowest: Vec<&FixtureResult> = results.iter().collect();
    slowest.sort_by(|a, b| b.wall_secs.total_cmp(&a.wall_secs));
    println!("  Slowest fixtures:");
    for r in slowest.into_iter().take(3) {
        let rtf = if r.audio_secs > 0.0 {
            r.wall_secs / r.audio_secs
        } else {
            0.0
        };
        println!(
            "    {:20} {:20}  {:.1}s wall  {:.1}s audio  RTF {:.3}",
            r.category, r.id, r.wall_secs, r.audio_secs, rtf
        );
    }

    println!();
}

#[derive(Default)]
struct PipelineOutcome {
    final_text: String,
    command_text: Option<String>,
    used_llm: bool,
    unresolved: Vec<String>,
}

async fn run_pipeline(
    fixture: &DomainFixture,
    harness: &BenchmarkHarness,
) -> Option<PipelineOutcome> {
    use lune_app::SessionManagerConfig;
    use lune_core::config::UserSettings;
    use tokio::sync::{Mutex, RwLock};

    let audio_stream =
        lune_audio::AudioStream::from_samples(&fixture.pcm, lune_audio::AudioConfig::default())
            .unwrap();
    let db = lune_persist::Database::open_in_memory().unwrap();
    let dict = lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
        .unwrap();

    let config = SessionManagerConfig {
        asr_backend_kind: harness.asr.kind(),
        pipeline: harness.asr.pipeline_config(),
        ..Default::default()
    };

    let settings = Arc::new(RwLock::new(UserSettings::default()));

    let mgr = lune_app::SessionManager::new(
        Arc::clone(&harness.asr),
        Arc::new(lune_commands::VoiceCommandRouter::new()),
        Arc::clone(&harness.llm),
        Arc::new(Mutex::new(dict)),
        Arc::new(Mutex::new(db)),
        Arc::new(lune_dict::corrector::HeuristicCorrector::new()),
        config,
        settings,
    );
    let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(256);

    tokio::task::LocalSet::new()
        .run_until(async {
            let _id = mgr
                .start_with_audio(event_tx, audio_stream)
                .await
                .expect("start session");
            let mut final_text = String::new();
            let mut command_text: Option<String> = None;
            let mut used_llm = false;
            let mut unresolved = Vec::new();
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(180);

            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(SessionEvent::FinalizedText(t))) => final_text.push_str(&t),
                    Ok(Some(SessionEvent::CommandDetected(payload))) => {
                        if command_text.is_none() {
                            command_text = command_text_from_payload(&payload);
                        }
                    }
                    Ok(Some(SessionEvent::GateDecision {
                        used_llm: u,
                        lang: _,
                        unresolved: w,
                    })) => {
                        used_llm = u;
                        unresolved = w;
                    }
                    Ok(Some(SessionEvent::Stopped)) => break,
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("  [warn] 180s safety timeout");
                        break;
                    }
                }
            }

            if final_text.is_empty() && command_text.is_none() {
                None
            } else {
                Some(PipelineOutcome {
                    final_text,
                    command_text,
                    used_llm,
                    unresolved,
                })
            }
        })
        .await
}

fn command_text_from_payload(payload: &str) -> Option<String> {
    let cmd: VoiceCommand = serde_json::from_str(payload).ok()?;
    Some(match cmd {
        VoiceCommand::StopDictation => "stop listening".to_string(),
        VoiceCommand::NewLine => "new line".to_string(),
        VoiceCommand::NewParagraph => "new paragraph".to_string(),
        VoiceCommand::ListItem => "list item".to_string(),
        VoiceCommand::ScratchThat => "scratch that".to_string(),
        VoiceCommand::SwitchStyle(style) => format!("switch to {style}"),
        VoiceCommand::InsertSnippet(name) => format!("insert {name}"),
        _ => return None,
    })
}
