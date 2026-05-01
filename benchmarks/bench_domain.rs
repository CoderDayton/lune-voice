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
use lune_core::session::SessionEvent;
use tokio::sync::mpsc;

#[derive(Clone)]
struct ManifestEntry {
    id: String,
    category: String,
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

struct FixtureResult {
    id: String,
    category: String,
    wer: f64,
    cer: f64,
    exact_match: bool,
    wall_secs: f64,
    hyp_words: usize,
    ref_words: usize,
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

        let start = Instant::now();
        let text = harness
            .rt
            .block_on(run_pipeline(fix, &harness))
            .unwrap_or_default();
        let wall = start.elapsed().as_secs_f64();

        let wer = word_error_rate(&text, &fix.reference) * 100.0;
        let cer = char_error_rate(&text, &fix.reference) * 100.0;
        let exact =
            bench_utils::normalize_text(&text) == bench_utils::normalize_text(&fix.reference);

        let ref_words = fix.reference.split_whitespace().count();
        let hyp_words = text.split_whitespace().count();

        println!(
            "  WER {:5.1}%  CER {:5.1}%  {:.1}s  {}",
            wer,
            cer,
            wall,
            if exact { "✓" } else { "" }
        );

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
            wall_secs: wall,
            hyp_words,
            ref_words,
        });
    }

    // ── Per-category summary ─────────────────────────────────────────
    println!();
    println!("═══ Per-Category Summary ═══");
    println!();
    println!(
        "  {:20}  {:>5}  {:>7}  {:>7}  {:>9}",
        "Category", "N", "WER", "CER", "Exact"
    );
    println!(
        "  {:20}  {:>5}  {:>7}  {:>7}  {:>9}",
        "────────", "─", "───", "───", "─────"
    );

    let mut by_cat: BTreeMap<&str, Vec<&FixtureResult>> = BTreeMap::new();
    for r in &results {
        by_cat.entry(&r.category).or_default().push(r);
    }

    for (cat, items) in &by_cat {
        let n = items.len() as f64;
        let avg_wer = items.iter().map(|r| r.wer).sum::<f64>() / n;
        let avg_cer = items.iter().map(|r| r.cer).sum::<f64>() / n;
        let exact_count = items.iter().filter(|r| r.exact_match).count();
        println!(
            "  {:20}  {:>5}  {:>6.1}%  {:>6.1}%  {:>4}/{:<4}",
            cat,
            items.len(),
            avg_wer,
            avg_cer,
            exact_count,
            items.len()
        );
    }

    // ── Overall ──────────────────────────────────────────────────────
    let n = results.len() as f64;
    let avg_wer = results.iter().map(|r| r.wer).sum::<f64>() / n;
    let avg_cer = results.iter().map(|r| r.cer).sum::<f64>() / n;
    let total_exact = results.iter().filter(|r| r.exact_match).count();

    println!();
    println!(
        "  Overall: WER {:.1}%  CER {:.1}%  Exact {}/{}",
        avg_wer,
        avg_cer,
        total_exact,
        results.len()
    );

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

    println!();
}

async fn run_pipeline(fixture: &DomainFixture, harness: &BenchmarkHarness) -> Option<String> {
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
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(180);

            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(SessionEvent::FinalizedText(t))) => final_text.push_str(&t),
                    Ok(Some(SessionEvent::Stopped)) => break,
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("  [warn] 180s safety timeout");
                        break;
                    }
                }
            }

            if final_text.is_empty() {
                None
            } else {
                Some(final_text)
            }
        })
        .await
}
