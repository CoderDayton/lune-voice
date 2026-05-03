//! LLM-only benchmark — runs the real pipeline with and without LLM to isolate
//! correction latency and measure text quality impact.
//!
//! GPU: cargo run --release --features cuda --example llm_quality -p lune-app
//! CPU: cargo run --release --example llm_quality -p lune-app

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, Fixture, char_error_rate, process_memory_usage, word_error_rate, wrap_print,
};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_llm::NullPostProcessor;
use tokio::sync::mpsc;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              LLM QUALITY BENCHMARK (A/B pipeline)              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .with_llm()
        .build();

    if let Some(mem) = process_memory_usage() {
        println!("  Memory after load: {}", mem.display());
    }
    if harness.fixtures.is_empty() {
        println!("[skip] No WAV fixtures found");
        return;
    }

    let null_llm: Arc<dyn TextPostProcessor> = Arc::new(NullPostProcessor::new());
    let mut results: Vec<FixtureResult> = Vec::new();

    for fixture in &harness.fixtures {
        println!();
        println!("── {} ({:.1}s audio) ──", fixture.name, fixture.audio_secs);

        let pipeline = harness.asr.pipeline_config();

        let (text_no_llm, time_no_llm) = harness
            .rt
            .block_on(run_pipeline(
                fixture,
                &harness,
                Arc::clone(&null_llm),
                pipeline.clone(),
            ))
            .unwrap_or_default();

        let (text_with_llm, time_with_llm) = harness
            .rt
            .block_on(run_pipeline(
                fixture,
                &harness,
                Arc::clone(&harness.llm),
                pipeline,
            ))
            .unwrap_or_default();

        let llm_delta = time_with_llm - time_no_llm;
        let bypassed = text_no_llm == text_with_llm;

        let wer_raw = word_error_rate(&text_no_llm, &fixture.reference) * 100.0;
        let cer_raw = char_error_rate(&text_no_llm, &fixture.reference) * 100.0;

        if bypassed {
            println!();
            println!("  [SAME] LLM made no changes (Δ {:.1}s)", llm_delta);
            println!("  {:>20}  {:>10.1}%", "WER", wer_raw);
            println!("  {:>20}  {:>10.1}%", "CER", cer_raw);
            println!();
            wrap_print("    ", &text_no_llm, 72);
            println!("  Reference:");
            wrap_print("    ", &fixture.reference, 72);

            results.push(FixtureResult {
                name: fixture.name,
                wer_raw,
                wer_llm: wer_raw,
                cer_raw,
                cer_llm: cer_raw,
                llm_delta: 0.0,
                bypassed: true,
            });
        } else {
            let wer_llm = word_error_rate(&text_with_llm, &fixture.reference) * 100.0;
            let cer_llm = char_error_rate(&text_with_llm, &fixture.reference) * 100.0;

            println!();
            println!("  {:>20}  {:>10}  {:>10}", "", "No LLM", "With LLM");
            println!("  {:>20}  {:>10.1}%  {:>10.1}%", "WER", wer_raw, wer_llm);
            println!("  {:>20}  {:>10.1}%  {:>10.1}%", "CER", cer_raw, cer_llm);
            println!(
                "  {:>20}  {:>10.3}s  {:>10.3}s",
                "Wall time", time_no_llm, time_with_llm
            );
            println!("  {:>20}  {:>10.3}s", "LLM delta", llm_delta);
            println!();
            println!("  ASR only:");
            wrap_print("    ", &text_no_llm, 72);
            println!("  ASR + LLM:");
            wrap_print("    ", &text_with_llm, 72);
            println!("  Reference:");
            wrap_print("    ", &fixture.reference, 72);

            results.push(FixtureResult {
                name: fixture.name,
                wer_raw,
                wer_llm,
                cer_raw,
                cer_llm,
                llm_delta,
                bypassed: false,
            });
        }
    }

    if results.len() > 1 {
        println!();
        println!("═══ Summary ═══");
        println!();
        println!(
            "  {:20}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}",
            "Fixture", "WER raw", "WER+LLM", "CER raw", "CER+LLM", "LLM Δ(s)"
        );
        println!(
            "  {:20}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}",
            "───────", "───────", "───────", "───────", "───────", "────────"
        );
        for r in &results {
            if r.bypassed {
                println!(
                    "  {:20}  {:>7.1}%  {:>8}  {:>7.1}%  {:>8}  {:>9}",
                    r.name, r.wer_raw, "BYPASS", r.cer_raw, "BYPASS", "—"
                );
            } else {
                println!(
                    "  {:20}  {:>7.1}%  {:>7.1}%  {:>7.1}%  {:>7.1}%  {:>9.3}",
                    r.name, r.wer_raw, r.wer_llm, r.cer_raw, r.cer_llm, r.llm_delta
                );
            }
        }
        let llm_results: Vec<_> = results.iter().filter(|r| !r.bypassed).collect();
        let n_llm = llm_results.len() as f64;
        let avg_delta = if n_llm > 0.0 {
            llm_results.iter().map(|r| r.llm_delta).sum::<f64>() / n_llm
        } else {
            0.0
        };
        let n = results.len() as f64;
        let avg_wer_raw = results.iter().map(|r| r.wer_raw).sum::<f64>() / n;
        let avg_wer_pipeline = results.iter().map(|r| r.wer_llm).sum::<f64>() / n;
        let avg_wer_llm_only = if n_llm > 0.0 {
            let raw_llm = llm_results.iter().map(|r| r.wer_raw).sum::<f64>() / n_llm;
            let post_llm = llm_results.iter().map(|r| r.wer_llm).sum::<f64>() / n_llm;
            format!("  LLM-only: {:.1}% → {:.1}%", raw_llm, post_llm)
        } else {
            String::new()
        };
        println!();
        println!(
            "  Avg WER: {:.1}% → {:.1}%  |  Avg LLM Δ: {:.3}s",
            avg_wer_raw, avg_wer_pipeline, avg_delta
        );
        if !avg_wer_llm_only.is_empty() {
            println!("{avg_wer_llm_only}");
        }
    }

    println!();
}

struct FixtureResult {
    name: &'static str,
    wer_raw: f64,
    wer_llm: f64,
    cer_raw: f64,
    cer_llm: f64,
    llm_delta: f64,
    bypassed: bool,
}

async fn run_pipeline(
    fixture: &Fixture,
    harness: &BenchmarkHarness,
    llm: Arc<dyn TextPostProcessor>,
    pipeline: lune_core::config::PipelineConfig,
) -> Option<(String, f64)> {
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
        pipeline,
        ..Default::default()
    };

    let user_settings = UserSettings {
        language: BenchmarkHarness::language_for_fixture(fixture.name).into(),
        ..Default::default()
    };
    let settings = Arc::new(RwLock::new(user_settings));

    let mgr = lune_app::SessionManager::new(
        Arc::clone(&harness.asr),
        Arc::new(lune_commands::VoiceCommandRouter::new()),
        llm,
        Arc::new(Mutex::new(dict)),
        Arc::new(Mutex::new(db)),
        Arc::new(lune_dict::corrector::HeuristicCorrector::new()),
        config,
        settings,
    );
    let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(256);

    let start = Instant::now();

    tokio::task::LocalSet::new()
        .run_until(async {
            let _id = mgr
                .start_with_audio(event_tx, audio_stream)
                .await
                .expect("start session");
            let mut final_text = String::new();
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);

            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(SessionEvent::FinalizedText(t))) => final_text.push_str(&t),
                    Ok(Some(SessionEvent::Stopped)) => break,
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("[warn] 120s safety timeout");
                        break;
                    }
                }
            }

            let wall = start.elapsed().as_secs_f64();
            if final_text.is_empty() {
                None
            } else {
                Some((final_text, wall))
            }
        })
        .await
}
