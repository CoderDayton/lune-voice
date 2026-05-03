//! Real pipeline benchmark — exercises the real SessionManager path with VAD, ASR, LLM.
//!
//! Run: cargo run --release --example pipeline_real -p lune-app
//! GPU: cargo run --release --features cuda --example pipeline_real -p lune-app
//! Force backend: cargo run --release --example pipeline_real -p lune-app -- --asr qwen3
//!
//! Unlike `pipeline_stages` (which calls stages individually), this feeds
//! audio through `SessionManager::start_with_audio()` and measures wall-time
//! from start to `FinalizedText` event — the same path as real dictation.

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, Fixture, char_error_rate, process_memory_usage, word_error_rate, wrap_print,
};
use lune_app::SessionManager;
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::config::UserSettings;
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use tokio::sync::{Mutex, RwLock, mpsc};

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          REAL PIPELINE BENCHMARK (SessionManager + VAD)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .with_llm()
        .build();

    println!("  Selected ASR backend: {:?}", harness.asr.kind());
    if let Some(mem) = process_memory_usage() {
        println!("  Memory after load: {}", mem.display());
    }

    if harness.fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found");
        return;
    }

    for fixture in &harness.fixtures {
        println!();
        println!("── {} ({:.1}s audio) ──", fixture.name, fixture.audio_secs);
        println!();

        // Optional marking/skip threshold overrides for ablation runs.
        // ITN is disabled by default in benchmarks because reference texts
        // use spelled-out numbers. Set LUNE_ITN=1 to enable.
        let mut pipeline = harness.asr.pipeline_config();
        pipeline.itn_enabled = std::env::var("LUNE_ITN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if let Ok(v) = std::env::var("LUNE_MARK_THRESHOLD")
            && let Ok(f) = v.parse::<f32>()
        {
            pipeline.word_confidence_threshold = Some(f);
        }
        if let Ok(v) = std::env::var("LUNE_SKIP_THRESHOLD") {
            if v == "off" {
                pipeline.llm_skip_threshold = None;
            } else if let Ok(f) = v.parse::<f32>() {
                pipeline.llm_skip_threshold = Some(f);
            }
        }

        let result = harness.rt.block_on(run_full_pipeline(
            fixture,
            Arc::clone(&harness.asr),
            Arc::clone(&harness.llm),
            pipeline,
        ));

        match result {
            Some((text, wall_time)) => {
                let w = word_error_rate(&text, &fixture.reference) * 100.0;
                let c = char_error_rate(&text, &fixture.reference) * 100.0;
                let rtf = wall_time / fixture.audio_secs;

                println!(
                    "  Wall time:   {:.3}s (VAD + ASR + LLM pipeline)",
                    wall_time
                );
                println!("  Audio:       {:.1}s", fixture.audio_secs);
                println!(
                    "  RTF:         {:.3}x ({:.1}x faster than real-time)",
                    rtf,
                    1.0 / rtf
                );
                println!("  WER:         {:.1}%", w);
                println!("  CER:         {:.1}%", c);
                if let Some(mem) = process_memory_usage() {
                    println!("  Memory:      {}", mem.display());
                }
                println!();

                println!("  Hypothesis:");
                wrap_print("    ", &text, 72);
                println!();
                println!("  Reference:");
                wrap_print("    ", &fixture.reference, 72);
            }
            None => {
                println!("  [error] No text produced");
            }
        }
    }

    println!();
}

async fn run_full_pipeline(
    fixture: &Fixture,
    asr: Arc<AsrRegistry>,
    llm: Arc<dyn TextPostProcessor>,
    pipeline: lune_core::config::PipelineConfig,
) -> Option<(String, f64)> {
    let audio_stream = AudioStream::from_samples(&fixture.pcm, AudioConfig::default()).unwrap();

    let db = lune_persist::Database::open_in_memory().unwrap();
    let dict = lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
        .unwrap();

    let config = lune_app::SessionManagerConfig {
        asr_backend_kind: asr.kind(),
        pipeline,
        ..Default::default()
    };

    let user_settings = UserSettings {
        language: BenchmarkHarness::language_for_fixture(fixture.name).into(),
        ..Default::default()
    };
    let settings = Arc::new(RwLock::new(user_settings));

    let mgr = SessionManager::new(
        asr,
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

    // Session detects offline audio exhaustion and proceeds to Phase B
    // automatically — no timeout needed. Just collect events until Stopped.
    tokio::task::LocalSet::new()
        .run_until(async {
            let _id = mgr
                .start_with_audio(event_tx, audio_stream)
                .await
                .expect("start session");

            let mut final_text = String::new();

            // Safety timeout: 120s max for the entire pipeline.
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);

            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(SessionEvent::FinalizedText(t))) => {
                        final_text.push_str(&t);
                    }
                    Ok(Some(SessionEvent::Stopped)) => break,
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("  [warn] 120s safety timeout reached");
                        break;
                    }
                }
            }

            let wall_time = start.elapsed().as_secs_f64();

            if final_text.is_empty() {
                None
            } else {
                Some((final_text, wall_time))
            }
        })
        .await
}
