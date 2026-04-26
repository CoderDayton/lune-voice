//! End-to-end session latency benchmark — timestamps every SessionEvent to
//! show per-phase timing, especially the gap between FinalizedText and Stopped.
//!
//! Run: cargo run --release --example session_latency -p lune-app

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    benchmark_asr_entry, init_ort, load_fixtures, process_memory_usage, workspace_root, wrap_print,
};
use lune_app::{SessionManager, SessionManagerConfig};
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::config::{DevicePolicy, UserSettings};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_llm::NullPostProcessor;
use tokio::sync::{Mutex, RwLock, mpsc};

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           SESSION LATENCY — per-event timing breakdown         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();
    let root = workspace_root();

    let device = if has_gpu {
        DevicePolicy::Cuda
    } else {
        DevicePolicy::Cpu
    };
    let asr_entry = match benchmark_asr_entry() {
        Ok(entry) => entry,
        Err(msg) => {
            println!("  [skip] {msg}");
            return;
        }
    };
    println!("  ASR backend: {}", asr_entry.display_name);
    print!("  Loading ... ");
    let t = Instant::now();
    let models_dir = bench_utils::models_dir();
    let asr = Arc::new(
        AsrRegistry::from_config(&asr_entry.config, device, &models_dir).expect("load ASR"),
    );
    println!("{:.1}s", t.elapsed().as_secs_f64());

    let llm: Arc<dyn TextPostProcessor> = Arc::new(NullPostProcessor::new());

    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found");
        return;
    }

    let fixture_filter = std::env::var("LUNE_BENCH_FIXTURE").ok();
    let rt = tokio::runtime::Runtime::new().unwrap();

    for fixture in &fixtures {
        if let Some(ref needle) = fixture_filter
            && !fixture.name.contains(needle.as_str())
        {
            continue;
        }

        println!();
        println!("── {} ({:.1}s audio) ──", fixture.name, fixture.audio_secs);
        println!();

        let audio_stream = AudioStream::from_samples(&fixture.pcm, AudioConfig::default()).unwrap();
        let db = lune_persist::Database::open_in_memory().unwrap();
        let dict =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();

        let mut pipeline = asr.pipeline_config();
        pipeline.itn_enabled = false;

        let config = SessionManagerConfig {
            asr_backend_kind: asr_entry.backend_kind,
            pipeline,
            ..Default::default()
        };
        let settings = Arc::new(RwLock::new(UserSettings::default()));

        let mgr = SessionManager::new(
            Arc::clone(&asr),
            Arc::new(lune_commands::VoiceCommandRouter::new()),
            Arc::clone(&llm),
            Arc::new(Mutex::new(dict)),
            Arc::new(Mutex::new(db)),
            config,
            settings,
        );

        let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(256);

        let events = rt.block_on(async {
            let t0 = Instant::now();

            let _id = mgr
                .start_with_audio(event_tx, audio_stream)
                .await
                .expect("start session");

            let mut events: Vec<(&str, f64)> = Vec::new();
            let mut final_text = String::new();
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);

            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(ref ev)) => {
                        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let tag = match ev {
                            SessionEvent::Started => "Started",
                            SessionEvent::PartialText => "PartialText",
                            SessionEvent::Processing => "Processing",
                            SessionEvent::FinalizedText(t) => {
                                final_text.push_str(t);
                                "FinalizedText"
                            }
                            SessionEvent::CorrectedText(_) => "CorrectedText",
                            SessionEvent::AudioLevel(_) => continue,
                            SessionEvent::CommandDetected(_) => "CommandDetected",
                            SessionEvent::Error(_) => "Error",
                            SessionEvent::Stopped => {
                                events.push(("Stopped", elapsed_ms));
                                break;
                            }
                            _ => "Unknown",
                        };
                        events.push((tag, elapsed_ms));
                    }
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("  [warn] 120s timeout");
                        break;
                    }
                }
            }
            (events, final_text)
        });

        let (events, final_text) = events;

        println!("  {:.<30} {:>10}", "Event", "Time (ms)");
        println!("  {}", "─".repeat(42));

        let mut prev_ms = 0.0_f64;
        for (tag, abs_ms) in &events {
            let delta = abs_ms - prev_ms;
            println!("  {:.<30} {:>7.1}ms  (+{:.1}ms)", tag, abs_ms, delta);
            prev_ms = *abs_ms;
        }

        let finalized_ms = events
            .iter()
            .find(|(t, _)| *t == "FinalizedText")
            .map(|(_, ms)| *ms);
        let stopped_ms = events
            .iter()
            .find(|(t, _)| *t == "Stopped")
            .map(|(_, ms)| *ms);

        if let (Some(fin), Some(stop)) = (finalized_ms, stopped_ms) {
            println!();
            println!("  FinalizedText → Stopped gap: {:.1}ms", stop - fin);
        }

        if let Some(fin) = finalized_ms {
            let rtf = (fin / 1000.0) / fixture.audio_secs;
            println!(
                "  Time to text:  {:.1}ms  (RTF {:.3}x, {:.0}x faster)",
                fin,
                rtf,
                1.0 / rtf
            );
        }

        if let Some(mem) = process_memory_usage() {
            println!("  Memory:        {}", mem.display());
        }

        if !final_text.is_empty() {
            println!();
            println!("  Output:");
            wrap_print("    ", &final_text, 72);
        }
    }

    println!();
}
