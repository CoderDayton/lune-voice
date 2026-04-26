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
    Fixture, benchmark_asr_entry, char_error_rate, init_ort, load_fixtures, process_memory_usage,
    word_error_rate, workspace_root, wrap_print,
};
use lune_app::{SessionManager, SessionManagerConfig};
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::asr::AsrBackendKind;
use lune_core::config::{DevicePolicy, UserSettings};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_llm::NullPostProcessor;
use tokio::sync::{Mutex, RwLock, mpsc};

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    // Honor RUST_LOG when set so sweep/debug runs can introspect the
    // correction path. No-op when RUST_LOG is unset.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          REAL PIPELINE BENCHMARK (SessionManager + VAD)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();
    let root = workspace_root();

    // Load ASR (Qwen preferred unless ASR/--asr forces Qwen3)
    let device = if has_gpu {
        DevicePolicy::Cuda
    } else {
        DevicePolicy::Cpu
    };
    let asr_entry = match benchmark_asr_entry(&root) {
        Ok(entry) => entry,
        Err(msg) => {
            println!("  [skip] {msg}");
            return;
        }
    };
    println!("  Selected ASR backend: {}", asr_entry.display_name);
    print!("  Loading ASR backend ... ");
    let t = Instant::now();
    let models_dir = root.join("models");
    let asr = AsrRegistry::from_config(&asr_entry.config, device, &models_dir).expect("load ASR");

    let asr = Arc::new(asr);
    println!("{:.1}s", t.elapsed().as_secs_f64());
    if let Some(mem) = process_memory_usage() {
        println!("  Memory after ASR load: {}", mem.display());
    }

    // The post-processor slot uses passthrough unless a correction backend is
    // wired into this benchmark.
    let llm: Arc<dyn TextPostProcessor> = Arc::new(NullPostProcessor::new());
    let _ = has_gpu; // still consumed by ASR loading above

    // Load fixtures
    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found");
        return;
    }

    // Run full pipeline for each fixture. Optional fixture filter via
    // `LUNE_BENCH_FIXTURE=<substring>` for sweep scripts that want a
    // single fixture.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture_filter = std::env::var("LUNE_BENCH_FIXTURE").ok();

    for fixture in &fixtures {
        if let Some(ref needle) = fixture_filter
            && !fixture.name.contains(needle.as_str())
        {
            continue;
        }
        println!();
        println!("── {} ({:.1}s audio) ──", fixture.name, fixture.audio_secs);
        println!();

        // Optional marking/skip threshold overrides for ablation runs.
        // ITN is disabled by default in benchmarks because reference texts
        // use spelled-out numbers. Set LUNE_ITN=1 to enable.
        let mut pipeline = asr.pipeline_config();
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

        let result = rt.block_on(run_full_pipeline(
            fixture,
            Arc::clone(&asr),
            Arc::clone(&llm),
            asr_entry.backend_kind,
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

fn language_from_fixture(name: &str) -> &str {
    if name.contains("french") {
        "fr"
    } else if name.contains("german") {
        "de"
    } else if name.contains("italian") {
        "it"
    } else if name.contains("spanish") {
        "es"
    } else {
        "en"
    }
}

async fn run_full_pipeline(
    fixture: &Fixture,
    asr: Arc<AsrRegistry>,

    llm: Arc<dyn TextPostProcessor>,
    backend_kind: AsrBackendKind,
    pipeline: lune_core::config::PipelineConfig,
) -> Option<(String, f64)> {
    let audio_stream = AudioStream::from_samples(&fixture.pcm, AudioConfig::default()).unwrap();

    let db = lune_persist::Database::open_in_memory().unwrap();
    let dict = lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
        .unwrap();

    let config = SessionManagerConfig {
        asr_backend_kind: backend_kind,
        pipeline,
        ..Default::default()
    };

    let mut user_settings = UserSettings::default();
    user_settings.language = language_from_fixture(&fixture.name).into();
    let settings = Arc::new(RwLock::new(user_settings));

    let mgr = SessionManager::new(
        asr,
        Arc::new(lune_commands::VoiceCommandRouter::new()),
        llm,
        Arc::new(Mutex::new(dict)),
        Arc::new(Mutex::new(db)),
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
