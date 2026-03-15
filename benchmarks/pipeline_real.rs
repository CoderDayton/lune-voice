//! Real pipeline benchmark — exercises the real SessionManager path with VAD, ASR, LLM.
//!
//! Run: cargo run --release --example pipeline_real -p lune-app
//! GPU: cargo run --release --features cuda --example pipeline_real -p lune-app
//! Force backend: cargo run --release --example pipeline_real -p lune-app -- --asr qwen
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
    Fixture, benchmark_aligner, benchmark_asr_entry, char_error_rate, init_ort, load_fixtures,
    process_memory_usage, word_error_rate, workspace_root, wrap_print,
};
use lune_app::{SessionManager, SessionManagerConfig};
use lune_asr::{AsrRegistry, OrtForcedAligner};
use lune_audio::{AudioConfig, AudioStream};
use lune_core::asr::AsrBackendKind;
use lune_core::config::{DevicePolicy, UserSettings};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_llm::{LlamaCppBackend, LlamaCppConfig};
use tokio::sync::{Mutex, RwLock, mpsc};

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          REAL PIPELINE BENCHMARK (SessionManager + VAD)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();
    let root = workspace_root();

    // Load ASR (Parakeet preferred unless ASR/--asr forces Qwen)
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
    let asr = AsrRegistry::from_config(&asr_entry.config, device).expect("load ASR");
    let aligner = benchmark_aligner(&asr, &root);
    let asr = Arc::new(asr);
    println!("{:.1}s", t.elapsed().as_secs_f64());
    if let Some(mem) = process_memory_usage() {
        println!("  Memory after ASR load: {}", mem.display());
    }

    // Load LLM
    let llm_model_path = root.join("models/llm-corrector-gguf/qwen2.5-1.5b-instruct.Q4_K_M.gguf");
    if !llm_model_path.exists() {
        println!("  [skip] LLM model not found: {}", llm_model_path.display());
        return;
    }
    print!("  Loading LLM backend ... ");
    let t = Instant::now();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let llm: Arc<dyn TextPostProcessor> = match rt.block_on(LlamaCppBackend::new(LlamaCppConfig {
        model_path: llm_model_path,
        n_gpu_layers: if has_gpu { 999 } else { 0 },
        ..Default::default()
    })) {
        Ok(b) => {
            println!("{:.1}s", t.elapsed().as_secs_f64());
            if let Some(mem) = process_memory_usage() {
                println!("  Memory after LLM load: {}", mem.display());
            }
            Arc::new(b)
        }
        Err(e) => {
            println!("  [error] LLM load failed: {e}");
            return;
        }
    };

    // Load fixtures
    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found");
        return;
    }

    // Run full pipeline for each fixture
    let rt = tokio::runtime::Runtime::new().unwrap();

    for fixture in &fixtures {
        println!();
        println!("── {} ({:.1}s audio) ──", fixture.name, fixture.audio_secs);
        println!();

        let result = rt.block_on(run_full_pipeline(
            fixture,
            Arc::clone(&asr),
            aligner.clone(),
            Arc::clone(&llm),
            asr_entry.backend_kind,
            asr.pipeline_config(),
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
    aligner: Option<Arc<OrtForcedAligner>>,
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

    let settings = Arc::new(RwLock::new(UserSettings::default()));

    let mgr = SessionManager::new(
        asr,
        Arc::new(lune_commands::VoiceCommandRouter::new()),
        llm,
        Arc::new(Mutex::new(dict)),
        Arc::new(Mutex::new(db)),
        config,
        settings,
    );
    let mgr = if let Some(aligner) = aligner {
        mgr.with_aligner(aligner)
    } else {
        mgr
    };

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
