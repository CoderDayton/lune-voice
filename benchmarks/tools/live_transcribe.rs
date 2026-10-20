// SPDX-License-Identifier: GPL-3.0-or-later

//! Live microphone transcription with delivery.
//!
//! Records from the default mic, runs the full ASR + LLM pipeline, and
//! delivers finalized text into the focused window via clipboard paste.
//! Also saves a .wav file (16kHz mono f32) and a .txt transcript.
//!
//! Run: cargo run --release --example live_transcribe -p lune-voice
//! GPU: cargo run --release --features whisper-cuda --example live_transcribe -p lune-voice
//! No delivery: DELIVER=0 cargo run --release --example live_transcribe -p lune-voice
//!
//! Press Ctrl+C to stop. The pipeline processes any remaining audio before
//! writing output files.

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use bench_utils::{benchmark_asr_entry, detect_gpu, workspace_root};
use cpal::StreamConfig;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use lune_asr::AsrRegistry;
use lune_core::config::{DevicePolicy, UserSettings};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_llm::NullPostProcessor;
use lune_voice::{DeliveryService, SessionManager, SessionManagerConfig};
use tokio::sync::{Mutex, RwLock, mpsc};

fn main() {
    eprintln!();
    eprintln!("  Live Transcription + Delivery");
    eprintln!("  Speak into your mic. Press Ctrl+C to stop.");
    eprintln!();

    let has_gpu = detect_gpu();
    let root = workspace_root();

    // ── Load ASR ─────────────────────────────────────────────────────────

    let device_policy = if has_gpu {
        DevicePolicy::Cuda
    } else {
        DevicePolicy::Cpu
    };

    let asr_entry = match benchmark_asr_entry() {
        Ok(entry) => entry,
        Err(msg) => {
            eprintln!("  [skip] {msg}");
            return;
        }
    };

    eprint!("  Loading ASR ... ");
    let t = Instant::now();
    let models_dir = bench_utils::models_dir();
    let cuda_available = lune_persist::AppPaths::resolve()
        .map(|p| lune_asr::gpu_backend::gpu_available(p.data_dir()))
        .unwrap_or(false);
    let asr = Arc::new(
        AsrRegistry::from_config(
            &asr_entry.config,
            device_policy,
            &models_dir,
            cuda_available,
        )
        .expect("load ASR"),
    );
    let backend_kind = asr.kind();
    eprintln!("{:.1}s ({:?})", t.elapsed().as_secs_f64(), backend_kind);

    // ── Post-processing slot ────────────────────────────────────────────
    // Passthrough unless a correction backend is wired in.
    let llm: Arc<dyn TextPostProcessor> = Arc::new(NullPostProcessor::new());
    let _ = has_gpu; // consumed by ASR loading above

    // ── Start mic recording (separate cpal stream for WAV output) ────────

    let pcm_buffer: Arc<StdMutex<Vec<f32>>> = Arc::new(StdMutex::new(Vec::with_capacity(
        16_000 * 120, // pre-alloc 120s
    )));

    let recording_stream = match start_recording(Arc::clone(&pcm_buffer)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  [error] Failed to start recording: {e}");
            eprintln!("  (PipeWire or PulseAudio required for concurrent mic access)");
            return;
        }
    };

    eprintln!();
    eprintln!("  ─── Recording + Transcribing ───");
    eprintln!();

    // ── Delivery ──────────────────────────────────────────────────────────

    let deliver = std::env::var("DELIVER").as_deref() != Ok("0");
    let delivery = if deliver {
        match DeliveryService::new(lune_core::config::TerminalMode::Auto) {
            Ok(svc) => {
                eprintln!("  Delivery: clipboard paste into focused window");
                Some(svc)
            }
            Err(e) => {
                eprintln!("  Delivery: failed to start ({e}), disabled");
                None
            }
        }
    } else {
        eprintln!("  Delivery: disabled (DELIVER=0)");
        None
    };

    // ── Run pipeline ─────────────────────────────────────────────────────

    let rt = tokio::runtime::Runtime::new().unwrap();
    let local = tokio::task::LocalSet::new();

    let (transcript, elapsed) = local.block_on(&rt, async {
        let db = lune_persist::Database::open_in_memory().unwrap();
        let dict = lune_dict::UserDictionary::default();

        let pipeline = asr.pipeline_config();
        let config = SessionManagerConfig {
            asr_backend_kind: backend_kind,
            pipeline,
            ..Default::default()
        };

        let settings = Arc::new(RwLock::new(UserSettings::default()));

        let mgr = SessionManager::new(
            asr,
            Arc::new(lune_dict::commands::VoiceCommandRouter::new()),
            llm,
            Arc::new(dict),
            Arc::new(Mutex::new(db)),
            Arc::new(lune_dict::corrector::LexicalGate::new()),
            config,
            settings,
        );

        let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(256);
        let start = Instant::now();

        mgr.start(
            event_tx,
            lune_voice::SessionAudio::Open,
            lune_voice::SessionKind::Dictate,
        )
        .await
        .expect("start session");

        let mut transcript = String::new();
        let mut stopping = false;

        loop {
            tokio::select! {
                biased;
                _ = tokio::signal::ctrl_c(), if !stopping => {
                    eprintln!("\n\n  Stopping (processing remaining audio)...");
                    mgr.stop().await;
                    stopping = true;
                }
                event = event_rx.recv() => {
                    match event {
                        Some(SessionEvent::FinalizedText { text: t, .. }) => {
                            eprint!("\r\x1b[2K  {t}\n");
                            if let Some(svc) = &delivery {
                                svc.deliver(
                                    None,
                                    &t,
                                    lune_core::config::DictationMode::TypeInPlace,
                                    false,
                                );
                            }
                            if !transcript.is_empty() {
                                transcript.push(' ');
                            }
                            transcript.push_str(&t);
                        }
                        Some(SessionEvent::Processing) => {
                            eprint!("\r\x1b[2K  \x1b[33m[processing...]\x1b[0m");
                        }
                        Some(SessionEvent::Stopped { .. }) | None => break,
                        _ => {}
                    }
                }
            }
        }

        (transcript, start.elapsed())
    });

    // ── Stop recording + save outputs ────────────────────────────────────

    drop(recording_stream);

    let pcm = pcm_buffer.lock().unwrap();
    let audio_secs = pcm.len() as f64 / 16_000.0;

    let wav_path = root.join("fixtures/audio/live_recording.wav");
    let txt_path = root.join("fixtures/audio/live_recording.txt");

    save_wav_f32(&pcm, &wav_path);
    std::fs::write(&txt_path, &transcript).expect("write transcript");

    eprintln!();
    eprintln!("  ─── Done ───");
    eprintln!();
    eprintln!(
        "  WAV: {} ({:.1}s, {} samples)",
        wav_path.display(),
        audio_secs,
        pcm.len()
    );
    eprintln!("  TXT: {} ({} chars)", txt_path.display(), transcript.len());
    eprintln!("  Wall time: {:.1}s", elapsed.as_secs_f64());
    eprintln!();
}

// ─── Recording helpers ───────────────────────────────────────────────────────

/// Open a cpal input stream at 16kHz mono f32 for WAV recording.
/// PipeWire/PulseAudio handles server-side resampling transparently.
fn start_recording(pcm: Arc<StdMutex<Vec<f32>>>) -> Result<cpal::Stream, String> {
    let host = lune_audio::capture_host();
    let device = host.default_input_device().ok_or("no input device found")?;
    let supported = device
        .default_input_config()
        .map_err(|e| format!("query input config: {e}"))?;

    eprintln!(
        "  Mic: {}",
        device
            .description()
            .map(|d| d.name().to_owned())
            .unwrap_or_else(|_| "unknown".into())
    );

    let config = StreamConfig {
        channels: 1,
        sample_rate: 16_000,
        buffer_size: lune_audio::capture_buffer_size(&host, 16_000, supported.buffer_size()),
    };

    let stream = device
        .build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut buf) = pcm.lock() {
                    buf.extend_from_slice(data);
                }
            },
            |err| eprintln!("  [warn] recording: {err}"),
            None,
        )
        .map_err(|e| format!("build input stream at 16kHz: {e}"))?;

    stream.play().map_err(|e| format!("play: {e}"))?;
    Ok(stream)
}

/// Write f32 PCM samples as a 16kHz mono float32 WAV.
fn save_wav_f32(samples: &[f32], path: &std::path::Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create WAV writer");
    for &s in samples {
        writer.write_sample(s).expect("write sample");
    }
    writer.finalize().expect("finalize WAV");
}
