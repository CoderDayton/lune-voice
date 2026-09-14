//! Single-WAV / dump ASR probe — feed recordings through the real ASR (and,
//! for WAVs, the full VAD+ASR+LLM pipeline) and print the hypothesis.
//! Throwaway diagnostic for the quiet short-utterance investigation.
//!
//! Run (GPU): cargo run --release --features whisper-cuda --example asr_probe -p lune-voice -- <path>
//!   <path> may be a single .wav, a single .f32 (raw LE f32, 16 kHz mono — the
//!   `LUNE_ASR_DUMP_DIR` dump format), or a DIRECTORY (all .f32/.wav in it).
//!   .f32 dumps are already post-VAD, so they default to DIRECT (push whole
//!   buffer straight to ASR). .wav defaults to the full pipeline.
//!   LUNE_PREGAIN=4.0 scales the input. LUNE_DIRECT=1 forces direct for all.

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bench_utils::BenchmarkHarness;
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use lune_core::config::UserSettings;
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_voice::SessionManager;
use tokio::sync::{Mutex, RwLock, mpsc};

fn main() {
    let arg = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("LUNE_WAV").ok())
        .expect("usage: asr_probe <wav|f32|dir>");
    let pregain: f32 = std::env::var("LUNE_PREGAIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let force_direct = std::env::var("LUNE_DIRECT")
        .map(|v| v == "1")
        .unwrap_or(false);

    let mut paths: Vec<PathBuf> = Vec::new();
    let p = PathBuf::from(&arg);
    if p.is_dir() {
        for e in std::fs::read_dir(&p).expect("read dir").flatten() {
            let ep = e.path();
            match ep.extension().and_then(|s| s.to_str()) {
                Some("f32") | Some("wav") => paths.push(ep),
                _ => {}
            }
        }
        paths.sort();
    } else {
        paths.push(p);
    }

    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .with_llm()
        .skip_fixtures()
        .build();
    println!();
    println!(
        "  backend={:?}  pregain={pregain}x  files={}",
        harness.asr.kind(),
        paths.len()
    );
    println!();

    let pipeline = harness.asr.pipeline_config();

    for path in &paths {
        let is_f32 = path.extension().and_then(|s| s.to_str()) == Some("f32");
        let pcm = load_pcm(path, is_f32, pregain);
        let secs = pcm.len() as f64 / 16_000.0;
        let direct = force_direct || is_f32;

        let text = if direct {
            harness
                .rt
                .block_on(run_direct(&pcm, Arc::clone(&harness.asr)))
        } else {
            harness.rt.block_on(run_full_pipeline(
                &pcm,
                Arc::clone(&harness.asr),
                Arc::clone(&harness.llm),
                pipeline.clone(),
            ))
        };
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        let mode = if direct { "direct" } else { "full" };
        println!(
            "  {name:<32} {secs:>4.2}s {mode:<6} -> {:?}",
            text.unwrap_or_default()
        );
    }
    println!();
}

fn load_pcm(path: &Path, is_f32: bool, pregain: f32) -> Vec<f32> {
    if is_f32 {
        let bytes = std::fs::read(path).expect("read f32");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * pregain)
            .map(|x| x.clamp(-1.0, 1.0))
            .collect()
    } else {
        let mut reader = hound::WavReader::open(path).expect("open wav");
        let spec = reader.spec();
        // Accept both float and integer WAVs (any bit depth) rather than
        // panicking on a non-float file.
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.expect("sample"))
                .collect(),
            hound::SampleFormat::Int => {
                let full = (1u64 << spec.bits_per_sample.saturating_sub(1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.expect("sample") as f32 / full)
                    .collect()
            }
        };
        // Downmix to mono if needed (the model is 16 kHz mono).
        let mono: Vec<f32> = if spec.channels > 1 {
            let ch = spec.channels as usize;
            samples
                .chunks(ch)
                .map(|f| f.iter().sum::<f32>() / ch as f32)
                .collect()
        } else {
            samples
        };
        mono.into_iter()
            .map(|x| (x * pregain).clamp(-1.0, 1.0))
            .collect()
    }
}

/// Push the whole clip straight to the ASR backend, no VAD, no LLM.
async fn run_direct(pcm: &[f32], asr: Arc<AsrRegistry>) -> Option<String> {
    tokio::task::LocalSet::new()
        .run_until(async {
            let session = asr
                .start_session(AsrSessionConfig::default())
                .await
                .expect("start asr session");
            asr.push_audio(&session, pcm).await.expect("push audio");
            let segs = asr.poll_final(&session).await.expect("poll final");
            asr.end_session(session).await.ok();
            let text = segs
                .into_iter()
                .map(|s| s.text)
                .collect::<Vec<_>>()
                .join(" ");
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        })
        .await
}

/// Full SessionManager path (VAD + ASR + LLM), mirroring `pipeline_real`.
async fn run_full_pipeline(
    pcm: &[f32],
    asr: Arc<AsrRegistry>,
    llm: Arc<dyn TextPostProcessor>,
    pipeline: lune_core::config::PipelineConfig,
) -> Option<String> {
    let audio_stream = AudioStream::from_samples(pcm, AudioConfig::default()).unwrap();
    let db = lune_persist::Database::open_in_memory().unwrap();
    let dict = lune_dict::UserDictionary::default();

    let config = lune_voice::SessionManagerConfig {
        asr_backend_kind: asr.kind(),
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

    tokio::task::LocalSet::new()
        .run_until(async {
            let _id = mgr
                .start(
                    event_tx,
                    lune_voice::SessionAudio::Owned(audio_stream),
                    lune_voice::SessionKind::Dictate,
                )
                .await
                .expect("start session");
            let mut final_text = String::new();
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);
            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(SessionEvent::FinalizedText { text: t, .. })) => {
                        final_text.push_str(&t)
                    }
                    Ok(Some(SessionEvent::Stopped { .. })) => break,
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => break,
                }
            }
            if final_text.trim().is_empty() {
                None
            } else {
                Some(final_text)
            }
        })
        .await
}
