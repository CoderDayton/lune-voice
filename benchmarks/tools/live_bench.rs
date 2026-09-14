//! Live transcription WER benchmark.
//!
//! Prompts you to read each reference line aloud, records it from the mic, runs
//! the real VAD + ASR + LLM pipeline on what you said, and reports per-line and
//! overall WER/CER against the reference. A real-voice accuracy harness — unlike
//! the fixture benchmarks, it measures your mic, your room, and your speech.
//!
//! Run (GPU): cargo run --release --features whisper-cuda --example live_bench -p lune-voice
//!   Quieter prompts: append `2>/dev/null` to hide model-load logs.
//!   Custom references: LUNE_REFS=path/to/lines.txt (one reference per line).

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex as StdMutex};

use bench_utils::{BenchmarkHarness, char_error_rate, word_error_rate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, StreamConfig};
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::config::UserSettings;
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_voice::SessionManager;
use tokio::sync::{Mutex, RwLock, mpsc};

/// Default reference set, spanning one word to a long multi-clause sentence so
/// short-utterance and long-form accuracy both get exercised. Written in the
/// expected *output* form (capitalized, punctuated) since the full pipeline
/// produces that, so WER reflects the real end-to-end result. Override with a
/// `LUNE_REFS` file.
const DEFAULT_REFS: &[&str] = &[
    // Coordinating-conjunction clause boundaries — the surface the punctuation
    // fix touches. A pause the ASR over-reads as a sentence break turns one of
    // these commas into a period (". But", ". So"), so WER here tracks exactly
    // that. No digits, so the score reflects punctuation, not number formatting.
    "I finished the report, so send it over whenever you're ready and I'll review it after lunch.",
    "The deployment went out last night and everything looks stable so far, but I want to keep an eye on the error rates for the next little while, just in case something slips through around the payment flow.",
    "Honestly, I'm not sure which approach is better, because the first one is simpler to build but harder to scale, while the second takes more work up front and pays off later, so we should weigh both before deciding.",
    // A GENUINE multi-sentence passage: the breaks here are real, so this checks
    // the pipeline still splits correctly without adding extra periods.
    "First, clone the repository and install the dependencies. Then open the config file and set your key. Finally, run the start script and the server should come up.",
];

fn references() -> Vec<String> {
    if let Ok(path) = std::env::var("LUNE_REFS") {
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let lines: Vec<String> = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(str::to_owned)
                    .collect();
                if !lines.is_empty() {
                    return lines;
                }
                eprintln!("  [warn] {path} had no non-empty lines; using defaults");
            }
            Err(e) => eprintln!("  [warn] couldn't read {path}: {e}; using defaults"),
        }
    }
    DEFAULT_REFS.iter().map(|s| s.to_string()).collect()
}

/// Continuously capture the default mic at 16 kHz mono f32 into `pcm`. The
/// stream runs until dropped; we clear `pcm` at the start of each utterance and
/// snapshot it at the end. (Mirrors `live_transcribe`'s capture.)
fn start_recording(pcm: Arc<StdMutex<Vec<f32>>>) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("no input device found")?;
    let config = StreamConfig {
        channels: 1,
        sample_rate: 16_000,
        buffer_size: BufferSize::Default,
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

fn main() {
    println!("\n  ══ Live transcription WER benchmark ══\n");
    let refs = references();

    let harness = BenchmarkHarness::builder()
        .with_tracing()
        .with_llm()
        .skip_fixtures()
        .build();
    println!(
        "  backend: {:?}  ·  {} reference line(s)\n",
        harness.asr.kind(),
        refs.len()
    );

    let pcm = Arc::new(StdMutex::new(Vec::<f32>::new()));
    let _stream = match start_recording(Arc::clone(&pcm)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  [error] microphone unavailable: {e}");
            return;
        }
    };

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let pipeline = harness.asr.pipeline_config();

    let mut rows: Vec<(String, String, f64, f64)> = Vec::new();

    for (i, reference) in refs.iter().enumerate() {
        println!("  [{}/{}] Read this aloud:", i + 1, refs.len());
        println!("      \"{reference}\"");
        print!("      ▶ press Enter to START… ");
        io::stdout().flush().ok();
        if lines.next().is_none() {
            break;
        }
        pcm.lock().unwrap().clear();
        print!("      🔴 recording — press Enter when DONE… ");
        io::stdout().flush().ok();
        if lines.next().is_none() {
            break;
        }
        let samples = pcm.lock().unwrap().clone();
        let secs = samples.len() as f64 / 16_000.0;
        if samples.is_empty() {
            println!("      [skip] no audio captured\n");
            continue;
        }

        let hyp = harness
            .rt
            .block_on(run_full_pipeline(
                &samples,
                Arc::clone(&harness.asr),
                Arc::clone(&harness.llm),
                pipeline.clone(),
            ))
            .unwrap_or_default();

        let wer = word_error_rate(&hyp, reference) * 100.0;
        let cer = char_error_rate(&hyp, reference) * 100.0;
        println!("      hyp ({secs:.1}s): \"{hyp}\"");
        println!("      WER {wer:.1}%   CER {cer:.1}%\n");
        rows.push((reference.clone(), hyp, wer, cer));
    }

    println!("\n  ══ Results ══");
    for (i, (r, h, wer, cer)) in rows.iter().enumerate() {
        println!("  [{:>2}]  WER {:>5.1}%   CER {:>5.1}%", i + 1, wer, cer);
        println!("        ref: {r}");
        println!("        hyp: {h}");
    }
    if rows.is_empty() {
        println!("\n  No lines scored.");
    } else {
        let n = rows.len() as f64;
        let wer = rows.iter().map(|r| r.2).sum::<f64>() / n;
        let cer = rows.iter().map(|r| r.3).sum::<f64>() / n;
        println!(
            "\n  Overall:  WER {wer:.1}%   CER {cer:.1}%   ({} line(s))",
            rows.len()
        );
    }
}

/// Full SessionManager path (VAD + ASR + LLM) on a recorded buffer, mirroring
/// `pipeline_real` / `asr_probe`.
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
