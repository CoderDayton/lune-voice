//! End-to-end session latency benchmark — timestamps every SessionEvent to
//! show per-phase timing, especially the gap between FinalizedText and Stopped.
//!
//! Run: cargo run --release --example session_latency -p lune-voice

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::time::Instant;

use bench_utils::{BenchmarkHarness, process_memory_usage, wrap_print};
use lune_audio::{AudioConfig, AudioStream};
use lune_core::session::SessionEvent;
use tokio::sync::mpsc;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           SESSION LATENCY — per-event timing breakdown         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let harness = BenchmarkHarness::builder().with_tracing().build();

    if harness.fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found");
        return;
    }

    for fixture in &harness.fixtures {
        println!();
        println!("── {} ({:.1}s audio) ──", fixture.name, fixture.audio_secs);
        println!();

        let audio_stream = AudioStream::from_samples(&fixture.pcm, AudioConfig::default()).unwrap();

        let mgr = harness.session_manager(harness.asr.pipeline_config());

        let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(256);

        let (events, final_text) = harness.rt.block_on(async {
            let t0 = Instant::now();

            let _id = mgr
                .start(
                    event_tx,
                    lune_voice::SessionAudio::Owned(audio_stream),
                    lune_voice::SessionKind::Dictate,
                )
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
                            SessionEvent::FinalizedText { text: t, .. } => {
                                final_text.push_str(t);
                                "FinalizedText"
                            }
                            SessionEvent::CorrectedText(_) => "CorrectedText",
                            SessionEvent::AudioLevel(_) => continue,
                            SessionEvent::CommandDetected { .. } => "CommandDetected",
                            SessionEvent::Error(_) => "Error",
                            SessionEvent::Stopped { .. } => {
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
