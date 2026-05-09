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
use lune_audio::{PcenAgc, PcenAgcConfig};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_llm::NullPostProcessor;
use tokio::sync::mpsc;

/// PCEN hop size (must match `crates/lune-audio/src/vad.rs` HOP_SIZE).
const PCEN_HOP_SAMPLES: usize = 256;
const PCEN_SAMPLE_RATE: u32 = 16_000;

/// Per-fixture acoustic summary computed offline from the raw PCM.
#[derive(Debug, Clone, Copy, Default)]
struct Acoustics {
    snr_p10_db: f32,
    snr_median_db: f32,
    snr_p90_db: f32,
    gain_p90: f32,
}

/// True if the ASR's detected-language label corresponds to the BCP-47 tag
/// configured for this fixture. Qwen3-ASR returns full English names
/// ("English", "Dutch", "Portuguese", …) while the bench harness uses
/// BCP-47 codes ("en", "nl", "pt", …); compare case-insensitively after
/// folding both sides to a canonical form. An empty detected label means
/// "no language committed" and counts as a non-match — that's the failure
/// mode we want surfaced.
fn lang_label_matches(detected: &str, configured_bcp47: &str) -> bool {
    if detected.trim().is_empty() {
        return false;
    }
    let det = detected.trim().to_ascii_lowercase();
    let cfg = configured_bcp47.trim().to_ascii_lowercase();
    if det == cfg {
        return true;
    }
    matches!(
        (cfg.as_str(), det.as_str()),
        ("en", "english")
            | ("fr", "french")
            | ("de", "german")
            | ("es", "spanish")
            | ("it", "italian")
            | ("nl", "dutch")
            | ("pt", "portuguese")
    )
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f32 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Run PCEN AGC offline over the fixture PCM and return SNR / gain percentiles.
///
/// Mirrors the runtime VAD frontend's PCEN settings so the SNR distribution
/// is comparable to what the live pipeline would observe.
fn compute_acoustics(pcm: &[f32]) -> Acoustics {
    if pcm.len() < PCEN_HOP_SAMPLES {
        return Acoustics::default();
    }
    let mut agc = match PcenAgc::new(PcenAgcConfig::default(), PCEN_SAMPLE_RATE, PCEN_HOP_SAMPLES) {
        Ok(a) => a,
        Err(_) => return Acoustics::default(),
    };
    let mut work = vec![0.0f32; PCEN_HOP_SAMPLES];
    let n_frames = pcm.len() / PCEN_HOP_SAMPLES;
    let mut snrs = Vec::with_capacity(n_frames);
    let mut gains = Vec::with_capacity(n_frames);
    for chunk in pcm.chunks_exact(PCEN_HOP_SAMPLES) {
        work.copy_from_slice(chunk);
        let stats = agc.process_frame(&mut work);
        if stats.snr_db.is_finite() {
            snrs.push(stats.snr_db);
        }
        if stats.gain_applied.is_finite() {
            gains.push(stats.gain_applied);
        }
    }
    snrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    gains.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Acoustics {
        snr_p10_db: percentile(&snrs, 0.10),
        snr_median_db: percentile(&snrs, 0.50),
        snr_p90_db: percentile(&snrs, 0.90),
        gain_p90: percentile(&gains, 0.90),
    }
}

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
        let acoustics = compute_acoustics(&fixture.pcm);
        let configured_lang = BenchmarkHarness::language_for_fixture(fixture.name);

        let (text_no_llm, time_no_llm, gate_no_llm) = harness
            .rt
            .block_on(run_pipeline(
                fixture,
                &harness,
                Arc::clone(&null_llm),
                pipeline.clone(),
            ))
            .unwrap_or_default();

        let (text_with_llm, time_with_llm, _gate_with_llm) = harness
            .rt
            .block_on(run_pipeline(
                fixture,
                &harness,
                Arc::clone(&harness.llm),
                pipeline,
            ))
            .unwrap_or_default();

        let detected_lang = gate_no_llm.lang.clone();
        let unresolved_count = gate_no_llm.unresolved_count;
        let lang_match = lang_label_matches(&detected_lang, configured_lang);
        let gate_invoked_llm = gate_no_llm.used_llm;

        let llm_delta = time_with_llm - time_no_llm;
        let bypassed = text_no_llm == text_with_llm;

        let wer_raw = word_error_rate(&text_no_llm, &fixture.reference) * 100.0;
        let cer_raw = char_error_rate(&text_no_llm, &fixture.reference) * 100.0;

        let acoustics_line = format!(
            "SNR p10={:>5.1} med={:>5.1} p90={:>5.1} dB · gain p90={:>5.2}× · lang cfg={} det={} {} · unresolved={}",
            acoustics.snr_p10_db,
            acoustics.snr_median_db,
            acoustics.snr_p90_db,
            acoustics.gain_p90,
            configured_lang,
            if detected_lang.is_empty() {
                "—"
            } else {
                detected_lang.as_str()
            },
            if lang_match { "ok" } else { "MISMATCH" },
            unresolved_count,
        );

        if bypassed {
            println!();
            println!("  [SAME] LLM made no changes (Δ {:.1}s)", llm_delta);
            println!("  {:>20}  {:>10.1}%", "WER", wer_raw);
            println!("  {:>20}  {:>10.1}%", "CER", cer_raw);
            println!("  acoustics: {}", acoustics_line);
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
                acoustics,
                detected_lang: detected_lang.clone(),
                configured_lang,
                lang_match,
                unresolved_count,
                gate_invoked_llm,
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
            println!("  acoustics: {}", acoustics_line);
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
                acoustics,
                detected_lang: detected_lang.clone(),
                configured_lang,
                lang_match,
                unresolved_count,
                gate_invoked_llm,
            });
        }
    }

    if results.len() > 1 {
        println!();
        println!("═══ Summary ═══");
        println!();
        println!(
            "  {:20}  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}  {:>7}  {:>7}  {:>5}  {:>5}  {:>4}",
            "Fixture",
            "gate",
            "WER raw",
            "WER+LLM",
            "CER raw",
            "CER+LLM",
            "LLM Δ(s)",
            "SNR p10",
            "SNR med",
            "g.p90",
            "lang",
            "unr",
        );
        println!(
            "  {:20}  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}  {:>7}  {:>7}  {:>5}  {:>5}  {:>4}",
            "───────",
            "────",
            "───────",
            "───────",
            "───────",
            "───────",
            "────────",
            "───────",
            "───────",
            "─────",
            "─────",
            "────",
        );
        for r in &results {
            let lang_tag = if r.lang_match {
                format!("{}=", r.configured_lang)
            } else {
                format!(
                    "{}≠{}",
                    r.configured_lang,
                    if r.detected_lang.is_empty() {
                        "—"
                    } else {
                        r.detected_lang.as_str()
                    }
                )
            };
            let gate_tag = if r.gate_invoked_llm { "LLM" } else { "skip" };
            if r.bypassed {
                println!(
                    "  {:20}  {:>4}  {:>7.1}%  {:>8}  {:>7.1}%  {:>8}  {:>9}  {:>7.1}  {:>7.1}  {:>5.2}  {:>5}  {:>4}",
                    r.name,
                    gate_tag,
                    r.wer_raw,
                    "BYPASS",
                    r.cer_raw,
                    "BYPASS",
                    "—",
                    r.acoustics.snr_p10_db,
                    r.acoustics.snr_median_db,
                    r.acoustics.gain_p90,
                    lang_tag,
                    r.unresolved_count,
                );
            } else {
                println!(
                    "  {:20}  {:>4}  {:>7.1}%  {:>7.1}%  {:>7.1}%  {:>7.1}%  {:>9.3}  {:>7.1}  {:>7.1}  {:>5.2}  {:>5}  {:>4}",
                    r.name,
                    gate_tag,
                    r.wer_raw,
                    r.wer_llm,
                    r.cer_raw,
                    r.cer_llm,
                    r.llm_delta,
                    r.acoustics.snr_p10_db,
                    r.acoustics.snr_median_db,
                    r.acoustics.gain_p90,
                    lang_tag,
                    r.unresolved_count,
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
    acoustics: Acoustics,
    detected_lang: String,
    configured_lang: &'static str,
    lang_match: bool,
    unresolved_count: usize,
    /// True if the session's gate decided to run the LLM (independent of
    /// whether the LLM actually changed the text).
    gate_invoked_llm: bool,
}

/// Gate-decision snapshot captured from the session's event stream.
#[derive(Debug, Clone, Default)]
struct GateSnapshot {
    lang: String,
    unresolved_count: usize,
    #[allow(dead_code)]
    used_llm: bool,
}

async fn run_pipeline(
    fixture: &Fixture,
    harness: &BenchmarkHarness,
    llm: Arc<dyn TextPostProcessor>,
    pipeline: lune_core::config::PipelineConfig,
) -> Option<(String, f64, GateSnapshot)> {
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
            let mut gate = GateSnapshot::default();
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);

            loop {
                match tokio::time::timeout_at(deadline, event_rx.recv()).await {
                    Ok(Some(SessionEvent::FinalizedText(t))) => final_text.push_str(&t),
                    Ok(Some(SessionEvent::GateDecision {
                        used_llm,
                        lang,
                        unresolved,
                    })) => {
                        gate = GateSnapshot {
                            lang,
                            unresolved_count: unresolved.len(),
                            used_llm,
                        };
                    }
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
                Some((final_text, wall, gate))
            }
        })
        .await
}
