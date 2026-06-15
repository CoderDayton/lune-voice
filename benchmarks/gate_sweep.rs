//! gate_sweep — calibrate the LLM-bypass gate.
//!
//! The domain benchmark showed the gate over-triggers: clean, already-correct
//! dictation is routed to the LLM (gate accuracy 55%, every miss a false-LLM).
//! This benchmark finds the right `llm_skip_threshold` empirically.
//!
//! It runs the REAL pipeline ONCE per categorized fixture (fixtures/bench +
//! manifest.json), and for each clip records BOTH:
//!   * the gate's decision inputs   — min_conf, avg_conf, all_resolved, needs_punct
//!   * the WER of BOTH arms         — det (deterministic bypass) and llm (LLM-always)
//!
//! Then it sweeps `llm_skip_threshold` OFFLINE (pure arithmetic on the cached
//! inputs — no re-running ASR/LLM) and, per threshold, reports the LLM-call
//! count, gate accuracy vs the manifest, the resulting routed WER, and the
//! "regret" (clips routed to the worse arm). The objective is not "match the
//! manifest" but "route each clip to whichever arm actually has lower WER with
//! the fewest wasted LLM calls."
//!
//! Run: cargo run --release --features cuda --example gate_sweep -p lune-app
//!      (LUNE_BENCH_PRINT=1 dumps per-fixture text.)
//!
//! NOTE: confidences come from direct full-clip ASR (the punctuation_ab
//! methodology), not the production VAD segmentation, so treat the recommended
//! threshold as a starting point to confirm on the live path.

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::path::Path;
use std::sync::Arc;

use bench_utils::{BenchmarkHarness, load_wav_f32, word_error_rate, workspace_root};
use lune_asr::AsrRegistry;
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::{PipelineConfig, StylePreset};
use lune_core::llm::PostProcessInput;
use lune_dict::corrector::{HWCE_FLAG_THRESHOLD, LexicalGate};
use lune_dict::punctuation::{decase_common_words, defuse_inconsistent, normalize};

/// One fixture's cached gate inputs + both-arm scores.
struct Probe {
    id: String,
    category: String,
    expect_bypass: Option<bool>,
    min_conf: f32,
    min_conf2: f32,
    avg_conf: f32,
    all_resolved: bool,
    needs_punct: bool,
    det_wer: f64,
    llm_wer: f64,
}

struct Entry {
    id: String,
    category: String,
    expect_bypass: Option<bool>,
}

fn parse_manifest(text: &str) -> Vec<Entry> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
        return Vec::new();
    };
    let Some(arr) = v.get("fixtures").and_then(|f| f.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|e| {
            Some(Entry {
                id: e.get("id")?.as_str()?.to_string(),
                category: e.get("category")?.as_str()?.to_string(),
                expect_bypass: e.get("expect_bypass").and_then(|v| v.as_bool()),
            })
        })
        .collect()
}

struct DomainFixture {
    entry: Entry,
    pcm: Vec<f32>,
    reference: String,
}

fn load_fixtures(root: &Path) -> Vec<DomainFixture> {
    let bench_dir = root.join("fixtures").join("bench");
    let Ok(manifest) = std::fs::read_to_string(bench_dir.join("manifest.json")) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in parse_manifest(&manifest) {
        let dir = bench_dir.join(&entry.category);
        let wav = dir.join(format!("{}.wav", entry.id));
        let txt = dir.join(format!("{}.txt", entry.id));
        if !wav.exists() || !txt.exists() {
            continue;
        }
        let pcm = load_wav_f32(&wav);
        let reference = std::fs::read_to_string(&txt)
            .expect("read reference")
            .trim()
            .to_string();
        out.push(DomainFixture {
            entry,
            pcm,
            reference,
        });
    }
    out
}

fn transcribe(asr: &AsrRegistry, rt: &tokio::runtime::Runtime, pcm: &[f32]) -> Vec<AsrSegment> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let cfg = AsrSessionConfig {
        language: Some("en".to_string()),
        ..Default::default()
    };
    let session = rt.block_on(asr.start_session(cfg)).expect("start session");
    rt.block_on(asr.push_audio(&session, pcm))
        .expect("push audio");
    let segments = rt.block_on(asr.poll_final(&session)).expect("poll final");
    let _ = rt.block_on(asr.end_session(session));
    segments
}

fn run_llm(
    harness: &BenchmarkHarness,
    preset: &Arc<StylePreset>,
    text: &str,
    lang: &str,
) -> String {
    let input = PostProcessInput {
        raw_text: text.to_owned(),
        style_preset: Arc::clone(preset),
        app_category: None,
        dictionary: vec![],
        language: lang.into(),
        correction_needed: true,
    };
    match harness.rt.block_on(harness.llm.transform(&input)) {
        Ok(o) => lune_dict::punctuation::capitalize_sentence_starts(&o.cleaned_text),
        Err(_) => text.to_owned(),
    }
}

fn main() {
    let harness = BenchmarkHarness::builder()
        .with_llm()
        .skip_fixtures()
        .build();
    let cfg = PipelineConfig::default();
    let corrector = LexicalGate::new();
    let preset = Arc::new(StylePreset::minimal()); // correction-only, no styling
    let print = std::env::var("LUNE_BENCH_PRINT").is_ok_and(|v| v != "0");

    let fixtures = load_fixtures(&workspace_root());
    if fixtures.is_empty() {
        println!("  [skip] no fixtures found in fixtures/bench/ (need manifest.json + wav/txt)");
        return;
    }

    println!();
    println!(
        "gate_sweep: {} fixtures, real ASR + LLM (minimal style)\n",
        fixtures.len()
    );

    let mut probes: Vec<Probe> = Vec::new();
    for fx in &fixtures {
        let segments = transcribe(&harness.asr, &harness.rt, &fx.pcm);
        if segments.is_empty() {
            continue;
        }
        let raw_text: String = segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let word_risk_scores: Vec<(&str, f32)> = segments
            .iter()
            .flat_map(|s| s.word_confidences.iter())
            .map(|wc| (wc.word.as_str(), wc.word_error_risk))
            .collect();
        let detected_lang = segments
            .first()
            .and_then(|s| s.detected_language.as_deref())
            .and_then(lune_core::lang::display_name)
            .unwrap_or("English");
        let pipeline_en = detected_lang == "English";

        // Shared corrector stage.
        let hc = corrector.correct_with_risk_threshold(
            &raw_text,
            &word_risk_scores,
            &[],
            detected_lang,
            HWCE_FLAG_THRESHOLD,
            true,
        );

        // Deterministic (bypass) arm — production order.
        let corrected = if pipeline_en {
            defuse_inconsistent(&hc.corrected_text)
        } else {
            hc.corrected_text.clone()
        };
        let punct = normalize(&corrected);
        let det_text = if pipeline_en {
            decase_common_words(&punct.text)
        } else {
            punct.text.clone()
        };

        // Gate inputs (byte-for-byte the production arithmetic).
        let mut risks: Vec<f32> = word_risk_scores.iter().map(|(_, r)| *r).collect();
        risks.sort_by(|a, b| b.total_cmp(a)); // descending risk
        let min_conf = 1.0 - risks.first().copied().unwrap_or(0.0);
        let min_conf2 = 1.0 - risks.get(1).copied().unwrap_or(0.0);
        let wc = raw_text.split_whitespace().count().max(1);
        let mean_risk = hc.word_risk_positions.iter().map(|(_, r)| *r).sum::<f32>() / wc as f32;
        let avg_conf = 1.0 - mean_risk;

        // LLM-always arm.
        let llm_text = run_llm(&harness, &preset, &hc.corrected_text, detected_lang);

        let det_wer = word_error_rate(&det_text, &fx.reference) * 100.0;
        let llm_wer = word_error_rate(&llm_text, &fx.reference) * 100.0;

        if print {
            println!("  [{}/{}]", fx.entry.category, fx.entry.id);
            println!("     det: {det_text:?}");
            println!("     llm: {llm_text:?}");
        }

        probes.push(Probe {
            id: fx.entry.id.clone(),
            category: fx.entry.category.clone(),
            expect_bypass: fx.entry.expect_bypass,
            min_conf,
            min_conf2,
            avg_conf,
            all_resolved: hc.all_resolved,
            needs_punct: punct.needs_llm,
            det_wer,
            llm_wer,
        });
    }

    print_per_fixture(&probes);
    sweep(&probes, &cfg);
}

/// Recompute the production gate decision at a candidate `llm_skip_threshold`.
/// Only `conf_attention` depends on the threshold; the rest is fixed per clip.
fn needs_llm(p: &Probe, conf: f32, threshold: f32, noisy: Option<f32>) -> bool {
    let too_broken = noisy.is_some_and(|n| p.avg_conf < n);
    let conf_attention = conf < threshold;
    let needs_correction = !p.all_resolved || p.needs_punct || conf_attention;
    needs_correction && !too_broken
}

/// The threshold-independent forced-LLM signal (unresolved word or run-on/
/// interrogative punctuation): these clips run the LLM at ANY threshold.
fn forced_llm(p: &Probe) -> bool {
    !p.all_resolved || p.needs_punct
}

fn print_per_fixture(probes: &[Probe]) {
    println!("── per-fixture gate inputs + both-arm WER ──");
    println!(
        "  {:22} {:>4} {:>6} {:>6} {:>6} {:>4} {:>5} {:>7} {:>7} {:>7}  trigger",
        "id", "exp", "minC", "minC2", "avgC", "res", "punct", "detWER", "llmWER", "Δllm"
    );
    for p in probes {
        let exp = match p.expect_bypass {
            Some(true) => "byp",
            Some(false) => "llm",
            None => "-",
        };
        // What forces this clip to the LLM at the DEFAULT 0.65 threshold?
        let trigger = if !p.all_resolved {
            "unresolved"
        } else if p.needs_punct {
            "needs_punct"
        } else if p.min_conf < 0.65 {
            "conf_attn"
        } else {
            "(bypass)"
        };
        println!(
            "  {:22} {:>4} {:>6.3} {:>6.3} {:>6.3} {:>4} {:>5} {:>6.1}% {:>6.1}% {:>+6.1}  {}",
            format!("{}/{}", p.category, p.id),
            exp,
            p.min_conf,
            p.min_conf2,
            p.avg_conf,
            if p.all_resolved { "Y" } else { "N" },
            if p.needs_punct { "Y" } else { "N" },
            p.det_wer,
            p.llm_wer,
            p.llm_wer - p.det_wer,
            trigger,
        );
    }
    println!();
}

fn sweep(probes: &[Probe], cfg: &PipelineConfig) {
    let n = probes.len() as f64;
    let forced = probes.iter().filter(|p| forced_llm(p)).count();
    println!(
        "── threshold sweep (noisy floor = {:?}; {}/{} clips forced-LLM regardless) ──",
        cfg.llm_skip_noisy_threshold,
        forced,
        probes.len()
    );
    // Oracle: route each clip to its better arm — the achievable floor.
    let oracle_calls = probes.iter().filter(|p| p.llm_wer < p.det_wer).count();
    let oracle_wer = probes.iter().map(|p| p.det_wer.min(p.llm_wer)).sum::<f64>() / n;
    println!(
        "  ORACLE (route to the better arm): {} LLM call(s), routed WER {:.2}%\n",
        oracle_calls, oracle_wer
    );

    sweep_design(
        probes,
        cfg,
        "A — conf_attn on the WEAKEST word (current design)",
        |p| p.min_conf,
    );
    sweep_design(
        probes,
        cfg,
        "B — conf_attn on the 2ND-weakest word (needs >=2 uncertain words)",
        |p| p.min_conf2,
    );
}

/// Sweep `llm_skip_threshold` for one confidence statistic (`conf`), reporting
/// per threshold: LLM-call count, gate accuracy vs the manifest, routed WER, and
/// regret (wasted = ran the LLM with no gain; missed = bypassed a helpful call).
fn sweep_design(probes: &[Probe], cfg: &PipelineConfig, label: &str, conf: impl Fn(&Probe) -> f32) {
    let noisy = cfg.llm_skip_noisy_threshold;
    let n = probes.len() as f64;
    const HELP_MARGIN: f64 = 0.5; // pp WER the LLM must beat the bypass arm by
    let helps = |p: &Probe| p.llm_wer + HELP_MARGIN < p.det_wer;

    println!("  {label}");
    println!(
        "  {:>6}  {:>9}  {:>9}  {:>10}  {:>7}  {:>7}",
        "thr", "LLM/N", "gateAcc", "routedWER", "wasted", "missed"
    );
    let grid = [0.40_f32, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80];
    let mut best: Option<(f32, f64, usize)> = None; // (thr, routed_wer, llm_calls)
    for &t in &grid {
        let mut llm_calls = 0usize;
        let mut routed_wer = 0.0_f64;
        let mut gate_total = 0usize;
        let mut gate_correct = 0usize;
        let mut wasted = 0usize;
        let mut missed = 0usize;
        for p in probes {
            let use_llm = needs_llm(p, conf(p), t, noisy);
            if use_llm {
                llm_calls += 1;
            }
            routed_wer += if use_llm { p.llm_wer } else { p.det_wer };
            if let Some(eb) = p.expect_bypass {
                gate_total += 1;
                if if eb { !use_llm } else { use_llm } {
                    gate_correct += 1;
                }
            }
            if use_llm && !helps(p) {
                wasted += 1;
            }
            if !use_llm && helps(p) {
                missed += 1;
            }
        }
        let avg_routed = routed_wer / n;
        let acc = if gate_total > 0 {
            format!(
                "{}/{} {:.0}%",
                gate_correct,
                gate_total,
                gate_correct as f64 / gate_total as f64 * 100.0
            )
        } else {
            "-".to_string()
        };
        let mark = if (t - 0.65).abs() < 1e-6 {
            " ← current"
        } else {
            ""
        };
        println!(
            "  {:>6.2}  {:>4}/{:<4}  {:>9}  {:>9.2}  {:>7}  {:>7}{}",
            t,
            llm_calls,
            probes.len(),
            acc,
            avg_routed,
            wasted,
            missed,
            mark
        );
        let better = match best {
            None => true,
            Some((_, bw, bc)) => {
                avg_routed < bw - 1e-9 || (avg_routed <= bw + 1e-9 && llm_calls < bc)
            }
        };
        if better {
            best = Some((t, avg_routed, llm_calls));
        }
    }
    if let Some((t, w, c)) = best {
        println!("  → best: routed WER {w:.2}% at thr {t:.2} ({c} LLM calls)\n");
    }
}
