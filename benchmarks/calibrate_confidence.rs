//! Fit an isotonic confidence calibrator from fixture runs.
//!
//! Runs the current ASR backend over each fixture in `test_fixtures/`,
//! aligns hypothesis words with the reference transcript via edit-distance
//! DP, and produces `(raw_confidence, is_correct)` samples. Fits an
//! `IsotonicCalibrator` via Pool Adjacent Violators and writes the result
//! to disk for `SessionManager::with_calibrator_path` to load at startup.
//!
//! Run: `cargo run --release --example calibrate_confidence -p lune-app`
//! GPU: `cargo run --release --features cuda --example calibrate_confidence -p lune-app`
//!
//! Optional env vars:
//!   ASR=parakeet|qwen          select backend (default: parakeet)
//!   LUNE_CALIBRATOR_OUT=/path  override output path (default: app data dir)

#[allow(dead_code)]
#[path = "bench_utils.rs"]
mod bench_utils;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    Fixture, benchmark_asr_entry, init_ort, load_fixtures, normalize_text, workspace_root,
};
use lune_asr::{AsrRegistry, IsotonicCalibrator};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig, WordConfidence};
use lune_core::config::DevicePolicy;
use tokio::runtime::Runtime;

// ─── Entry point ────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              ISOTONIC CONFIDENCE CALIBRATOR FIT                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let has_gpu = init_ort();
    let root = workspace_root();
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
    println!("  Backend: {}", asr_entry.display_name);

    print!("  Loading ASR ... ");
    let t = Instant::now();
    let asr = AsrRegistry::from_config(&asr_entry.config, device).expect("load ASR");
    let asr = Arc::new(asr);
    println!("{:.1}s", t.elapsed().as_secs_f64());

    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        println!("  [skip] No WAV fixtures found; cannot fit calibrator.");
        return;
    }

    // ── Collect samples across all fixtures ─────────────────────────────
    let rt = Runtime::new().expect("build tokio runtime");
    let mut all_samples: Vec<(f32, bool)> = Vec::new();
    let mut total_words = 0usize;
    let mut total_correct = 0usize;

    for fixture in &fixtures {
        println!();
        println!("  ▸ {} ({:.1}s audio)", fixture.name, fixture.audio_secs);
        let segments = match run_asr(&asr, fixture, &rt) {
            Ok(s) => s,
            Err(e) => {
                println!("    [error] ASR failed: {e}");
                continue;
            }
        };
        let hyp_words = collect_hyp_words(&segments);
        if hyp_words.is_empty() {
            println!("    [warn] no hypothesis words with per-word confidence");
            continue;
        }
        let ref_words = tokenize(&fixture.reference);
        let fixture_samples = align_and_sample(&hyp_words, &ref_words);
        let n_correct = fixture_samples.iter().filter(|s| s.1).count();
        let n = fixture_samples.len();
        println!(
            "    samples = {n}  correct = {n_correct}  wer ≈ {:.1}%",
            100.0 * (1.0 - n_correct as f32 / n.max(1) as f32)
        );
        total_words += n;
        total_correct += n_correct;
        all_samples.extend(fixture_samples);
    }

    if all_samples.is_empty() {
        println!();
        println!("  [error] no samples collected — cannot fit calibrator");
        return;
    }

    println!();
    println!(
        "  Collected {} samples across {} fixtures ({} correct, empirical accuracy {:.2}%)",
        total_words,
        fixtures.len(),
        total_correct,
        100.0 * total_correct as f32 / total_words.max(1) as f32
    );

    // ── Fit calibrator ──────────────────────────────────────────────────
    let calibrator = IsotonicCalibrator::fit(&all_samples);
    println!(
        "  Fit complete: {} breakpoints",
        calibrator.breakpoint_count()
    );

    // ── Reliability diagram ─────────────────────────────────────────────
    print_reliability(&all_samples, &calibrator);

    // ── Save to disk ────────────────────────────────────────────────────
    let out_path = resolve_output_path();
    match save_calibrator(&calibrator, &out_path) {
        Ok(()) => {
            println!();
            println!("  ✓ Wrote calibrator to {}", out_path.display());
            println!("    Next dictation session will load it automatically.");
        }
        Err(e) => {
            println!();
            println!("  [error] failed to write calibrator: {e}");
        }
    }
    println!();
}

// ─── ASR invocation ─────────────────────────────────────────────────────────

fn run_asr(asr: &AsrRegistry, fixture: &Fixture, rt: &Runtime) -> Result<Vec<AsrSegment>, String> {
    rt.block_on(async {
        let handle = asr
            .start_session(AsrSessionConfig::default())
            .await
            .map_err(|e| format!("{e}"))?;
        asr.push_audio(&handle, &fixture.pcm)
            .await
            .map_err(|e| format!("{e}"))?;
        let segments = asr.poll_final(&handle).await.map_err(|e| format!("{e}"))?;
        asr.end_session(handle).await.map_err(|e| format!("{e}"))?;
        Ok(segments)
    })
}

fn collect_hyp_words(segments: &[AsrSegment]) -> Vec<WordConfidence> {
    segments
        .iter()
        .flat_map(|s| s.word_confidences.iter())
        .map(|w| WordConfidence::new(normalize_text(&w.word), w.confidence))
        .filter(|w| !w.word.is_empty())
        .collect()
}

fn tokenize(reference: &str) -> Vec<String> {
    normalize_text(reference)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

// ─── Alignment ──────────────────────────────────────────────────────────────

/// Edit-distance alignment with traceback. Returns one sample per
/// hypothesis word: match → (conf, true), substitution → (conf, false),
/// insertion → (conf, false). Deletions (ref words with no hyp match) do
/// not contribute samples — there is no confidence to label.
fn align_and_sample(hyp: &[WordConfidence], refw: &[String]) -> Vec<(f32, bool)> {
    let m = hyp.len();
    let n = refw.len();

    // Cost table
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let sub_cost = if hyp[i - 1].word == refw[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j - 1] + sub_cost)
                .min(dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1);
        }
    }

    // Traceback — in reverse — emitting a sample per hyp word.
    let mut samples: Vec<(f32, bool)> = Vec::with_capacity(m);
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let diag_cost = dp[i - 1][j - 1] + usize::from(hyp[i - 1].word != refw[j - 1]);
            if dp[i][j] == diag_cost {
                let is_match = hyp[i - 1].word == refw[j - 1];
                samples.push((hyp[i - 1].confidence, is_match));
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            // Insertion: hyp word has no corresponding ref word → wrong.
            samples.push((hyp[i - 1].confidence, false));
            i -= 1;
            continue;
        }
        // Deletion: ref word has no hyp counterpart → no sample.
        j -= 1;
    }
    samples.reverse();
    samples
}

// ─── Reliability diagram ────────────────────────────────────────────────────

fn print_reliability(samples: &[(f32, bool)], calibrator: &IsotonicCalibrator) {
    println!();
    println!("  Reliability diagram (empirical vs calibrated, 10 bins):");
    println!(
        "    {:^8}  {:^10}  {:^10}  {:^10}",
        "bin", "empirical", "calibrated", "count"
    );
    println!("    {:─<8}  {:─<10}  {:─<10}  {:─<10}", "", "", "", "");
    for b in 0..10 {
        let lo = b as f32 / 10.0;
        let hi = (b + 1) as f32 / 10.0;
        let in_bin: Vec<&(f32, bool)> = samples
            .iter()
            .filter(|&&(c, _)| c >= lo && c < hi + 1e-6)
            .collect();
        if in_bin.is_empty() {
            continue;
        }
        let n_correct = in_bin.iter().filter(|(_, ok)| *ok).count();
        let empirical = n_correct as f32 / in_bin.len() as f32;
        let mid = (lo + hi) * 0.5;
        let calibrated = calibrator.calibrate(mid);
        println!(
            "    [{lo:.1},{hi:.1})  {empirical:>8.3}    {calibrated:>8.3}    {:>8}",
            in_bin.len()
        );
    }
}

// ─── Output path resolution ─────────────────────────────────────────────────

fn resolve_output_path() -> PathBuf {
    if let Ok(override_path) = std::env::var("LUNE_CALIBRATOR_OUT") {
        return PathBuf::from(override_path);
    }
    let data_dir = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local/share"))
        })
        .unwrap_or_else(|| PathBuf::from("."));
    data_dir.join("lune-voice").join("calibrator.json")
}

fn save_calibrator(cal: &IsotonicCalibrator, path: &PathBuf) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(cal)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}
