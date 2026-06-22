//! Frame-level false-positive probe for the Earshot VAD stage, comparing the
//! raw per-frame decision against temporal (K-of-M) smoothing.
//!
//! Run: cargo run --release --features cuda --example vad_fp_probe -p lune-app
//!
//! We have no human frame labels, but we DO have clean→noisy pairs: the
//! `*_noise_snr*` clips are the clean fixtures with reproducible Gaussian noise
//! added (same speech, same positions). The clean clip's Silero gives a pseudo-
//! ground-truth speech mask; a stage firing on the NOISY clip in a clean-silent
//! frame is a true noise false positive.
//!
//! Question: can K-of-M smoothing of Earshot's decision (require K of the last M
//! frames above threshold) cut its isolated-blip false positives WITHOUT losing
//! speech recall? Silero FP is shown for reference (it is the precise stage).
//!
//! Caveat: "truth" is clean-Silero, not a human; the mask is dilated a couple of
//! frames so onset/offset ramps aren't miscounted.

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use bench_utils::{load_wav_f32, workspace_root};
use lune_app::vad::{DEFAULT_VAD_CONFIG, HybridVad};
use lune_audio::AudioChunk;

const PAIRS: &[&str] = &["librispeech_clean", "custom_recording"];
const SNRS: &[i32] = &[0, 5, 10];
const DILATE: usize = 2;
/// (M, K) smoothing configs to compare against raw (1,1).
const SMOOTH: &[(usize, usize)] = &[(1, 1), (3, 2), (5, 3)];

fn frame_scores(hybrid: &mut HybridVad, pcm: &[f32]) -> Vec<(f32, f32)> {
    hybrid.reset().expect("reset");
    let mut out = Vec::new();
    for chunk in pcm.chunks(1280) {
        let c = AudioChunk {
            samples: chunk.to_vec(),
            timestamp_ms: 0,
        };
        out.extend(hybrid.debug_stage_scores(&c));
    }
    out
}

fn speech_mask(clean: &[(f32, f32)], offset: f32) -> Vec<bool> {
    let raw: Vec<bool> = clean.iter().map(|&(_, p)| p >= offset).collect();
    let mut mask = vec![false; raw.len()];
    for (i, m) in mask.iter_mut().enumerate() {
        let lo = i.saturating_sub(DILATE);
        let hi = (i + DILATE + 1).min(raw.len());
        *m = raw[lo..hi].iter().any(|&b| b);
    }
    mask
}

/// Causal K-of-M smoothing of `es >= thr`: active iff at least K of the last M
/// frames cleared the threshold. `(1, 1)` is the raw decision.
fn smoothed_active(scores: &[(f32, f32)], thr: f32, m: usize, k: usize) -> Vec<bool> {
    let raw: Vec<bool> = scores.iter().map(|&(es, _)| es >= thr).collect();
    (0..raw.len())
        .map(|i| {
            let lo = (i + 1).saturating_sub(m);
            raw[lo..=i].iter().filter(|&&b| b).count() >= k
        })
        .collect()
}

/// (FP%, recall%) of an active-bit vector against the speech mask.
fn rate(active: &[bool], mask: &[bool]) -> (f64, f64) {
    let n = active.len().min(mask.len());
    let (mut fp, mut tp, mut n_sil, mut n_sp) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..n {
        if mask[i] {
            n_sp += 1;
            if active[i] {
                tp += 1;
            }
        } else {
            n_sil += 1;
            if active[i] {
                fp += 1;
            }
        }
    }
    let pct = |a: usize, b: usize| {
        if b == 0 {
            0.0
        } else {
            a as f64 / b as f64 * 100.0
        }
    };
    (pct(fp, n_sil), pct(tp, n_sp))
}

fn main() {
    let cfg = DEFAULT_VAD_CONFIG;
    let mut hybrid = HybridVad::with_config(cfg, true).expect("build hybrid vad");
    let dir = workspace_root().join("fixtures/audio");
    let es_thr = cfg.earshot_activity_threshold;

    println!();
    println!(
        "  Earshot smoothing probe   (thr {:.2}; columns: FP% / recall% per K-of-M)",
        es_thr
    );
    println!("  FP = fired in a clean-SILENT frame; lower FP + same recall = win.\n");
    print!("  {:<22} {:>6}", "clip", "SNR");
    for (m, k) in SMOOTH {
        let label = if (*m, *k) == (1, 1) {
            "raw".to_string()
        } else {
            format!("{k}of{m}")
        };
        print!("  {label:>13}");
    }
    println!("   {:>8}", "silFP");

    for base in PAIRS {
        let clean_path = dir.join(format!("{base}.wav"));
        if !clean_path.exists() {
            continue;
        }
        let clean = frame_scores(&mut hybrid, &load_wav_f32(&clean_path));
        let mask = speech_mask(&clean, cfg.silero_offset);

        let mut rows: Vec<(String, Vec<(f32, f32)>)> = Vec::new(); // (label, scores)
        rows.push(("clean".to_string(), clean.clone()));
        for &snr in SNRS {
            let p = dir.join(format!("{base}_noise_snr{snr}.wav"));
            if p.exists() {
                rows.push((
                    format!("{snr}dB"),
                    frame_scores(&mut hybrid, &load_wav_f32(&p)),
                ));
            }
        }

        for (label, scores) in &rows {
            print!("  {base:<22} {label:>6}");
            for &(m, k) in SMOOTH {
                let active = smoothed_active(scores, es_thr, m, k);
                let (fp, rec) = rate(&active, &mask);
                print!("  {fp:>5.0}/{rec:>5.1}");
            }
            // Silero FP for reference (no smoothing).
            let sil: Vec<bool> = scores.iter().map(|&(_, p)| p >= cfg.silero_onset).collect();
            let (sfp, _) = rate(&sil, &mask);
            println!("   {sfp:>8.1}");
        }
        println!();
    }
    println!("  done.");
}
