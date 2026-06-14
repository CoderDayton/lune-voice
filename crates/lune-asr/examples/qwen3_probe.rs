//! Word confidence validation — runs ASR on all fixtures, sends transcripts
//! to the LLM without fix_words, diffs to find which words the LLM changed,
//! then correlates changes with ASR confidence scores.
//!
//! Usage: cargo run --release --features cuda --example qwen3_probe -p lune-asr

use std::path::PathBuf;
use std::sync::Arc;

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn word_diff(asr_text: &str, llm_text: &str) -> Vec<(String, bool)> {
    let asr_words: Vec<&str> = asr_text.split_whitespace().collect();
    let llm_words: Vec<&str> = llm_text.split_whitespace().collect();

    let n = asr_words.len();
    let m = llm_words.len();
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        row[0] = i as u32;
    }
    for (j, col) in dp[0].iter_mut().enumerate().take(m + 1) {
        *col = j as u32;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if normalize(asr_words[i - 1]) == normalize(llm_words[j - 1]) {
                0
            } else {
                1
            };
            dp[i][j] = dp[i - 1][j - 1]
                .wrapping_add(cost)
                .min(dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1);
        }
    }

    let mut result: Vec<(String, bool)> = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && normalize(asr_words[i - 1]) == normalize(llm_words[j - 1]) {
            result.push((asr_words[i - 1].to_string(), false));
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            result.push((asr_words[i - 1].to_string(), true));
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            result.push((asr_words[i - 1].to_string(), true));
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let _ = ort::init().commit();

    let data_dir = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".local/share")
        });
    let models_dir = data_dir.join("lune-voice/models");

    let asr_model_dir = models_dir.join(lune_core::models::DIR_QWEN3_ASR);
    let asr_config = lune_asr::Qwen3AsrConfig {
        model_dir: asr_model_dir,
        use_gpu: true,
        cuda_required: false,
    };
    let backend = lune_asr::Qwen3AsrBackend::new(asr_config).expect("load ASR");

    let llm_dir = models_dir.join(lune_core::models::DIR_STYLE_LLM);
    let ort_llm = lune_llm::OrtStyleLlm::new(lune_llm::OrtStyleLlmConfig {
        model_dir: llm_dir,
        use_gpu: true,
        ..Default::default()
    })
    .expect("load LLM");
    let llm: Arc<dyn lune_core::llm::TextPostProcessor> = Arc::new(
        lune_llm::CorrectionPipeline::new(Arc::new(ort_llm) as Arc<dyn lune_core::llm::LlmService>),
    );

    let fixtures_dir = root.join("fixtures/audio");
    let mut fixtures: Vec<(String, Vec<f32>)> = Vec::new();
    for entry in std::fs::read_dir(&fixtures_dir)
        .expect("read fixtures")
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wav") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let reader = hound::WavReader::open(&path).expect("open wav");
        let spec = reader.spec();
        let pcm: Vec<f32> = if spec.bits_per_sample == 16 {
            reader
                .into_samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect()
        } else {
            reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
        };
        fixtures.push((stem, pcm));
    }
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    if let Ok(only) = std::env::var("FIXTURE") {
        fixtures.retain(|(name, _)| name.contains(&only));
    }
    println!("Fixtures: {}\n", fixtures.len());

    let rt = tokio::runtime::Runtime::new().unwrap();
    let style = Arc::new(lune_core::config::StylePreset::casual());

    let mut all_words: Vec<(String, f32, f32, bool)> = Vec::new();

    for (name, pcm) in &fixtures {
        let word_confs = rt.block_on(async {
            use lune_core::asr::{AsrBackend, AsrSessionConfig};
            let session = backend
                .start_session(AsrSessionConfig::default())
                .await
                .unwrap();
            backend.push_audio(&session, pcm).await.unwrap();
            let results = backend.poll_final(&session).await.unwrap();
            backend.end_session(session).await.unwrap();
            if results.is_empty() {
                return (String::new(), None, vec![]);
            }
            let seg = &results[0];
            (
                seg.text.clone(),
                seg.confidence,
                seg.word_confidences.clone(),
            )
        });

        let (asr_text, seg_conf, wcs) = word_confs;
        if asr_text.is_empty() {
            continue;
        }

        let fix_words: Vec<String> = wcs
            .iter()
            .filter(|wc| wc.word_error_risk >= 0.50 && !wc.word.is_empty())
            .map(|wc| wc.word.clone())
            .collect();

        const BYPASS_CONFIDENCE: f32 = 0.92;
        let needs_llm = seg_conf.is_none_or(|c| c < BYPASS_CONFIDENCE)
            || wcs.iter().any(|wc| wc.word_error_risk >= 0.50);

        let llm_output = if needs_llm {
            rt.block_on(async {
                use lune_core::llm::PostProcessInput;
                let input = PostProcessInput {
                    raw_text: asr_text.clone(),
                    style_preset: Arc::clone(&style),
                    app_category: None,
                    dictionary: vec![],
                    language: "en".into(),
                    correction_needed: true,
                };
                llm.transform(&input).await.unwrap().cleaned_text
            })
        } else {
            println!(
                "  [BYPASS] confidence={:.2}, no flagged words",
                seg_conf.unwrap_or(0.0)
            );
            asr_text.clone()
        };
        if !fix_words.is_empty() {
            println!("  fix_words: {:?}", fix_words);
        }

        let diff = word_diff(&asr_text, &llm_output);

        println!("── {} ──", name);
        println!("  ASR: {}", asr_text);
        println!("  LLM: {}", llm_output);

        // Build set of changed words from diff
        let mut changed_set: Vec<String> = Vec::new();
        for (dw, changed) in &diff {
            if *changed {
                changed_set.push(
                    dw.trim_matches(|c: char| c.is_ascii_punctuation())
                        .to_lowercase(),
                );
            }
        }

        // Print ALL words with scores, mark changed ones
        println!(
            "  {:>20} {:>6} {:>6} {:>7}",
            "word", "conf", "fix", "status"
        );
        for wc in &wcs {
            let stripped = wc
                .word
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase();
            let is_changed = changed_set.contains(&stripped);
            let status = if is_changed {
                "CHANGED"
            } else if wc.word_error_risk >= 0.30 {
                "FLAG"
            } else {
                ""
            };
            if wc.word_error_risk >= 0.10 || is_changed {
                println!(
                    "  {:>20} {:>6.2} {:>6.2} {:>7}",
                    wc.word, wc.confidence, wc.word_error_risk, status
                );
            }
            // Always collect for sweep
            all_words.push((
                wc.word.clone(),
                wc.confidence,
                wc.word_error_risk,
                is_changed,
            ));
        }
        println!();
    }

    println!("═══ Threshold Sweep (LLM-validated) ═══\n");
    println!("\n── word_error_risk sweep (flag if word_error_risk >= thresh) ──\n");
    println!(
        "{:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}",
        "Thresh", "TP", "FP", "FN", "TN", "Prec", "F1"
    );

    let mut best_f1 = 0.0f64;
    let mut best_thresh = 0.0f32;

    let mut t = 0.30f32;
    while t <= 0.95 {
        let (mut tp, mut fp, mut r#fn, mut tn) = (0u32, 0u32, 0u32, 0u32);
        for (_, _, word_error_risk, changed) in &all_words {
            let flagged = *word_error_risk >= t;
            match (flagged, changed) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => r#fn += 1,
                (false, false) => tn += 1,
            }
        }
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + r#fn > 0 {
            tp as f64 / (tp + r#fn) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!(
            "{:>6.2} {:>6} {:>6} {:>6} {:>6} {:>8.3} {:>8.3}",
            t, tp, fp, r#fn, tn, precision, f1
        );
        if f1 > best_f1 {
            best_f1 = f1;
            best_thresh = t;
        }
        t += 0.05;
    }

    println!("\nBest F1: {:.3} at threshold {:.2}", best_f1, best_thresh);

    let mut changed_fs: Vec<f32> = all_words
        .iter()
        .filter(|(_, _, _, c)| *c)
        .map(|(_, _, fs, _)| *fs)
        .collect();
    let mut unchanged_fs: Vec<f32> = all_words
        .iter()
        .filter(|(_, _, _, c)| !*c)
        .map(|(_, _, fs, _)| *fs)
        .collect();
    changed_fs.sort_by(|a, b| a.total_cmp(b));
    unchanged_fs.sort_by(|a, b| a.total_cmp(b));
    println!(
        "\nUnchanged word_error_risks ({}): min={:.3} p5={:.3} median={:.3}",
        unchanged_fs.len(),
        unchanged_fs.first().copied().unwrap_or(0.0),
        percentile(&unchanged_fs, 5.0),
        percentile(&unchanged_fs, 50.0)
    );
    println!(
        "Changed word_error_risks   ({}): max={:.3} p95={:.3} median={:.3}",
        changed_fs.len(),
        changed_fs.last().copied().unwrap_or(0.0),
        percentile(&changed_fs, 95.0),
        percentile(&changed_fs, 50.0)
    );
}

fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
