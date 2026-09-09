//! punctuation_ab — does the deterministic punctuation layer (`punctuation.rs`)
//! help or hurt? Runs the REAL ASR + LLM once per fixture and compares four
//! post-ASR variants against the cased, punctuated reference:
//!
//!   raw  = corrector only (no punctuation.rs, no LLM)          [floor]
//!   det  = corrector + defuse/normalize/decase (no LLM)        [deterministic layer alone]
//!   llm  = corrector + LLM always (no punctuation.rs)          [LLM-only ceiling]
//!   full = the shipped pipeline (punctuation.rs + gate + LLM)  [production]
//!
//! Corpus: `fixtures/audio` (read speech) + `fixtures/bench/<category>` (the
//! casing/punctuation/proper-noun/fusion stress set). `punctuation.rs` only acts
//! on English, so the verdict is computed on the ENGLISH subset; non-English
//! fixtures are still scored for the table but excluded from the verdict.
//!
//! Metrics vs reference: WER, cased-CER (casing), punct-F1 (punctuation marks),
//! cat (fused words), spc (spacing). LLM arms also report %bypass + latency.
//!
//! Run: cargo run --release --example punctuation_ab -p lune-voice
//!      (LUNE_BENCH_PRINT=1 dumps per-fixture text; LUNE_BENCH_FIXTURE=<sub> filters.)

#[allow(dead_code)]
#[path = "../bench_utils.rs"]
mod bench_utils;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bench_utils::{
    BenchmarkHarness, Fixture, cased_cer, concat_errors, load_wav_f32, percentile, spacing_delta,
    word_error_rate, workspace_root,
};
use lune_asr::AsrRegistry;
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::{PipelineConfig, StylePreset};
use lune_core::llm::PostProcessInput;
use lune_dict::corrector::{HWCE_FLAG_THRESHOLD, LexicalGate};
use lune_dict::formatting::{
    capitalize_sentence_starts, decase_common_words, defuse_inconsistent, normalize,
};

/// One arm's score for one fixture.
#[derive(Clone, Copy)]
struct ArmScore {
    wer: f64,
    cer: f64,
    pf1: f64,
    cat: f64,
    spc: f64,
    used_llm: bool,
    latency_ms: f64,
}

impl ArmScore {
    fn new(hyp: &str, ref_: &str, used_llm: bool, latency_ms: f64) -> Self {
        ArmScore {
            wer: word_error_rate(hyp, ref_),
            cer: cased_cer(hyp, ref_),
            pf1: punct_f1(hyp, ref_),
            cat: concat_errors(hyp, ref_) as f64,
            spc: spacing_delta(hyp, ref_) as f64,
            used_llm,
            latency_ms,
        }
    }
}

/// Every arm's score for one fixture, plus its category and language.
struct FxResult {
    category: String,
    is_en: bool,
    raw: ArmScore,
    det: ArmScore,
    llm: ArmScore,
    full: ArmScore,
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Punctuation-mark F1: multiset agreement of `. , ? ! ; :` between hyp and ref.
/// A direct proxy for "did the punctuation come out right" (1.0 = perfect).
fn punct_f1(hyp: &str, ref_: &str) -> f64 {
    fn counts(s: &str) -> HashMap<char, i32> {
        let mut m = HashMap::new();
        for c in s
            .chars()
            .filter(|c| matches!(c, '.' | ',' | '?' | '!' | ';' | ':'))
        {
            *m.entry(c).or_insert(0) += 1;
        }
        m
    }
    let (h, r) = (counts(hyp), counts(ref_));
    let tp: i32 = r
        .iter()
        .map(|(c, &rc)| h.get(c).copied().unwrap_or(0).min(rc))
        .sum();
    let (hn, rn): (i32, i32) = (h.values().sum(), r.values().sum());
    if hn == 0 && rn == 0 {
        return 1.0;
    }
    let prec = if hn == 0 { 0.0 } else { tp as f64 / hn as f64 };
    let rec = if rn == 0 { 0.0 } else { tp as f64 / rn as f64 };
    if prec + rec == 0.0 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    }
}

/// Transcribe one fixture into ASR segments (with per-word confidences and the
/// detected language), mirroring the production session's ASR call.
fn transcribe_segments(
    asr: &AsrRegistry,
    rt: &tokio::runtime::Runtime,
    pcm: &[f32],
    lang: &str,
) -> Vec<AsrSegment> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let cfg = AsrSessionConfig {
        language: Some(lang.to_string()),
        ..Default::default()
    };
    let session = rt.block_on(asr.start_session(cfg)).expect("start session");
    rt.block_on(asr.push_audio(&session, pcm))
        .expect("push audio");
    let segments = rt.block_on(asr.poll_final(&session)).expect("poll final");
    let _ = rt.block_on(asr.end_session(session));
    segments
}

/// `"category/stem"` -> `"category"`; bare names (the read-speech set) -> `"audio"`.
fn category_of(name: &str) -> &str {
    name.split_once('/').map_or("audio", |(c, _)| c)
}

/// Load the `fixtures/bench/<category>/*.wav` + `.txt` stress set, tagged
/// `"category/stem"`, matching the vad_tune loader.
fn load_bench_fixtures(root: &Path) -> Vec<Fixture> {
    let bench_dir = root.join("fixtures/bench");
    let mut out = Vec::new();
    let Ok(cats) = std::fs::read_dir(&bench_dir) else {
        return out;
    };
    let mut cat_dirs: Vec<std::path::PathBuf> = cats
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    cat_dirs.sort();
    for cat_dir in cat_dirs {
        let category = cat_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let Ok(entries) = std::fs::read_dir(&cat_dir) else {
            continue;
        };
        let mut wavs: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
            .collect();
        wavs.sort();
        for wav in wavs {
            let txt = wav.with_extension("txt");
            if !txt.exists() {
                continue;
            }
            let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let pcm = load_wav_f32(&wav);
            let reference = std::fs::read_to_string(&txt)
                .expect("read reference")
                .trim()
                .to_string();
            let audio_secs = pcm.len() as f64 / 16_000.0;
            let name: &'static str = Box::leak(format!("{category}/{stem}").into_boxed_str());
            out.push(Fixture {
                name,
                pcm,
                reference,
                audio_secs,
            });
        }
    }
    out
}

/// Run the real LLM correction pass on `text` and return (capitalized output, latency_ms).
fn run_llm(
    harness: &BenchmarkHarness,
    preset: Option<&Arc<StylePreset>>,
    text: &str,
    lang: &str,
    correction_needed: bool,
) -> (String, f64) {
    // The LLM is styling-only now; there is no correction/edit gate, so this
    // flag no longer routes anything. Kept in the signature so the arm
    // bookkeeping below reads the same.
    let _ = correction_needed;
    // No style selected (production default): the styling pass never runs, so
    // the arm mirrors its input verbatim with zero LLM latency.
    let Some(preset) = preset else {
        return (text.to_owned(), 0.0);
    };
    let input = PostProcessInput {
        raw_text: text.to_owned(),
        style_preset: Arc::clone(preset),
        dictionary: vec![],
        language: lang.into(),
    };
    let t0 = Instant::now();
    let out = harness.rt.block_on(harness.llm.transform(&input));
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let text = match out {
        Ok(o) => capitalize_sentence_starts(&o.cleaned_text),
        Err(_) => text.to_owned(),
    };
    (text, ms)
}

fn main() {
    let harness = BenchmarkHarness::builder().with_llm().build();
    let cfg = PipelineConfig::default();
    let corrector = LexicalGate::new();
    // Styling is now optional and OFF by default (production default_style =
    // None), and the LLM pass is styling-only — there is no correction mode.
    // With no style selected the llm/full arms below bypass the LLM entirely
    // and mirror their no-LLM inputs (llm == raw, full == det).
    let preset: Option<Arc<StylePreset>> = None;
    let llm_ran = preset.is_some();

    let bench = load_bench_fixtures(&workspace_root());
    let needle = std::env::var("LUNE_BENCH_FIXTURE").ok();
    let fixtures: Vec<&Fixture> = harness
        .fixtures
        .iter()
        .chain(bench.iter())
        .filter(|f| needle.as_deref().is_none_or(|n| f.name.contains(n)))
        .collect();
    let print = std::env::var("LUNE_BENCH_PRINT").is_ok_and(|v| v != "0");
    println!(
        "punctuation_ab: {} fixtures (audio + bench), real ASR + LLM, styling off\n",
        fixtures.len()
    );

    let mut results: Vec<FxResult> = Vec::new();
    for fx in fixtures {
        let lang = BenchmarkHarness::language_for_fixture(fx.name);
        let segments = transcribe_segments(&harness.asr, &harness.rt, &fx.pcm, lang);
        if segments.is_empty() {
            continue;
        }
        let raw_text: String = segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let word_risk_scores: Vec<(&str, f32, f32)> = segments
            .iter()
            .flat_map(|s| s.word_confidences.iter())
            .map(|wc| (wc.word.as_str(), wc.word_error_risk, wc.mean_confidence))
            .collect();
        let detected_lang = segments
            .first()
            .and_then(|s| s.detected_language.as_deref())
            .and_then(lune_core::lang::display_name)
            .unwrap_or("English");
        let pipeline_en = detected_lang == "English";
        let truth_en = BenchmarkHarness::language_for_fixture(fx.name) == "en";

        // Stage shared by every arm: the lexical corrector.
        let hc = corrector.correct_with_risk_threshold(
            &raw_text,
            &word_risk_scores,
            &[],
            detected_lang,
            HWCE_FLAG_THRESHOLD,
            true,
        );

        // Deterministic punctuation layer (English-gated, production order).
        let corrected = if pipeline_en {
            defuse_inconsistent(&hc.corrected_text)
        } else {
            hc.corrected_text.clone()
        };
        let punct = normalize(&corrected);
        let text_after_heuristic = if pipeline_en {
            decase_common_words(&punct.text)
        } else {
            punct.text.clone()
        };

        // Gate decision — inlined because gate::decide is crate-private. This is
        // byte-for-byte the production arithmetic (wants_styling = false).
        let max_risk = word_risk_scores
            .iter()
            .map(|(_, r, _)| *r)
            .fold(0.0_f32, f32::max);
        let min_conf = 1.0 - max_risk;
        let conf_attention = cfg.llm_skip_threshold.is_some_and(|t| min_conf < t);
        let wc = raw_text.split_whitespace().count().max(1);
        let mean_risk = hc.word_risk_positions.iter().map(|(_, r)| *r).sum::<f32>() / wc as f32;
        let avg_conf = 1.0 - mean_risk;
        let too_broken = cfg.llm_skip_noisy_threshold.is_some_and(|t| avg_conf < t);
        let needs_correction = !hc.all_resolved || punct.needs_llm || conf_attention;
        let needs_llm = needs_correction && !too_broken;

        // Arm llm: corrector output -> LLM always (no deterministic punctuation).
        let (llm_text, llm_ms) = run_llm(
            &harness,
            preset.as_ref(),
            &hc.corrected_text,
            detected_lang,
            true,
        );

        // Arm full: the shipped pipeline (deterministic + gate + optional LLM).
        let (full_text, full_ms, full_used) = if needs_llm && llm_ran {
            let (t, ms) = run_llm(
                &harness,
                preset.as_ref(),
                &text_after_heuristic,
                detected_lang,
                needs_correction,
            );
            (t, ms, true)
        } else {
            (text_after_heuristic.clone(), 0.0, false)
        };

        if print {
            println!("  [{}]  (en={truth_en}, used_llm={full_used})", fx.name);
            println!("     ref : {:?}", fx.reference);
            println!("     raw : {:?}", hc.corrected_text);
            println!("     det : {:?}", text_after_heuristic);
            println!("     llm : {:?}", llm_text);
            println!("     full: {:?}", full_text);
        }

        results.push(FxResult {
            category: category_of(fx.name).to_string(),
            is_en: truth_en,
            raw: ArmScore::new(&hc.corrected_text, &fx.reference, false, 0.0),
            det: ArmScore::new(&text_after_heuristic, &fx.reference, false, 0.0),
            llm: ArmScore::new(&llm_text, &fx.reference, llm_ran, llm_ms),
            full: ArmScore::new(&full_text, &fx.reference, full_used, full_ms),
        });
    }

    let all: Vec<&FxResult> = results.iter().collect();
    let en: Vec<&FxResult> = results.iter().filter(|r| r.is_en).collect();

    println!(
        "(lower WER / cerCASE / cat / |spc| = better; higher punctF1 = better; bypass = % skipping the LLM)\n"
    );
    section("ALL fixtures", &all);
    section(
        &format!(
            "ENGLISH only ({} fixtures — punctuation.rs acts here)",
            en.len()
        ),
        &en,
    );
    per_category(&en);
    verdict(&en);
}

fn section(title: &str, rows: &[&FxResult]) {
    println!("── {title} ──");
    report(rows, "raw  (corrector only)", |r| &r.raw, false);
    report(rows, "det  (+ punctuation.rs, no LLM)", |r| &r.det, false);
    report(rows, "llm  (corrector + LLM always)", |r| &r.llm, true);
    report(
        rows,
        "full (shipped: punctuation.rs+gate+LLM)",
        |r| &r.full,
        true,
    );
    println!();
}

fn report(rows: &[&FxResult], label: &str, pick: impl Fn(&FxResult) -> &ArmScore, llm_arm: bool) {
    if rows.is_empty() {
        return;
    }
    let s: Vec<&ArmScore> = rows.iter().map(|r| pick(r)).collect();
    let m = |f: &dyn Fn(&ArmScore) -> f64| mean(&s.iter().map(|a| f(a)).collect::<Vec<_>>());
    let (bypass, p50, p90) = if llm_arm {
        let n = s.len();
        let calls = s.iter().filter(|a| a.used_llm).count();
        let mut lat: Vec<f64> = s.iter().map(|a| a.latency_ms).collect();
        lat.sort_by(|x, y| x.partial_cmp(y).unwrap());
        (
            format!("{:>3.0}%", 100.0 * (n - calls) as f64 / n as f64),
            percentile(&lat, 50.0),
            percentile(&lat, 90.0),
        )
    } else {
        ("  --".to_string(), 0.0, 0.0)
    };
    println!(
        "  {label:<42} WER {:>5.1}%  cerCASE {:>5.1}%  punctF1 {:>4.2}  cat {:>4.1}  spc {:>+5.1}  bypass {bypass}  p50 {p50:>5.0}ms  p90 {p90:>5.0}ms",
        m(&|a| a.wer) * 100.0,
        m(&|a| a.cer) * 100.0,
        m(&|a| a.pf1),
        m(&|a| a.cat),
        m(&|a| a.spc),
    );
}

/// Per-category English breakdown of the casing/punctuation metrics — shows
/// where punctuation.rs moves the needle (raw -> det) and how the arms compare.
fn per_category(en: &[&FxResult]) {
    if en.is_empty() {
        return;
    }
    let mut by_cat: BTreeMap<&str, Vec<&FxResult>> = BTreeMap::new();
    for r in en {
        by_cat.entry(r.category.as_str()).or_default().push(r);
    }
    println!("── per-category (English): cerCASE% / punctF1 / cat ──");
    println!(
        "  {:<16} {:>18} {:>18} {:>18} {:>18}",
        "category", "raw", "det", "llm", "full"
    );
    for (cat, rows) in by_cat {
        let cell = |pick: fn(&FxResult) -> &ArmScore| {
            let c = mean(&rows.iter().map(|r| pick(r).cer).collect::<Vec<_>>()) * 100.0;
            let f = mean(&rows.iter().map(|r| pick(r).pf1).collect::<Vec<_>>());
            let g = mean(&rows.iter().map(|r| pick(r).cat).collect::<Vec<_>>());
            format!("{c:>4.1}/{f:>4.2}/{g:>3.1}")
        };
        println!(
            "  {:<16} {:>18} {:>18} {:>18} {:>18}  (n={})",
            cat,
            cell(|r| &r.raw),
            cell(|r| &r.det),
            cell(|r| &r.llm),
            cell(|r| &r.full),
            rows.len()
        );
    }
    println!();
}

fn verdict(en: &[&FxResult]) {
    if en.is_empty() {
        println!("── verdict ──\n  No English fixtures — cannot judge punctuation.rs.");
        return;
    }
    let m = |rows: &[&FxResult], pick: fn(&FxResult) -> &ArmScore, f: fn(&ArmScore) -> f64| {
        mean(&rows.iter().map(|r| f(pick(r))).collect::<Vec<_>>())
    };
    let (raw_cer, det_cer) = (m(en, |r| &r.raw, |a| a.cer), m(en, |r| &r.det, |a| a.cer));
    let (raw_pf1, det_pf1) = (m(en, |r| &r.raw, |a| a.pf1), m(en, |r| &r.det, |a| a.pf1));
    let (raw_cat, det_cat) = (m(en, |r| &r.raw, |a| a.cat), m(en, |r| &r.det, |a| a.cat));
    let (llm_wer, full_wer) = (m(en, |r| &r.llm, |a| a.wer), m(en, |r| &r.full, |a| a.wer));
    let (llm_cer, full_cer) = (m(en, |r| &r.llm, |a| a.cer), m(en, |r| &r.full, |a| a.cer));
    let (llm_pf1, full_pf1) = (m(en, |r| &r.llm, |a| a.pf1), m(en, |r| &r.full, |a| a.pf1));
    let bypass = 100.0 * en.iter().filter(|r| !r.full.used_llm).count() as f64 / en.len() as f64;

    println!("── verdict (English only) ──");
    println!(
        "  punctuation.rs alone (det − raw):  cerCASE {:+.1}pp   punctF1 {:+.2}   cat {:+.1}",
        (det_cer - raw_cer) * 100.0,
        det_pf1 - raw_pf1,
        det_cat - raw_cat,
    );
    println!(
        "  full vs llm-only:                  WER {:+.1}pp   cerCASE {:+.1}pp   punctF1 {:+.2}   bypass {bypass:.0}%",
        (full_wer - llm_wer) * 100.0,
        (full_cer - llm_cer) * 100.0,
        full_pf1 - llm_pf1,
    );

    let det_helps =
        det_cer < raw_cer - 0.005 || det_pf1 > raw_pf1 + 0.02 || det_cat < raw_cat - 0.1;
    let full_quality_ok = full_wer <= llm_wer + 0.005 && full_cer <= llm_cer + 0.01;
    println!();
    if det_helps && full_quality_ok {
        println!(
            "  → punctuation.rs HELPS: the deterministic layer improves casing/punctuation on its own,\n    and the full pipeline matches LLM-only quality while bypassing the LLM on {bypass:.0}% of utterances."
        );
    } else if !full_quality_ok {
        println!(
            "  → punctuation.rs may HURT: full-pipeline quality trails LLM-only — the deterministic edits\n    appear to degrade what the LLM would produce. Inspect with LUNE_BENCH_PRINT=1."
        );
    } else {
        println!(
            "  → punctuation.rs is ~NEUTRAL on quality; its value is the {bypass:.0}% LLM bypass (latency saved)."
        );
    }
    println!("  (Heuristic read — trust the tables; small fixture sets are noisy.)");
}
