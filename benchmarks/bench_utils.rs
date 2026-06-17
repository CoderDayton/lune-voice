//! Shared helpers for pipeline benchmarks.
//!
//! Included via `#[path = "bench_utils.rs"] mod bench_utils;` from each example.

use std::path::PathBuf;
use std::sync::Arc;

use lune_core::asr::AsrBackendKind;
use lune_core::config::{AsrBackendConfig, DevicePolicy, UserSettings};
use lune_core::llm::TextPostProcessor;
use lune_persist::AppPaths;

pub fn models_dir() -> PathBuf {
    AppPaths::resolve().expect("resolve XDG paths").models_dir()
}

// ─── Fixture ────────────────────────────────────────────────────────────────

pub struct Fixture {
    pub name: &'static str,
    pub pcm: Vec<f32>,
    pub reference: String,
    pub audio_secs: f64,
}

// ─── Path helpers ───────────────────────────────────────────────────────────

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

// ─── Process memory ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryUsage {
    pub rss_mb: f64,
    pub peak_mb: f64,
}

impl MemoryUsage {
    pub fn display(self) -> String {
        format!("RSS {:.0} MB  Peak {:.0} MB", self.rss_mb, self.peak_mb)
    }
}

pub fn process_memory_usage() -> Option<MemoryUsage> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss_kb = None;
    let mut peak_kb = None;

    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kb = rest.split_whitespace().next()?.parse::<f64>().ok();
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            peak_kb = rest.split_whitespace().next()?.parse::<f64>().ok();
        }
    }

    Some(MemoryUsage {
        rss_mb: rss_kb? / 1024.0,
        peak_mb: peak_kb.unwrap_or(rss_kb?) / 1024.0,
    })
}

// ─── ASR benchmark selection ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkAsr {
    Qwen3,
}

impl BenchmarkAsr {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "qwen3" | "qwen3-asr" | "qwen" => Some(Self::Qwen3),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchmarkAsrEntry {
    pub display_name: &'static str,
    pub backend_kind: AsrBackendKind,
    pub config: AsrBackendConfig,
}

pub fn requested_benchmark_asr() -> Option<BenchmarkAsr> {
    if let Ok(raw) = std::env::var("ASR") {
        if let Some(choice) = BenchmarkAsr::parse(&raw) {
            return Some(choice);
        }
        eprintln!("  [warn] Ignoring unsupported ASR={raw:?}; expected qwen3");
    }

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--asr=") {
            if let Some(choice) = BenchmarkAsr::parse(value) {
                return Some(choice);
            }
            eprintln!("  [warn] Ignoring unsupported --asr={value:?}; expected qwen3");
            return None;
        }
        if arg == "--asr" {
            if let Some(value) = args.next() {
                if let Some(choice) = BenchmarkAsr::parse(&value) {
                    return Some(choice);
                }
                eprintln!("  [warn] Ignoring unsupported --asr {value:?}; expected qwen3");
            } else {
                eprintln!("  [warn] Ignoring bare --asr; expected qwen3");
            }
            return None;
        }
    }

    None
}

pub fn benchmark_asr_entries() -> Vec<BenchmarkAsrEntry> {
    let _requested = requested_benchmark_asr();
    let mut entries = Vec::new();

    let qwen3_dir = models_dir().join(lune_core::models::DIR_QWEN3_ASR);
    if qwen3_dir.join("tokenizer.json").is_file()
        && qwen3_dir.join("qwen3-asr-encoder.onnx").is_file()
        && qwen3_dir.join("qwen3-asr-embed-tokens.onnx").is_file()
        && qwen3_dir.join("qwen3-asr-decoder.onnx").is_file()
    {
        entries.push(BenchmarkAsrEntry {
            display_name: "Qwen3-ASR-0.6B-int4",
            backend_kind: AsrBackendKind::Qwen3Asr,
            config: AsrBackendConfig::Qwen3Asr {},
        });
    }

    entries
}

pub fn benchmark_asr_entry() -> Result<BenchmarkAsrEntry, String> {
    let requested = requested_benchmark_asr();
    benchmark_asr_entries()
        .into_iter()
        .next()
        .ok_or_else(|| match requested {
            Some(choice) => format!(
                "Requested ASR backend {:?} is unavailable. Expected qwen3-asr-encoder.onnx, qwen3-asr-embed-tokens.onnx, qwen3-asr-decoder.onnx, and tokenizer.json under {:?}.",
                choice,
                models_dir().join(lune_core::models::DIR_QWEN3_ASR)
            ),
            None => format!(
                "No complete Qwen3-ASR ONNX model found under {:?}.",
                models_dir().join(lune_core::models::DIR_QWEN3_ASR)
            ),
        })
}

/// Names that must come first when present, in this order. Everything else
/// is appended alphabetically so the noisified variants cluster under their
/// clean source (`librispeech_clean`, `librispeech_clean_noise_snr10`, …).
const PRIORITY_FIXTURE_NAMES: &[&str] =
    &["librispeech_clean", "librispeech_40744", "custom_recording"];

pub fn load_fixtures(root: &std::path::Path) -> Vec<Fixture> {
    let fixtures_dir = root.join("fixtures/audio");
    let Ok(entries) = std::fs::read_dir(&fixtures_dir) else {
        return Vec::new();
    };

    // Collect `(name, wav_path, txt_path)` tuples for every `.wav` that has
    // a matching `.txt`. Skip orphans silently so stray audio files don't
    // break the benchmark suite.
    let mut discovered: Vec<(String, std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let wav_path = entry.path();
        if wav_path.extension().and_then(|s| s.to_str()) != Some("wav") {
            continue;
        }
        let stem = match wav_path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let txt_path = wav_path.with_extension("txt");
        if !txt_path.exists() {
            continue;
        }
        discovered.push((stem, wav_path, txt_path));
    }

    // Stable ordering: priority names first in their declared order, then
    // the remainder alphabetically. This keeps the ablation output readable
    // while still picking up any new fixture generated by `noisify_fixtures`.
    discovered.sort_by(|(a, _, _), (b, _, _)| {
        let rank = |name: &str| {
            PRIORITY_FIXTURE_NAMES
                .iter()
                .position(|&n| n == name)
                .unwrap_or(usize::MAX)
        };
        rank(a).cmp(&rank(b)).then_with(|| a.cmp(b))
    });

    let mut fixtures = Vec::new();
    for (name, wav_path, txt_path) in discovered {
        let pcm = load_wav_f32(&wav_path);
        let reference = std::fs::read_to_string(&txt_path)
            .expect("read reference")
            .trim()
            .to_string();
        let audio_secs = pcm.len() as f64 / 16_000.0;
        println!(
            "  Loaded {}: {:.1}s ({} samples)",
            name,
            audio_secs,
            pcm.len()
        );
        // Leak the name so `Fixture::name` can stay `&'static str` without
        // restructuring the struct. These live for the whole benchmark run
        // which is a single process, so the leak is bounded (~few dozen
        // strings, ~10 bytes each).
        let name_static: &'static str = Box::leak(name.into_boxed_str());
        fixtures.push(Fixture {
            name: name_static,
            pcm,
            reference,
            audio_secs,
        });
    }
    fixtures
}

// ─── Audio I/O ──────────────────────────────────────────────────────────────

pub fn load_wav_f32(path: &std::path::Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono WAV");
    assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz WAV");
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.expect("read sample"))
            .collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.expect("read sample") as f32 / max)
                .collect()
        }
    }
}

// ─── Text metrics ───────────────────────────────────────────────────────────

/// Lowercase, strip punctuation, collapse whitespace.
pub fn normalize_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn levenshtein<T: Eq>(a: &[T], b: &[T]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp: Vec<usize> = (0..=n).collect();
    for i in 1..=m {
        let mut prev = dp[0];
        dp[0] = i;
        for j in 1..=n {
            let temp = dp[j];
            dp[j] = if a[i - 1] == b[j - 1] {
                prev
            } else {
                1 + prev.min(dp[j]).min(dp[j - 1])
            };
            prev = temp;
        }
    }
    dp[n]
}

pub fn word_error_rate(hyp: &str, ref_: &str) -> f64 {
    let h = normalize_text(hyp);
    let r = normalize_text(ref_);
    let hw: Vec<&str> = h.split_whitespace().collect();
    let rw: Vec<&str> = r.split_whitespace().collect();
    if rw.is_empty() {
        return 0.0;
    }
    levenshtein(&hw, &rw) as f64 / rw.len() as f64
}

pub fn char_error_rate(hyp: &str, ref_: &str) -> f64 {
    let h = normalize_text(hyp);
    let r = normalize_text(ref_);
    let hc: Vec<char> = h.chars().collect();
    let rc: Vec<char> = r.chars().collect();
    if rc.is_empty() {
        return 0.0;
    }
    levenshtein(&hc, &rc) as f64 / rc.len() as f64
}

// ─── Case/spacing-aware metrics ──────────────────────────────────────────────
//
// WER and `char_error_rate` both `normalize_text` first (lowercase, strip
// punctuation, collapse whitespace), so they are blind to the exact failures
// users hit: stray mid-word capitals and fused words (`MakeSure`). These three
// metrics see what WER erases.

/// Character error rate with **no** normalization — case, spacing, and
/// punctuation differences all count. Unlike [`char_error_rate`], a pure case
/// change (`Abc` vs `abc`) is a real error here.
pub fn cased_cer(hyp: &str, ref_: &str) -> f64 {
    let hc: Vec<char> = hyp.chars().collect();
    let rc: Vec<char> = ref_.chars().collect();
    if rc.is_empty() {
        return 0.0;
    }
    levenshtein(&hc, &rc) as f64 / rc.len() as f64
}

/// Count spurious fused-word boundaries — a lowercase letter immediately
/// followed by an uppercase letter with no space (`MakeSure`) — present in
/// `hyp` but not in `ref_`. The numeric signature of the VAD-splice regression.
/// camelCase already in the reference (`iPhone`) is subtracted out, not flagged.
pub fn concat_errors(hyp: &str, ref_: &str) -> usize {
    /// Count adjacent `[a-z][A-Z]` char pairs. The space requirement is
    /// implicit: a space between the letters breaks adjacency, so only
    /// genuinely fused words contribute.
    fn fused_boundaries(s: &str) -> usize {
        let chars: Vec<char> = s.chars().collect();
        chars
            .windows(2)
            .filter(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase())
            .count()
    }
    fused_boundaries(hyp).saturating_sub(fused_boundaries(ref_))
}

/// Signed word-count difference (`hyp − ref_`), counting whitespace tokens that
/// contain at least one alphanumeric character (so trailing punctuation like
/// `world.` is never counted as a separate word). Negative means `hyp` merged
/// words (`MakeSure`); positive means it split them.
pub fn spacing_delta(hyp: &str, ref_: &str) -> i64 {
    /// Count whitespace-separated tokens carrying at least one alphanumeric
    /// char, so trailing punctuation (`world.`) never changes the tally.
    fn word_count(s: &str) -> i64 {
        s.split_whitespace()
            .filter(|tok| tok.chars().any(|c| c.is_alphanumeric()))
            .count() as i64
    }
    word_count(hyp) - word_count(ref_)
}

// ─── Display helpers ────────────────────────────────────────────────────────

pub fn wrap_print(prefix: &str, text: &str, width: usize) {
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            println!("{prefix}{line}");
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        println!("{prefix}{line}");
    }
}

/// Compute the p-th percentile of a slice (linear interpolation).
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ─── ORT init ───────────────────────────────────────────────────────────────

/// Initialize ORT runtime. Returns `true` if GPU (CUDA EP) is active.
pub fn init_ort() -> bool {
    let force_cpu = std::env::var("FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false);
    if force_cpu {
        let _ = ort::init().commit();
        println!("  [info] ORT: CPU only (FORCE_CPU=1)");
        return false;
    }
    // CPU-only global init. ASR sessions register their own CUDA EP
    // via build_ort_session — keeps VAD on CPU automatically.
    let _ = ort::init().commit();

    #[cfg(feature = "cuda")]
    let has_gpu = {
        use ort::ep::ExecutionProvider;
        let ok = (|| -> std::result::Result<(), Box<dyn std::error::Error>> {
            let mut b = ort::session::Session::builder()?;
            ort::ep::CUDA::default().register(&mut b)?;
            Ok(())
        })();
        ok.is_ok()
    };
    #[cfg(not(feature = "cuda"))]
    let has_gpu = false;

    if has_gpu {
        println!("  [info] ORT: CPU global, CUDA per-session for ASR");
    } else {
        println!("  [info] ORT: CPU only");
    }
    has_gpu
}

// ─── Brent's method — 1D minimizer ─────────────────────────────────────────

/// Brent's method — 1D minimizer.
/// Converges at ~1.32 order near optimum vs golden section's linear.
/// Falls back to golden section when parabola is unreliable.
pub fn brent_minimize<F>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    const GOLDEN: f64 = 0.3819660112501051; // 1 - 1/φ
    let mut x = a + GOLDEN * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0f64;
    let mut e = 0.0f64;

    for _ in 0..100 {
        let mid = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        if (x - mid).abs() <= tol2 - 0.5 * (b - a) {
            return x;
        }

        let mut use_golden = true;

        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let mut q2 = 2.0 * (q - r);

            if q2 > 0.0 {
            } else {
                q2 = -q2;
            }
            let p2 = if q2 > 0.0 { -p } else { p };

            if p2.abs() < (0.5 * q2 * e).abs() && p2 > q2 * (a - x) && p2 < q2 * (b - x) {
                d = p2 / q2;
                use_golden = false;
                if (x + d - a) < tol2 || (b - x - d) < tol2 {
                    d = if mid > x { tol1 } else { -tol1 };
                }
            }
        }

        if use_golden {
            e = if x >= mid { a - x } else { b - x };
            d = GOLDEN * e;
        }

        let u = x + if d.abs() >= tol1 {
            d
        } else {
            tol1 * d.signum()
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

// ─── BenchmarkHarness ────────────────────────────────────────────────────────

/// Consolidated setup for benchmark binaries.
///
/// Build via [`BenchmarkHarness::builder()`]. Owns the tokio runtime, loaded
/// ASR and LLM backends, and the pre-filtered fixture list. All fields are
/// `pub` for direct use in benchmark code.
/// `GATE=<tau>` enables the open-vocab LLM-edit gate at threshold `tau` (any
/// parseable f32, including 0 to admit every above-zero-margin edit); unset,
/// empty, or `off` leaves it disabled. The gate lives in the correction
/// pipeline, so every benchmark routing through `harness.llm` honours it.
fn edit_gate_tau() -> Option<f32> {
    let v = std::env::var("GATE").ok()?;
    let v = v.trim();
    if v.is_empty() || v.eq_ignore_ascii_case("off") {
        return None;
    }
    v.parse::<f32>().ok()
}

pub struct BenchmarkHarness {
    pub rt: tokio::runtime::Runtime,
    pub asr: Arc<lune_asr::AsrRegistry>,
    pub llm: Arc<dyn TextPostProcessor>,
    pub fixtures: Vec<Fixture>,
    pub has_gpu: bool,
    pub models_dir: PathBuf,
}

impl BenchmarkHarness {
    pub fn builder() -> HarnessBuilder {
        HarnessBuilder::default()
    }

    /// Create a fresh `SessionManager` backed by an in-memory DB.
    ///
    /// Call once per fixture run; each call gets its own isolated DB and
    /// dictionary so runs don't bleed state.
    pub fn session_manager(
        &self,
        pipeline: lune_core::config::PipelineConfig,
    ) -> lune_app::SessionManager<lune_asr::AsrRegistry> {
        use std::sync::Arc;
        use tokio::sync::{Mutex, RwLock};

        let db = lune_persist::Database::open_in_memory().expect("in-memory DB");
        let dict = lune_dict::DictionaryManager::new(
            lune_persist::Database::open_in_memory().expect("in-memory DB for dict"),
        )
        .expect("DictionaryManager");

        let config = lune_app::SessionManagerConfig {
            asr_backend_kind: self.asr.kind(),
            pipeline,
            ..Default::default()
        };

        lune_app::SessionManager::new(
            Arc::clone(&self.asr),
            Arc::new(lune_dict::commands::VoiceCommandRouter::new()),
            Arc::clone(&self.llm),
            Arc::new(Mutex::new(dict)),
            Arc::new(Mutex::new(db)),
            Arc::new(lune_dict::corrector::LexicalGate::new()),
            config,
            Arc::new(RwLock::new(UserSettings::default())),
        )
    }

    /// Map a fixture name to a BCP-47 language tag.
    ///
    /// Checks for language keywords in the fixture name; defaults to `"en"`.
    pub fn language_for_fixture(name: &str) -> &'static str {
        if name.contains("french") {
            "fr"
        } else if name.contains("german") {
            "de"
        } else if name.contains("spanish") {
            "es"
        } else if name.contains("italian") {
            "it"
        } else if name.contains("dutch") {
            "nl"
        } else if name.contains("portuguese") {
            "pt"
        } else {
            "en"
        }
    }
}

/// Builder for [`BenchmarkHarness`].
#[derive(Default)]
pub struct HarnessBuilder {
    with_llm: bool,
    with_tracing: bool,
    skip_fixtures: bool,
}

impl HarnessBuilder {
    /// Load ORT-style LLM (falls back to `NullPostProcessor` on error or
    /// missing model). Without this the harness uses `NullPostProcessor`.
    pub fn with_llm(mut self) -> Self {
        self.with_llm = true;
        self
    }

    /// Init a `tracing_subscriber` with `RUST_LOG` env filter.
    pub fn with_tracing(mut self) -> Self {
        self.with_tracing = true;
        self
    }

    pub fn skip_fixtures(mut self) -> Self {
        self.skip_fixtures = true;
        self
    }

    /// Consume the builder and construct the harness.
    ///
    /// Panics (with a descriptive message) if required models are missing.
    pub fn build(self) -> BenchmarkHarness {
        if self.with_tracing {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
                )
                .try_init();
        }

        let has_gpu = init_ort();
        let device = if has_gpu {
            DevicePolicy::Cuda
        } else {
            DevicePolicy::Cpu
        };

        let models_dir = models_dir();

        // ASR
        let asr_entry = benchmark_asr_entry().unwrap_or_else(|msg| {
            panic!("BenchmarkHarness: cannot load ASR — {msg}");
        });
        let asr = lune_asr::AsrRegistry::from_config(&asr_entry.config, device, &models_dir)
            .unwrap_or_else(|e| panic!("BenchmarkHarness: ASR load failed — {e}"));
        let asr = Arc::new(asr);

        // LLM
        let llm: Arc<dyn TextPostProcessor> = if self.with_llm {
            let llm_dir = models_dir.join(lune_core::models::DIR_STYLE_LLM);
            let llm_off = std::env::var("LUNE_LLM")
                .map(|v| v == "off" || v == "0")
                .unwrap_or(false);
            if !llm_off && llm_dir.join("tokenizer.json").exists() {
                match lune_llm::OrtStyleLlm::new(lune_llm::OrtStyleLlmConfig {
                    model_dir: llm_dir,
                    use_gpu: has_gpu,
                    ..Default::default()
                }) {
                    Ok(ort_llm) => {
                        let pipeline = lune_llm::CorrectionPipeline::new(
                            Arc::new(ort_llm) as Arc<dyn lune_core::llm::LlmService>
                        );
                        if let Some(tau) = edit_gate_tau() {
                            pipeline.update_edit_gate(true, tau);
                            eprintln!("[bench] LLM-edit gate ENABLED (tau={tau})");
                        }
                        Arc::new(pipeline) as Arc<dyn TextPostProcessor>
                    }
                    Err(e) => {
                        eprintln!("[warn] LLM load failed ({e}), using NullPostProcessor");
                        Arc::new(lune_llm::NullPostProcessor::new())
                    }
                }
            } else {
                Arc::new(lune_llm::NullPostProcessor::new())
            }
        } else {
            Arc::new(lune_llm::NullPostProcessor::new())
        };

        // Fixtures
        let fixtures = if self.skip_fixtures {
            Vec::new()
        } else {
            let root = workspace_root();
            let mut fx = load_fixtures(&root);
            if let Ok(needle) = std::env::var("LUNE_BENCH_FIXTURE") {
                fx.retain(|f| f.name.contains(needle.as_str()));
            }
            fx
        };

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

        BenchmarkHarness {
            rt,
            asr,
            llm,
            fixtures,
            has_gpu,
            models_dir,
        }
    }
}

// ─── Metric tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod metric_tests {
    use super::{cased_cer, char_error_rate, concat_errors, spacing_delta};

    #[test]
    fn cased_cer_is_zero_for_identical() {
        assert_eq!(cased_cer("Make sure", "Make sure"), 0.0);
    }

    #[test]
    fn cased_cer_counts_case_differences() {
        // The whole point: a pure case change is a real error here, while the
        // case-folded char_error_rate is blind to it.
        assert!((cased_cer("Abc", "abc") - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(char_error_rate("Abc", "abc"), 0.0);
    }

    #[test]
    fn concat_errors_zero_when_no_fusion() {
        assert_eq!(concat_errors("Make sure", "Make sure"), 0);
    }

    #[test]
    fn concat_errors_flags_fused_words() {
        // The MakeSure signature: lowercase immediately followed by uppercase.
        assert_eq!(concat_errors("MakeSure", "Make sure"), 1);
        assert_eq!(concat_errors("oneTwoThree", "one two three"), 2);
    }

    #[test]
    fn concat_errors_ignores_casing_only_defect() {
        // "So Let's" is a double-capital defect (space-separated), not a fusion;
        // that belongs to cased_cer, not here.
        assert_eq!(concat_errors("So Let's", "So let's"), 0);
    }

    #[test]
    fn concat_errors_preserves_legit_camelcase() {
        // camelCase already in the reference is not a regression.
        assert_eq!(concat_errors("iPhone app", "iPhone app"), 0);
    }

    #[test]
    fn spacing_delta_zero_for_identical() {
        assert_eq!(spacing_delta("Make sure it works", "Make sure it works"), 0);
    }

    #[test]
    fn spacing_delta_negative_for_merge() {
        assert_eq!(spacing_delta("MakeSure it works", "Make sure it works"), -1);
    }

    #[test]
    fn spacing_delta_positive_for_split() {
        assert_eq!(spacing_delta("a b c", "a b"), 1);
    }

    #[test]
    fn spacing_delta_ignores_punctuation() {
        assert_eq!(spacing_delta("hello, world.", "hello world"), 0);
    }
}
