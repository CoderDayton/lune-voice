//! Shared helpers for pipeline benchmarks.
//!
//! Included via `#[path = "bench_utils.rs"] mod bench_utils;` from each example.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lune_asr::{AsrRegistry, OrtForcedAligner};
use lune_core::asr::AsrBackendKind;
use lune_core::config::{AsrBackendConfig, UserSettings};

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
    Parakeet,
    Qwen,
}

impl BenchmarkAsr {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "parakeet" | "parakeet-tdt" => Some(Self::Parakeet),
            "qwen" | "ortqwen" | "ort-qwen" | "qwen3-asr" => Some(Self::Qwen),
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
        eprintln!("  [warn] Ignoring unsupported ASR={raw:?}; expected qwen or parakeet");
    }

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--asr=") {
            if let Some(choice) = BenchmarkAsr::parse(value) {
                return Some(choice);
            }
            eprintln!("  [warn] Ignoring unsupported --asr={value:?}; expected qwen or parakeet");
            return None;
        }
        if arg == "--asr" {
            if let Some(value) = args.next() {
                if let Some(choice) = BenchmarkAsr::parse(&value) {
                    return Some(choice);
                }
                eprintln!(
                    "  [warn] Ignoring unsupported --asr {value:?}; expected qwen or parakeet"
                );
            } else {
                eprintln!("  [warn] Ignoring bare --asr; expected qwen or parakeet");
            }
            return None;
        }
    }

    None
}

pub fn benchmark_asr_entries(root: &std::path::Path) -> Vec<BenchmarkAsrEntry> {
    let requested = requested_benchmark_asr();
    let parakeet_dir = root.join("models/parakeet-tdt-0.6b");
    let qwen_dir = root.join("models/qwen3-asr-0.6b-onnx");

    let mut entries = Vec::new();

    let push_if_selected = |entries: &mut Vec<BenchmarkAsrEntry>,
                            choice: BenchmarkAsr,
                            present: bool,
                            entry: BenchmarkAsrEntry| {
        if !present {
            return;
        }
        if requested.is_none() || requested == Some(choice) {
            entries.push(entry);
        }
    };

    push_if_selected(
        &mut entries,
        BenchmarkAsr::Parakeet,
        parakeet_dir.exists(),
        BenchmarkAsrEntry {
            display_name: "Parakeet-TDT-int8",
            backend_kind: AsrBackendKind::ParakeetTdt,
            config: AsrBackendConfig::ParakeetTdt {
                model_dir: parakeet_dir,
            },
        },
    );

    push_if_selected(
        &mut entries,
        BenchmarkAsr::Qwen,
        qwen_dir.exists(),
        BenchmarkAsrEntry {
            display_name: "OrtQwen-INT4-batch",
            backend_kind: AsrBackendKind::OrtQwenAsr,
            config: AsrBackendConfig::OrtQwenAsr {
                model_dir: qwen_dir,
            },
        },
    );

    entries
}

pub fn benchmark_asr_entry(root: &std::path::Path) -> Result<BenchmarkAsrEntry, String> {
    let requested = requested_benchmark_asr();
    benchmark_asr_entries(root)
        .into_iter()
        .next()
        .ok_or_else(|| match requested {
            Some(choice) => format!(
                "Requested ASR backend {:?} is unavailable. Expected model dir under models/.",
                choice
            ),
            None => "No ASR model dir found under models/.".to_string(),
        })
}

pub fn benchmark_aligner_enabled() -> bool {
    if let Ok(raw) = std::env::var("ALIGNER") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => return true,
            "0" | "false" | "off" => return false,
            _ => {
                eprintln!("  [warn] Ignoring unsupported ALIGNER={raw:?}; expected 0/1/false/true")
            }
        }
    }

    UserSettings::default()
        .aligner
        .as_ref()
        .map(|cfg| cfg.enabled)
        .unwrap_or(false)
}

pub fn benchmark_aligner(asr: &AsrRegistry, root: &Path) -> Option<Arc<OrtForcedAligner>> {
    if !benchmark_aligner_enabled() {
        return None;
    }

    let Some(backend) = asr.as_ort_qwen() else {
        return None;
    };

    let fa_dir = root.join("models/wav2vec2-fa");
    let fa_model = fa_dir.join("model.onnx");
    let fa_config = fa_dir.join("fa_config.json");

    if !fa_model.exists() || !fa_config.exists() {
        eprintln!(
            "  [warn] ALIGNER enabled but {} is missing model.onnx/fa_config.json; skipping",
            fa_dir.display()
        );
        return None;
    }

    match backend.make_forced_aligner(&fa_dir) {
        Ok(fa) => {
            let fa = Arc::new(fa);
            backend.attach_aligner(Arc::clone(&fa));
            println!("  Forced aligner: attached ({})", fa_dir.display());
            Some(fa)
        }
        Err(error) => {
            eprintln!(
                "  [warn] forced aligner load failed at {}: {error}",
                fa_dir.display()
            );
            None
        }
    }
}

pub fn maybe_attach_benchmark_aligner(asr: &AsrRegistry, root: &Path) {
    let _ = benchmark_aligner(asr, root);
}

pub fn load_fixtures(root: &std::path::Path) -> Vec<Fixture> {
    let pairs = [
        (
            "librispeech_clean",
            "test_fixtures/librispeech_clean.wav",
            "test_fixtures/librispeech_clean.txt",
        ),
        (
            "librispeech_40744",
            "test_fixtures/librispeech_40744.wav",
            "test_fixtures/librispeech_40744.txt",
        ),
        (
            "custom_recording",
            "test_fixtures/custom_recording.wav",
            "test_fixtures/custom_recording.txt",
        ),
    ];
    let mut fixtures = Vec::new();
    for (name, wav, txt) in &pairs {
        let wav_path = root.join(wav);
        let txt_path = root.join(txt);
        if wav_path.exists() && txt_path.exists() {
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
            fixtures.push(Fixture {
                name,
                pcm,
                reference,
                audio_secs,
            });
        }
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

/// Strip punctuation, uppercase, collapse whitespace.
pub fn normalize_text(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphabetic() {
                c.to_ascii_uppercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
    let ok = ort::init()
        .with_execution_providers([ort::ep::CUDA::default().build()])
        .commit();
    if ok {
        println!("  [info] ORT: CUDA EP active");
    } else {
        let _ = ort::init().commit();
        println!("  [info] ORT: CPU only (CUDA not available)");
    }
    ok
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
