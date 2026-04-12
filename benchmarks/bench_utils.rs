//! Shared helpers for pipeline benchmarks.
//!
//! Included via `#[path = "bench_utils.rs"] mod bench_utils;` from each example.

use std::path::PathBuf;

use lune_core::asr::AsrBackendKind;
use lune_core::config::AsrBackendConfig;

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
}

impl BenchmarkAsr {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "parakeet" | "parakeet-tdt" => Some(Self::Parakeet),
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
        eprintln!("  [warn] Ignoring unsupported ASR={raw:?}; expected parakeet");
    }

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--asr=") {
            if let Some(choice) = BenchmarkAsr::parse(value) {
                return Some(choice);
            }
            eprintln!("  [warn] Ignoring unsupported --asr={value:?}; expected parakeet");
            return None;
        }
        if arg == "--asr" {
            if let Some(value) = args.next() {
                if let Some(choice) = BenchmarkAsr::parse(&value) {
                    return Some(choice);
                }
                eprintln!("  [warn] Ignoring unsupported --asr {value:?}; expected parakeet");
            } else {
                eprintln!("  [warn] Ignoring bare --asr; expected parakeet");
            }
            return None;
        }
    }

    None
}

pub fn benchmark_asr_entries(root: &std::path::Path) -> Vec<BenchmarkAsrEntry> {
    let parakeet_dir = root.join("models/parakeet-tdt-0.6b");

    let mut entries = Vec::new();
    if parakeet_dir.exists() {
        entries.push(BenchmarkAsrEntry {
            display_name: "Parakeet-TDT-int8",
            backend_kind: AsrBackendKind::ParakeetTdt,
            config: AsrBackendConfig::ParakeetTdt {},
        });
    }
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

/// Names that must come first when present, in this order. Everything else
/// is appended alphabetically so the noisified variants cluster under their
/// clean source (`librispeech_clean`, `librispeech_clean_noise_snr10`, …).
const PRIORITY_FIXTURE_NAMES: &[&str] =
    &["librispeech_clean", "librispeech_40744", "custom_recording"];

pub fn load_fixtures(root: &std::path::Path) -> Vec<Fixture> {
    let fixtures_dir = root.join("test_fixtures");
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
