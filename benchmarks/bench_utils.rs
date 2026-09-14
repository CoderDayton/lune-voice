//! Shared helpers for pipeline benchmarks.
//!
//! Included via `#[path = "../bench_utils.rs"] mod bench_utils;` from each
//! example (benchmarks live in pipeline/, quality/, tuning/, and tools/).
//!
//! ASR is whisper.cpp (`AsrBackendConfig::WhisperCpp`). The styling LLM is a
//! remote OpenAI-compatible endpoint (`LUNE_BENCH_LLM_*` env vars — point it
//! at Ollama, llama-server, LM Studio, or a hosted API) wrapped in
//! [`lune_llm::StylePipeline`]. There is no ORT runtime and no
//! correction/edit gate anymore.

use std::path::PathBuf;
use std::sync::Arc;

use lune_core::asr::AsrBackendKind;
use lune_core::config::{AsrBackendConfig, DevicePolicy, UserSettings, WhisperModel};
use lune_core::llm::{LlmService, TextPostProcessor};
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

// ─── Device memory ──────────────────────────────────────────────────────────

/// Device memory this process holds, in MiB.
///
/// Per-process rather than per-GPU: a desktop has a compositor and a browser on
/// the same card, and their allocations would swamp the signal. `nvidia-smi` is
/// the only source for this that does not require linking NVML.
///
/// `None` means the question could not be answered — no `nvidia-smi`, or no
/// compute context yet (before the first CUDA allocation the process is not
/// listed at all, which is not zero and must not be reported as zero).
pub fn vram_mb() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let pid = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|line| {
            let (p, used) = line.split_once(',')?;
            (p.trim() == pid).then(|| used.trim().parse::<f64>().ok())?
        })
}

/// [`vram_mb`] or a dash, so a CPU run still prints a readable table.
pub fn vram_str() -> String {
    match vram_mb() {
        Some(mb) => format!("{mb:>7.0} MB"),
        None => "      — MB".to_string(),
    }
}

// ─── GPU detection ───────────────────────────────────────────────────────────

/// Whether this build can offload to CUDA. There is no ORT runtime probe
/// anymore — whisper.cpp CUDA is a compile-time decision gated on the
/// `whisper-cuda` feature. Returns `true` when compiled in; the backend still
/// falls back to CPU at runtime if CUDA is absent.
pub fn detect_gpu() -> bool {
    let has_gpu = cfg!(feature = "whisper-cuda");
    if has_gpu {
        println!("  [info] CUDA feature compiled in (whisper offload enabled)");
    } else {
        println!("  [info] CPU-only build (no CUDA feature)");
    }
    has_gpu
}

// ─── ASR benchmark selection ────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BenchmarkAsrEntry {
    pub display_name: &'static str,
    pub backend_kind: AsrBackendKind,
    pub config: AsrBackendConfig,
    /// The model actually loaded, and how it was chosen. Reported in every
    /// [`BenchReport`] so two runs are never compared without knowing whether
    /// the model was pinned or picked up off disk.
    pub model: WhisperModel,
    pub model_source: ModelSource,
}

/// How the benchmarked whisper model was chosen.
///
/// Only [`ModelSource::Env`] and [`ModelSource::Flag`] are reproducible: the
/// other two depend on what happens to be installed, so two machines running
/// the same command can silently benchmark different models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSource {
    /// Pinned via the `WHISPER_MODEL` env var.
    Env,
    /// Pinned via `--whisper=<v>` / `--whisper <v>`.
    Flag,
    /// Unpinned — the first model found on disk, ascending by size.
    DiskFallback,
    /// Unpinned and nothing on disk — `WhisperModel::default()`.
    Default,
}

impl ModelSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Env => "WHISPER_MODEL",
            Self::Flag => "--whisper",
            Self::DiskFallback => "disk-fallback",
            Self::Default => "compiled-default",
        }
    }

    /// True when the caller named the model explicitly, so the run is
    /// reproducible on another machine.
    pub fn is_pinned(self) -> bool {
        matches!(self, Self::Env | Self::Flag)
    }

    fn from_label(raw: &str) -> Option<Self> {
        match raw {
            "WHISPER_MODEL" => Some(Self::Env),
            "--whisper" => Some(Self::Flag),
            "disk-fallback" => Some(Self::DiskFallback),
            "compiled-default" => Some(Self::Default),
            _ => None,
        }
    }
}

/// Parse a whisper model override from `WHISPER_MODEL` (or `--whisper` /
/// `--whisper=<v>`). Accepts `small`, `medium`, and `large`/`turbo`.
fn parse_whisper_model(raw: &str) -> Option<WhisperModel> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "small" => Some(WhisperModel::Small),
        "medium" => Some(WhisperModel::Medium),
        "large" | "largev3turbo" | "large-v3-turbo" | "turbo" => Some(WhisperModel::LargeV3Turbo),
        _ => None,
    }
}

fn requested_whisper_model() -> Option<(WhisperModel, ModelSource)> {
    if let Ok(raw) = std::env::var("WHISPER_MODEL") {
        if let Some(m) = parse_whisper_model(&raw) {
            return Some((m, ModelSource::Env));
        }
        eprintln!(
            "  [warn] Ignoring unsupported WHISPER_MODEL={raw:?}; expected small|medium|large"
        );
    }
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--whisper=") {
            return parse_whisper_model(value).map(|m| (m, ModelSource::Flag));
        }
        if arg == "--whisper" {
            if let Some(value) = args.next() {
                return parse_whisper_model(&value).map(|m| (m, ModelSource::Flag));
            }
        }
    }
    None
}

/// True when the GGML file for `model` is present under `models/`.
fn whisper_model_on_disk(model: WhisperModel) -> bool {
    models_dir()
        .join(model.dir())
        .join(model.ggml_file())
        .is_file()
}

/// Pick the whisper model to benchmark: an explicit `WHISPER_MODEL` override if
/// set, else the first model present on disk (ascending size), else the
/// compiled default. Returns how the choice was made so callers can warn when
/// the run is not reproducible.
fn selected_whisper_model() -> (WhisperModel, ModelSource) {
    if let Some(picked) = requested_whisper_model() {
        return picked;
    }
    match WhisperModel::ALL
        .into_iter()
        .find(|&m| whisper_model_on_disk(m))
    {
        Some(m) => (m, ModelSource::DiskFallback),
        None => (WhisperModel::default(), ModelSource::Default),
    }
}

/// Build the WhisperCpp ASR entry for the benchmark, or describe why it can't.
///
/// Prints the selected model and its provenance. An unpinned selection is
/// warned about loudly: it depends on what is installed, so the same command
/// on another machine can benchmark a different model and produce numbers that
/// look comparable but are not.
pub fn benchmark_asr_entry() -> Result<BenchmarkAsrEntry, String> {
    let (model, model_source) = selected_whisper_model();
    if !whisper_model_on_disk(model) {
        return Err(format!(
            "Whisper model {} not found. Expected {:?} under {:?}.",
            model.display_name(),
            model.ggml_file(),
            models_dir().join(model.dir())
        ));
    }
    if model_source.is_pinned() {
        println!(
            "  [info] Whisper model: {} (pinned via {})",
            model.display_name(),
            model_source.label()
        );
    } else {
        eprintln!(
            "  [warn] Whisper model: {} chosen by {} — NOT reproducible.\n         \
             Results are only comparable across runs that used the same model.\n         \
             Pin it: WHISPER_MODEL=small|medium|large (or --whisper=<v>).",
            model.display_name(),
            model_source.label()
        );
    }
    Ok(BenchmarkAsrEntry {
        display_name: model.display_name(),
        backend_kind: AsrBackendKind::WhisperCpp,
        config: AsrBackendConfig::WhisperCpp { model },
        model,
        model_source,
    })
}

/// Names that must come first when present, in this order. Everything else
/// is appended alphabetically so the noisified variants cluster under their
/// clean source (`librispeech_clean`, `librispeech_clean_noise_snr10`, …).
const PRIORITY_FIXTURE_NAMES: &[&str] = &["librispeech_clean", "librispeech_40744"];

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

// ─── Result reporting & regression gate ─────────────────────────────────────
//
// Benchmarks print to stdout, which cannot fail a build. These types add the
// two missing halves: a machine-readable report, and a comparison against a
// stored baseline that yields a non-zero exit code when accuracy regresses.

/// One fixture's scores. `wer`/`cer` are percentages (0–100), matching what the
/// benches already print.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricRecord {
    pub id: String,
    pub category: String,
    pub wer: f64,
    pub cer: f64,
    pub exact_match: bool,
}

/// A whole benchmark run, keyed by the model it used.
#[derive(Clone, Debug)]
pub struct BenchReport {
    pub bench: String,
    pub model: String,
    pub model_source: ModelSource,
    pub fixtures: Vec<MetricRecord>,
}

/// A metric that got worse than the baseline allows.
#[derive(Clone, Debug, PartialEq)]
pub struct Regression {
    pub id: String,
    pub metric: &'static str,
    pub baseline: f64,
    pub current: f64,
}

impl std::fmt::Display for Regression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.metric {
            "missing" => write!(f, "{}: present in baseline, absent from this run", self.id),
            "exact_match" => write!(f, "{}: lost exact match", self.id),
            m => write!(
                f,
                "{}: {m} {:.1}% -> {:.1}% (+{:.1})",
                self.id,
                self.baseline,
                self.current,
                self.current - self.baseline
            ),
        }
    }
}

impl BenchReport {
    /// Unweighted mean WER across fixtures, matching the per-fixture average
    /// the benches print as "Overall".
    pub fn overall_wer(&self) -> f64 {
        if self.fixtures.is_empty() {
            return 0.0;
        }
        self.fixtures.iter().map(|r| r.wer).sum::<f64>() / self.fixtures.len() as f64
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "bench": self.bench,
            "model": self.model,
            "model_source": self.model_source.label(),
            "overall_wer": self.overall_wer(),
            "fixtures": self.fixtures.iter().map(|r| serde_json::json!({
                "id": r.id,
                "category": r.category,
                "wer": r.wer,
                "cer": r.cer,
                "exact_match": r.exact_match,
            })).collect::<Vec<_>>(),
        })
    }

    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        let fixtures = v
            .get("fixtures")?
            .as_array()?
            .iter()
            .map(|e| {
                Some(MetricRecord {
                    id: e.get("id")?.as_str()?.to_string(),
                    category: e.get("category")?.as_str()?.to_string(),
                    wer: e.get("wer")?.as_f64()?,
                    cer: e.get("cer")?.as_f64()?,
                    exact_match: e.get("exact_match")?.as_bool()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            bench: v.get("bench")?.as_str()?.to_string(),
            model: v.get("model")?.as_str()?.to_string(),
            model_source: v
                .get("model_source")
                .and_then(|s| s.as_str())
                .and_then(ModelSource::from_label)
                .unwrap_or(ModelSource::Default),
            fixtures,
        })
    }
}

/// Compare `current` against `baseline`, allowing each fixture's WER and CER to
/// worsen by up to `tolerance_pts` percentage points.
///
/// Errors when the two runs used different models: those numbers are not
/// comparable, and silently passing the comparison would be worse than not
/// having one. Fixtures new in `current` are ignored; fixtures missing from it
/// are reported, so losing coverage cannot read as a pass.
pub fn compare_to_baseline(
    current: &BenchReport,
    baseline: &BenchReport,
    tolerance_pts: f64,
) -> Result<Vec<Regression>, String> {
    if current.model != baseline.model {
        return Err(format!(
            "Model mismatch: baseline ran {:?}, this run ran {:?}. \
             WER is not comparable across models — re-record the baseline, \
             or pin WHISPER_MODEL to {:?}.",
            baseline.model, current.model, baseline.model
        ));
    }

    let mut regressions = Vec::new();
    for base in &baseline.fixtures {
        let Some(cur) = current.fixtures.iter().find(|r| r.id == base.id) else {
            regressions.push(Regression {
                id: base.id.clone(),
                metric: "missing",
                baseline: base.wer,
                current: f64::NAN,
            });
            continue;
        };
        if cur.wer > base.wer + tolerance_pts {
            regressions.push(Regression {
                id: base.id.clone(),
                metric: "wer",
                baseline: base.wer,
                current: cur.wer,
            });
        }
        if cur.cer > base.cer + tolerance_pts {
            regressions.push(Regression {
                id: base.id.clone(),
                metric: "cer",
                baseline: base.cer,
                current: cur.cer,
            });
        }
        if base.exact_match && !cur.exact_match {
            regressions.push(Regression {
                id: base.id.clone(),
                metric: "exact_match",
                baseline: 1.0,
                current: 0.0,
            });
        }
    }
    Ok(regressions)
}

/// Env var naming where to write this run's JSON report.
pub const LUNE_BENCH_JSON: &str = "LUNE_BENCH_JSON";
/// Env var naming a previous report to gate against.
pub const LUNE_BENCH_BASELINE: &str = "LUNE_BENCH_BASELINE";
/// Env var overriding the allowed per-fixture worsening, in points.
pub const LUNE_BENCH_TOLERANCE: &str = "LUNE_BENCH_TOLERANCE";

fn baseline_tolerance() -> f64 {
    std::env::var(LUNE_BENCH_TOLERANCE)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(1.0)
}

/// Write `report` to `LUNE_BENCH_JSON` when set, then gate it against
/// `LUNE_BENCH_BASELINE` when set. Returns the process exit code — `1` if the
/// baseline comparison found regressions or could not be performed.
///
/// Both steps are opt-in, so an interactive run behaves exactly as before.
pub fn finish_report(report: &BenchReport) -> i32 {
    if let Ok(path) = std::env::var(LUNE_BENCH_JSON) {
        let json = serde_json::to_string_pretty(&report.to_json()).expect("serialize report");
        match std::fs::write(&path, json) {
            Ok(()) => println!("  Wrote report: {path}"),
            Err(e) => eprintln!("  [warn] Could not write {path}: {e}"),
        }
    }

    let Ok(baseline_path) = std::env::var(LUNE_BENCH_BASELINE) else {
        return 0;
    };

    let baseline = match std::fs::read_to_string(&baseline_path)
        .map_err(|e| format!("read {baseline_path}: {e}"))
        .and_then(|t| {
            serde_json::from_str::<serde_json::Value>(&t)
                .map_err(|e| format!("parse {baseline_path}: {e}"))
        })
        .and_then(|v| {
            BenchReport::from_json(&v).ok_or_else(|| format!("{baseline_path}: not a bench report"))
        }) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  [FAIL] Baseline unusable — {e}");
            return 1;
        }
    };

    let tolerance = baseline_tolerance();
    match compare_to_baseline(report, &baseline, tolerance) {
        Err(e) => {
            eprintln!("  [FAIL] {e}");
            1
        }
        Ok(regressions) if regressions.is_empty() => {
            println!(
                "  Baseline OK: {} fixtures within {:.1} pts of {}",
                baseline.fixtures.len(),
                tolerance,
                baseline_path
            );
            0
        }
        Ok(regressions) => {
            eprintln!(
                "  [FAIL] {} regression(s) beyond {:.1} pts vs {}:",
                regressions.len(),
                tolerance,
                baseline_path
            );
            for r in &regressions {
                eprintln!("    {r}");
            }
            1
        }
    }
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

// ─── BenchmarkHarness ────────────────────────────────────────────────────────

/// Consolidated setup for benchmark binaries.
///
/// Build via [`BenchmarkHarness::builder()`]. Owns the tokio runtime, loaded
/// whisper.cpp ASR and (optional) remote OpenAI-compatible styling LLM, and
/// the pre-filtered fixture list. All fields are `pub` for direct use in
/// benchmark code.
pub struct BenchmarkHarness {
    pub rt: tokio::runtime::Runtime,
    pub asr: Arc<lune_asr::AsrRegistry>,
    /// The resolved ASR selection, including which whisper model was loaded and
    /// how it was chosen. Read this instead of calling [`benchmark_asr_entry`]
    /// again — a second call re-resolves and re-prints the selection.
    pub asr_entry: BenchmarkAsrEntry,
    pub llm: Arc<dyn TextPostProcessor>,
    pub fixtures: Vec<Fixture>,
    pub has_gpu: bool,
    pub models_dir: PathBuf,
}

/// Env vars that configure the remote OpenAI-compatible styling LLM the
/// benchmarks talk to — point them at Ollama, llama-server, LM Studio, or a
/// hosted API. Named consts so the required configuration lives in one place.
pub const LUNE_BENCH_LLM_BASE_URL: &str = "LUNE_BENCH_LLM_BASE_URL";
pub const LUNE_BENCH_LLM_MODEL: &str = "LUNE_BENCH_LLM_MODEL";
pub const LUNE_BENCH_LLM_API_KEY: &str = "LUNE_BENCH_LLM_API_KEY";

/// Build a remote styling-LLM config from the `LUNE_BENCH_LLM_*` env vars.
/// Returns `None` when the endpoint or model is unset, so callers can skip
/// the styling pass with an explicit message instead of failing the run.
pub fn remote_llm_config_from_env() -> Option<lune_llm::OpenAiLlmConfig> {
    let base_url = std::env::var(LUNE_BENCH_LLM_BASE_URL).ok()?;
    let model = std::env::var(LUNE_BENCH_LLM_MODEL).ok()?;
    let api_key = std::env::var(LUNE_BENCH_LLM_API_KEY).unwrap_or_default();
    Some(lune_llm::OpenAiLlmConfig {
        base_url,
        api_key: lune_core::config::RedactedString::new(api_key),
        model,
        max_tokens: 512,
        temperature: 0.2,
    })
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
    ) -> lune_voice::SessionManager<lune_asr::AsrRegistry> {
        use std::sync::Arc;
        use tokio::sync::{Mutex, RwLock};

        let db = lune_persist::Database::open_in_memory().expect("in-memory DB");
        let dict = lune_dict::UserDictionary::default();

        let config = lune_voice::SessionManagerConfig {
            asr_backend_kind: self.asr.kind(),
            pipeline,
            ..Default::default()
        };

        lune_voice::SessionManager::new(
            Arc::clone(&self.asr),
            Arc::new(lune_dict::commands::VoiceCommandRouter::new()),
            Arc::clone(&self.llm),
            Arc::new(dict),
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
    /// Load the remote OpenAI-compatible styling LLM (configured via
    /// `LUNE_BENCH_LLM_*` env vars) wrapped in [`lune_llm::StylePipeline`]
    /// (falls back to `NullPostProcessor` when unset). Without this the
    /// harness uses `NullPostProcessor`.
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

        let has_gpu = detect_gpu();
        // `Auto` defers to what ggml actually registered, so a CPU-only build
        // (or a machine with no plugin installed) stays on the CPU path.
        let device = DevicePolicy::Auto;

        let models_dir = models_dir();
        let cuda_available = AppPaths::resolve()
            .map(|p| lune_asr::gpu_backend::cuda_available(p.data_dir()))
            .unwrap_or(false);

        // ASR — whisper.cpp.
        let asr_entry = benchmark_asr_entry().unwrap_or_else(|msg| {
            panic!("BenchmarkHarness: cannot load ASR — {msg}");
        });
        let asr = lune_asr::AsrRegistry::from_config(
            &asr_entry.config,
            device,
            &models_dir,
            cuda_available,
        )
        .unwrap_or_else(|e| panic!("BenchmarkHarness: ASR load failed — {e}"));
        let asr = Arc::new(asr);

        // LLM — remote OpenAI-compatible styling backend behind StylePipeline.
        let llm: Arc<dyn TextPostProcessor> = if self.with_llm {
            let llm_off = std::env::var("LUNE_LLM")
                .map(|v| v == "off" || v == "0")
                .unwrap_or(false);
            match (llm_off, remote_llm_config_from_env()) {
                (false, Some(config)) => {
                    let svc: Arc<dyn LlmService> = Arc::new(
                        lune_llm::OpenAiLlmBackend::new(config)
                            .expect("bench LLM config must be valid"),
                    );
                    Arc::new(lune_llm::StylePipeline::new(svc))
                }
                (false, None) => {
                    eprintln!(
                        "[warn] {LUNE_BENCH_LLM_BASE_URL}/{LUNE_BENCH_LLM_MODEL} not set, using NullPostProcessor"
                    );
                    Arc::new(lune_llm::NullPostProcessor::new())
                }
                (true, _) => Arc::new(lune_llm::NullPostProcessor::new()),
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
            asr_entry,
            llm,
            fixtures,
            has_gpu,
            models_dir,
        }
    }
}

// ─── Baseline gate tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod baseline_tests {
    use super::{BenchReport, MetricRecord, ModelSource, compare_to_baseline};

    /// CER is held at zero so a WER assertion isolates WER; the CER path gets
    /// its own test via [`record_cer`].
    fn record(id: &str, wer: f64, exact: bool) -> MetricRecord {
        MetricRecord {
            id: id.to_string(),
            category: "quiet_dictation".to_string(),
            wer,
            cer: 0.0,
            exact_match: exact,
        }
    }

    fn record_cer(id: &str, cer: f64) -> MetricRecord {
        MetricRecord {
            cer,
            ..record(id, 0.0, true)
        }
    }

    fn report(model: &str, records: Vec<MetricRecord>) -> BenchReport {
        BenchReport {
            bench: "domain".to_string(),
            model: model.to_string(),
            model_source: ModelSource::Env,
            fixtures: records,
        }
    }

    #[test]
    fn identical_reports_have_no_regressions() {
        let base = report("Small", vec![record("a", 5.0, true)]);
        let cur = report("Small", vec![record("a", 5.0, true)]);
        assert_eq!(compare_to_baseline(&cur, &base, 1.0).unwrap(), Vec::new());
    }

    #[test]
    fn wer_within_tolerance_is_not_a_regression() {
        let base = report("Small", vec![record("a", 5.0, true)]);
        let cur = report("Small", vec![record("a", 5.9, true)]);
        assert_eq!(compare_to_baseline(&cur, &base, 1.0).unwrap(), Vec::new());
    }

    #[test]
    fn wer_beyond_tolerance_is_a_regression() {
        let base = report("Small", vec![record("a", 5.0, true)]);
        let cur = report("Small", vec![record("a", 7.5, true)]);
        let regs = compare_to_baseline(&cur, &base, 1.0).unwrap();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].id, "a");
        assert_eq!(regs[0].metric, "wer");
    }

    #[test]
    fn cer_beyond_tolerance_is_a_regression() {
        let base = report("Small", vec![record_cer("a", 2.0)]);
        let cur = report("Small", vec![record_cer("a", 4.0)]);
        let regs = compare_to_baseline(&cur, &base, 1.0).unwrap();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].metric, "cer");
    }

    #[test]
    fn improvement_is_never_a_regression() {
        let base = report("Small", vec![record("a", 9.0, true)]);
        let cur = report("Small", vec![record("a", 1.0, true)]);
        assert_eq!(compare_to_baseline(&cur, &base, 1.0).unwrap(), Vec::new());
    }

    #[test]
    fn losing_an_exact_match_is_a_regression() {
        // WER is unchanged, so only the exact-match flag catches this.
        let base = report("Small", vec![record("a", 0.0, true)]);
        let cur = report("Small", vec![record("a", 0.0, false)]);
        let regs = compare_to_baseline(&cur, &base, 1.0).unwrap();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].metric, "exact_match");
    }

    #[test]
    fn gaining_an_exact_match_is_not_a_regression() {
        let base = report("Small", vec![record("a", 0.0, false)]);
        let cur = report("Small", vec![record("a", 0.0, true)]);
        assert_eq!(compare_to_baseline(&cur, &base, 1.0).unwrap(), Vec::new());
    }

    #[test]
    fn a_dropped_fixture_is_a_regression() {
        // Silently losing coverage must not read as a pass.
        let base = report(
            "Small",
            vec![record("a", 5.0, true), record("b", 5.0, true)],
        );
        let cur = report("Small", vec![record("a", 5.0, true)]);
        let regs = compare_to_baseline(&cur, &base, 1.0).unwrap();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].id, "b");
        assert_eq!(regs[0].metric, "missing");
    }

    #[test]
    fn a_new_fixture_is_not_a_regression() {
        let base = report("Small", vec![record("a", 5.0, true)]);
        let cur = report(
            "Small",
            vec![record("a", 5.0, true), record("b", 90.0, false)],
        );
        assert_eq!(compare_to_baseline(&cur, &base, 1.0).unwrap(), Vec::new());
    }

    #[test]
    fn comparing_across_models_is_refused() {
        // The whole point of pinning the model: Small-vs-Medium numbers are
        // not comparable, so this must be an error, never a silent pass.
        let base = report("Small", vec![record("a", 5.0, true)]);
        let cur = report("Medium", vec![record("a", 5.0, true)]);
        let err = compare_to_baseline(&cur, &base, 1.0).unwrap_err();
        assert!(
            err.contains("Small"),
            "error names the baseline model: {err}"
        );
        assert!(
            err.contains("Medium"),
            "error names the current model: {err}"
        );
    }

    #[test]
    fn report_survives_a_json_round_trip() {
        let original = report(
            "Small",
            vec![record("a", 5.0, true), record("b", 0.0, false)],
        );
        let parsed = BenchReport::from_json(&original.to_json()).expect("parse");
        assert_eq!(parsed.bench, original.bench);
        assert_eq!(parsed.model, original.model);
        assert_eq!(parsed.fixtures, original.fixtures);
    }

    #[test]
    fn overall_wer_averages_fixtures() {
        let r = report(
            "Small",
            vec![record("a", 4.0, true), record("b", 6.0, true)],
        );
        assert!((r.overall_wer() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn overall_wer_of_an_empty_report_is_zero() {
        assert_eq!(report("Small", Vec::new()).overall_wer(), 0.0);
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
