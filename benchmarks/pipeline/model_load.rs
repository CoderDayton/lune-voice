// SPDX-License-Identifier: GPL-3.0-or-later

//! Model load / unload timing probe.
//!
//! Answers the one question an idle-unload feature cannot be designed without:
//! **what does the user pay on the first hotkey press after the model was
//! dropped?** `prepare_manager_for_start` rebuilds the backend *before* the
//! session opens, so that cost lands between the keypress and the mic — it is
//! silence the user speaks into, not background work.
//!
//! Three numbers per model, because they are not interchangeable:
//!
//! - **cold** — the model file evicted from the OS page cache first. This is
//!   the honest worst case: a machine that idled long enough to unload has
//!   usually also had the file evicted under memory pressure.
//! - **warm** — file still cached. The best case, and the one a short timeout
//!   on a roomy machine actually hits.
//! - **unload** — how long `drop` takes. `release_backend` awaits this on a
//!   blocking thread because a CUDA free of gigabytes is not instant.
//!
//! Timing covers `AsrRegistry::from_config`, which loads the whisper model
//! *and* the silero VAD — exactly the pair the app reloads, so the number
//! needs no adjustment before being used as a timeout budget.
//!
//! Run: `cgs bench --env TYPE=modelload` (add `--env REPEATS=n` for more
//! samples, `--env WHISPER_MODEL=base|small|medium|large` to probe one model
//! instead of every model on disk).

#[path = "../bench_utils.rs"]
#[allow(dead_code)]
mod bench_utils;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bench_utils::{models_dir, percentile, process_memory_usage, vram_mb};
use lune_asr::AsrRegistry;
use lune_core::config::{AsrBackendConfig, DevicePolicy, WhisperModel};

/// Samples per phase. A load is seconds long and its variance is dominated by
/// the disk and the allocator, so a handful of runs pins the median without
/// making the probe take a coffee break.
const DEFAULT_REPEATS: usize = 3;

fn model_file(model: WhisperModel) -> PathBuf {
    models_dir().join(model.dir()).join(model.ggml_file())
}

fn vad_file() -> PathBuf {
    models_dir()
        .join(lune_core::models::DIR_WHISPER_VAD)
        .join(lune_core::models::WHISPER_VAD_GGML)
}

fn file_mb(path: &Path) -> f64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f64 / (1024.0 * 1024.0)
}

/// Drop a file's pages from the OS page cache, so the next read hits the disk.
///
/// `POSIX_FADV_DONTNEED` does this for one file without root — unlike
/// `vm.drop_caches`, which needs privileges and empties the cache for the
/// whole system. It only evicts *clean* pages, which is all a model file ever
/// has: it is written once by the downloader and read-only thereafter.
///
/// Best-effort by design. `false` means the pages may still be cached, so the
/// "cold" number is really a second warm one — reported rather than hidden,
/// because a cold row that is secretly warm would understate the budget.
#[cfg(unix)]
fn evict_page_cache(path: &Path) -> bool {
    use std::os::unix::io::AsRawFd;
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    // SAFETY: `file` owns a live descriptor for the whole call, and
    // `posix_fadvise` only advises the kernel about it — no memory is touched.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    rc == 0
}

#[cfg(not(unix))]
fn evict_page_cache(_path: &Path) -> bool {
    false
}

/// Where the model is loaded. Taken from `DEVICE` so a single CUDA build can
/// measure both halves of the "unload it, or run it on the CPU?" question —
/// two builds would differ in more than the device and not be comparable.
///
/// `Cuda` still falls back to the CPU when no device registered, which is why
/// the header prints what actually happened rather than what was asked for.
fn device_policy() -> DevicePolicy {
    match std::env::var("DEVICE")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => DevicePolicy::Cpu,
        "cuda" | "gpu" => DevicePolicy::Cuda,
        "" | "auto" => DevicePolicy::Auto,
        other => {
            eprintln!("  [warn] ignoring DEVICE={other:?}; expected cpu|cuda|auto");
            DevicePolicy::Auto
        }
    }
}

/// The device the run is pinned to, plus what the machine can actually offer.
/// Carried together because `use_gpu` is decided from both.
#[derive(Clone, Copy)]
struct Target {
    device: DevicePolicy,
    cuda: bool,
}

impl Target {
    /// What the load will really do, as opposed to what `DEVICE` asked for.
    fn on_gpu(self) -> bool {
        self.device.use_gpu(self.cuda)
    }

    fn label(self) -> &'static str {
        if self.on_gpu() { "CUDA" } else { "CPU" }
    }
}

fn load(model: WhisperModel, target: Target) -> AsrRegistry {
    AsrRegistry::from_config(
        &AsrBackendConfig::WhisperCpp { model },
        target.device,
        &models_dir(),
        target.cuda,
    )
    .unwrap_or_else(|e| panic!("load {}: {e}", model.display_name()))
}

fn on_disk(model: WhisperModel) -> bool {
    model_file(model).is_file()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// What one model cost, in the shape the summary table prints.
struct Timing {
    model: WhisperModel,
    size_mb: f64,
    cold_ms: Option<f64>,
    warm_ms: f64,
    unload_ms: f64,
    rss_delta_mb: f64,
    vram_delta_mb: Option<f64>,
}

/// Time `repeats` load/unload pairs, evicting the page cache first when
/// `cold`. Returns the median load and median unload in ms.
///
/// The backend is dropped inside the loop, not collected and freed at the end:
/// holding several would stack whole models in device memory and turn a timing
/// probe into an OOM.
fn time_phase(model: WhisperModel, target: Target, repeats: usize, cold: bool) -> (f64, f64, bool) {
    let mut loads = Vec::with_capacity(repeats);
    let mut unloads = Vec::with_capacity(repeats);
    let mut evicted = true;
    for _ in 0..repeats {
        if cold {
            // The VAD is loaded by the same call, so leaving it cached would
            // make the cold row a blend of a cold model and a warm VAD.
            evicted &= evict_page_cache(&model_file(model));
            evicted &= evict_page_cache(&vad_file());
        }
        let t0 = Instant::now();
        let asr = load(model, target);
        loads.push(ms(t0.elapsed()));
        let t1 = Instant::now();
        drop(asr);
        unloads.push(ms(t1.elapsed()));
    }
    (
        percentile(&loads, 50.0),
        percentile(&unloads, 50.0),
        cold && evicted,
    )
}

/// RSS and VRAM held by one loaded model, measured on a dedicated load so the
/// deltas are not contaminated by a previous backend still being freed.
fn measure_footprint(model: WhisperModel, target: Target) -> (f64, Option<f64>) {
    let rss_before = process_memory_usage().map(|m| m.rss_mb).unwrap_or(0.0);
    let vram_before = vram_mb();
    let asr = load(model, target);
    let rss_after = process_memory_usage().map(|m| m.rss_mb).unwrap_or(0.0);
    let vram_after = vram_mb();
    drop(asr);
    let vram_delta = match (vram_before, vram_after) {
        // `None` before the first CUDA allocation means "not listed yet", which
        // is not zero — but as the *baseline* it is, so treat it as such.
        (before, Some(after)) => Some(after - before.unwrap_or(0.0)),
        _ => None,
    };
    (rss_after - rss_before, vram_delta)
}

fn probe(model: WhisperModel, target: Target, repeats: usize) -> Timing {
    let path = model_file(model);
    let size_mb = file_mb(&path);
    let heading = format!("── {} ({size_mb:.0} MB) ", model.display_name());
    println!(
        "{heading}{}",
        "─".repeat(60usize.saturating_sub(heading.chars().count()))
    );

    let (cold_load, _, evicted) = time_phase(model, target, repeats, true);
    let (warm_load, unload, _) = time_phase(model, target, repeats, false);
    let (rss_delta_mb, vram_delta_mb) = measure_footprint(model, target);

    if evicted {
        println!("  cold load    {cold_load:>8.0} ms   (page cache evicted)");
    } else {
        println!(
            "  cold load    {cold_load:>8.0} ms   [warn] eviction failed — this is a warm number"
        );
    }
    println!("  warm load    {warm_load:>8.0} ms   (median of {repeats})");
    println!("  unload       {unload:>8.0} ms");
    print!("  footprint    {rss_delta_mb:>8.0} MB RSS");
    match vram_delta_mb {
        Some(v) => println!("   {v:.0} MB VRAM"),
        None => println!("   — VRAM"),
    }
    println!();

    Timing {
        model,
        size_mb,
        cold_ms: evicted.then_some(cold_load),
        warm_ms: warm_load,
        unload_ms: unload,
        rss_delta_mb,
        vram_delta_mb,
    }
}

fn main() {
    let cuda = lune_asr::gpu_backend::gpu_available(
        lune_persist::AppPaths::resolve()
            .expect("resolve XDG paths")
            .data_dir(),
    );
    let repeats: usize = std::env::var("REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(DEFAULT_REPEATS);

    let target = Target {
        device: device_policy(),
        cuda,
    };

    println!("\n=== Model load / unload timing ===\n");
    println!(
        "  device: {}   repeats: {repeats}   models: {}\n",
        target.label(),
        models_dir().display()
    );
    if matches!(target.device, DevicePolicy::Cuda) && !cuda {
        println!("  [warn] DEVICE=cuda but no CUDA device registered — this ran on the CPU\n");
    }
    if !vad_file().is_file() {
        println!("  [warn] silero VAD missing — loads exclude it and read low\n");
    }

    // An explicit `WHISPER_MODEL` probes one model; otherwise every model on
    // disk, so a single run produces the whole table the timeout is chosen from.
    let requested = std::env::var("WHISPER_MODEL").ok().and_then(|v| {
        match v.trim().to_ascii_lowercase().as_str() {
            "base" => Some(WhisperModel::Base),
            "small" => Some(WhisperModel::Small),
            "medium" => Some(WhisperModel::Medium),
            "large" | "large-v3-turbo" | "turbo" => Some(WhisperModel::LargeV3Turbo),
            other => {
                eprintln!("  [warn] ignoring WHISPER_MODEL={other:?}");
                None
            }
        }
    });

    let models: Vec<WhisperModel> = match requested {
        Some(m) => vec![m],
        None => WhisperModel::ALL.into_iter().collect(),
    };
    let present: Vec<WhisperModel> = models.into_iter().filter(|&m| on_disk(m)).collect();
    if present.is_empty() {
        println!("No whisper models on disk under {:?}.", models_dir());
        return;
    }

    let timings: Vec<Timing> = present
        .into_iter()
        .map(|m| probe(m, target, repeats))
        .collect();

    summary(&timings);
}

fn summary(timings: &[Timing]) {
    println!("── Summary ─────────────────────────────────────────────────\n");
    println!(
        "  {:<30} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "model", "size", "cold", "warm", "unload", "RSS", "VRAM"
    );
    for t in timings {
        let cold = match t.cold_ms {
            Some(c) => format!("{c:.0}ms"),
            None => "—".to_string(),
        };
        let vram = match t.vram_delta_mb {
            Some(v) => format!("{v:.0}MB"),
            None => "—".to_string(),
        };
        println!(
            "  {:<30} {:>5.0}MB {:>8} {:>6.0}ms {:>6.0}ms {:>6.0}MB {:>8}",
            t.model.display_name(),
            t.size_mb,
            cold,
            t.warm_ms,
            t.unload_ms,
            t.rss_delta_mb,
            vram,
        );
    }

    // The reason the probe exists: `cold` is what the first hotkey press after
    // an idle unload costs, and it is paid before the mic opens.
    let worst = timings
        .iter()
        .filter_map(|t| t.cold_ms)
        .fold(f64::NAN, f64::max);
    println!();
    if worst.is_finite() {
        println!("  Worst cold load: {worst:.0} ms of silence between the hotkey and the mic,");
        println!("  unless the start path buffers audio while the backend rebuilds.");
    } else {
        println!("  No cold numbers (page-cache eviction unavailable) — warm times only.");
    }
    println!();
}
