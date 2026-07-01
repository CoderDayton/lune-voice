use std::hint::black_box;
use std::time::{Duration, Instant};

use lune_dict::corrector::LexicalGate;

const DEFAULT_WARMUP_ITERS: usize = 2;
const DEFAULT_MEASURE_ITERS: usize = 10;

struct Workload {
    name: &'static str,
    text: &'static str,
    // (word, risk, mean_confidence). `mean_confidence = 1.0 - risk` keeps
    // the mean-vs-min corroboration in corrector.rs a no-op for these
    // synthetic workloads (matches today's pre-corroboration behavior).
    risks: &'static [(&'static str, f32, f32)],
    replacements: &'static [(&'static str, &'static str)],
    lang: &'static str,
}

struct BenchResult {
    name: &'static str,
    iterations: usize,
    chars_per_iter: usize,
    total: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    checksum: usize,
}

fn main() {
    lune_dict::lexicon::warm();
    let corrector = LexicalGate::new();
    let warmup_iters = env_usize("LUNE_BENCH_WARMUP", DEFAULT_WARMUP_ITERS);
    let measure_iters = env_usize("LUNE_BENCH_ITERS", DEFAULT_MEASURE_ITERS);

    println!("lune-dict corrector benchmark");
    println!("warmup_iters={warmup_iters} measure_iters={measure_iters}");
    println!();
    println!(
        "{:<22} {:>10} {:>12} {:>10} {:>10} {:>10} {:>12}",
        "workload", "chars", "chars/sec", "p50_us", "p95_us", "p99_us", "checksum"
    );

    for workload in workloads() {
        let result = run_workload(&corrector, &workload, warmup_iters, measure_iters);
        let chars_per_sec =
            (result.chars_per_iter * result.iterations) as f64 / result.total.as_secs_f64();
        println!(
            "{:<22} {:>10} {:>12.0} {:>10} {:>10} {:>10} {:>12}",
            result.name,
            result.chars_per_iter,
            chars_per_sec,
            result.p50.as_micros(),
            result.p95.as_micros(),
            result.p99.as_micros(),
            result.checksum,
        );
    }
}

fn run_workload(
    corrector: &LexicalGate,
    workload: &Workload,
    warmup_iters: usize,
    measure_iters: usize,
) -> BenchResult {
    let risks = owned_pairs_f32(workload.risks);
    let replacements = owned_pairs_str(workload.replacements);

    for _ in 0..warmup_iters {
        let result = corrector.correct(
            black_box(workload.text),
            black_box(&risks),
            black_box(&replacements),
            black_box(workload.lang),
        );
        black_box(result.corrected_text.len());
    }

    let mut samples = Vec::with_capacity(measure_iters);
    let mut checksum = 0usize;
    let total_start = Instant::now();

    for _ in 0..measure_iters {
        let start = Instant::now();
        let result = corrector.correct(
            black_box(workload.text),
            black_box(&risks),
            black_box(&replacements),
            black_box(workload.lang),
        );
        let elapsed = start.elapsed();
        checksum = checksum.wrapping_add(result.corrected_text.len());
        checksum = checksum.wrapping_add(result.unresolved_words.len());
        checksum = checksum.wrapping_add(usize::from(result.all_resolved));
        samples.push(elapsed);
    }

    let total = total_start.elapsed();
    samples.sort_unstable();

    BenchResult {
        name: workload.name,
        iterations: measure_iters,
        chars_per_iter: workload.text.len(),
        total,
        p50: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        p99: percentile(&samples, 99),
        checksum,
    }
}

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((samples.len() - 1) * percentile) / 100;
    samples[idx]
}

fn owned_pairs_f32(pairs: &[(&str, f32, f32)]) -> Vec<(String, f32, f32)> {
    pairs
        .iter()
        .map(|(word, risk, mean)| ((*word).to_owned(), *risk, *mean))
        .collect()
}

fn owned_pairs_str(pairs: &[(&str, &str)]) -> Vec<lune_dict::DictMatch> {
    pairs
        .iter()
        .map(|(from, to)| lune_dict::DictMatch {
            matched: (*from).to_owned(),
            replacement: (*to).to_owned(),
            fuzzy: false,
        })
        .collect()
}

fn workloads() -> [Workload; 5] {
    [
        Workload {
            name: "known_passthrough",
            text: "the quick brown fox jumps over the lazy dog and then writes a short note about the weather forecast",
            risks: &[],
            replacements: &[],
            lang: "en",
        },
        Workload {
            name: "common_typos",
            text: "teh quick brwon fox jumps ovre the lazi dog becuase the weahter report was wrong",
            risks: &[],
            replacements: &[],
            lang: "en",
        },
        Workload {
            name: "flagged_domain",
            text: "the websocket conection failed after roblox refreshed the auth token for lune voice",
            risks: &[
                ("websocket", 0.75, 0.25),
                ("conection", 0.90, 0.10),
                ("roblox", 0.70, 0.30),
                ("lune", 0.65, 0.35),
            ],
            replacements: &[],
            lang: "en",
        },
        Workload {
            name: "dict_replacements",
            text: "helo world please open teh github issue and tag lune voice in the summary",
            risks: &[("helo", 0.82, 0.18), ("teh", 0.78, 0.22)],
            replacements: &[("helo", "hello"), ("github", "GitHub")],
            lang: "en",
        },
        Workload {
            name: "french_passthrough",
            text: "bonjour le monde je voudrais dicter un message court avec une phrase simple",
            risks: &[],
            replacements: &[],
            lang: "fr",
        },
    ]
}
