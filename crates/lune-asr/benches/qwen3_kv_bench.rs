//! Qwen3 decoder KV-cache benchmark.
//!
//! Compares normal decoder runs against GPU/ORT IO binding using the same
//! fixtures and prints Qwen-internal stage timings.
//!
//! Run: cargo run --release --features cuda --example qwen3_kv_bench -p lune-asr
//! Filter: LUNE_BENCH_FIXTURE=librispeech cargo run --release --features cuda --example qwen3_kv_bench -p lune-asr

use std::path::{Path, PathBuf};

use lune_core::asr::{AsrBackend, AsrSessionConfig};

struct Fixture {
    name: String,
    pcm: Vec<f32>,
    audio_secs: f64,
}

struct RunRow {
    fixture: String,
    text: String,
    profile: lune_asr::Qwen3AsrProfile,
}

fn main() {
    let _ = ort::init().commit();

    let root = workspace_root();
    let fixtures = load_fixtures(&root);
    if fixtures.is_empty() {
        eprintln!("[skip] no WAV fixtures found in fixtures/audio/");
        return;
    }

    let data_dir = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".local/share")
        });
    let model_dir = data_dir
        .join("lune-voice/models")
        .join(lune_core::models::DIR_QWEN3_ASR);
    let use_gpu = cfg!(feature = "cuda");
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    println!("Qwen3 KV benchmark");
    println!("  model: {}", model_dir.display());
    println!("  device: {}", if use_gpu { "cuda" } else { "cpu" });
    println!("  fixtures: {}", fixtures.len());
    println!();

    let plain = run_variant(&rt, &model_dir, use_gpu, false, &fixtures);
    let bound = run_variant(&rt, &model_dir, use_gpu, true, &fixtures);
    print_comparison(&plain, &bound);
}

fn run_variant(
    rt: &tokio::runtime::Runtime,
    model_dir: &Path,
    use_gpu: bool,
    decoder_io_binding: bool,
    fixtures: &[Fixture],
) -> Vec<RunRow> {
    println!(
        "── decoder_io_binding={} ──",
        if decoder_io_binding { "on" } else { "off" }
    );
    let backend = lune_asr::Qwen3AsrBackend::new_with_decoder_io_binding(
        lune_asr::Qwen3AsrConfig {
            model_dir: model_dir.to_path_buf(),
            use_gpu,
        },
        decoder_io_binding,
    )
    .expect("load Qwen3 ASR");

    let mut rows = Vec::with_capacity(fixtures.len());
    for fixture in fixtures {
        let t0 = std::time::Instant::now();
        let text = rt.block_on(async {
            let session = backend
                .start_session(AsrSessionConfig {
                    language: Some("en".into()),
                    ..Default::default()
                })
                .await
                .expect("start session");
            backend
                .push_audio(&session, &fixture.pcm)
                .await
                .expect("push audio");
            let segments = backend.poll_final(&session).await.expect("poll final");
            backend.end_session(session).await.expect("end session");
            segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ")
        });
        let profile = backend
            .last_profile()
            .unwrap_or_else(|| lune_asr::Qwen3AsrProfile {
                audio_ms: (fixture.audio_secs * 1000.0).round() as usize,
                total_ms: t0.elapsed().as_secs_f64() * 1000.0,
                decoder_io_binding,
                ..Default::default()
            });
        println!(
            "{:22} total={:8.1}ms prefill={:7.1}ms decode={:7.1}ms steps={:3} tok_ms={:6.2}",
            fixture.name,
            profile.total_ms,
            profile.prefill_ms,
            profile.decode_ms,
            profile.decode_steps,
            per_token_ms(profile),
        );
        rows.push(RunRow {
            fixture: fixture.name.clone(),
            text,
            profile,
        });
    }
    println!();
    rows
}

fn print_comparison(plain: &[RunRow], bound: &[RunRow]) {
    println!("── comparison ──");
    println!(
        "{:22}{:>10}{:>10}{:>10}{:>10}{:>8}",
        "fixture", "off_ms", "on_ms", "delta", "tok_delta", "same"
    );
    for off in plain {
        let Some(on) = bound.iter().find(|row| row.fixture == off.fixture) else {
            continue;
        };
        let total_delta = off.profile.total_ms - on.profile.total_ms;
        let token_delta = per_token_ms(off.profile) - per_token_ms(on.profile);
        println!(
            "{:22}{:10.1}{:10.1}{:10.1}{:10.2}{:>8}",
            off.fixture,
            off.profile.total_ms,
            on.profile.total_ms,
            total_delta,
            token_delta,
            if off.text == on.text { "yes" } else { "no" },
        );
    }
}

fn per_token_ms(profile: lune_asr::Qwen3AsrProfile) -> f64 {
    if profile.decode_steps == 0 {
        0.0
    } else {
        profile.decode_ms / profile.decode_steps as f64
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn load_fixtures(root: &Path) -> Vec<Fixture> {
    let fixtures_dir = root.join("fixtures/audio");
    let Ok(entries) = std::fs::read_dir(&fixtures_dir) else {
        return Vec::new();
    };
    let filter = std::env::var("LUNE_BENCH_FIXTURE").ok();
    let mut fixtures = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wav") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()).map(str::to_owned) else {
            continue;
        };
        if filter.as_ref().is_some_and(|needle| !name.contains(needle)) {
            continue;
        }
        let pcm = load_wav_f32(&path);
        fixtures.push(Fixture {
            name,
            audio_secs: pcm.len() as f64 / 16_000.0,
            pcm,
        });
    }
    fixtures.sort_by(|a, b| a.name.cmp(&b.name));
    fixtures
}

fn load_wav_f32(path: &Path) -> Vec<f32> {
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
