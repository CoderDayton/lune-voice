//! Qwen3-ASR diagnostic probe — feeds audio directly through the ONNX pipeline
//! at varying lengths to isolate where transcription breaks down.
//!
//! Usage: cargo run --release --example qwen3_probe -p lune-asr

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    // Load a test fixture
    let wav_path = root.join("test_fixtures/librispeech_clean.wav");
    if !wav_path.exists() {
        eprintln!("Missing {}", wav_path.display());
        return;
    }

    let mut reader = hound::WavReader::open(&wav_path).expect("open wav");
    let spec = reader.spec();
    let pcm: Vec<f32> = if spec.bits_per_sample == 16 {
        reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    } else {
        reader.samples::<f32>().map(|s| s.unwrap()).collect()
    };

    let total_secs = pcm.len() as f64 / 16_000.0;
    println!("Loaded: {:.1}s ({} samples)", total_secs, pcm.len());

    // Init ORT
    let _ = ort::init().commit();

    let model_dir = root.join("models/qwen3-asr-0.6b-onnx");
    let config = lune_asr::Qwen3AsrConfig {
        model_dir,
        use_gpu: false,
    };
    let backend = lune_asr::Qwen3AsrBackend::new(config).expect("load backend");

    // Probe at different durations
    // Fine-grained sweep around known boundaries
    let mut durations_sec: Vec<f64> = Vec::new();
    // Sweep 2.0-4.0 in 0.25s steps to find exact breakpoint
    let mut d = 2.0;
    while d <= 4.0 {
        durations_sec.push(d);
        d += 0.25;
    }
    // Sweep 7.0-9.0
    d = 7.0;
    while d <= 9.0 {
        durations_sec.push(d);
        d += 0.25;
    }
    durations_sec.push(total_secs);

    let rt = tokio::runtime::Runtime::new().unwrap();

    for &dur in &durations_sec {
        let samples = (dur * 16_000.0) as usize;
        if samples > pcm.len() {
            continue;
        }
        let slice = &pcm[..samples];

        // Run through the backend directly
        rt.block_on(async {
            use lune_core::asr::{AsrBackend, AsrSessionConfig};

            let session = backend
                .start_session(AsrSessionConfig::default())
                .await
                .unwrap();
            backend.push_audio(&session, slice).await.unwrap();
            let results = backend.poll_final(&session).await.unwrap();
            backend.end_session(session).await.unwrap();

            if results.is_empty() {
                println!("{:5.1}s → [empty]", dur);
            } else {
                let seg = &results[0];
                let words = seg.text.split_whitespace().count();
                let conf = seg
                    .confidence
                    .map(|c| format!("{:.2}", c))
                    .unwrap_or("-".into());
                let preview: String = seg.text.chars().take(80).collect();
                println!(
                    "{:5.1}s → {} words, conf={}, {:?}",
                    dur, words, conf, preview
                );
            }
        });
    }
}
