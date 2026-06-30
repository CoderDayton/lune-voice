//! Real-model integration test for whisper.cpp's built-in (silero ggml) VAD.
//!
//! Ignored by default (needs ggml models on disk + the C++ build). Run with:
//!   LIBCLANG_PATH=/lib/x86_64-linux-gnu \
//!   LUNE_TEST_WHISPER_GGML=$HOME/.local/share/lune-voice/models/whisper-small-q5_1/ggml-small-q5_1.bin \
//!   LUNE_TEST_VAD_GGML=/path/to/ggml-silero-v5.1.2.bin \
//!   LUNE_TEST_SPEECH_WAV=fixtures/audio/librispeech_40744.wav \
//!   cargo test -p lune-asr --features whisper-cpp --test whisper_vad -- --ignored --nocapture

#![cfg(feature = "whisper-cpp")]

use std::path::PathBuf;

use lune_asr::{WhisperCppBackend, WhisperCppConfig};
use lune_core::asr::{AsrBackend, AsrSessionConfig};

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn load_wav_16k_mono(path: &PathBuf) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let mono: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
    };
    // Fixtures used here are already 16 kHz mono; downmix only if needed.
    if spec.channels == 1 {
        mono
    } else {
        mono.chunks(spec.channels as usize)
            .map(|f| f.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    }
}

fn backend(use_vad: bool) -> Option<WhisperCppBackend> {
    let model_path = env_path("LUNE_TEST_WHISPER_GGML")?;
    let vad_model_path = if use_vad {
        Some(env_path("LUNE_TEST_VAD_GGML")?)
    } else {
        None
    };
    WhisperCppBackend::new(WhisperCppConfig {
        model_path,
        use_gpu: false,
        vad_model_path,
    })
    .ok()
}

/// The capture pipeline drops any segment below this confidence as garbage
/// (see `lune_app::capture::GARBAGE_DROP_THRESHOLD`).
const GARBAGE_DROP_THRESHOLD: f32 = 0.25;

async fn transcribe(b: &WhisperCppBackend, pcm: &[f32]) -> Vec<lune_core::asr::AsrSegment> {
    let session = b
        .start_session(AsrSessionConfig {
            language: Some("en".into()),
            ..Default::default()
        })
        .await
        .expect("start_session");
    b.push_audio(&session, pcm).await.expect("push_audio");
    let segs = b.poll_final(&session).await.expect("poll_final");
    b.end_session(session).await.ok();
    segs
}

fn joined(segs: &[lune_core::asr::AsrSegment]) -> String {
    segs.iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

#[tokio::test]
#[ignore = "needs ggml whisper + VAD models (LUNE_TEST_WHISPER_GGML / LUNE_TEST_VAD_GGML)"]
async fn vad_passes_speech_through() {
    let Some(b) = backend(true) else {
        eprintln!("models not set; skipping");
        return;
    };
    let wav = env_path("LUNE_TEST_SPEECH_WAV").expect("LUNE_TEST_SPEECH_WAV");
    let pcm = load_wav_16k_mono(&wav);
    let text = joined(&transcribe(&b, &pcm).await);
    eprintln!("SPEECH+VAD => {text:?}");
    assert!(
        !text.is_empty(),
        "VAD should pass speech through to a transcript"
    );
}

#[tokio::test]
#[ignore = "needs ggml whisper + VAD models"]
async fn vad_drops_silence() {
    let Some(b) = backend(true) else {
        return;
    };
    // 3 s of digital silence. Ideally VAD yields no segments; whisper may still
    // hallucinate a short low-confidence token (the classic "you"). Either is
    // acceptable because the capture pipeline drops sub-threshold segments — so
    // assert "no surviving segment": empty, or every segment below the garbage
    // confidence threshold.
    let silence = vec![0.0f32; 16_000 * 3];
    let segs = transcribe(&b, &silence).await;
    for s in &segs {
        eprintln!("SILENCE+VAD seg conf={:?} text={:?}", s.confidence, s.text);
    }
    let surviving = segs.iter().any(|s| {
        !s.text.trim().is_empty() && s.confidence.unwrap_or(0.0) >= GARBAGE_DROP_THRESHOLD
    });
    assert!(
        !surviving,
        "silence produced a high-confidence segment the pipeline would keep: {:?}",
        joined(&segs)
    );
}
