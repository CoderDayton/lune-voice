//! End-to-end integration tests against real ONNX model files.
//!
//! Models are loaded once (shared across tests) — timings reflect pure inference.
//!
//! Requires model directories under `models/` at the workspace root:
//!   - models/qwen3-asr-0.6b-onnx/
//!   - models/qwen3-0.6b-onnx/
//!
//! Run with: `cargo test -p lune-app --test e2e_ort_models -- --ignored`
//! GPU:     `cargo test -p lune-app --features cuda --test e2e_ort_models -- --ignored`

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use lune_asr::{OrtQwenAsrBackend, OrtQwenAsrConfig};
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use lune_core::config::{DevicePolicy, StylePreset};
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_llm::{OrtLlmBackend, OrtLlmConfig};

// ── Shared backends (loaded once) ───────────────────────────────────────────

fn use_gpu() -> bool {
    DevicePolicy::Auto.use_gpu(cfg!(feature = "cuda"))
}

static ASR: OnceLock<Option<Arc<OrtQwenAsrBackend>>> = OnceLock::new();
static LLM: OnceLock<Option<Arc<OrtLlmBackend>>> = OnceLock::new();

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn get_asr() -> Option<&'static Arc<OrtQwenAsrBackend>> {
    ASR.get_or_init(|| {
        let dir = workspace_root().join("models/qwen3-asr-0.6b-onnx");
        if !dir.exists() {
            return None;
        }
        let t = Instant::now();
        let backend = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
            model_dir: dir,
            use_gpu: use_gpu(),
            ..Default::default()
        })
        .expect("ASR backend should load");
        eprintln!(
            "[setup] ASR loaded in {:?} (gpu={})",
            t.elapsed(),
            use_gpu()
        );
        Some(Arc::new(backend))
    })
    .as_ref()
}

fn get_llm() -> Option<&'static Arc<OrtLlmBackend>> {
    LLM.get_or_init(|| {
        let dir = workspace_root().join("models/qwen3-0.6b-onnx");
        if !dir.exists() {
            return None;
        }
        let t = Instant::now();
        let backend = OrtLlmBackend::new(OrtLlmConfig {
            model_dir: dir,
            use_gpu: use_gpu(),
            temperature: 0.0,
            max_output_tokens: 64,
            ..Default::default()
        })
        .expect("LLM backend should load");
        eprintln!(
            "[setup] LLM loaded in {:?} (gpu={})",
            t.elapsed(),
            use_gpu()
        );
        Some(Arc::new(backend))
    })
    .as_ref()
}

// ── Audio generators ────────────────────────────────────────────────────────

fn silence_16khz(duration_sec: f32) -> Vec<f32> {
    vec![0.0f32; (16_000.0 * duration_sec) as usize]
}

fn tone_16khz(freq_hz: f32, duration_sec: f32) -> Vec<f32> {
    let n = (16_000.0 * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.3
        })
        .collect()
}

fn llm_input(raw_text: &str) -> PostProcessInput {
    PostProcessInput {
        raw_text: raw_text.into(),
        style_preset: Arc::new(StylePreset::general()),
        app_category: None,
        dictionary: vec![],
    }
}

// ── ASR Tests ───────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires models/qwen3-asr-0.6b-onnx on disk"]
async fn asr_silence() {
    let asr = match get_asr() {
        Some(b) => b,
        None => return,
    };
    let handle = asr
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();

    let audio = silence_16khz(2.0);
    let t = Instant::now();
    asr.push_audio(&handle, &audio).await.unwrap();
    let segments = asr.poll_results(&handle).await.unwrap();
    eprintln!(
        "[asr_silence] inference: {:?}, segments: {}",
        t.elapsed(),
        segments.len()
    );

    asr.end_session(handle).await.unwrap();
}

#[tokio::test]
#[ignore = "requires models/qwen3-asr-0.6b-onnx on disk"]
async fn asr_tone() {
    let asr = match get_asr() {
        Some(b) => b,
        None => return,
    };
    let handle = asr
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();

    let audio = tone_16khz(440.0, 3.0);
    let t = Instant::now();
    asr.push_audio(&handle, &audio).await.unwrap();
    let segments = asr.poll_results(&handle).await.unwrap();
    eprintln!(
        "[asr_tone] inference: {:?}, segments: {segments:?}",
        t.elapsed()
    );

    asr.end_session(handle).await.unwrap();
}

#[tokio::test]
#[ignore = "requires models/qwen3-asr-0.6b-onnx on disk"]
async fn asr_reset_and_reuse() {
    let asr = match get_asr() {
        Some(b) => b,
        None => return,
    };
    let handle = asr
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();

    // First utterance.
    asr.push_audio(&handle, &silence_16khz(1.0)).await.unwrap();
    let t = Instant::now();
    let _ = asr.poll_results(&handle).await.unwrap();
    let first = t.elapsed();

    asr.reset_context(&handle).await.unwrap();

    // Second utterance after reset.
    asr.push_audio(&handle, &tone_16khz(440.0, 1.0))
        .await
        .unwrap();
    let t = Instant::now();
    let segments = asr.poll_results(&handle).await.unwrap();
    let second = t.elapsed();
    eprintln!("[asr_reset] first: {first:?}, second: {second:?}, segments: {segments:?}");

    asr.end_session(handle).await.unwrap();
}

// ── LLM Tests ───────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires models/qwen3-0.6b-onnx on disk"]
async fn llm_cleans_disfluencies() {
    let llm = match get_llm() {
        Some(b) => b,
        None => return,
    };

    let t = Instant::now();
    let output = llm
        .transform(&llm_input("um hello world uh"))
        .await
        .unwrap();
    let elapsed = t.elapsed();
    eprintln!("[llm_disfluency] {:?} → {:?}", elapsed, output.cleaned_text);

    assert!(!output.cleaned_text.is_empty());
    assert!(output.cleaned_text.to_lowercase().contains("hello"));
}

#[tokio::test]
#[ignore = "requires models/qwen3-0.6b-onnx on disk"]
async fn llm_streaming_matches_non_streaming() {
    let llm = match get_llm() {
        Some(b) => b,
        None => return,
    };

    let input = llm_input("so like I was thinking we should probably go to the park");

    let t = Instant::now();
    let direct = llm.transform(&input).await.unwrap();
    let direct_time = t.elapsed();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let t = Instant::now();
    let streaming = llm.transform_streaming(&input, tx).await.unwrap();
    let stream_time = t.elapsed();

    let mut chunks = Vec::new();
    while let Ok(chunk) = rx.try_recv() {
        chunks.push(chunk);
    }

    eprintln!(
        "[llm_streaming] direct: {:?} ({:?}), stream: {:?} ({:?}), chunks: {}",
        direct.cleaned_text,
        direct_time,
        streaming.cleaned_text,
        stream_time,
        chunks.len()
    );

    assert_eq!(direct.cleaned_text, streaming.cleaned_text);
    assert!(!chunks.is_empty());
}

// ── Full Pipeline ───────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires both model directories on disk"]
async fn full_pipeline_asr_to_llm() {
    let asr = match get_asr() {
        Some(b) => b,
        None => return,
    };
    let llm = match get_llm() {
        Some(b) => b,
        None => return,
    };

    // ASR pass.
    let handle = asr
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();
    let audio = tone_16khz(440.0, 3.0);
    asr.push_audio(&handle, &audio).await.unwrap();

    let t = Instant::now();
    let segments = asr.poll_results(&handle).await.unwrap();
    let asr_time = t.elapsed();
    asr.end_session(handle).await.unwrap();

    let raw_text = if segments.is_empty() || segments[0].text.trim().is_empty() {
        "um so yeah the uh meeting is at three".to_string()
    } else {
        segments
            .into_iter()
            .map(|s| s.text)
            .collect::<Vec<_>>()
            .join(" ")
    };

    // LLM pass.
    let t = Instant::now();
    let output = llm.transform(&llm_input(&raw_text)).await.unwrap();
    let llm_time = t.elapsed();

    eprintln!(
        "[pipeline] asr: {:?}, llm: {:?}, total: {:?}",
        asr_time,
        llm_time,
        asr_time + llm_time
    );
    eprintln!(
        "[pipeline] input: {raw_text:?} → output: {:?}",
        output.cleaned_text
    );

    assert!(!output.cleaned_text.is_empty());
}
