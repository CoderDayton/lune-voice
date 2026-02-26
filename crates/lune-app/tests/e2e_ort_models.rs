//! End-to-end integration tests against real ONNX model files.
//!
//! These tests are `#[ignore]` because they require multi-GB model directories
//! under `models/` at the workspace root:
//!   - models/qwen3-asr-0.6b-onnx/  (encoder, embed_tokens, decoder, tokenizer)
//!   - models/qwen3-0.6b-onnx/      (decoder, tokenizer)
//!
//! Run with: `cargo test -p lune-app --test e2e_ort_models -- --ignored`

use std::path::PathBuf;
use std::sync::Arc;

use lune_asr::{OrtQwenAsrBackend, OrtQwenAsrConfig};
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use lune_core::config::StylePreset;
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_llm::{OrtLlmBackend, OrtLlmConfig};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn asr_model_dir() -> PathBuf {
    workspace_root().join("models/qwen3-asr-0.6b-onnx")
}

fn llm_model_dir() -> PathBuf {
    workspace_root().join("models/qwen3-0.6b-onnx")
}

/// Generate 2 seconds of 16 kHz silence — enough to exercise the full ASR
/// pipeline (mel → encoder → decoder) without producing meaningful text.
fn silence_16khz(duration_sec: f32) -> Vec<f32> {
    vec![0.0f32; (16_000.0 * duration_sec) as usize]
}

/// Generate a 16 kHz sine tone. Not real speech, but exercises the full
/// encoder path with non-zero energy.
fn tone_16khz(freq_hz: f32, duration_sec: f32) -> Vec<f32> {
    let n = (16_000.0 * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.3
        })
        .collect()
}

// ── ASR Tests ────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires models/qwen3-asr-0.6b-onnx on disk"]
async fn asr_model_loads() {
    let dir = asr_model_dir();
    if !dir.exists() {
        eprintln!("skip: {dir:?} not found");
        return;
    }
    let backend = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
        model_dir: dir,
        use_gpu: false,
        ..Default::default()
    })
    .expect("ASR backend should load");

    let handle = backend
        .start_session(AsrSessionConfig::default())
        .await
        .expect("start_session should succeed");

    backend
        .end_session(handle)
        .await
        .expect("end_session should succeed");
}

#[tokio::test]
#[ignore = "requires models/qwen3-asr-0.6b-onnx on disk"]
async fn asr_silence_produces_no_crash() {
    let dir = asr_model_dir();
    if !dir.exists() {
        return;
    }
    let backend = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
        model_dir: dir,
        use_gpu: false,
        ..Default::default()
    })
    .unwrap();

    let handle = backend
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();

    // Push 2s of silence — should run through mel + encoder + decoder without panic.
    let audio = silence_16khz(2.0);
    backend.push_audio(&handle, &audio).await.unwrap();

    let segments = backend.poll_results(&handle).await.unwrap();
    // Silence may produce empty or whitespace-only segments; either is acceptable.
    for seg in &segments {
        eprintln!("asr silence segment: {:?}", seg);
    }

    backend.end_session(handle).await.unwrap();
}

#[tokio::test]
#[ignore = "requires models/qwen3-asr-0.6b-onnx on disk"]
async fn asr_tone_produces_segments() {
    let dir = asr_model_dir();
    if !dir.exists() {
        return;
    }
    let backend = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
        model_dir: dir,
        use_gpu: false,
        ..Default::default()
    })
    .unwrap();

    let handle = backend
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();

    // Push 3s of 440 Hz tone — encoder should produce non-trivial features.
    let audio = tone_16khz(440.0, 3.0);
    backend.push_audio(&handle, &audio).await.unwrap();

    let segments = backend.poll_results(&handle).await.unwrap();
    eprintln!("asr tone segments: {segments:?}");
    // Tone isn't speech, so we only verify the pipeline completed without error.

    backend.end_session(handle).await.unwrap();
}

#[tokio::test]
#[ignore = "requires models/qwen3-asr-0.6b-onnx on disk"]
async fn asr_reset_and_reuse() {
    let dir = asr_model_dir();
    if !dir.exists() {
        return;
    }
    let backend = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
        model_dir: dir,
        use_gpu: false,
        ..Default::default()
    })
    .unwrap();

    let handle = backend
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();

    // First utterance.
    backend
        .push_audio(&handle, &silence_16khz(1.0))
        .await
        .unwrap();
    let _ = backend.poll_results(&handle).await.unwrap();

    // Reset context (simulates end-of-utterance).
    backend.reset_context(&handle).await.unwrap();

    // Second utterance after reset.
    backend
        .push_audio(&handle, &tone_16khz(440.0, 1.0))
        .await
        .unwrap();
    let segments = backend.poll_results(&handle).await.unwrap();
    eprintln!("after reset: {segments:?}");

    backend.end_session(handle).await.unwrap();
}

// ── LLM Tests ────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires models/qwen3-0.6b-onnx on disk"]
async fn llm_model_loads() {
    let dir = llm_model_dir();
    if !dir.exists() {
        eprintln!("skip: {dir:?} not found");
        return;
    }
    let _backend = OrtLlmBackend::new(OrtLlmConfig {
        model_dir: dir,
        use_gpu: false,
        ..Default::default()
    })
    .expect("LLM backend should load");
}

#[tokio::test]
#[ignore = "requires models/qwen3-0.6b-onnx on disk"]
async fn llm_cleans_disfluencies() {
    let dir = llm_model_dir();
    if !dir.exists() {
        return;
    }
    let backend = OrtLlmBackend::new(OrtLlmConfig {
        model_dir: dir,
        use_gpu: false,
        temperature: 0.0, // greedy for determinism
        max_output_tokens: 64,
        ..Default::default()
    })
    .unwrap();

    let input = PostProcessInput {
        raw_text: "um hello world uh".into(),
        style_preset: Arc::new(StylePreset::general()),
        app_category: None,
        dictionary: vec![],
    };

    let output = backend.transform(&input).await.unwrap();
    eprintln!("llm output: {:?}", output.cleaned_text);

    // The model should at minimum remove fillers and produce non-empty output.
    assert!(
        !output.cleaned_text.is_empty(),
        "LLM output must not be empty"
    );
    let lower = output.cleaned_text.to_lowercase();
    assert!(
        lower.contains("hello"),
        "output should contain 'hello', got: {lower}"
    );
}

#[tokio::test]
#[ignore = "requires models/qwen3-0.6b-onnx on disk"]
async fn llm_streaming_matches_non_streaming() {
    let dir = llm_model_dir();
    if !dir.exists() {
        return;
    }
    let backend = OrtLlmBackend::new(OrtLlmConfig {
        model_dir: dir,
        use_gpu: false,
        temperature: 0.0,
        max_output_tokens: 64,
        ..Default::default()
    })
    .unwrap();

    let input = PostProcessInput {
        raw_text: "so like I was thinking we should probably go to the park".into(),
        style_preset: Arc::new(StylePreset::general()),
        app_category: None,
        dictionary: vec![],
    };

    // Non-streaming.
    let direct = backend.transform(&input).await.unwrap();

    // Streaming — collect chunks.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let streaming = backend.transform_streaming(&input, tx).await.unwrap();

    let mut chunks = Vec::new();
    while let Ok(chunk) = rx.try_recv() {
        chunks.push(chunk);
    }

    eprintln!("direct:    {:?}", direct.cleaned_text);
    eprintln!("streaming: {:?}", streaming.cleaned_text);
    eprintln!("chunks:    {chunks:?}");

    // Final text from both paths should match.
    assert_eq!(
        direct.cleaned_text, streaming.cleaned_text,
        "streaming and non-streaming should produce identical final text"
    );
    // At least one chunk should have been emitted.
    assert!(
        !chunks.is_empty(),
        "streaming should emit at least one chunk"
    );
}

// ── Full Pipeline ────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires both model directories on disk"]
async fn full_pipeline_asr_to_llm() {
    let asr_dir = asr_model_dir();
    let llm_dir = llm_model_dir();
    if !asr_dir.exists() || !llm_dir.exists() {
        eprintln!("skip: model directories not found");
        return;
    }

    // 1. Build backends.
    let asr = OrtQwenAsrBackend::new(OrtQwenAsrConfig {
        model_dir: asr_dir,
        use_gpu: false,
        ..Default::default()
    })
    .unwrap();

    let llm = OrtLlmBackend::new(OrtLlmConfig {
        model_dir: llm_dir,
        use_gpu: false,
        temperature: 0.0,
        max_output_tokens: 64,
        ..Default::default()
    })
    .unwrap();

    // 2. Run ASR on synthetic audio.
    let handle = asr
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();

    // Push 3 seconds of a tone — won't be real speech, but exercises the path.
    let audio = tone_16khz(440.0, 3.0);
    asr.push_audio(&handle, &audio).await.unwrap();
    let segments = asr.poll_results(&handle).await.unwrap();
    asr.end_session(handle).await.unwrap();

    eprintln!("ASR segments: {segments:?}");

    // 3. Feed ASR output (or fallback text) into LLM.
    let raw_text = if segments.is_empty() || segments[0].text.trim().is_empty() {
        // Tone isn't speech — use a canned string to still exercise LLM.
        "um so yeah the uh meeting is at three".to_string()
    } else {
        segments
            .into_iter()
            .map(|s| s.text)
            .collect::<Vec<_>>()
            .join(" ")
    };

    eprintln!("LLM input:  {raw_text:?}");

    let output = llm
        .transform(&PostProcessInput {
            raw_text,
            style_preset: Arc::new(StylePreset::general()),
            app_category: None,
            dictionary: vec![],
        })
        .await
        .unwrap();

    eprintln!("LLM output: {:?}", output.cleaned_text);
    assert!(
        !output.cleaned_text.is_empty(),
        "pipeline should produce non-empty output"
    );
}
