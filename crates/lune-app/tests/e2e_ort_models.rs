//! End-to-end integration tests against real ONNX model files.
//!
//! Models are loaded once (shared across tests) — timings reflect pure inference.
//!
//! Requires model directories under `models/` at the workspace root:
//!   - models/qwen3-asr-0.6b-onnx/
//!   - models/llm-corrector-gguf/
//!
//! Run with: `cargo test -p lune-app --test e2e_ort_models -- --ignored`
//! GPU:     `cargo test -p lune-app --features cuda --test e2e_ort_models -- --ignored`

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::sync::OnceCell;

use lune_asr::{OrtQwenAsrBackend, OrtQwenAsrConfig};
use lune_core::asr::{AsrBackend, AsrSessionConfig};
use lune_core::config::{DevicePolicy, StylePreset};
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_llm::{LlamaCppBackend, LlamaCppConfig};

// ── Shared backends (loaded once) ───────────────────────────────────────────

fn use_gpu() -> bool {
    DevicePolicy::Auto.use_gpu(cfg!(feature = "cuda"))
}

static ORT_INIT: OnceLock<()> = OnceLock::new();

fn ensure_ort_init() {
    ORT_INIT.get_or_init(|| {
        #[cfg(feature = "cuda")]
        {
            if ort::init()
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .commit()
            {
                eprintln!("[setup] ORT initialized with CUDA EP");
                return;
            }
            eprintln!("[setup] CUDA init failed, falling back to CPU");
        }
        let _ = ort::init().commit();
    });
}

static ASR: OnceLock<Option<Arc<OrtQwenAsrBackend>>> = OnceLock::new();
static LLM: OnceCell<Option<Arc<LlamaCppBackend>>> = OnceCell::const_new();

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn get_asr() -> Option<&'static Arc<OrtQwenAsrBackend>> {
    ensure_ort_init();
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

        // Attach forced aligner for timestamp-based chunk merging if model files exist.
        let fa_dir = workspace_root().join("models/wav2vec2-fa");
        let fa_model = fa_dir.join("model.onnx");
        let fa_config = fa_dir.join("fa_config.json");
        if fa_model.exists() && fa_config.exists() {
            match backend.make_forced_aligner(&fa_dir) {
                Ok(fa) => {
                    let fa = Arc::new(fa);
                    backend.attach_aligner(fa);
                    eprintln!("[setup] FA attached for chunk merging");
                }
                Err(e) => {
                    eprintln!("[setup] FA load failed (non-fatal): {e}");
                }
            }
        } else {
            eprintln!(
                "[setup] FA model not found at {}, skipping",
                fa_dir.display()
            );
        }

        Some(Arc::new(backend))
    })
    .as_ref()
}

async fn get_llm() -> Option<&'static Arc<LlamaCppBackend>> {
    LLM.get_or_init(|| async {
        let path = workspace_root()
            .join("models/qwen2.5-1.5b-instruct-gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf");
        if !path.exists() {
            return None;
        }
        let t = Instant::now();
        let n_gpu_layers = if use_gpu() { 1000 } else { 0 };
        let backend = LlamaCppBackend::new(LlamaCppConfig {
            model_path: path,
            n_gpu_layers,
            max_output_tokens: 64,
            ..Default::default()
        })
        .await
        .expect("LLM backend should load");
        eprintln!(
            "[setup] LLM loaded in {:?} (gpu={})",
            t.elapsed(),
            use_gpu()
        );
        Some(Arc::new(backend))
    })
    .await
    .as_ref()
}

// ── Audio helpers ───────────────────────────────────────────────────────────

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

/// Load 16 kHz mono f32 PCM from a WAV file.
fn load_wav_f32(path: &std::path::Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono");
    assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz");
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

fn llm_input(raw_text: &str) -> PostProcessInput {
    PostProcessInput {
        raw_text: raw_text.into(),
        style_preset: Arc::new(StylePreset::casual()),
        app_category: None,
        dictionary: vec![],
        context: None,
        language: "en".into(),
    }
}

use lune_core::asr::AsrSegment;

/// Run ASR on audio using incremental push/poll (same cadence as the real session loop).
async fn transcribe(asr: &OrtQwenAsrBackend, audio: &[f32]) -> Vec<AsrSegment> {
    let handle = asr
        .start_session(AsrSessionConfig::default())
        .await
        .unwrap();
    let chunk_size = 16_000; // 1.0s push cadence (batch mode — only poll_final transcribes)
    let mut segments = Vec::new();
    for chunk in audio.chunks(chunk_size) {
        asr.push_audio(&handle, chunk).await.unwrap();
        segments.extend(asr.poll_results(&handle).await.unwrap());
    }
    // Flush tail with 1s silence (>= chunk_samples) so speech-end detection
    // fires and commits all rollback-buffered tokens.
    asr.push_audio(&handle, &vec![0.0f32; 16_000])
        .await
        .unwrap();
    segments.extend(asr.poll_final(&handle).await.unwrap());
    asr.end_session(handle).await.unwrap();
    segments
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
#[ignore = "requires models/llm-corrector-gguf on disk"]
async fn llm_cleans_disfluencies() {
    let llm = match get_llm().await {
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
    assert!(
        output.cleaned_text.to_lowercase().contains("hello"),
        "output should preserve core content, got: {:?}",
        output.cleaned_text
    );
}

#[tokio::test]
#[ignore = "requires models/llm-corrector-gguf on disk"]
async fn llm_streaming_matches_non_streaming() {
    let llm = match get_llm().await {
        Some(b) => b,
        None => return,
    };

    let input = llm_input("so like I was thinkin we should probably go to the park");

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

/// Verify that providing prior context doesn't break LLM output and the context
/// text appears in the generated prompt (checked via non-empty, non-hallucinated output).
#[tokio::test]
#[ignore = "requires models/llm-corrector-gguf on disk"]
async fn llm_with_context_produces_valid_output() {
    let llm = match get_llm().await {
        Some(b) => b,
        None => return,
    };

    let context_text =
        "Context: The queen mother gave the French the most affectionate reception.\n";
    let input_with_ctx = PostProcessInput {
        raw_text: "she had suffered too much unhappyness in England".into(),
        style_preset: Arc::new(StylePreset::casual()),
        app_category: None,
        dictionary: vec![],
        context: Some(context_text.into()),
        language: "en".into(),
    };
    let input_without_ctx = PostProcessInput {
        context: None,
        ..input_with_ctx.clone()
    };

    let t = Instant::now();
    let with_ctx = llm.transform(&input_with_ctx).await.unwrap();
    let ctx_time = t.elapsed();

    let t = Instant::now();
    let without_ctx = llm.transform(&input_without_ctx).await.unwrap();
    let no_ctx_time = t.elapsed();

    eprintln!(
        "[llm_context] with_ctx ({ctx_time:?}): {:?}",
        with_ctx.cleaned_text
    );
    eprintln!(
        "[llm_context] without_ctx ({no_ctx_time:?}): {:?}",
        without_ctx.cleaned_text
    );

    // Both should produce non-empty output containing key words.
    assert!(!with_ctx.cleaned_text.is_empty());
    assert!(!without_ctx.cleaned_text.is_empty());
    let lower = with_ctx.cleaned_text.to_lowercase();
    assert!(
        lower.contains("england") || lower.contains("suffered"),
        "context output should contain key words from input, got: {:?}",
        with_ctx.cleaned_text
    );
}

/// Full pipeline using the real `SessionManager` with file audio.
///
/// Exercises the complete path: AudioStream → VAD → ASR → confidence gate →
/// LLM (with sliding context) → commit_final_text → session_text.
/// The only difference from a live session is the audio source.
#[tokio::test]
#[ignore = "requires both model directories on disk"]
async fn full_pipeline_real_session() {
    use lune_app::{SessionManager, SessionManagerConfig};
    use lune_audio::{AudioConfig, AudioStream};
    use lune_core::asr::AsrBackendKind;
    use lune_core::config::UserSettings;
    use lune_core::session::SessionEvent;
    use tokio::sync::{Mutex, RwLock, mpsc};

    let asr = match get_asr() {
        Some(b) => Arc::clone(b),
        None => return,
    };
    let llm: Arc<dyn TextPostProcessor> = match get_llm().await {
        Some(b) => Arc::clone(b) as _,
        None => return,
    };

    // Prefer the 30s chapter-40744 clip; fall back to the original 23s clip.
    let wav_path = [
        "test_fixtures/librispeech_40744.wav",
        "test_fixtures/librispeech_clean.wav",
    ]
    .iter()
    .map(|p| workspace_root().join(p))
    .find(|p| p.exists());
    let Some(wav_path) = wav_path else {
        eprintln!("[real_session] skipping — no wav file");
        return;
    };
    eprintln!("[real_session] using {}", wav_path.display());
    let samples = load_wav_f32(&wav_path);
    let audio_stream = AudioStream::from_samples(&samples, AudioConfig::default()).unwrap();

    let db = lune_persist::Database::open_in_memory().unwrap();
    let dict = lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
        .unwrap();
    let settings = Arc::new(RwLock::new(UserSettings::default()));

    let config = SessionManagerConfig {
        asr_backend_kind: AsrBackendKind::OrtQwenAsr,
        ..SessionManagerConfig::default()
    };

    let mgr = SessionManager::new(
        asr,
        Arc::new(lune_commands::VoiceCommandRouter::new()),
        llm,
        Arc::new(Mutex::new(dict)),
        Arc::new(Mutex::new(db)),
        config,
        settings,
    );

    let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(256);

    tokio::task::LocalSet::new()
        .run_until(async {
            let _id = mgr
                .start_with_audio(event_tx, audio_stream)
                .await
                .expect("start session");

            let mut final_text = String::new();
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);

            loop {
                let event = tokio::time::timeout_at(deadline, event_rx.recv()).await;
                match event {
                    Ok(Some(SessionEvent::FinalizedText(t))) => {
                        eprintln!("[real_session] chunk: {t:?}");
                        final_text.push_str(&t);
                    }
                    Ok(Some(SessionEvent::Stopped)) => break,
                    Ok(Some(_)) => {} // PartialText, Started, etc.
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("[real_session] timeout waiting for events");
                        mgr.stop().await;
                        break;
                    }
                }
            }

            eprintln!("[real_session] output: {final_text:?}");

            assert!(
                !final_text.is_empty(),
                "real pipeline should produce text from speech audio"
            );
            // Verify no stray spaces before punctuation (the commit_final_text fix).
            assert!(
                !final_text.contains(" ,") && !final_text.contains(" ."),
                "should not have spaces before punctuation, got: {final_text:?}"
            );
        })
        .await;
}

// ── Full Pipeline ───────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires both model directories on disk"]
async fn full_pipeline_asr_to_llm() {
    let asr = match get_asr() {
        Some(b) => b,
        None => return,
    };
    let llm = match get_llm().await {
        Some(b) => b,
        None => return,
    };

    // Use real speech if available, otherwise fall back to synthetic.
    let wav_path = workspace_root().join("test_fixtures/librispeech_clean.wav");
    let (audio, ground_truth) = if wav_path.exists() {
        let gt =
            std::fs::read_to_string(workspace_root().join("test_fixtures/librispeech_clean.txt"))
                .unwrap_or_default();
        (load_wav_f32(&wav_path), Some(gt))
    } else {
        (tone_16khz(440.0, 3.0), None)
    };
    let total_secs = audio.len() as f64 / 16_000.0;

    // ASR pass.
    let t = Instant::now();
    let segments = transcribe(asr, &audio).await;
    let asr_time = t.elapsed();

    if let Some(gt) = &ground_truth {
        eprintln!("[pipeline] truth: {gt}");
    }

    // Confidence gating: send segments with uncertain words to LLM for correction.
    let word_threshold = 0.85;

    let non_empty: Vec<&AsrSegment> = segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .collect();
    let raw_text: String = non_empty
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");

    for seg in &non_empty {
        eprintln!(
            "[pipeline] conf={:.3} min_tok={:.3} words={}: {:?}",
            seg.confidence.unwrap_or(0.0),
            seg.min_token_confidence.unwrap_or(0.0),
            seg.word_confidences.len(),
            seg.text.trim(),
        );
        for wc in &seg.word_confidences {
            eprintln!("  word={:?} conf={:.3}", wc.word, wc.confidence);
        }
    }

    // Send segments with uncertain words to LLM for correction.
    let has_uncertain = non_empty
        .iter()
        .any(|s| s.has_uncertain_words(word_threshold));

    let mut llm_time = std::time::Duration::ZERO;
    let session_text = if !has_uncertain || raw_text.is_empty() {
        eprintln!("[pipeline] no uncertain words → BYPASS LLM");
        raw_text
    } else {
        let mut corrected_parts: Vec<String> = Vec::new();
        for seg in &non_empty {
            let text = seg.text.trim();
            if !seg.has_uncertain_words(word_threshold) {
                corrected_parts.push(text.to_owned());
                continue;
            }
            let t = Instant::now();
            let output = llm.transform(&llm_input(text)).await.unwrap();
            llm_time += t.elapsed();
            eprintln!(
                "[pipeline] LLM correction ({:?}): {:?} → {:?}",
                t.elapsed(),
                text,
                output.cleaned_text
            );
            corrected_parts.push(output.cleaned_text);
        }
        corrected_parts.join(" ")
    };

    // Remove fillers deterministically after LLM correction.
    let session_text =
        lune_llm::fillers::remove_fillers(&session_text, "en").unwrap_or(session_text);
    eprintln!("[pipeline] after filler removal: {session_text:?}");

    eprintln!(
        "[pipeline] asr={asr_time:?} ({:.2}x RT), llm={llm_time:?}, total={:?}",
        asr_time.as_secs_f64() / total_secs,
        asr_time + llm_time,
    );
    eprintln!("[pipeline] output: {session_text:?}");

    assert!(
        !session_text.is_empty() || segments.is_empty(),
        "pipeline produced no text from non-empty segments"
    );
}
