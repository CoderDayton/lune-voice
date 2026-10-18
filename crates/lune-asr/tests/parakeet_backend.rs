// SPDX-License-Identifier: GPL-3.0-or-later

//! The Parakeet backend against the real native library.
//!
//! The always-on tests load `fixtures/parakeet/for-tests-ggml-parakeet-tdt.bin`,
//! a 16 KB random-weight Parakeet TDT model from whisper.cpp v1.9.4's own test
//! suite (MIT, `models/for-tests-ggml-parakeet-tdt.bin`). It cannot transcribe
//! anything, but every byte goes through the real loader, encoder, TDT decoder
//! and teardown, which is what these tests are about: the FFI contract, the
//! reused decoder state, and the edge cases around it.
//!
//! The ignored test measures the real model. Run with:
//!   LUNE_TEST_PARAKEET_GGML=$HOME/.local/share/lune-voice/models/parakeet-tdt-0.6b-v3-q8_0/ggml-parakeet-tdt-0.6b-v3-q8_0.bin \
//!   LUNE_TEST_VAD_GGML=$HOME/.local/share/lune-voice/models/whisper-vad-silero/ggml-silero-v5.1.2.bin \
//!   cargo test -p lune-asr --features whisper-cpp --test parakeet_backend -- --ignored --nocapture

#![cfg(feature = "whisper-cpp")]

use std::path::{Path, PathBuf};

use lune_asr::{ParakeetCppBackend, ParakeetCppConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::ParakeetModel;

const SECOND: usize = 16_000;

/// One decode at a time, as in the app. ggml's CPU threads spin while they
/// wait for work, so several decodes in parallel test threads, each with a
/// full thread pool, starve one another on a small CI runner and turn a 2 s
/// suite into minutes.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn tiny_model() -> PathBuf {
    fixture("fixtures/parakeet/for-tests-ggml-parakeet-tdt.bin")
}

fn backend_at(model_path: PathBuf, vad_model_path: Option<PathBuf>) -> ParakeetCppBackend {
    load_compute_backends();
    ParakeetCppBackend::new(ParakeetCppConfig {
        model: ParakeetModel::TdtV3Q8,
        model_path,
        use_gpu: false,
        vad_model_path,
    })
    .expect("load the Parakeet model")
}

/// Register ggml's compute backends, as the app does at startup. A no-op for a
/// static build; under `GGML_BACKEND_DL` it loads the CPU plugins staged next
/// to the test binary (`cgs stage-runtime`), without which there is nothing to
/// decode on. Idempotent.
fn load_compute_backends() {
    let data_dir = std::env::temp_dir().join("lune-parakeet-test-data");
    let _ = lune_asr::gpu_backend::load_plugins(&data_dir);
}

/// Deterministic broadband noise, loud enough for the encoder to act on.
fn noise(samples: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..samples)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.4
        })
        .collect()
}

async fn transcribe(backend: &ParakeetCppBackend, pcm: &[f32]) -> Vec<AsrSegment> {
    let session = backend
        .start_session(AsrSessionConfig::default())
        .await
        .expect("start session");
    backend.push_audio(&session, pcm).await.expect("push audio");
    let segments = backend.poll_final(&session).await.expect("decode");
    backend.end_session(session).await.expect("end session");
    segments
}

fn texts(segments: &[AsrSegment]) -> Vec<&str> {
    segments.iter().map(|s| s.text.as_str()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_test_model_loads_decodes_and_frees() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    let segments = transcribe(&backend, &noise(3 * SECOND, 1)).await;
    for seg in &segments {
        assert!(seg.is_final);
        assert!(!seg.text.trim().is_empty(), "empty segments are dropped");
        let (start, end) = (seg.start_ms.unwrap(), seg.end_ms.unwrap());
        assert!(start <= end, "{start} > {end}");
        assert!(end <= 3_000 + 50, "timestamps stay inside the clip: {end}");
        for w in &seg.word_confidences {
            assert!((0.0..=1.0).contains(&w.confidence), "{w:?}");
            assert!((0.0..=1.0).contains(&w.mean_confidence), "{w:?}");
        }
    }
    drop(backend);
}

/// The decoder state is built once and reused. Whatever came before, the same
/// audio has to decode the same way — the property a fresh state per session
/// would give for free.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reused_decoder_state_decodes_the_same_audio_the_same_way() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    let clip = noise(2 * SECOND, 7);
    let first = transcribe(&backend, &clip).await;
    // Something different in between, long enough to resize the encoder graph.
    let _ = transcribe(&backend, &noise(4 * SECOND, 99)).await;
    let again = transcribe(&backend, &clip).await;
    assert_eq!(texts(&first), texts(&again));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_decodes_each_utterance_once() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    let session = backend
        .start_session(AsrSessionConfig::default())
        .await
        .expect("start");
    backend
        .push_audio(&session, &noise(2 * SECOND, 3))
        .await
        .expect("push");
    let first = backend.poll_final(&session).await.expect("decode");
    // Drained: polling again has nothing left to decode.
    assert!(backend.poll_final(&session).await.expect("poll").is_empty());
    backend
        .push_audio(&session, &noise(2 * SECOND, 3))
        .await
        .expect("push");
    let second = backend.poll_final(&session).await.expect("decode");
    assert_eq!(texts(&first), texts(&second));
    let ids: Vec<u64> = first.iter().chain(&second).map(|s| s.id).collect();
    let mut unique = ids.clone();
    unique.dedup();
    assert_eq!(ids, unique, "segment ids are unique within a session");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_audio_decodes_to_nothing() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    assert!(transcribe(&backend, &[]).await.is_empty());
}

/// A clip a few samples long encodes to almost no frames. It is padded up to
/// a workable length instead of being handed to the encoder as is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_clip_of_a_few_samples_decodes_without_failing() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    for len in [1, 160, 1_600, 7_999] {
        let _ = transcribe(&backend, &noise(len, len as u32)).await;
    }
}

/// The test model's encoder window is 3200 mel frames (32 s). A longer clip
/// with no VAD (so no pause to cut at) is split evenly to fit, rather than
/// handed to the library's dynamic encoder path, which drops speech.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dictation_longer_than_the_encoder_window_decodes() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    let segments = transcribe(&backend, &noise(33 * SECOND, 11)).await;
    for seg in &segments {
        assert!(seg.end_ms.unwrap() <= 33_000 + 50);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unsupported_dictation_language_is_refused_at_session_start() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    let refused = backend
        .start_session(AsrSessionConfig {
            language: Some("ja".into()),
            ..Default::default()
        })
        .await;
    assert!(refused.is_err());
    for language in [None, Some("de".to_string()), Some("en-US".to_string())] {
        assert!(
            backend
                .start_session(AsrSessionConfig {
                    language,
                    ..Default::default()
                })
                .await
                .is_ok()
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_dictation_language_is_reported_on_each_segment() {
    let _serial = SERIAL.lock().await;
    let backend = backend_at(tiny_model(), None);
    let session = backend
        .start_session(AsrSessionConfig {
            language: Some("de".into()),
            ..Default::default()
        })
        .await
        .expect("start");
    backend
        .push_audio(&session, &noise(3 * SECOND, 5))
        .await
        .expect("push");
    for seg in backend.poll_final(&session).await.expect("decode") {
        assert_eq!(seg.detected_language.as_deref(), Some("de"));
    }
}

#[test]
fn a_missing_model_is_an_error_not_a_crash() {
    let _serial = SERIAL.blocking_lock();
    load_compute_backends();
    let result = ParakeetCppBackend::new(ParakeetCppConfig {
        model: ParakeetModel::TdtV3Q8,
        model_path: std::env::temp_dir().join("lune-parakeet-absent.bin"),
        use_gpu: false,
        vad_model_path: None,
    });
    assert!(result.is_err());
}

/// A download cut short after its checksum was verified. The native loader
/// would abort the process on this, so it must be stopped before it.
#[test]
fn a_truncated_model_is_an_error_not_a_crash() {
    let _serial = SERIAL.blocking_lock();
    load_compute_backends();
    let bytes = std::fs::read(tiny_model()).expect("read fixture");
    let path = std::env::temp_dir().join("lune-parakeet-truncated.bin");
    std::fs::write(&path, &bytes[..2_048]).expect("write");
    let result = ParakeetCppBackend::new(ParakeetCppConfig {
        model: ParakeetModel::TdtV3Q8,
        model_path: path.clone(),
        use_gpu: false,
        vad_model_path: None,
    });
    let _ = std::fs::remove_file(path);
    let err = result.err().expect("a 2 KB model must be refused");
    assert!(err.to_string().contains("partial download"), "{err}");
}

/// A VAD file that cannot be used loses the endpoint trim, not the backend.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unusable_vad_degrades_to_decoding_the_whole_clip() {
    let _serial = SERIAL.lock().await;
    let path = std::env::temp_dir().join("lune-parakeet-bad-vad.bin");
    std::fs::write(&path, b"not a model").expect("write");
    let backend = backend_at(tiny_model(), Some(path.clone()));
    let _ = std::fs::remove_file(path);
    let _ = transcribe(&backend, &noise(2 * SECOND, 2)).await;
}

// ── Real model ──────────────────────────────────────────────────────────────

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn load_wav_16k_mono(path: &Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "fixtures are 16 kHz");
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
    };
    samples
        .chunks(spec.channels as usize)
        .map(|f| f.iter().sum::<f32>() / f.len() as f32)
        .collect()
}

fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Word error rate by edit distance over normalized words.
fn wer(reference: &[String], hypothesis: &[String]) -> f32 {
    let mut prev: Vec<usize> = (0..=hypothesis.len()).collect();
    for (i, r) in reference.iter().enumerate() {
        let mut row = vec![i + 1];
        for (j, h) in hypothesis.iter().enumerate() {
            let sub = prev[j] + usize::from(r != h);
            row.push(sub.min(prev[j + 1] + 1).min(row[j] + 1));
        }
        prev = row;
    }
    prev[hypothesis.len()] as f32 / reference.len().max(1) as f32
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the real Parakeet model on disk; see the module docs"]
async fn the_real_model_transcribes_the_librispeech_fixtures() {
    let _serial = SERIAL.lock().await;
    let Some(model_path) = env_path("LUNE_TEST_PARAKEET_GGML") else {
        panic!("set LUNE_TEST_PARAKEET_GGML to the downloaded Parakeet model");
    };
    let backend = backend_at(model_path, env_path("LUNE_TEST_VAD_GGML"));
    for name in [
        "librispeech_clean",
        "librispeech_40744",
        "librispeech_clean_noise_snr10",
    ] {
        let audio = load_wav_16k_mono(&fixture(&format!("fixtures/audio/{name}.wav")));
        let reference = std::fs::read_to_string(fixture(&format!("fixtures/audio/{name}.txt")))
            .expect("reference transcript");
        let started = std::time::Instant::now();
        let segments = transcribe(&backend, &audio).await;
        let elapsed = started.elapsed();
        let hypothesis: String = texts(&segments).join(" ");
        let rate = wer(&normalize(&reference), &normalize(&hypothesis));
        let rtf = elapsed.as_secs_f32() / (audio.len() as f32 / SECOND as f32);
        println!(
            "{name}: WER {:.1}%  RTF {rtf:.3}\n  {hypothesis}",
            rate * 100.0
        );
        assert!(
            rate < 0.15,
            "{name}: WER {rate} on a clean read-speech clip"
        );
        // Parakeet punctuates and capitalizes on its own.
        assert!(hypothesis.chars().any(char::is_uppercase), "{hypothesis}");
    }
}

/// A dictation past the 60 s window is cut at a pause and decoded in pieces.
/// The pieces must read back as one transcript, in order, with timestamps that
/// keep rising across the cut.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the real Parakeet model on disk; see the module docs"]
async fn the_real_model_decodes_a_dictation_longer_than_one_window() {
    let _serial = SERIAL.lock().await;
    let Some(model_path) = env_path("LUNE_TEST_PARAKEET_GGML") else {
        panic!("set LUNE_TEST_PARAKEET_GGML to the downloaded Parakeet model");
    };
    let Some(vad_path) = env_path("LUNE_TEST_VAD_GGML") else {
        panic!("set LUNE_TEST_VAD_GGML: the cut needs VAD speech spans to find a pause");
    };
    let backend = backend_at(model_path, Some(vad_path));
    let names = [
        "librispeech_40744",
        "librispeech_clean",
        "librispeech_40744",
    ];
    let mut audio = Vec::new();
    let mut reference = String::new();
    for name in names {
        audio.extend(load_wav_16k_mono(&fixture(&format!(
            "fixtures/audio/{name}.wav"
        ))));
        // A full second of silence between clips: the pause a cut lands in.
        audio.extend(std::iter::repeat_n(0.0, SECOND));
        reference.push_str(
            &std::fs::read_to_string(fixture(&format!("fixtures/audio/{name}.txt")))
                .expect("reference"),
        );
        reference.push(' ');
    }
    assert!(audio.len() > 60 * SECOND, "the clip must exceed one window");

    let segments = transcribe(&backend, &audio).await;
    assert!(segments.len() >= 2, "decoded in more than one piece");
    for pair in segments.windows(2) {
        assert!(
            pair[0].start_ms <= pair[1].start_ms,
            "timestamps run in order across the cut: {:?} then {:?}",
            pair[0].start_ms,
            pair[1].start_ms
        );
    }
    let total_ms = (audio.len() * 1000 / SECOND) as u64;
    assert!(segments.last().unwrap().end_ms.unwrap() <= total_ms);
    let hypothesis = texts(&segments).join(" ");
    for seg in &segments {
        println!("  [{:?}..{:?} ms] {}", seg.start_ms, seg.end_ms, seg.text);
    }
    let rate = wer(&normalize(&reference), &normalize(&hypothesis));
    println!(
        "{:.0} s in {} pieces: WER {:.1}%",
        audio.len() as f32 / SECOND as f32,
        segments.len(),
        rate * 100.0
    );
    assert!(rate < 0.10, "WER {rate} across chunk boundaries");
}

/// Auto-detect: Parakeet picks the language itself, with no hint given.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the real Parakeet model on disk; see the module docs"]
async fn the_real_model_detects_european_languages_by_itself() {
    let _serial = SERIAL.lock().await;
    let Some(model_path) = env_path("LUNE_TEST_PARAKEET_GGML") else {
        panic!("set LUNE_TEST_PARAKEET_GGML to the downloaded Parakeet model");
    };
    let backend = backend_at(model_path, env_path("LUNE_TEST_VAD_GGML"));
    for name in [
        "mls_german",
        "mls_french",
        "mls_spanish",
        "mls_italian",
        "mls_portuguese",
        "mls_dutch",
    ] {
        let audio = load_wav_16k_mono(&fixture(&format!("fixtures/audio/{name}.wav")));
        let reference = std::fs::read_to_string(fixture(&format!("fixtures/audio/{name}.txt")))
            .expect("reference transcript");
        let hypothesis = texts(&transcribe(&backend, &audio).await).join(" ");
        let rate = wer(&normalize(&reference), &normalize(&hypothesis));
        println!("{name}: WER {:.1}%\n  {hypothesis}", rate * 100.0);
        // MLS read speech in auto mode; a wrong-language decode is ~100%.
        assert!(rate < 0.35, "{name}: WER {rate}");
    }
}
