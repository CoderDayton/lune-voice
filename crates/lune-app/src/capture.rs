//! Audio capture + VAD gating stage.
//!
//! Owns the VAD, the preroll / pause-injection state, and the capture loop that
//! streams VAD-passed audio into the ASR backend — reconstructing the pauses
//! the VAD gate strips so the model still hears prosody for punctuation.
//! Extracted from `session.rs` so orchestration and the capture/VAD stage stay
//! separable; the session calls [`run_capture`] and consumes its
//! [`CaptureResult`].

use std::collections::VecDeque;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AudioConfig, AudioPull, AudioStream, Vad, VadConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::audio::SAMPLE_RATE;
use lune_core::config::PipelineConfig;
use lune_core::session::{SessionEvent, SessionStatus};

use crate::session::{SessionAudio, SessionControl};

const MAX_SEGMENTS: usize = 500;

const GARBAGE_DROP_THRESHOLD: f32 = 0.25;

/// Number of pre-speech chunks (~80 ms each at 16 kHz / 1280 samples) kept
/// in a rolling buffer and flushed to ASR when VAD first fires. Sized to
/// cover word-onset consonants/breaths that score below `threshold` but
/// belong to the same word as the vowel that did fire — without enough
/// pre-roll, hard-onset words like "hello" arrive at ASR as "ello".
const PREROLL_CHUNKS: usize = 4;

/// Minimum total acoustic gap (last speech → next speech) that counts as an
/// intentional pause rather than a breath. Defined on the TOTAL gap, not the
/// post-hangover silence, so its meaning is stable under UI/calibrator VAD
/// changes and any future hangover tuning. 650 ms proved too eager in real
/// dictation — every compositional hesitation became a comma — so only
/// gaps of a full second and up count as deliberate pauses.
const PAUSE_TOTAL_MIN_SAMPLES: usize = SAMPLE_RATE; // 1000 ms
/// Ceiling on the total pause the model hears. Longer pauses carry no extra
/// punctuation information and only cost encoder frames.
const PAUSE_TOTAL_CAP_SAMPLES: usize = (SAMPLE_RATE * 3) / 2; // 1500 ms

/// How much silence to feed ASR at a silence→speech transition so the model
/// hears the pause the VAD gate removed — exactly.
///
/// `gap_fed_samples` is MEASURED, not configured: every chunk fed live while
/// `is_speech` held but the VAD score sat below `speech_threshold_end` (the
/// hangover tail) is counted by the caller, so the math is exact under any
/// UI/calibrator threshold or hangover setting and any chunk size. The true
/// gated gap is `gap_fed + silence`; injecting `silence − preroll` (capped)
/// makes the heard pause `min(true gap, PAUSE_TOTAL_CAP_SAMPLES)`
/// sample-for-sample.
fn pause_injection_samples(
    silence_samples: usize,
    preroll_samples: usize,
    gap_fed_samples: usize,
) -> usize {
    if silence_samples == 0 {
        // Speech resumed within the hangover: the entire gap was fed live.
        return 0;
    }
    let total_gap = gap_fed_samples + silence_samples;
    if total_gap < PAUSE_TOTAL_MIN_SAMPLES {
        return 0;
    }
    let already_heard = gap_fed_samples + preroll_samples;
    silence_samples
        .saturating_sub(preroll_samples)
        .min(PAUSE_TOTAL_CAP_SAMPLES.saturating_sub(already_heard))
}

fn append_segments(
    all_segments: &mut Vec<AsrSegment>,
    joined_text: &mut String,
    new_segments: Vec<AsrSegment>,
) -> bool {
    let mut appended = false;
    for seg in new_segments {
        let trimmed = seg.text.trim();
        if trimmed.is_empty() {
            all_segments.push(seg);
            continue;
        }

        if let Some(conf) = seg.confidence
            && conf < GARBAGE_DROP_THRESHOLD
        {
            warn!(
                confidence = conf,
                text = trimmed,
                "dropping garbage segment"
            );
            continue;
        }

        if !joined_text.is_empty() {
            joined_text.push(' ');
        }
        joined_text.push_str(trimmed);
        appended = true;

        if all_segments.len() >= MAX_SEGMENTS {
            warn!("segment limit ({MAX_SEGMENTS}) reached, dropping oldest");
            all_segments.remove(0);
        }
        all_segments.push(seg);
    }
    appended
}

/// Hard ceiling on ASR backend init. A hung Qwen3/Parakeet ONNX session
/// would otherwise block the session forever — caller would never see an
/// error event, hotkey would feel broken until process restart.
const ASR_START_TIMEOUT: Duration = Duration::from_secs(15);

/// Resolved per-session runtime objects built before the capture loop starts.
struct SessionRuntime<H> {
    audio_source: Box<dyn AudioPull>,
    asr_session: H,
    vad: Vad,
}

/// Open audio capture; not async because all current variants are sync.
fn init_audio_source(
    audio_override: SessionAudio,
) -> std::result::Result<Box<dyn AudioPull>, String> {
    let source: Box<dyn AudioPull> = match audio_override {
        SessionAudio::Owned(stream) => Box::new(stream),
        SessionAudio::Shared(handle) => {
            // Keep ~500ms of pre-session audio so speech captured between
            // hotkey press and this task scheduling isn't lost.
            handle.drain_keep_last(6);
            Box::new(handle)
        }
        SessionAudio::Open => Box::new(
            AudioStream::new(AudioConfig::default()).map_err(|e| format!("audio init: {e}"))?,
        ),
    };
    Ok(source)
}

/// Start the ASR backend session with a hard timeout.
async fn init_asr_session<A: AsrBackend + 'static>(
    asr: &A,
    asr_config: AsrSessionConfig,
) -> std::result::Result<A::SessionHandle, String> {
    match tokio::time::timeout(ASR_START_TIMEOUT, asr.start_session(asr_config)).await {
        Ok(Ok(s)) => Ok(s),
        Ok(Err(e)) => Err(format!("ASR init: {e}")),
        Err(_) => Err(format!(
            "ASR init: backend did not initialize within {ASR_START_TIMEOUT:?}"
        )),
    }
}

fn init_vad(pipeline: &PipelineConfig, vad_pcen_enabled: bool) -> std::result::Result<Vad, String> {
    let mut vad_config = VadConfig {
        threshold: pipeline.speech_threshold,
        threshold_end: pipeline.speech_threshold_end,
        ..VadConfig::default()
    };
    if !vad_pcen_enabled {
        vad_config.pcen = None;
    }
    Vad::new(vad_config, SAMPLE_RATE as u32).map_err(|e| format!("VAD init: {e}"))
}

async fn init_session_runtime<A: AsrBackend + 'static>(
    asr: &A,
    asr_config: AsrSessionConfig,
    audio_override: SessionAudio,
    pipeline: &PipelineConfig,
    vad_pcen_enabled: bool,
) -> std::result::Result<SessionRuntime<A::SessionHandle>, String> {
    let audio_source = init_audio_source(audio_override)?;
    let asr_session = init_asr_session(asr, asr_config).await?;
    let vad = init_vad(pipeline, vad_pcen_enabled)?;
    Ok(SessionRuntime {
        audio_source,
        asr_session,
        vad,
    })
}

/// Outcome of the capture + VAD-gating stage, consumed by `run_session`.
pub(crate) struct CaptureResult {
    pub segments: Vec<AsrSegment>,
    pub raw_text: String,
    pub speech_samples: usize,
    pub cancelled: bool,
    pub first_asr_error: Option<String>,
    pub saw_asr_success: bool,
    pub status: SessionStatus,
}

/// Run the audio-capture + VAD-gating stage: build the runtime, stream
/// VAD-passed audio (with reconstructed pauses) into the ASR backend until a
/// Stop/Cancel/deadline/max-speech/starvation condition, then finalize the
/// backend (`poll_final`) into segments. Returns `Err(msg)` if the runtime
/// fails to initialize (the caller surfaces it via `fail_init`).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_capture<A: AsrBackend + 'static>(
    session_id: Uuid,
    asr: &A,
    asr_config: AsrSessionConfig,
    audio_override: SessionAudio,
    pipeline: &PipelineConfig,
    vad_pcen_enabled: bool,
    segment_capacity_hint: usize,
    event_tx: &mpsc::Sender<SessionEvent>,
    control_rx: &mut mpsc::UnboundedReceiver<SessionControl>,
) -> std::result::Result<CaptureResult, String> {
    let mut status = SessionStatus::Completed;

    // ── Initialize audio capture, ASR session, and VAD ──────────────────
    let SessionRuntime {
        audio_source,
        asr_session,
        mut vad,
    } = init_session_runtime(asr, asr_config, audio_override, pipeline, vad_pcen_enabled).await?;
    let _ = event_tx.send(SessionEvent::Started).await;

    let mut segments = Vec::with_capacity(segment_capacity_hint);
    let mut raw_text = String::with_capacity(256);

    let mut speech_samples: usize = 0;
    let mut first_asr_error: Option<String> = None;
    let mut saw_asr_success = false;
    let mut cancelled = false;

    let session_deadline = tokio::time::Instant::now()
        + Duration::from_secs((pipeline.max_speech_samples / SAMPLE_RATE) as u64 * 3);

    let mut audio_poll_tick = tokio::time::interval(Duration::from_millis(10));
    audio_poll_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut next_level_emit = tokio::time::Instant::now();
    let level_emit_interval = Duration::from_millis(67); // ~15 fps
    let mut peak_rms: f32 = 0.0;
    let mut preroll: VecDeque<Vec<f32>> = VecDeque::with_capacity(PREROLL_CHUNKS);
    let mut was_speech = false;
    // Silence samples accumulated since the last speech burst ended. The VAD
    // gate strips inter-burst silence from the audio fed to ASR, which also
    // strips the model's punctuation evidence — Qwen3-ASR (like Whisper)
    // places commas and sentence breaks from pause prosody. A bounded slice
    // of each real pause is re-injected so the model hears it.
    let mut silence_samples: usize = 0;
    // Samples fed to ASR during the hangover tail (is_speech still true but
    // VAD score below the end threshold): real silence the model already
    // heard, measured exactly so injection completes the gap rather than
    // double-counting it.
    let mut gap_fed_samples: usize = 0;
    let mut last_audio_at = tokio::time::Instant::now();
    let session_start = tokio::time::Instant::now();
    let mut first_audio_logged = false;
    let mut first_speech_logged = false;
    let audio_starvation_limit = Duration::from_secs(5);
    'capture: loop {
        tokio::select! {
            biased;

            control = control_rx.recv() => {
                match control {
                    Some(SessionControl::Stop) => {
                        debug!(%session_id, "stop signal received during recording");
                        break 'capture;
                    }
                    Some(SessionControl::Cancel) => {
                        debug!(%session_id, "cancel signal received during recording");
                        cancelled = true;
                        break 'capture;
                    }
                    None => break 'capture,
                }
            }

            _ = tokio::time::sleep_until(session_deadline) => {
                warn!(%session_id, "session wall-clock deadline reached, forcing stop");
                break 'capture;
            }

            _ = audio_poll_tick.tick() => {
                let mut saw_audio = false;
                while let Some(chunk) = audio_source.try_next_chunk() {
                    saw_audio = true;
                    if !first_audio_logged {
                        let elapsed_ms = session_start.elapsed().as_millis();
                        info!(
                            %session_id,
                            elapsed_ms,
                            samples = chunk.samples.len(),
                            "vad: first audio chunk received",
                        );
                        first_audio_logged = true;
                    }
                    let vad_result = vad.process_chunk(&chunk.samples);
                    if vad_result.energy > peak_rms { peak_rms = vad_result.energy; }
                    if !vad_result.is_speech {
                        if speech_samples > 0 {
                            silence_samples += chunk.samples.len();
                        }
                        if preroll.len() == PREROLL_CHUNKS {
                            preroll.pop_front();
                        }
                        preroll.push_back(chunk.samples);
                        was_speech = false;
                        continue;
                    }

                    if !first_speech_logged {
                        info!(
                            %session_id,
                            elapsed_ms = session_start.elapsed().as_millis() as u64,
                            "vad: first speech detected",
                        );
                        first_speech_logged = true;
                    }
                    if vad_result.vad_score < pipeline.speech_threshold_end {
                        // Hangover tail: silence fed live as "speech".
                        gap_fed_samples += chunk.samples.len();
                    } else if was_speech {
                        gap_fed_samples = 0;
                    }
                    if !was_speech {
                        let inject = pause_injection_samples(
                            silence_samples,
                            preroll.iter().map(Vec::len).sum(),
                            gap_fed_samples,
                        );
                        if inject > 0 {
                            let gap_ms =
                                (gap_fed_samples + silence_samples) * 1000 / SAMPLE_RATE;
                            info!(%session_id, gap_ms, inject_ms = inject * 1000 / SAMPLE_RATE,
                                "vad: re-injecting pause for punctuation prosody");
                            let _ = asr.push_audio(&asr_session, &vec![0.0f32; inject]).await;
                        }
                        for pre in preroll.drain(..) {
                            let _ = asr.push_audio(&asr_session, &pre).await;
                        }
                        gap_fed_samples = 0;
                    }
                    silence_samples = 0;
                    was_speech = true;
                    speech_samples += chunk.samples.len();

                    if let Err(e) = asr.push_audio(&asr_session, &chunk.samples).await {
                        let msg = e.to_string();
                        warn!(error = %msg, "push_audio error during streaming ASR");
                        first_asr_error.get_or_insert(msg);
                        continue;
                    }

                    if speech_samples >= pipeline.max_speech_samples {
                        info!(
                            %session_id,
                            speech_secs = speech_samples / SAMPLE_RATE,
                            "max speech duration reached; stopping session"
                        );
                        break 'capture;
                    }
                }

                // Throttled audio level emission (~15 fps)
                let now = tokio::time::Instant::now();
                if now >= next_level_emit && peak_rms > 0.0 {
                    // Clamp to [0, 1] — typical speech RMS is 0.01–0.15
                    let normalized = (peak_rms * 6.0).min(1.0);
                    let _ = event_tx.try_send(SessionEvent::AudioLevel(normalized));
                    peak_rms = 0.0;
                    next_level_emit = now + level_emit_interval;
                }

                if saw_audio {
                    last_audio_at = tokio::time::Instant::now();
                } else if !audio_source.is_offline()
                    && last_audio_at.elapsed() > audio_starvation_limit
                {
                    warn!(%session_id, "no audio received for 5s — mic may have disconnected");
                    let _ = event_tx.send(SessionEvent::Error("audio device stopped responding".into())).await;
                    status = SessionStatus::Failed("audio starvation".into());
                    break 'capture;
                }

                if !saw_audio && audio_source.is_offline() {
                    debug!(%session_id, "offline audio exhausted");
                    break 'capture;
                }
            }
        }
    }

    // Recover the producer-side audio tail: the capture callback may hold up
    // to one chunk (~80 ms) of audio past the last full-chunk boundary —
    // often the final phonemes of the last word. Ask the producer to flush,
    // give the audio thread one callback cycle to deliver, then feed whatever
    // arrives to ASR. Only worth doing when speech was in progress.
    if !cancelled && was_speech {
        audio_source.request_flush();
        tokio::time::sleep(Duration::from_millis(30)).await;
        while let Some(chunk) = audio_source.try_next_chunk() {
            speech_samples += chunk.samples.len();
            if let Err(e) = asr.push_audio(&asr_session, &chunk.samples).await {
                warn!(error = %e, "push_audio error during tail flush");
                break;
            }
        }
    }

    if !cancelled {
        let _ = event_tx.send(SessionEvent::Processing).await;
    }

    if cancelled {
        if let Err(e) = asr.reset_context(&asr_session).await {
            warn!(error = %e, "reset_context error during cancel");
        }
    } else {
        let poll_final_t0 = tokio::time::Instant::now();
        let poll_final_result = asr.poll_final(&asr_session).await;
        let poll_final_ms = poll_final_t0.elapsed().as_millis() as u64;
        info!(%session_id, poll_final_ms, "asr: poll_final complete");
        match poll_final_result {
            Ok(new_segments) => {
                if append_segments(&mut segments, &mut raw_text, new_segments) {
                    saw_asr_success = true;
                }
            }
            Err(e) => {
                let msg = e.to_string();
                warn!(error = %msg, "poll_final error");
                first_asr_error.get_or_insert(msg);
            }
        }
    }

    if let Err(e) = asr.end_session(asr_session).await {
        warn!(error = %e, "end_session error");
        status = SessionStatus::Failed(e.to_string());
    }

    Ok(CaptureResult {
        segments,
        raw_text,
        speech_samples,
        cancelled,
        first_asr_error,
        saw_asr_success,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Chunk-quantized fed-gap values as the session actually measures them:
    // hangover rounds UP to whole 80 ms chunks (3 chunks for 240 ms, 4 for
    // a 250 ms setting, 1 for 30 ms), preroll is 4 × 80 ms chunks.
    const CHUNK: usize = SAMPLE_RATE * 80 / 1000;
    const PREROLL: usize = 4 * CHUNK;

    #[test]
    fn pause_injection_breath_gaps_skipped() {
        // Gap fully absorbed by hangover: everything was fed live.
        assert_eq!(pause_injection_samples(0, 0, 3 * CHUNK), 0);
        // Total gap 240 + 320 = 560 ms < 650 ms floor: a breath.
        assert_eq!(pause_injection_samples(PREROLL, PREROLL, 3 * CHUNK), 0);
    }

    #[test]
    fn pause_injection_heard_equals_true_gap() {
        // heard = gap_fed + inject + preroll must equal the measured gap
        // exactly, for any fed-gap size (i.e. any hangover configuration).
        for gap_fed in [CHUNK, 3 * CHUNK, 6 * CHUNK] {
            let silence = SAMPLE_RATE - gap_fed; // 1 s total gap
            let inject = pause_injection_samples(silence, PREROLL, gap_fed);
            assert_eq!(gap_fed + inject + PREROLL, SAMPLE_RATE, "gap_fed={gap_fed}");
        }
    }

    #[test]
    fn pause_injection_caps_total_heard() {
        // 5 s gap: heard must be exactly the 1500 ms total cap.
        for gap_fed in [CHUNK, 3 * CHUNK] {
            let silence = SAMPLE_RATE * 5 - gap_fed;
            let inject = pause_injection_samples(silence, PREROLL, gap_fed);
            assert_eq!(
                gap_fed + inject + PREROLL,
                PAUSE_TOTAL_CAP_SAMPLES,
                "gap_fed={gap_fed}"
            );
        }
        // Fed gap already beyond the cap: never underflow, inject nothing.
        assert_eq!(pause_injection_samples(SAMPLE_RATE, 0, SAMPLE_RATE * 2), 0);
    }
}
