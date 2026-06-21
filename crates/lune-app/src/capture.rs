//! Audio capture + VAD gating stage.
//!
//! Owns the hybrid VAD gate ([`crate::vad::HybridVad`]: Earshot → Silero →
//! chunker) and the capture loop that streams gate-verified audio into the ASR
//! backend. The gate buffers each speech segment and emits [`GateAction`]s —
//! verified-speech pushes plus optional reconstructed inter-segment pauses (so
//! the batch model still hears prosody for punctuation). Extracted from
//! `session.rs` so orchestration and the capture/VAD stage stay separable; the
//! session calls [`run_capture`] and consumes its [`CaptureResult`].

use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AudioConfig, AudioPull, AudioStream};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::audio::SAMPLE_RATE;
use lune_core::config::PipelineConfig;
use lune_core::session::{SessionEvent, SessionStatus};

use crate::session::{SessionAudio, SessionControl};
use crate::vad::{GateAction, HybridVad};

const MAX_SEGMENTS: usize = 500;

const GARBAGE_DROP_THRESHOLD: f32 = 0.25;

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

/// Apply the gate's actions to the ASR stream, returning how many speech
/// samples were pushed (preroll + segment audio; injected pauses don't count).
async fn apply_gate_actions<A: AsrBackend + 'static>(
    asr: &A,
    asr_session: &A::SessionHandle,
    actions: Vec<GateAction>,
    first_asr_error: &mut Option<String>,
) -> usize {
    let mut pushed = 0;
    for action in actions {
        match action {
            GateAction::PushSamples(samples) => {
                pushed += samples.len();
                if let Err(e) = asr.push_audio(asr_session, &samples).await {
                    let msg = e.to_string();
                    warn!(error = %msg, "push_audio error during streaming ASR");
                    first_asr_error.get_or_insert(msg);
                }
            }
            GateAction::SegmentFinalized => {}
        }
    }
    pushed
}

/// Hard ceiling on ASR backend init. A hung Qwen3/Parakeet ONNX session
/// would otherwise block the session forever — caller would never see an
/// error event, hotkey would feel broken until process restart.
const ASR_START_TIMEOUT: Duration = Duration::from_secs(15);

/// Resolved per-session runtime objects built before the capture loop starts.
struct SessionRuntime<H> {
    audio_source: Box<dyn AudioPull>,
    asr_session: H,
    hybrid: HybridVad,
}

/// Open audio capture; not async because all current variants are sync.
fn init_audio_source(
    audio_override: SessionAudio,
    input_device: Option<String>,
) -> std::result::Result<Box<dyn AudioPull>, String> {
    let source: Box<dyn AudioPull> = match audio_override {
        SessionAudio::Owned(stream) => Box::new(stream),
        SessionAudio::Shared(handle) => {
            // Keep ~500ms of pre-session audio so speech captured between
            // hotkey press and this task scheduling isn't lost.
            handle.drain_keep_last(6);
            Box::new(handle)
        }
        SessionAudio::Open => {
            let config = AudioConfig {
                input_device,
                ..AudioConfig::default()
            };
            Box::new(AudioStream::new(config).map_err(|e| format!("audio init: {e}"))?)
        }
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

async fn init_session_runtime<A: AsrBackend + 'static>(
    asr: &A,
    asr_config: AsrSessionConfig,
    audio_override: SessionAudio,
    pipeline: &PipelineConfig,
    vad_pcen_enabled: bool,
    input_device: Option<String>,
) -> std::result::Result<SessionRuntime<A::SessionHandle>, String> {
    let audio_source = init_audio_source(audio_override, input_device)?;
    let asr_session = init_asr_session(asr, asr_config).await?;
    let hybrid = HybridVad::new(pipeline, vad_pcen_enabled)?;
    Ok(SessionRuntime {
        audio_source,
        asr_session,
        hybrid,
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
/// gate-verified audio (with reconstructed pauses) into the ASR backend until a
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
    input_device: Option<String>,
    segment_capacity_hint: usize,
    event_tx: &mpsc::Sender<SessionEvent>,
    control_rx: &mut mpsc::UnboundedReceiver<SessionControl>,
) -> std::result::Result<CaptureResult, String> {
    let mut status = SessionStatus::Completed;

    // ── Initialize audio capture, ASR session, and the hybrid VAD gate ──
    let SessionRuntime {
        audio_source,
        asr_session,
        mut hybrid,
    } = init_session_runtime(
        asr,
        asr_config,
        audio_override,
        pipeline,
        vad_pcen_enabled,
        input_device,
    )
    .await?;
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

                    let pushed = apply_gate_actions(
                        asr,
                        &asr_session,
                        hybrid.process_chunk(&chunk),
                        &mut first_asr_error,
                    )
                    .await;
                    speech_samples += pushed;
                    if pushed > 0 && !first_speech_logged {
                        info!(
                            %session_id,
                            elapsed_ms = session_start.elapsed().as_millis() as u64,
                            "vad: first verified speech",
                        );
                        first_speech_logged = true;
                    }
                    peak_rms = peak_rms.max(hybrid.take_peak_energy());

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
    // give the audio thread one callback cycle to deliver, then run whatever
    // arrives through the gate. Only worth doing when a segment is open.
    if !cancelled && hybrid.in_segment() {
        audio_source.request_flush();
        tokio::time::sleep(Duration::from_millis(30)).await;
        while let Some(chunk) = audio_source.try_next_chunk() {
            speech_samples += apply_gate_actions(
                asr,
                &asr_session,
                hybrid.process_chunk(&chunk),
                &mut first_asr_error,
            )
            .await;
        }
    }

    // Flush any segment still open in the gate (the common case for a short
    // utterance ended by Stop before the hangover elapses).
    if !cancelled {
        speech_samples +=
            apply_gate_actions(asr, &asr_session, hybrid.finish(), &mut first_asr_error).await;
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
