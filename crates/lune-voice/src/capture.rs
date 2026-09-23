// SPDX-License-Identifier: GPL-3.0-or-later

//! Audio capture stage.
//!
//! Streams microphone audio into the ASR backend, which buffers the whole
//! utterance and decodes it on stop. An optional PCEN adaptive-gain frontend
//! (the `vad_pcen` setting) boosts quiet mics toward a target speech level
//! before the audio reaches the recognizer; there is no online VAD in this
//! stage. Extracted from `session.rs` so
//! orchestration and the capture stage stay separable; the session calls
//! [`run_capture`] and consumes its [`CaptureResult`].

use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AudioConfig, AudioPull, AudioStream, PcenAgc, PcenAgcConfig};
use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::audio::SAMPLE_RATE;
use lune_core::config::PipelineConfig;
use lune_core::session::{SessionEvent, SessionStatus};

use crate::redact::Transcript;
use crate::session::{SessionAudio, SessionControl, send_bounded};

const MAX_SEGMENTS: usize = 500;

const GARBAGE_DROP_THRESHOLD: f32 = 0.25;

/// Hop size for the adaptive-gain frontend, shared with the VAD's own PCEN pass
/// so both re-frame audio the way the gain envelope was tuned for.
const AGC_HOP: usize = lune_audio::PCEN_HOP_SIZE;

/// Minimum fraction of "speech-like" characters (letters, digits, and the
/// punctuation that occurs in real speech) a segment must contain to be kept.
const MIN_SPEECH_CONTENT_RATIO: f32 = 0.3;

/// Reject ASR segments that are almost certainly hallucinations on silence: a
/// segment with no alphanumerics ("...", "[BLANK_AUDIO]"-style residue, lone
/// punctuation) or one that is mostly symbols. Mirrors vocalinux's non-speech
/// filter. Pure-text and ordinary punctuated speech always pass.
fn is_speech_content(text: &str) -> bool {
    let probe = text.trim();
    if probe.is_empty() {
        return false;
    }
    // No letters or digits at all → not speech.
    if !probe.chars().any(char::is_alphanumeric) {
        return false;
    }
    let non_space = probe.chars().filter(|c| !c.is_whitespace()).count();
    if non_space == 0 {
        return false;
    }
    let speechy = probe
        .chars()
        .filter(|c| c.is_alphanumeric() || ".,!?-'\"".contains(*c))
        .count();
    (speechy as f32 / non_space as f32) >= MIN_SPEECH_CONTENT_RATIO
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
                text = %Transcript::new(trimmed),
                "dropping garbage segment"
            );
            continue;
        }

        if !is_speech_content(trimmed) {
            warn!(text = %Transcript::new(trimmed), "dropping non-speech segment");
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

/// Root-mean-square amplitude of a chunk, for the UI level meter. Typical
/// speech RMS is 0.01–0.15.
fn rms_of(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Hard ceiling on ASR backend init. A hung local model session
/// would otherwise block the session forever — caller would never see an
/// error event, hotkey would feel broken until process restart.
const ASR_START_TIMEOUT: Duration = Duration::from_secs(15);

/// Base ceiling on the final decode; [`final_decode_ceiling`] adds the clip's
/// duration.
///
/// Deliberately generous: a long utterance on a large model genuinely takes
/// tens of seconds on CPU, and cutting a real transcription short is worse than
/// waiting. The point is that "forever" is not an option — the remote backend
/// retries with backoff and would otherwise sit here indefinitely with the pill
/// still up. Cancellation, not this ceiling, is what a user actually reaches
/// for; the two are raced together.
const POLL_FINAL_TIMEOUT: Duration = Duration::from_secs(180);

/// The final-decode ceiling for a clip of `speech_samples`:
/// [`POLL_FINAL_TIMEOUT`] plus the clip's duration.
///
/// The remote backend gives each request its own deadline of a base plus the
/// clip's duration, so a fixed ceiling would cut a long request off before
/// that deadline, however much the user raised `max_speech_secs`.
fn final_decode_ceiling(speech_samples: usize) -> Duration {
    POLL_FINAL_TIMEOUT + Duration::from_secs((speech_samples / SAMPLE_RATE) as u64)
}

/// How the final decode ended.
enum PollOutcome {
    Finished(lune_core::Result<Vec<AsrSegment>>),
    TimedOut,
    Cancelled,
}

/// Resolved per-session runtime objects built before the capture loop starts.
struct SessionRuntime<H> {
    audio_source: Box<dyn AudioPull>,
    asr_session: H,
}

/// Open audio capture; not async because all current variants are sync.
/// Optional PCEN adaptive-gain frontend applied to mic audio before it reaches
/// the recognizer. Quiet mics are boosted toward a target speech level so soft
/// word onsets (notably the first word) aren't lost. Disabled => passthrough.
///
/// PCEN updates its envelopes per fixed hop, so variable-length input chunks are
/// re-framed to [`AGC_HOP`]; the sub-hop remainder is held until the next
/// `push` and emitted by [`MicAgc::flush`] at end of utterance.
struct MicAgc {
    agc: Option<PcenAgc>,
    pending: Vec<f32>,
}

impl MicAgc {
    fn new(enabled: bool) -> Self {
        let agc = enabled
            .then(|| PcenAgc::new(PcenAgcConfig::default(), SAMPLE_RATE as u32, AGC_HOP))
            .transpose()
            .unwrap_or_else(|e| {
                warn!(error = %e, "failed to init PCEN AGC; using raw mic audio");
                None
            });
        Self {
            agc,
            pending: Vec::with_capacity(AGC_HOP * 4),
        }
    }

    /// Append gain-adjusted samples for `input` to `out`. Passthrough when disabled.
    fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let Some(agc) = self.agc.as_mut() else {
            out.extend_from_slice(input);
            return;
        };
        self.pending.extend_from_slice(input);
        let mut frame = [0f32; AGC_HOP];
        while self.pending.len() >= AGC_HOP {
            frame.copy_from_slice(&self.pending[..AGC_HOP]);
            agc.process_frame(&mut frame);
            out.extend_from_slice(&frame);
            self.pending.drain(..AGC_HOP);
        }
    }

    /// Emit the held sub-hop remainder, scaled by the last applied gain so the
    /// final phonemes match the level of the preceding audio.
    fn flush(&mut self, out: &mut Vec<f32>) {
        match self.agc.as_ref() {
            Some(agc) => {
                let g = agc.last_gain();
                out.extend(self.pending.drain(..).map(|s| s * g));
            }
            None => out.append(&mut self.pending),
        }
    }
}

fn init_audio_source(
    audio_override: SessionAudio,
    input_device: Option<String>,
) -> std::result::Result<Box<dyn AudioPull>, String> {
    let source: Box<dyn AudioPull> = match audio_override {
        SessionAudio::Owned(stream) => Box::new(stream),
        SessionAudio::Shared(handle) => {
            // The ring is drained at key-press time (see runtime.rs start path),
            // anchoring the pre-roll to the press rather than to this task's
            // scheduling. Draining again here would re-trim to the last ~480 ms
            // and discard audio captured during pipeline warmup — exactly the
            // lead-in we want to keep — so we deliberately don't. A large
            // backlog here means the caller skipped that drain and we'd
            // transcribe stale idle audio, so surface the contract violation.
            let fill = handle.buffer_fill();
            if fill > 0.75 {
                warn!(
                    buffer_fill = fill,
                    "shared audio handle backlogged at session start; caller did not drain — stale audio may be transcribed"
                );
            }
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
    input_device: Option<String>,
) -> std::result::Result<SessionRuntime<A::SessionHandle>, String> {
    let audio_source = init_audio_source(audio_override, input_device)?;
    let asr_session = init_asr_session(asr, asr_config).await?;
    Ok(SessionRuntime {
        audio_source,
        asr_session,
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
    vad_pcen: bool,
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
    } = init_session_runtime(asr, asr_config, audio_override, input_device).await?;
    // From here until this function returns, evictions in the ring mean *we*
    // fell behind. Outside the guard a pre-warmed stream's ring is just a
    // rolling pre-roll window, and its evictions are expected, not a stall.
    let _consuming = audio_source.consume();
    let _ = send_bounded(event_tx, SessionEvent::Started).await;

    let mut segments = Vec::with_capacity(segment_capacity_hint);
    let mut raw_text = String::with_capacity(256);

    let mut speech_samples: usize = 0;
    let mut first_asr_error: Option<String> = None;
    let mut saw_asr_success = false;
    let mut cancelled = false;

    // Adaptive-gain frontend (PCEN): boosts quiet mics so soft onsets aren't
    // lost. Disabled => passthrough. `framed` is the reusable output scratch.
    let mut agc = MicAgc::new(vad_pcen);
    let mut framed: Vec<f32> = Vec::with_capacity(AGC_HOP * 8);

    // `.max(1)`: a misconfigured max_speech_samples below one second of audio
    // would otherwise yield a zero deadline and end every session instantly.
    let session_deadline = tokio::time::Instant::now()
        + Duration::from_secs((pipeline.max_speech_samples / SAMPLE_RATE).max(1) as u64 * 3);

    let mut audio_poll_tick = tokio::time::interval(Duration::from_millis(10));
    audio_poll_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut next_level_emit = tokio::time::Instant::now();
    let level_emit_interval = Duration::from_millis(67); // ~15 fps
    let mut peak_rms: f32 = 0.0;
    let mut last_audio_at = tokio::time::Instant::now();
    let session_start = tokio::time::Instant::now();
    let mut first_audio_logged = false;
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

                    // Adaptive-gain (if enabled) before the recognizer; whisper
                    // decodes the full buffer (no online VAD in this stage).
                    framed.clear();
                    agc.push(&chunk.samples, &mut framed);
                    if let Err(e) = asr.push_audio(&asr_session, &framed).await {
                        let msg = e.to_string();
                        warn!(error = %msg, "push_audio error during capture");
                        first_asr_error.get_or_insert(msg);
                    }
                    speech_samples += chunk.samples.len();
                    // Meter the POST-AGC signal (`framed`) so the bar reflects
                    // what the recognizer hears — a quiet mic that the AGC boosts
                    // now reads correctly instead of barely moving.
                    peak_rms = peak_rms.max(rms_of(&framed));

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
                    // Map the post-AGC level onto [0, 1] relative to the AGC's
                    // target RMS: subtract the noise floor so silence reads empty,
                    // and scale so normal speech (≈ target) nearly fills the bar
                    // with headroom for louder peaks. (With AGC off this meters
                    // the raw signal, so a quiet mic legitimately reads low.)
                    let floor = lune_audio::DEFAULT_NOISE_CEILING_RMS;
                    let span = (lune_audio::EXPECTED_OUTPUT_RMS - floor).max(f32::EPSILON);
                    let normalized = ((peak_rms - floor) / span * 0.9).clamp(0.0, 1.0);
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
                    let _ = send_bounded(
                        event_tx,
                        SessionEvent::Error("audio device stopped responding".into()),
                    )
                    .await;
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
    // to one chunk (~80 ms) of audio past the last full-chunk boundary — often
    // the final phonemes of the last word. Ask the producer to flush, give the
    // audio thread one callback cycle to deliver, then push whatever arrives.
    if !cancelled {
        audio_source.request_flush();
        tokio::time::sleep(Duration::from_millis(30)).await;
        while let Some(chunk) = audio_source.try_next_chunk() {
            framed.clear();
            agc.push(&chunk.samples, &mut framed);
            if let Err(e) = asr.push_audio(&asr_session, &framed).await {
                let msg = e.to_string();
                warn!(error = %msg, "push_audio error during tail flush");
                first_asr_error.get_or_insert(msg);
            }
            speech_samples += chunk.samples.len();
        }
        // Emit the AGC's held sub-hop remainder so the final phonemes survive.
        framed.clear();
        agc.flush(&mut framed);
        if !framed.is_empty()
            && let Err(e) = asr.push_audio(&asr_session, &framed).await
        {
            let msg = e.to_string();
            warn!(error = %msg, "push_audio error during agc tail flush");
            first_asr_error.get_or_insert(msg);
        }
    }

    // Audio lost to a full ring is invisible downstream — the transcript simply
    // starts mid-sentence — so say so here, where the cause is still known.
    let dropped = audio_source.dropped_chunks();
    if dropped > 0 {
        warn!(
            %session_id,
            dropped,
            "audio chunks were discarded during capture; the start of this dictation is missing"
        );
    }

    if !cancelled {
        let _ = send_bounded(event_tx, SessionEvent::Processing).await;
    }

    if cancelled {
        if let Err(e) = asr.reset_context(&asr_session).await {
            warn!(error = %e, "reset_context error during cancel");
        }
    } else {
        let poll_final_t0 = tokio::time::Instant::now();
        // Race the decode against a cancel and a hard ceiling. Unraced, this is
        // the one unbounded await in the session: the remote backend retries
        // with backoff and can sit here for minutes with the pill up, and a
        // Cancel arriving meanwhile would go unread until it returned.
        //
        // This only bites the remote backend, and only it can be interrupted.
        // `WhisperCppBackend::poll_final` has no await points — it runs the
        // whole decode inside `block_in_place` — so its first poll returns
        // `Ready` and neither the ceiling nor a late Cancel ever gets a turn.
        // Making a `!Send` whisper decode interruptible is a backend redesign,
        // and the local decode is bounded by the utterance anyway; the remote
        // one, which is genuinely unbounded, is the case that needed this.
        let decode = asr.poll_final(&asr_session);
        let ceiling_after = final_decode_ceiling(speech_samples);
        let ceiling = tokio::time::sleep(ceiling_after);
        tokio::pin!(decode, ceiling);
        let mut listening = true;
        let outcome = loop {
            tokio::select! {
                biased;

                control = control_rx.recv(), if listening => match control {
                    Some(SessionControl::Cancel) => break PollOutcome::Cancelled,
                    // A stop is what got us here; asking twice changes nothing.
                    Some(SessionControl::Stop) => {}
                    // The manager is gone. Nothing more will arrive, so stop
                    // polling this branch instead of spinning on it.
                    None => listening = false,
                },

                result = &mut decode => break PollOutcome::Finished(result),

                () = &mut ceiling => break PollOutcome::TimedOut,
            }
        };
        let poll_final_ms = poll_final_t0.elapsed().as_millis() as u64;
        info!(%session_id, poll_final_ms, "asr: poll_final complete");
        match outcome {
            PollOutcome::Finished(Ok(new_segments)) => {
                if append_segments(&mut segments, &mut raw_text, new_segments) {
                    saw_asr_success = true;
                }
            }
            PollOutcome::Finished(Err(e)) => {
                let msg = e.to_string();
                warn!(error = %msg, "poll_final error");
                first_asr_error.get_or_insert(msg);
            }
            PollOutcome::TimedOut => {
                warn!(%session_id, poll_final_ms, "transcription timed out");
                first_asr_error.get_or_insert(format!(
                    "transcription did not finish within {}s",
                    ceiling_after.as_secs()
                ));
            }
            PollOutcome::Cancelled => {
                debug!(%session_id, "cancel signal received while transcribing");
                cancelled = true;
                if let Err(e) = asr.reset_context(&asr_session).await {
                    warn!(error = %e, "reset_context error during cancel");
                }
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
    use super::{POLL_FINAL_TIMEOUT, final_decode_ceiling, is_speech_content};
    use lune_core::audio::SAMPLE_RATE;
    use std::time::Duration;

    #[test]
    fn a_long_utterance_extends_the_final_decode_ceiling() {
        assert_eq!(final_decode_ceiling(0), POLL_FINAL_TIMEOUT);
        // A remote request's own deadline is 30 s plus the clip, so the outer
        // ceiling has to grow with the clip too or it cuts that request short.
        let clip = 200 * SAMPLE_RATE;
        assert_eq!(
            final_decode_ceiling(clip),
            POLL_FINAL_TIMEOUT + Duration::from_secs(200)
        );
    }

    #[test]
    fn keeps_ordinary_speech() {
        assert!(is_speech_content("Hello, world."));
        assert!(is_speech_content(
            "That's not even a GPU, it's pretty fast, huh?"
        ));
        assert!(is_speech_content("testing123"));
        assert!(is_speech_content("ok"));
    }

    #[test]
    fn drops_hallucinations_on_silence() {
        assert!(!is_speech_content(""));
        assert!(!is_speech_content("   "));
        assert!(!is_speech_content("..."));
        assert!(!is_speech_content("?!"));
        assert!(!is_speech_content("[ ... ]"));
        assert!(!is_speech_content("- -- ---"));
    }
}
