//! Microphone calibration session.
//!
//! Owns its own `AudioStream` + `Vad` and emits per-frame VAD scores so a
//! settings UI can derive recommended start/end thresholds for the user's
//! mic + voice + room. Independent of the dictation pipeline.
//!
//! Lifecycle: caller invokes [`CalibrationSession::start`] to begin the
//! `Noise` phase, then [`CalibrationSession::advance_to_speech`] when the
//! user is ready to read the prompt. The task auto-finalizes after the
//! speech phase elapses, emitting either a [`CalibrationFrame::Result`]
//! or [`CalibrationFrame::Error`].

use std::sync::Arc;
use std::time::Duration;

use lune_audio::{
    AudioConfig, AudioStream, CalibrationInput, CalibrationResult, Vad, VadConfig,
    calibrate::{self, CalibrationError},
};
use tokio::sync::{Notify, mpsc};
use tracing::{debug, warn};

const NOISE_DURATION: Duration = Duration::from_millis(3_000);
const SPEECH_DURATION: Duration = Duration::from_millis(12_000);
const POLL_INTERVAL: Duration = Duration::from_millis(33);
const FRAME_EMIT_INTERVAL: Duration = Duration::from_millis(33);

/// Injectable timing knobs so tests can run the state machine in milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationTimings {
    pub noise: Duration,
    pub speech: Duration,
}

impl Default for CalibrationTimings {
    fn default() -> Self {
        Self {
            noise: NOISE_DURATION,
            speech: SPEECH_DURATION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationPhase {
    Idle,
    Noise,
    Waiting,
    Speech,
    Done,
    Cancelled,
    Failed,
}

impl CalibrationPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Noise => "noise",
            Self::Waiting => "waiting",
            Self::Speech => "speech",
            Self::Done => "done",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub enum CalibrationFrame {
    PhaseChanged(CalibrationPhase),
    Frame {
        phase: CalibrationPhase,
        vad_score: f32,
        energy: f32,
    },
    Result(CalibrationResult),
    Error(String),
}

pub struct CalibrationSession {
    advance: Arc<Notify>,
    cancel: Arc<Notify>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl CalibrationSession {
    /// Spawn the calibration task with a freshly-opened mic. Frames are sent
    /// on `event_tx`; the task closes the channel when finalized.
    pub fn start(event_tx: mpsc::Sender<CalibrationFrame>, input_device: Option<String>) -> Self {
        Self::start_with(event_tx, None, CalibrationTimings::default(), input_device)
    }

    /// Spawn the calibration task using a caller-supplied `AudioStream`.
    /// Used by tests to inject a file-backed stream. `input_device` selects the
    /// capture device when `stream` is `None` (ignored when a stream is given).
    pub fn start_with(
        event_tx: mpsc::Sender<CalibrationFrame>,
        stream: Option<AudioStream>,
        timings: CalibrationTimings,
        input_device: Option<String>,
    ) -> Self {
        let advance = Arc::new(Notify::new());
        let cancel = Arc::new(Notify::new());
        let advance_task = Arc::clone(&advance);
        let cancel_task = Arc::clone(&cancel);

        let handle = tokio::spawn(async move {
            run_calibration(
                event_tx,
                advance_task,
                cancel_task,
                stream,
                timings,
                input_device,
            )
            .await;
        });

        Self {
            advance,
            cancel,
            handle: Some(handle),
        }
    }

    /// Move the state machine from `Waiting` (after Noise) into `Speech`.
    /// Idempotent; called by the UI when the user clicks "Start reading".
    pub fn advance_to_speech(&self) {
        self.advance.notify_waiters();
    }

    /// Cancel the session. Drops the audio stream; the spawned task exits
    /// after observing the cancel signal.
    pub fn cancel(&self) {
        self.cancel.notify_waiters();
    }
}

impl Drop for CalibrationSession {
    fn drop(&mut self) {
        self.cancel.notify_waiters();
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn run_calibration(
    event_tx: mpsc::Sender<CalibrationFrame>,
    advance: Arc<Notify>,
    cancel: Arc<Notify>,
    stream: Option<AudioStream>,
    timings: CalibrationTimings,
    input_device: Option<String>,
) {
    let audio_config = AudioConfig {
        input_device,
        ..AudioConfig::default()
    };
    let frame_duration_ms = audio_config.chunk_duration_ms;
    let stream = match stream {
        Some(s) => s,
        None => match AudioStream::new(audio_config) {
            Ok(s) => s,
            Err(e) => {
                let _ = event_tx
                    .send(CalibrationFrame::Error(format!("failed to open mic: {e}")))
                    .await;
                return;
            }
        },
    };
    let mut vad = match Vad::new(VadConfig::default(), 16_000) {
        Ok(v) => v,
        Err(e) => {
            let _ = event_tx
                .send(CalibrationFrame::Error(format!("vad init: {e}")))
                .await;
            return;
        }
    };

    // Phase 1 — Noise.
    let _ = event_tx
        .send(CalibrationFrame::PhaseChanged(CalibrationPhase::Noise))
        .await;
    let PhaseOutcome::Done(noise_scores) = collect_phase(
        &stream,
        &mut vad,
        &event_tx,
        &cancel,
        CalibrationPhase::Noise,
        timings.noise,
    )
    .await
    else {
        finish_cancelled(&event_tx).await;
        return;
    };

    // Wait for the user to confirm they're ready to read.
    let _ = event_tx
        .send(CalibrationFrame::PhaseChanged(CalibrationPhase::Waiting))
        .await;
    tokio::select! {
        _ = advance.notified() => {}
        _ = cancel.notified() => {
            finish_cancelled(&event_tx).await;
            return;
        }
    }

    // Phase 2 — Speech.
    let _ = event_tx
        .send(CalibrationFrame::PhaseChanged(CalibrationPhase::Speech))
        .await;
    let PhaseOutcome::Done(speech_scores) = collect_phase(
        &stream,
        &mut vad,
        &event_tx,
        &cancel,
        CalibrationPhase::Speech,
        timings.speech,
    )
    .await
    else {
        finish_cancelled(&event_tx).await;
        return;
    };

    // Compute and emit recommendation.
    let input = CalibrationInput {
        noise_scores,
        speech_scores,
        frame_duration_ms,
    };
    match calibrate::recommend(&input) {
        Ok(result) => {
            let _ = event_tx
                .send(CalibrationFrame::PhaseChanged(CalibrationPhase::Done))
                .await;
            let _ = event_tx.send(CalibrationFrame::Result(result)).await;
        }
        Err(e) => {
            let msg = match e {
                CalibrationError::InsufficientSignal(speech, noise, gap) => format!(
                    "Mic level too low or background too loud (speech {speech:.2} − noise {noise:.2} < {gap:.2}). Move closer or quieter, then retry."
                ),
                other => other.to_string(),
            };
            warn!(error = %msg, "calibration recommendation failed");
            let _ = event_tx
                .send(CalibrationFrame::PhaseChanged(CalibrationPhase::Failed))
                .await;
            let _ = event_tx.send(CalibrationFrame::Error(msg)).await;
        }
    }
}

enum PhaseOutcome {
    Done(Vec<f32>),
    Cancelled,
}

async fn collect_phase(
    stream: &AudioStream,
    vad: &mut Vad,
    event_tx: &mpsc::Sender<CalibrationFrame>,
    cancel: &Notify,
    phase: CalibrationPhase,
    duration: Duration,
) -> PhaseOutcome {
    let deadline = tokio::time::Instant::now() + duration;
    let mut next_emit = tokio::time::Instant::now();
    let mut tick = tokio::time::interval(POLL_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut scores: Vec<f32> = Vec::with_capacity(duration.as_millis() as usize / 16);

    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => {
                debug!(phase = phase.as_str(), frames = scores.len(), "phase complete");
                return PhaseOutcome::Done(scores);
            }
            _ = cancel.notified() => {
                return PhaseOutcome::Cancelled;
            }
            _ = tick.tick() => {
                let mut peak_score = 0.0f32;
                let mut peak_energy = 0.0f32;
                while let Some(chunk) = stream.try_next_chunk() {
                    let r = vad.process_chunk(&chunk.samples);
                    scores.push(r.vad_score);
                    if r.vad_score > peak_score { peak_score = r.vad_score; }
                    if r.energy > peak_energy { peak_energy = r.energy; }
                }
                let now = tokio::time::Instant::now();
                if now >= next_emit && peak_energy > 0.0 {
                    let _ = event_tx.try_send(CalibrationFrame::Frame {
                        phase,
                        vad_score: peak_score,
                        energy: peak_energy,
                    });
                    next_emit = now + FRAME_EMIT_INTERVAL;
                }
            }
        }
    }
}

async fn finish_cancelled(event_tx: &mpsc::Sender<CalibrationFrame>) {
    let _ = event_tx
        .send(CalibrationFrame::PhaseChanged(CalibrationPhase::Cancelled))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_timings() -> CalibrationTimings {
        CalibrationTimings {
            noise: Duration::from_millis(50),
            speech: Duration::from_millis(50),
        }
    }

    fn make_stream(samples: &[f32]) -> AudioStream {
        AudioStream::from_samples(samples, AudioConfig::default()).unwrap()
    }

    /// Drain frames until either the predicate matches or timeout elapses.
    async fn collect_until<F>(
        rx: &mut mpsc::Receiver<CalibrationFrame>,
        mut stop: F,
    ) -> Vec<CalibrationFrame>
    where
        F: FnMut(&CalibrationFrame) -> bool,
    {
        let mut out = Vec::new();
        while let Ok(Some(frame)) = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await {
            let matched = stop(&frame);
            out.push(frame);
            if matched {
                break;
            }
        }
        out
    }

    fn phases(frames: &[CalibrationFrame]) -> Vec<CalibrationPhase> {
        frames
            .iter()
            .filter_map(|f| match f {
                CalibrationFrame::PhaseChanged(p) => Some(*p),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn phase_str_round_trip() {
        for p in [
            CalibrationPhase::Idle,
            CalibrationPhase::Noise,
            CalibrationPhase::Waiting,
            CalibrationPhase::Speech,
            CalibrationPhase::Done,
            CalibrationPhase::Cancelled,
            CalibrationPhase::Failed,
        ] {
            assert!(!p.as_str().is_empty());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn full_run_progresses_noise_waiting_speech_then_emits_result_or_error() {
        // 3s of mixed noise + speech samples at 16 kHz so both phases find data.
        let mut samples = vec![0.001f32; 16_000];
        for i in 0..32_000 {
            samples.push(((i as f32 * 0.05).sin()) * 0.6);
        }
        let stream = make_stream(&samples);
        let (tx, mut rx) = mpsc::channel(128);
        let session = CalibrationSession::start_with(tx, Some(stream), fast_timings(), None);

        // Wait for Waiting, then advance.
        let pre = collect_until(&mut rx, |f| {
            matches!(f, CalibrationFrame::PhaseChanged(CalibrationPhase::Waiting))
        })
        .await;
        assert!(phases(&pre).contains(&CalibrationPhase::Noise));
        assert!(phases(&pre).contains(&CalibrationPhase::Waiting));

        session.advance_to_speech();

        let post = collect_until(&mut rx, |f| {
            matches!(
                f,
                CalibrationFrame::PhaseChanged(CalibrationPhase::Done | CalibrationPhase::Failed)
            )
        })
        .await;
        let post_phases = phases(&post);
        assert!(post_phases.contains(&CalibrationPhase::Speech));
        assert!(
            post_phases
                .iter()
                .any(|p| matches!(p, CalibrationPhase::Done | CalibrationPhase::Failed))
        );
        drop(session);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancel_during_noise_emits_cancelled_phase() {
        let stream = make_stream(&vec![0.001f32; 16_000]);
        let (tx, mut rx) = mpsc::channel(128);
        let session = CalibrationSession::start_with(
            tx,
            Some(stream),
            CalibrationTimings {
                noise: Duration::from_secs(60),
                speech: Duration::from_secs(60),
            },
            None,
        );

        // Wait for the Noise phase to be entered.
        let _ = collect_until(&mut rx, |f| {
            matches!(f, CalibrationFrame::PhaseChanged(CalibrationPhase::Noise))
        })
        .await;

        session.cancel();

        let after = collect_until(&mut rx, |f| {
            matches!(
                f,
                CalibrationFrame::PhaseChanged(CalibrationPhase::Cancelled)
            )
        })
        .await;
        assert!(phases(&after).contains(&CalibrationPhase::Cancelled));
        drop(session);
    }
}
