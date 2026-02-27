use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use lune_asr::OrtForcedAligner;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AudioConfig, AudioStream, Vad, VadConfig};
use lune_commands::VoiceCommandRouter;
use lune_core::Result;
use tokio::sync::RwLock;

use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSessionConfig};
use lune_core::config::{StylePreset, UserSettings, builtin_registry};
use lune_core::error::Error;
use lune_core::llm::TextPostProcessor;
use lune_core::session::{DictationSession, SessionEvent, SessionStatus};
use lune_dict::DictionaryManager;
use lune_output::TextOutputController;
use lune_persist::Database;

use crate::llm_worker::{self, LlmRequest};

/// Timeout for waiting on final LLM response during teardown.
const TEARDOWN_LLM_TIMEOUT: Duration = Duration::from_secs(3);
/// Hard cap for a dictation session.
const MAX_SESSION_LENGTH: Duration = Duration::from_secs(60);
/// Minimum new samples before calling poll_results (~200 ms at 16 kHz).
const MIN_SAMPLES_BEFORE_POLL: usize = 3_200;

/// Silence duration (in samples at 16 kHz) after speech that triggers a flush.
/// 200 ms = 3_200 samples — matches natural conversation turn-taking gap
/// (Stivers et al. 2009: 208 ms mean across 10 languages). Short enough
/// for responsive end-of-utterance detection while clearing inter-word pauses.
pub const SILENCE_FLUSH_SAMPLES: usize = 3_200;
/// Keep forwarding a short silence tail after speech so brief words can still
/// reach backend chunking thresholds and get committed.
const ASR_PUSH_HANGOVER_SAMPLES: usize = 8_000; // 500ms @ 16kHz

/// How far back (in samples) to search for the acoustic onset when VAD fires.
/// 160ms covers TEN-VAD's worst-case confirmation latency (~48ms = 3 frames)
/// plus a comfortable margin for slow fricative onsets.
const ONSET_LOOKBACK_SAMPLES: usize = 2_560; // 160ms @ 16kHz
/// Windowed RMS threshold for the silence/speech crossing in the onset scan.
/// Sits between typical room noise (~0.001) and quiet speech (~0.02).
const ONSET_ENERGY_THRESHOLD: f32 = 0.005; // ≈ -46 dBFS
/// RMS window size for onset scan — matches TEN-VAD's internal hop size.
const ONSET_SCAN_WINDOW: usize = 256; // 16ms @ 16kHz
/// Suppress immediate duplicate flushes from repeated backend emits.
const DUPLICATE_FLUSH_WINDOW: Duration = Duration::from_millis(1200);

/// Number of recent committed utterances to include as LLM context.
const CONTEXT_UTTERANCE_COUNT: usize = 5;
/// Maximum words of context to keep (tail-trimmed). ~100 words ≈ ~130 tokens.
/// Fits within the 2048 n_ctx budget (system ~300 + few-shot ~400 + context ~130 + input ~100 + output ~256 = ~1186).
const CONTEXT_MAX_WORDS: usize = 100;

/// >= 99% confidence: speculative paste — ~75ms (1200 samples).
pub const SILENCE_TIER_SPECULATIVE: usize = 1_200;
/// >= 92% confidence: aggressive — ~80ms (1280 samples).
pub const SILENCE_TIER_HIGH: usize = 1_280;
/// >= 80% confidence: moderate — ~120ms (1920 samples).
pub const SILENCE_TIER_MODERATE: usize = 1_920;
/// < 80% or unknown: conservative — ~320ms (5120 samples).
/// Combined with 30ms VAD hangover, effective wait is ~350ms — provides
/// 100ms total tail protection for uncertain segments where a false flush
/// would cause the LLM correction pass to fail.
pub const SILENCE_TIER_DEFAULT: usize = 5_120;

#[inline]
pub fn dynamic_silence_threshold(confidence: Option<f32>, enabled: bool) -> usize {
    if !enabled {
        return SILENCE_FLUSH_SAMPLES;
    }
    match confidence {
        Some(c) if c >= 0.99 => SILENCE_TIER_SPECULATIVE,
        Some(c) if c >= 0.92 => SILENCE_TIER_HIGH,
        Some(c) if c >= 0.80 => SILENCE_TIER_MODERATE,
        _ => SILENCE_TIER_DEFAULT,
    }
}

/// Configuration for the session pipeline.
#[derive(Clone)]
pub struct SessionManagerConfig {
    pub asr_config: AsrSessionConfig,
    pub asr_backend_kind: AsrBackendKind,
    /// Confidence threshold for LLM bypass. `None` = always send to LLM.
    pub confidence_threshold: Option<f32>,
    /// Min single-token confidence threshold. `None` = no word-level gate.
    pub min_token_threshold: Option<f32>,
    /// Enable confidence-weighted dynamic silence window.
    pub dynamic_silence: Option<bool>,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            asr_config: AsrSessionConfig::default(),
            asr_backend_kind: AsrBackendKind::ParakeetTdt,
            confidence_threshold: Some(0.95),
            min_token_threshold: Some(0.7),
            dynamic_silence: Some(true),
        }
    }
}

struct ActiveHandle {
    stop_tx: oneshot::Sender<()>,
    session_id: Uuid,
}

#[derive(Default)]
struct SessionCounters {
    teardown_forced_flushes: u32,
}

/// Orchestrates the full dictation pipeline.
///
/// Only one session may be active at a time. Call `start()` to begin and
/// `stop()` to finalize gracefully. Events are delivered over the channel
/// passed to `start()`.
pub struct SessionManager<A: AsrBackend> {
    asr: Arc<A>,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    config: SessionManagerConfig,
    settings: Arc<RwLock<UserSettings>>,
    active: Arc<Mutex<Option<ActiveHandle>>>,
    /// Optional post-utterance forced aligner for word-order correction.
    /// When `Some`, spawned as a blocking task after each silence flush.
    aligner: Option<Arc<OrtForcedAligner>>,
}

impl<A: AsrBackend + 'static> SessionManager<A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        asr: Arc<A>,
        router: Arc<VoiceCommandRouter>,
        post_processor: Arc<dyn TextPostProcessor>,
        dict: Arc<Mutex<DictionaryManager>>,
        persist: Arc<Mutex<Database>>,
        config: SessionManagerConfig,
        settings: Arc<RwLock<UserSettings>>,
    ) -> Self {
        Self {
            asr,
            router,
            post_processor,
            dict,
            persist,
            config,
            settings,
            active: Arc::new(Mutex::new(None)),
            aligner: None,
        }
    }

    /// Attach a pre-built forced aligner to this session manager.
    /// When set, the aligner runs after each silence flush.
    pub fn with_aligner(mut self, aligner: OrtForcedAligner) -> Self {
        self.aligner = Some(Arc::new(aligner));
        self
    }

    /// Start a new session. Returns `Err` if one is already active.
    pub async fn start(&self, event_tx: mpsc::Sender<SessionEvent>) -> Result<Uuid> {
        self.start_inner(event_tx, None).await
    }

    /// Start a session with a pre-built audio stream (e.g. from a file).
    pub async fn start_with_audio(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        audio: AudioStream,
    ) -> Result<Uuid> {
        self.start_inner(event_tx, Some(audio)).await
    }

    async fn start_inner(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        audio_override: Option<AudioStream>,
    ) -> Result<Uuid> {
        let mut lock = self.active.lock().await;
        if lock.is_some() {
            return Err(Error::Session("session already active".into()));
        }

        let session_id = Uuid::new_v4();
        let (stop_tx, stop_rx) = oneshot::channel();

        let current_style_id = {
            let settings = self.settings.read().await;
            settings.default_style.clone()
        };
        let style_preset = builtin_registry()
            .get(&current_style_id)
            .unwrap_or_else(|| {
                builtin_registry()
                    .get("general")
                    .expect("builtin 'general' preset must exist")
            });

        {
            let session = DictationSession {
                id: session_id,
                started_at: Utc::now(),
                ended_at: None,
                style: style_preset.id.clone(),
                backend: self.config.asr_backend_kind,
                status: SessionStatus::Active,
                transcript: None,
            };
            if let Err(e) = self.persist.lock().await.insert_session(&session) {
                warn!(error = %e, "failed to persist session start");
            }
        }

        let asr = Arc::clone(&self.asr);
        let router = Arc::clone(&self.router);
        let post_processor = Arc::clone(&self.post_processor);
        let dict = Arc::clone(&self.dict);
        let persist = Arc::clone(&self.persist);
        let config = self.config.clone();
        let active_arc = Arc::clone(&self.active);
        let aligner = self.aligner.as_ref().map(Arc::clone);

        tokio::task::spawn_local(run_session(
            session_id,
            asr,
            router,
            post_processor,
            dict,
            persist,
            config,
            style_preset,
            event_tx,
            stop_rx,
            active_arc,
            audio_override,
            aligner,
        ));

        *lock = Some(ActiveHandle {
            stop_tx,
            session_id,
        });
        info!(%session_id, "session started");
        Ok(session_id)
    }

    pub async fn stop(&self) {
        let mut lock = self.active.lock().await;
        if let Some(handle) = lock.take() {
            debug!(session_id = %handle.session_id, "sending stop signal");
            let _ = handle.stop_tx.send(());
        }
    }

    #[inline]
    pub async fn cancel(&self) {
        self.stop().await;
    }

    #[inline]
    pub async fn active_session_id(&self) -> Option<Uuid> {
        self.active.lock().await.as_ref().map(|h| h.session_id)
    }
}

// ── Pipeline task ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn run_session<A: AsrBackend + 'static>(
    session_id: Uuid,
    asr: Arc<A>,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    config: SessionManagerConfig,
    style_preset: Arc<StylePreset>,
    event_tx: mpsc::Sender<SessionEvent>,
    mut stop_rx: oneshot::Receiver<()>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
    audio_override: Option<AudioStream>,
    aligner: Option<Arc<OrtForcedAligner>>,
) {
    let mut status = SessionStatus::Completed;
    let mut style_preset = style_preset;

    let (llm_tx, mut llm_rx) = llm_worker::spawn_llm_worker(Arc::clone(&post_processor));

    // ── Start audio capture ───────────────────────────────────────────────
    let audio_stream = if let Some(stream) = audio_override {
        stream
    } else {
        match AudioStream::new(AudioConfig::default()) {
            Ok(s) => s,
            Err(e) => {
                let _ = event_tx.send(SessionEvent::Error(e.to_string())).await;
                finalize_db(
                    session_id,
                    &persist,
                    SessionStatus::Failed(e.to_string()),
                    None,
                    None,
                )
                .await;
                active_arc.lock().await.take();
                return;
            }
        }
    };

    // ── Start ASR session ─────────────────────────────────────────────────
    let asr_session = match asr.start_session(config.asr_config).await {
        Ok(s) => s,
        Err(e) => {
            let _ = event_tx.send(SessionEvent::Error(e.to_string())).await;
            finalize_db(
                session_id,
                &persist,
                SessionStatus::Failed(e.to_string()),
                None,
                None,
            )
            .await;
            active_arc.lock().await.take();
            return;
        }
    };

    let mut vad = Vad::new(VadConfig::default(), 16_000);

    let _ = event_tx.send(SessionEvent::Started).await;

    let mut output_ctrl = TextOutputController::new(event_tx.clone());
    let mut session_text = String::with_capacity(512);
    let mut context_utterances: VecDeque<String> =
        VecDeque::with_capacity(CONTEXT_UTTERANCE_COUNT + 1);
    let mut pending_nonfinal = String::with_capacity(128);
    let mut counters = SessionCounters::default();
    let mut sent_to_llm = false;
    let mut last_flushed_text = String::new();
    let mut last_flush_at = tokio::time::Instant::now() - DUPLICATE_FLUSH_WINDOW;
    // Streaming LLM state: first chunk of a new response needs a separator
    // from previous session text; subsequent chunks are typed as-is.
    let mut llm_first_chunk = true;

    // ── Ordered commit buffer ────────────────────────────────────────────
    // When an LLM request is in-flight, bypass-committed segments are held
    // and drained only after the LLM response arrives. This prevents the
    // confidence-gate ordering race (fast bypass overtaking slow LLM).
    let mut llm_pending = false;
    let mut held_commits: Vec<String> = Vec::new();

    // ── DIY timestamps (VAD-level) ───────────────────────────────────────
    // Track absolute audio position for segment-level timing. Not per-word,
    // but sufficient for ordering and re-transcription detection.
    let mut total_samples_pushed: u64 = 0;
    let mut speech_start_sample: Option<u64> = None;
    // Lookback ring buffer for hybrid energy-onset detection.
    // Holds up to ONSET_LOOKBACK_SAMPLES of recent audio so that when TEN-VAD
    // confirms speech, we can scan backward to find the true acoustic onset
    // rather than using the VAD detection frame (which lags by 1–3 frames).
    let mut onset_lookback: VecDeque<f32> = VecDeque::with_capacity(ONSET_LOOKBACK_SAMPLES + 1_280);

    // Reused each audio frame to avoid allocation on the ~50Hz partial path.
    let mut partial_buf = String::with_capacity(512);
    let mut audio_batch: Vec<f32> = Vec::with_capacity(4096);
    let mut audio_poll_tick = tokio::time::interval(Duration::from_millis(10));
    audio_poll_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Shared pipeline gate: accumulate samples + VAD speech flag before calling poll_results.
    let mut samples_since_last_poll: usize = 0;
    let mut had_speech_since_last_poll: bool = false;

    // VAD-driven speech→silence tracking (no wall-clock timers).
    let mut was_speaking = false;
    let mut silence_samples: usize = 0;
    let mut asr_push_hangover_samples: usize = 0;
    // Absolute session cap to avoid indefinitely active toggle sessions.
    let session_max_dur = tokio::time::sleep(MAX_SESSION_LENGTH);
    tokio::pin!(session_max_dur);
    // Resetting Qwen context on every utterance causes repeated streaming
    // cold-start behavior; keep it warm within a session.
    let reset_asr_context_on_flush = !matches!(config.asr_backend_kind, AsrBackendKind::QwenAsr);
    let dynamic_silence_enabled = config.dynamic_silence.unwrap_or(true);
    let mut last_asr_confidence: Option<f32> = None;

    // ── Forced aligner state ──────────────────────────────────────────────
    // `utterance_audio` accumulates raw PCM pushed to ASR for the current
    // speech segment. `utterance_committed` tracks text committed (pasted)
    // during this utterance, in emission order. Both reset on speech start.
    let mut utterance_audio: Vec<f32> = Vec::new();
    let mut utterance_committed: String = String::new();
    let (align_tx, mut align_rx) = mpsc::channel::<String>(4);

    // ── Main loop ─────────────────────────────────────────────────────────
    loop {
        tokio::select! {
            biased;

            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }

            // ── Hard session cap reached → stop session ────────────────
            () = &mut session_max_dur => {
                info!(
                    %session_id,
                    max_session_secs = MAX_SESSION_LENGTH.as_secs(),
                    "max session length reached; stopping session"
                );
                break;
            }

            // ── Forced-aligner correction ─────────────────────────
            Some(corrected) = align_rx.recv() => {
                debug!(%session_id, text = %corrected, "forced aligner correction");
                let _ = event_tx.try_send(SessionEvent::CorrectedText(corrected));
            }

            // ── LLM streaming chunk or completion ────────────────────
            Some(resp) = llm_rx.recv() => {
                if !resp.corrected_text.is_empty() {
                    if llm_first_chunk {
                        // First chunk: insert separator from previous session text.
                        commit_final_text(&mut output_ctrl, &mut session_text, &resp.corrected_text, None, &mut context_utterances).await;
                        llm_first_chunk = false;
                    } else {
                        // Subsequent streaming chunks: already word-spaced.
                        output_ctrl.emit_final(resp.corrected_text.clone()).await;
                        session_text.push_str(&resp.corrected_text);
                        push_context_utterance(&mut context_utterances, resp.corrected_text.trim().to_owned());
                    }
                }
                if resp.is_final {
                    llm_first_chunk = true;
                    llm_pending = false;
                    // Drain held segments that were buffered while LLM was in-flight.
                    for held in held_commits.drain(..) {
                        commit_final_text(&mut output_ctrl, &mut session_text, &held, None, &mut context_utterances).await;
                    }
                }
            }

            // ── Poll captured audio → feed ASR, process speech ───────
            _ = audio_poll_tick.tick() => {
                let Some(first_chunk) = audio_stream.try_next_chunk() else {
                    continue;
                };

                audio_batch.clear();
                audio_batch.extend_from_slice(&first_chunk.samples);
                while let Some(extra) = audio_stream.try_next_chunk() {
                    audio_batch.extend_from_slice(&extra.samples);
                    if audio_batch.len() >= 16_000 * 2 {
                        break;
                    }
                }

                // TEN VAD is the sole speech authority — no energy pre-gate.
                let vad_result = vad.process_chunk(&audio_batch);
                let batch_has_speech = vad_result.is_speech;

                // DIY timestamps: hybrid energy-onset detection.
                // When TEN-VAD first confirms speech, scan backward through
                // the lookback buffer + current batch to find the actual
                // acoustic onset — where energy first crosses the noise floor.
                // This corrects for TEN-VAD's 1–3 frame confirmation latency.
                // Reset per-utterance buffers when speech starts so each flush
                // carries only the audio + text from the current utterance.
                if batch_has_speech && !was_speaking {
                    utterance_audio.clear();
                    utterance_committed.clear();
                }

                if batch_has_speech && speech_start_sample.is_none() {
                    let onset = find_speech_onset(
                        &onset_lookback,
                        &audio_batch,
                        total_samples_pushed,
                        ONSET_ENERGY_THRESHOLD,
                        ONSET_SCAN_WINDOW,
                    );
                    speech_start_sample = Some(onset);
                    debug!(
                        %session_id,
                        onset_ms = onset * 1000 / 16_000,
                        vad_fire_ms = total_samples_pushed * 1000 / 16_000,
                        lookback_ms = (total_samples_pushed.saturating_sub(onset)) * 1000 / 16_000,
                        "speech onset detected (energy lookback)"
                    );
                }
                total_samples_pushed += audio_batch.len() as u64;

                // Update onset lookback — keep the most recent ONSET_LOOKBACK_SAMPLES.
                onset_lookback.extend(audio_batch.iter().copied());
                let trim = onset_lookback.len().saturating_sub(ONSET_LOOKBACK_SAMPLES);
                onset_lookback.drain(..trim);

                if batch_has_speech {
                    asr_push_hangover_samples = ASR_PUSH_HANGOVER_SAMPLES;
                } else {
                    asr_push_hangover_samples =
                        asr_push_hangover_samples.saturating_sub(audio_batch.len());
                }
                let should_push_audio = batch_has_speech || asr_push_hangover_samples > 0;

                if should_push_audio
                    && let Err(e) = asr.push_audio(&asr_session, &audio_batch).await
                {
                    warn!(error = %e, "push_audio error");
                    continue;
                }

                // Accumulate utterance audio for forced aligner.
                if should_push_audio {
                    utterance_audio.extend_from_slice(&audio_batch);
                }

                // Shared pipeline gate: only poll for audio actually sent to ASR.
                if should_push_audio {
                    samples_since_last_poll += audio_batch.len();
                }
                had_speech_since_last_poll |= batch_has_speech;

                let segments = if samples_since_last_poll >= MIN_SAMPLES_BEFORE_POLL {
                    let should_poll = had_speech_since_last_poll || should_push_audio;
                    samples_since_last_poll = 0;
                    had_speech_since_last_poll = false;

                    if should_poll {
                        let poll_start = std::time::Instant::now();
                        match asr.poll_results(&asr_session).await {
                            Ok(s) => {
                                debug!(
                                    elapsed_ms = poll_start.elapsed().as_millis(),
                                    segment_count = s.len(),
                                    "poll_results"
                                );
                                s
                            }
                            Err(e) => { warn!(error = %e, "poll_results error"); continue; }
                        }
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };

                if !segments.is_empty() {
                    for seg in &segments {
                        let route = router.route_full(&seg.text);
                        let asr_text = if route.commands.is_empty() {
                            seg.text.as_str()
                        } else {
                            route.remaining.as_str()
                        };
                        let remaining = asr_text.trim();
                        for cmd in &route.commands {
                            if let lune_core::command::VoiceCommand::SwitchStyle(style_id) = cmd {
                                if let Some(next_preset) = builtin_registry().get(style_id) {
                                    style_preset = next_preset;
                                    info!(
                                        %session_id,
                                        style = %style_preset.id,
                                        "switched style preset via voice command"
                                    );
                                } else {
                                    warn!(
                                        %session_id,
                                        style = %style_id,
                                        "unknown style preset in voice command"
                                    );
                                }
                            }
                            let payload = serde_json::to_string(cmd)
                                .unwrap_or_else(|_| format!("{cmd:?}"));
                            let _ = event_tx
                                .try_send(SessionEvent::CommandDetected(payload));
                        }

                        if let Some(c) = seg.confidence {
                            last_asr_confidence = Some(c);
                        }

                        if seg.is_final {
                            if !remaining.is_empty() {
                                // Two-level confidence gating: sentence mean AND
                                // min-token check (arXiv:2407.21414). High-confidence
                                // segments bypass LLM to avoid overcorrection.
                                let gate_pass = config
                                    .confidence_threshold
                                    .is_some_and(|t| {
                                        let min_t = config.min_token_threshold.unwrap_or(0.0);
                                        seg.is_confident_gated(t, min_t)
                                    });

                                if gate_pass {
                                    let applied = {
                                        let guard = dict.lock().await;
                                        guard.apply(remaining)
                                    };
                                    if is_retranscription(&session_text, &applied) {
                                        debug!(
                                            text = %applied,
                                            "retranscription detected, skipping"
                                        );
                                    } else {
                                        debug!(
                                            confidence = ?seg.confidence,
                                            text = %applied,
                                            "confidence gate: bypassing LLM"
                                        );
                                        eprintln!(
                                            "\x1b[36m[GATE]\x1b[0m conf={:.3} bypass LLM: {applied}",
                                            seg.confidence.unwrap_or(1.0)
                                        );
                                        if llm_pending {
                                            // Hold until pending LLM response arrives
                                            // to preserve audio-timeline ordering.
                                            held_commits.push(applied);
                                        } else {
                                            commit_final_text(
                                                &mut output_ctrl,
                                                &mut session_text,
                                                &applied,
                                                Some(&mut utterance_committed),
                                                &mut context_utterances,
                                            )
                                            .await;
                                        }
                                    }
                                } else {
                                    let ctx = build_llm_context(&context_utterances);
                                    let flushed = flush_to_llm_if_new(
                                        remaining,
                                        &dict,
                                        &style_preset,
                                        &llm_tx,
                                        &mut last_flushed_text,
                                        &mut last_flush_at,
                                        ctx,
                                    )
                                    .await;
                                    sent_to_llm |= flushed;
                                    llm_pending |= flushed;
                                }
                            }
                            pending_nonfinal.clear();
                        } else {
                            partial_buf.clear();
                            partial_buf.push_str(&session_text);
                            if !remaining.is_empty() {
                                if !partial_buf.is_empty() { partial_buf.push(' '); }
                                partial_buf.push_str(remaining);
                                pending_nonfinal.clear();
                                pending_nonfinal.push_str(remaining);
                            } else {
                                pending_nonfinal.clear();
                            }
                            let _ = event_tx.try_send(SessionEvent::PartialText(partial_buf.clone()));
                        }
                    }
                }

                // ── VAD end-of-speech flush ──────────────────────
                // Use VAD (not energy) for speech state tracking so
                // background noise can't keep was_speaking stuck on.
                if batch_has_speech {
                    silence_samples = 0;
                    was_speaking = true;
                } else if was_speaking {
                    silence_samples += audio_batch.len();
                    let flush_threshold = dynamic_silence_threshold(
                        last_asr_confidence,
                        dynamic_silence_enabled,
                    );
                    if silence_samples >= flush_threshold {
                        if !pending_nonfinal.is_empty() {
                            let speculative = dynamic_silence_enabled
                                && last_asr_confidence.is_some_and(|c| c >= 0.99);
                            if speculative {
                                // High confidence: bypass LLM, paste directly after dictionary.
                                let applied = { dict.lock().await.apply(&pending_nonfinal) };
                                if is_retranscription(&session_text, &applied) {
                                    debug!(
                                        text = %applied,
                                        "speculative retranscription detected, skipping"
                                    );
                                } else {
                                    debug!(
                                        %session_id,
                                        confidence = ?last_asr_confidence,
                                        text = %applied,
                                        "dynamic silence flush: speculative paste (>=99%)"
                                    );
                                    if llm_pending {
                                        held_commits.push(applied);
                                    } else {
                                        commit_final_text(
                                            &mut output_ctrl,
                                            &mut session_text,
                                            &applied,
                                            Some(&mut utterance_committed),
                                            &mut context_utterances,
                                        )
                                        .await;
                                    }
                                }
                            } else {
                                debug!(
                                    %session_id,
                                    confidence = ?last_asr_confidence,
                                    flush_ms = flush_threshold * 1000 / 16_000,
                                    "dynamic silence flush"
                                );
                                let ctx = build_llm_context(&context_utterances);
                                let flushed = flush_to_llm_if_new(
                                    &pending_nonfinal,
                                    &dict,
                                    &style_preset,
                                    &llm_tx,
                                    &mut last_flushed_text,
                                    &mut last_flush_at,
                                    ctx,
                                )
                                .await;
                                sent_to_llm |= flushed;
                                llm_pending |= flushed;
                            }
                            pending_nonfinal.clear();
                            if reset_asr_context_on_flush
                                && let Err(e) = asr.reset_context(&asr_session).await
                            {
                                warn!(error = %e, "reset_context error");
                            }
                        }
                        // ── Forced aligner ──────────────────────────────
                        // Spawn after flush so utterance_committed includes
                        // any speculative paste from this silence boundary.
                        if let Some(al) = &aligner {
                            if !utterance_committed.is_empty() {
                                let al = Arc::clone(al);
                                let audio = utterance_audio.clone();
                                let text = utterance_committed.clone();
                                let tx = align_tx.clone();
                                tokio::task::spawn_blocking(move || {
                                    if let Ok(words) = al.align(&audio, &text) {
                                        let reordered = words
                                            .iter()
                                            .map(|w| w.text.as_str())
                                            .collect::<Vec<_>>()
                                            .join(" ");
                                        if reordered != text {
                                            let _ = tx.blocking_send(reordered);
                                        }
                                    }
                                });
                            }
                        }

                        // Always reset — even if nothing to flush — so VAD
                        // inference stops running on every tick after speech.
                        if let Some(start) = speech_start_sample.take() {
                            debug!(
                                %session_id,
                                start_ms = start * 1000 / 16_000,
                                end_ms = total_samples_pushed * 1000 / 16_000,
                                "speech segment ended"
                            );
                        }
                        was_speaking = false;
                        silence_samples = 0;
                        last_asr_confidence = None;
                    }
                }
            }
        }
    }

    // ── Teardown ──────────────────────────────────────────────────────────

    // Drain any remaining captured audio through ASR to capture the last word(s).
    while let Some(chunk) = audio_stream.try_next_chunk() {
        let _ = asr.push_audio(&asr_session, &chunk.samples).await;
    }
    if let Ok(segments) = asr.poll_results(&asr_session).await {
        for seg in &segments {
            let route = router.route_full(&seg.text);
            let asr_text = if route.commands.is_empty() {
                seg.text.as_str()
            } else {
                route.remaining.as_str()
            };
            let text = asr_text.trim();

            for cmd in &route.commands {
                let payload = serde_json::to_string(cmd).unwrap_or_else(|_| format!("{cmd:?}"));
                let _ = event_tx.try_send(SessionEvent::CommandDetected(payload));
            }

            if !text.is_empty() {
                if seg.is_final {
                    let gate_pass = config
                        .confidence_threshold
                        .is_some_and(|t| seg.is_confident(t));

                    if gate_pass {
                        let applied = {
                            let guard = dict.lock().await;
                            guard.apply(text)
                        };
                        if !is_retranscription(&session_text, &applied) {
                            if llm_pending {
                                held_commits.push(applied);
                            } else {
                                commit_final_text(
                                    &mut output_ctrl,
                                    &mut session_text,
                                    &applied,
                                    None,
                                    &mut context_utterances,
                                )
                                .await;
                            }
                        }
                    } else {
                        let ctx = build_llm_context(&context_utterances);
                        let flushed = flush_to_llm_if_new(
                            text,
                            &dict,
                            &style_preset,
                            &llm_tx,
                            &mut last_flushed_text,
                            &mut last_flush_at,
                            ctx,
                        )
                        .await;
                        sent_to_llm |= flushed;
                        llm_pending |= flushed;
                    }
                } else {
                    pending_nonfinal.clear();
                    pending_nonfinal.push_str(text);
                }
            }
        }
    }

    if !pending_nonfinal.is_empty() {
        let ctx = build_llm_context(&context_utterances);
        let flushed = flush_to_llm_if_new(
            &pending_nonfinal,
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
            ctx,
        )
        .await;
        pending_nonfinal.clear();
        if flushed {
            sent_to_llm = true;
            counters.teardown_forced_flushes = counters.teardown_forced_flushes.saturating_add(1);
        }
    }

    // Drop sender so the LLM worker knows no more requests are coming.
    drop(llm_tx);

    if sent_to_llm {
        let _ = tokio::time::timeout(TEARDOWN_LLM_TIMEOUT, async {
            while let Some(resp) = llm_rx.recv().await {
                if !resp.corrected_text.is_empty() {
                    if llm_first_chunk {
                        commit_final_text(
                            &mut output_ctrl,
                            &mut session_text,
                            &resp.corrected_text,
                            None,
                            &mut context_utterances,
                        )
                        .await;
                        llm_first_chunk = false;
                    } else {
                        output_ctrl.emit_final(resp.corrected_text.clone()).await;
                        session_text.push_str(&resp.corrected_text);
                        push_context_utterance(
                            &mut context_utterances,
                            resp.corrected_text.trim().to_owned(),
                        );
                    }
                }
                if resp.is_final {
                    llm_first_chunk = true;
                    // Drain held segments from teardown.
                    for held in held_commits.drain(..) {
                        commit_final_text(
                            &mut output_ctrl,
                            &mut session_text,
                            &held,
                            None,
                            &mut context_utterances,
                        )
                        .await;
                    }
                }
            }
        })
        .await;
    }

    if let Err(e) = asr.end_session(asr_session).await {
        warn!(error = %e, "end_session error");
        status = SessionStatus::Failed(e.to_string());
    }

    let transcript = if session_text.is_empty() {
        None
    } else {
        Some(session_text)
    };
    finalize_db(session_id, &persist, status, Some(Utc::now()), transcript).await;

    info!(
        %session_id,
        teardown_forced_flushes = counters.teardown_forced_flushes,
        "session processing counters"
    );

    let _ = event_tx.send(SessionEvent::Stopped).await;
    active_arc.lock().await.take();
    info!(%session_id, "session stopped");
}

/// Build LLM context from recent committed utterances.
/// Returns `None` if no utterances exist (first utterance).
/// Joins the last `CONTEXT_UTTERANCE_COUNT` utterances, tail-trimmed to
/// `CONTEXT_MAX_WORDS` words so the context fits the 2048 n_ctx budget.
fn build_llm_context(utterances: &VecDeque<String>) -> Option<String> {
    if utterances.is_empty() {
        return None;
    }
    let joined: String = utterances
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if joined.is_empty() {
        return None;
    }
    let words: Vec<&str> = joined.split_whitespace().collect();
    let ctx = if words.len() > CONTEXT_MAX_WORDS {
        words[words.len() - CONTEXT_MAX_WORDS..].join(" ")
    } else {
        joined
    };
    Some(format!("Context: {ctx}\n"))
}

/// Push a committed text segment into the utterance context ring buffer.
#[inline]
fn push_context_utterance(utterances: &mut VecDeque<String>, text: String) {
    if utterances.len() >= CONTEXT_UTTERANCE_COUNT {
        utterances.pop_front();
    }
    utterances.push_back(text);
}

/// Detect re-transcription: returns true when `new_text` (trimmed, ≥10 chars)
/// already appears verbatim in `session_text` (case-insensitive).
/// Guards against the 0.6B streaming decoder re-generating content when the
/// encoder sliding window covers already-transcribed audio.
#[inline]
fn is_retranscription(session_text: &str, new_text: &str) -> bool {
    let new = new_text.trim();
    // Too short to be a meaningful duplicate — single words like "the" would false-positive.
    if new.len() < 10 {
        return false;
    }
    // ASCII-lowercase comparison is sufficient for English ASR output.
    session_text
        .to_ascii_lowercase()
        .contains(&new.to_ascii_lowercase())
}

/// Forward ASR text to the LLM worker without ASR-side post-processing.
async fn flush_to_llm(
    utterance: &str,
    dict: &Mutex<DictionaryManager>,
    style_preset: &Arc<StylePreset>,
    llm_tx: &mpsc::Sender<LlmRequest>,
    context: Option<String>,
) {
    let (text, dictionary) = {
        let guard = dict.lock().await;
        let applied = guard.apply(utterance);
        let relevant = guard
            .top_relevant(&applied, 20)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        (applied, relevant)
    };
    let req = LlmRequest {
        text,
        style_preset: Arc::clone(style_preset),
        dictionary,
        context,
    };
    if let Err(e) = llm_tx.send(req).await {
        warn!(error = %e, "LLM worker unavailable");
    }
}

async fn flush_to_llm_if_new(
    utterance: &str,
    dict: &Mutex<DictionaryManager>,
    style_preset: &Arc<StylePreset>,
    llm_tx: &mpsc::Sender<LlmRequest>,
    last_flushed_text: &mut String,
    last_flush_at: &mut tokio::time::Instant,
    context: Option<String>,
) -> bool {
    let normalized = normalize_whitespace(utterance);
    if normalized.is_empty() {
        return false;
    }
    let now = tokio::time::Instant::now();
    if normalized == *last_flushed_text
        && now.duration_since(*last_flush_at) <= DUPLICATE_FLUSH_WINDOW
    {
        debug!(text = %normalized, "suppressing duplicate ASR flush");
        return false;
    }

    flush_to_llm(&normalized, dict, style_preset, llm_tx, context).await;
    *last_flushed_text = normalized;
    *last_flush_at = now;
    true
}

#[inline]
fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Commit corrected text without ever emitting a standalone leading space token.
///
/// `utterance_committed` is an optional per-utterance accumulator for the
/// forced aligner. Pass `Some(&mut string)` in the main loop, `None` in teardown.
async fn commit_final_text(
    output_ctrl: &mut TextOutputController,
    session_text: &mut String,
    corrected_text: &str,
    utterance_committed: Option<&mut String>,
    context_utterances: &mut VecDeque<String>,
) {
    let trimmed = corrected_text.trim_start();
    if trimmed.is_empty() {
        return;
    }

    // Skip the separator when the new text starts with punctuation that
    // attaches to the preceding word (commas, periods, etc.).
    let starts_with_punct = trimmed
        .as_bytes()
        .first()
        .is_some_and(|&b| matches!(b, b',' | b'.' | b'!' | b'?' | b';' | b':' | b')' | b']'));
    let needs_space = !session_text.is_empty() && !starts_with_punct;
    let mut payload = String::with_capacity(trimmed.len() + usize::from(needs_space));
    if needs_space {
        payload.push(' ');
    }
    payload.push_str(trimmed);

    output_ctrl.emit_final(payload.clone()).await;
    session_text.push_str(&payload);

    // Track for LLM context window.
    push_context_utterance(context_utterances, trimmed.to_owned());

    if let Some(uc) = utterance_committed {
        if !uc.is_empty() {
            uc.push(' ');
        }
        uc.push_str(trimmed);
    }
}

#[cold]
async fn finalize_db(
    session_id: Uuid,
    persist: &Mutex<Database>,
    status: SessionStatus,
    ended_at: Option<chrono::DateTime<Utc>>,
    transcript: Option<String>,
) {
    let mut db = persist.lock().await;
    match db.get_session(session_id) {
        Ok(Some(mut s)) => {
            s.status = status;
            s.ended_at = ended_at;
            s.transcript = transcript;
            if let Err(e) = db.update_session(&s) {
                warn!(error = %e, %session_id, "failed to update session in DB");
            }
        }
        Ok(None) => warn!(%session_id, "session not found in DB during finalize"),
        Err(e) => warn!(error = %e, %session_id, "DB error during finalize"),
    }
}

/// Hybrid speech onset detector.
///
/// TEN-VAD confirms speech, but needs 1–3 × 16 ms frames to do so. By the
/// time `is_speech = true`, the acoustic onset has already passed. This
/// function scans backward through the lookback ring buffer (audio from
/// previous batches) and the current batch to find where energy first
/// emerged from silence, giving sub-frame onset precision.
///
/// Algorithm: walk backward through `lookback ++ batch` in 16ms (256-sample)
/// windows. The first window whose RMS drops below `threshold` marks the
/// end of silence — the onset is the start of the next window.
///
/// # Arguments
/// * `lookback`  — ring buffer of recent audio preceding the current batch
/// * `batch`     — the audio batch on which VAD just fired
/// * `batch_start_sample` — absolute sample index of `batch[0]`
/// * `threshold` — RMS silence/speech boundary (recommend 0.005 ≈ −46 dBFS)
/// * `window`    — RMS window width in samples (recommend 256 = 16ms)
///
/// # Returns
/// Absolute sample index of the detected onset. Clamped to
/// `batch_start_sample - lookback.len()` at minimum (never before our view).
fn find_speech_onset(
    lookback: &VecDeque<f32>,
    batch: &[f32],
    batch_start_sample: u64,
    threshold: f32,
    window: usize,
) -> u64 {
    // Build a contiguous search buffer: [lookback | batch].
    // Allocate only once; the lookback is at most 160ms + one 80ms batch.
    let search_len = lookback.len() + batch.len();
    let mut search: Vec<f32> = Vec::with_capacity(search_len);
    search.extend(lookback.iter().copied());
    search.extend_from_slice(batch);

    // Absolute sample of search[0].
    let search_start_sample = batch_start_sample.saturating_sub(lookback.len() as u64);

    // Walk backward in `window`-sized steps.
    // We're looking for the silence→speech boundary: the first silent window
    // encountered while walking backward *after* already seeing energetic frames.
    // This avoids false triggers when the whole buffer is silent (VAD glitch).
    let mut pos = search_len;
    let mut seen_speech = false;
    while pos >= window {
        let start = pos - window;
        let rms = rms_slice(&search[start..pos]);
        if rms >= threshold {
            seen_speech = true;
        } else if seen_speech {
            // Silent window found after energetic frames — onset begins at `pos`.
            return search_start_sample.saturating_add(pos as u64);
        }
        pos -= window;
    }
    // Remaining partial window at the front.
    if pos > 0 && rms_slice(&search[..pos]) < threshold && seen_speech {
        return search_start_sample + pos as u64;
    }

    // Either the whole buffer was energetic (onset before our view)
    // or it was all silent (VAD glitch — fall back to batch start).
    if seen_speech {
        search_start_sample
    } else {
        batch_start_sample
    }
}

/// Compute RMS of a slice of f32 PCM samples.
#[inline]
fn rms_slice(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::asr::{AsrSegment, AsrSessionConfig};
    use lune_core::llm::{PostProcessInput, PostProcessOutput};
    use tokio::sync::mpsc;

    struct NoopAsr;

    #[async_trait::async_trait]
    impl AsrBackend for NoopAsr {
        type SessionHandle = ();

        async fn start_session(&self, _: AsrSessionConfig) -> Result<()> {
            Ok(())
        }
        async fn push_audio(&self, _: &(), _: &[f32]) -> Result<()> {
            Ok(())
        }
        async fn poll_results(&self, _: &()) -> Result<Vec<AsrSegment>> {
            Ok(vec![])
        }
        async fn end_session(&self, _: ()) -> Result<()> {
            Ok(())
        }
    }

    struct EchoPostProcessor;

    #[async_trait::async_trait]
    impl TextPostProcessor for EchoPostProcessor {
        async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
            Ok(PostProcessOutput {
                cleaned_text: input.raw_text.clone(),
                metadata: None,
            })
        }
    }

    fn make_manager() -> SessionManager<NoopAsr> {
        let db = lune_persist::Database::open_in_memory().unwrap();
        let dict =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        let settings = Arc::new(RwLock::new(lune_core::config::UserSettings::default()));

        SessionManager::new(
            Arc::new(NoopAsr),
            Arc::new(VoiceCommandRouter::new()),
            Arc::new(EchoPostProcessor),
            Arc::new(Mutex::new(dict)),
            Arc::new(Mutex::new(db)),
            SessionManagerConfig::default(),
            settings,
        )
    }

    fn local_set() -> tokio::task::LocalSet {
        tokio::task::LocalSet::new()
    }

    #[tokio::test]
    async fn start_returns_session_id() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx, _rx) = mpsc::channel(16);
                let id = mgr.start(tx).await.unwrap();
                assert!(!id.is_nil());
                mgr.stop().await;
            })
            .await;
    }

    #[tokio::test]
    async fn double_start_errors() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx1, _rx1) = mpsc::channel(16);
                let (tx2, _rx2) = mpsc::channel(16);
                mgr.start(tx1).await.unwrap();
                let err = mgr.start(tx2).await;
                assert!(err.is_err());
                mgr.stop().await;
            })
            .await;
    }

    #[tokio::test]
    async fn stop_clears_active() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx, _rx) = mpsc::channel(16);
                mgr.start(tx).await.unwrap();
                mgr.stop().await;
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                assert!(mgr.active_session_id().await.is_none());
            })
            .await;
    }

    #[tokio::test]
    async fn flush_to_llm_applies_dictionary_and_includes_relevant_entries() {
        let mut dict_mgr =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        dict_mgr.insert("lune voice", "Lune Voice", 10).unwrap();
        let dict = Mutex::new(dict_mgr);

        let (llm_tx, mut llm_rx) = mpsc::channel(1);
        let style_preset = Arc::new(StylePreset::general());

        flush_to_llm(
            "i use lune voice daily",
            &dict,
            &style_preset,
            &llm_tx,
            None,
        )
        .await;

        let req = llm_rx.try_recv().expect("request should be enqueued");
        assert_eq!(req.text, "i use Lune Voice daily");
        assert_eq!(req.dictionary.len(), 1);
        assert_eq!(req.dictionary[0].phrase, "lune voice");
        assert_eq!(req.dictionary[0].replacement, "Lune Voice");
    }

    #[tokio::test]
    async fn commit_final_text_uses_single_separator_between_commits() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut output_ctrl = TextOutputController::new(tx);
        let mut session_text = String::new();
        let mut ctx_utt = VecDeque::new();

        commit_final_text(
            &mut output_ctrl,
            &mut session_text,
            "Hello",
            None,
            &mut ctx_utt,
        )
        .await;
        commit_final_text(
            &mut output_ctrl,
            &mut session_text,
            "world",
            None,
            &mut ctx_utt,
        )
        .await;

        let first = rx.try_recv().expect("first final event");
        let second = rx.try_recv().expect("second final event");
        assert!(matches!(first, SessionEvent::FinalizedText(t) if t == "Hello"));
        assert!(matches!(second, SessionEvent::FinalizedText(t) if t == " world"));
        assert_eq!(session_text, "Hello world");
    }

    #[tokio::test]
    async fn commit_final_text_no_space_before_punctuation() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut output_ctrl = TextOutputController::new(tx);
        let mut session_text = String::new();
        let mut ctx_utt = VecDeque::new();

        commit_final_text(
            &mut output_ctrl,
            &mut session_text,
            "Hello",
            None,
            &mut ctx_utt,
        )
        .await;
        commit_final_text(
            &mut output_ctrl,
            &mut session_text,
            ", world",
            None,
            &mut ctx_utt,
        )
        .await;
        commit_final_text(
            &mut output_ctrl,
            &mut session_text,
            ". Done",
            None,
            &mut ctx_utt,
        )
        .await;

        let first = rx.try_recv().expect("first");
        let second = rx.try_recv().expect("second");
        let third = rx.try_recv().expect("third");
        assert!(matches!(first, SessionEvent::FinalizedText(t) if t == "Hello"));
        assert!(matches!(second, SessionEvent::FinalizedText(t) if t == ", world"));
        assert!(matches!(third, SessionEvent::FinalizedText(t) if t == ". Done"));
        assert_eq!(session_text, "Hello, world. Done");
    }

    #[tokio::test]
    async fn flush_to_llm_if_new_suppresses_duplicate_within_window() {
        let dict =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        let dict = Mutex::new(dict);
        let style_preset = Arc::new(StylePreset::general());
        let (llm_tx, mut llm_rx) = mpsc::channel(4);
        let mut last_flushed_text = String::new();
        let mut last_flush_at = tokio::time::Instant::now() - DUPLICATE_FLUSH_WINDOW;

        let first = flush_to_llm_if_new(
            "hello there",
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
            None,
        )
        .await;
        let second = flush_to_llm_if_new(
            "hello there",
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
            None,
        )
        .await;

        assert!(first);
        assert!(!second);
        let req = llm_rx.try_recv().expect("first request should be present");
        assert_eq!(req.text, "hello there");
        assert!(llm_rx.try_recv().is_err());

        last_flush_at =
            tokio::time::Instant::now() - DUPLICATE_FLUSH_WINDOW - Duration::from_millis(5);
        let third = flush_to_llm_if_new(
            "hello there",
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
            None,
        )
        .await;
        assert!(third);
    }

    #[test]
    fn build_llm_context_empty_returns_none() {
        let empty: VecDeque<String> = VecDeque::new();
        assert!(build_llm_context(&empty).is_none());

        let whitespace: VecDeque<String> = VecDeque::from(["   ".to_owned()]);
        assert!(build_llm_context(&whitespace).is_none());
    }

    #[test]
    fn build_llm_context_short_text_returns_full() {
        let utt = VecDeque::from(["Hello world.".to_owned()]);
        let ctx = build_llm_context(&utt).unwrap();
        assert_eq!(ctx, "Context: Hello world.\n");
    }

    #[test]
    fn build_llm_context_truncates_at_word_limit() {
        // 200 words across 3 utterances — exceeds CONTEXT_MAX_WORDS (100).
        let u1 = (0..80)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let u2 = (80..160)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let u3 = (160..200)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let utt = VecDeque::from([u1, u2, u3]);
        let ctx = build_llm_context(&utt).unwrap();
        assert!(ctx.starts_with("Context: "));
        assert!(ctx.ends_with('\n'));
        let inner = ctx
            .strip_prefix("Context: ")
            .unwrap()
            .strip_suffix('\n')
            .unwrap();
        let word_count = inner.split_whitespace().count();
        assert_eq!(word_count, CONTEXT_MAX_WORDS);
        // Should contain the tail words (w100..w199).
        assert!(inner.contains("w100"));
        assert!(inner.contains("w199"));
        assert!(!inner.contains("w0 "));
    }

    #[test]
    fn push_context_utterance_evicts_oldest() {
        let mut utt: VecDeque<String> = VecDeque::new();
        for i in 0..CONTEXT_UTTERANCE_COUNT + 3 {
            push_context_utterance(&mut utt, format!("u{i}"));
        }
        assert_eq!(utt.len(), CONTEXT_UTTERANCE_COUNT);
        // Oldest entries (u0, u1, u2) should be evicted.
        assert_eq!(utt.front().unwrap(), "u3");
    }

    #[test]
    fn normalize_whitespace_collapses_runs() {
        assert_eq!(
            normalize_whitespace("  hello   world \n test\t"),
            "hello world test"
        );
    }

    #[test]
    fn dynamic_silence_threshold_disabled_returns_static() {
        assert_eq!(
            dynamic_silence_threshold(Some(1.0), false),
            SILENCE_FLUSH_SAMPLES,
        );
        assert_eq!(
            dynamic_silence_threshold(None, false),
            SILENCE_FLUSH_SAMPLES,
        );
    }

    #[test]
    fn dynamic_silence_threshold_tiers() {
        // >= 0.99 → speculative
        assert_eq!(
            dynamic_silence_threshold(Some(0.99), true),
            SILENCE_TIER_SPECULATIVE
        );
        assert_eq!(
            dynamic_silence_threshold(Some(1.0), true),
            SILENCE_TIER_SPECULATIVE
        );
        // >= 0.92 → high
        assert_eq!(
            dynamic_silence_threshold(Some(0.92), true),
            SILENCE_TIER_HIGH
        );
        assert_eq!(
            dynamic_silence_threshold(Some(0.98), true),
            SILENCE_TIER_HIGH
        );
        // >= 0.80 → moderate
        assert_eq!(
            dynamic_silence_threshold(Some(0.80), true),
            SILENCE_TIER_MODERATE
        );
        assert_eq!(
            dynamic_silence_threshold(Some(0.91), true),
            SILENCE_TIER_MODERATE
        );
        // < 0.80 → default
        assert_eq!(
            dynamic_silence_threshold(Some(0.79), true),
            SILENCE_TIER_DEFAULT
        );
        assert_eq!(
            dynamic_silence_threshold(Some(0.0), true),
            SILENCE_TIER_DEFAULT
        );
        // None → default
        assert_eq!(dynamic_silence_threshold(None, true), SILENCE_TIER_DEFAULT);
    }

    #[test]
    fn is_retranscription_detects_duplicate() {
        let session = "Eleven o'clock had struck it was a fine clear night";
        assert!(is_retranscription(session, "eleven o'clock"));
        assert!(is_retranscription(session, "Eleven o'clock"));
        assert!(is_retranscription(session, "a fine clear night"));
    }

    #[test]
    fn is_retranscription_ignores_short_text() {
        let session = "Eleven o'clock had struck";
        // Below 10-char threshold — common short phrases should not be filtered.
        assert!(!is_retranscription(session, "the road"));
        assert!(!is_retranscription(session, "had"));
        assert!(!is_retranscription(session, ""));
    }

    #[test]
    fn is_retranscription_no_false_positive_on_novel_text() {
        let session = "Eleven o'clock had struck";
        assert!(!is_retranscription(
            session,
            "they sauntered leisurely along"
        ));
        assert!(!is_retranscription(
            session,
            "the good hidalgo was far from thinking"
        ));
    }

    #[test]
    fn speculative_silence_tier_is_75ms() {
        // 75ms at 16 kHz = 1200 samples
        assert_eq!(SILENCE_TIER_SPECULATIVE, 1_200);
    }

    // ── find_speech_onset ─────────────────────────────────────────────────

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    fn speech(n: usize) -> Vec<f32> {
        // Sine wave at 0.1 amplitude — well above ONSET_ENERGY_THRESHOLD.
        (0..n).map(|i| 0.1 * (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn onset_speech_starts_at_batch_boundary() {
        // Lookback = silence, batch = all speech.
        // Onset should be at the very start of the batch.
        let lookback: VecDeque<f32> = silence(1280).into();
        let batch = speech(1280);
        let onset = find_speech_onset(&lookback, &batch, 1280, ONSET_ENERGY_THRESHOLD, 256);
        // onset should be <= batch_start_sample (1280) since energy starts there
        assert_eq!(onset, 1280);
    }

    #[test]
    fn onset_found_in_lookback() {
        // Lookback = [silence(512) | speech(768)], batch = speech.
        // Onset should be 512 samples into the search = sample 512 from session start.
        let mut lookback: VecDeque<f32> = VecDeque::new();
        lookback.extend(silence(512));
        lookback.extend(speech(768));
        let batch = speech(1280);
        // batch_start_sample = 1280 (lookback came before it)
        let onset = find_speech_onset(&lookback, &batch, 1280, ONSET_ENERGY_THRESHOLD, 256);
        // search_start = 1280 - 1280 = 0; onset at index 512 → absolute 512
        assert_eq!(onset, 512);
    }

    #[test]
    fn onset_empty_lookback_falls_back_to_batch_start() {
        // No lookback yet (session just started). Whole batch is speech.
        let lookback: VecDeque<f32> = VecDeque::new();
        let batch = speech(1280);
        let onset = find_speech_onset(&lookback, &batch, 0, ONSET_ENERGY_THRESHOLD, 256);
        assert_eq!(onset, 0);
    }

    #[test]
    fn onset_whole_buffer_silent_falls_back_to_batch_start() {
        // Edge case: VAD fired but energy is below threshold everywhere (glitch).
        // Should fall back to batch_start_sample, not panic or return wrong value.
        let lookback: VecDeque<f32> = silence(1280).into();
        let batch = silence(1280);
        let onset = find_speech_onset(&lookback, &batch, 1280, ONSET_ENERGY_THRESHOLD, 256);
        // No speech seen anywhere → fallback to batch_start_sample = 1280.
        assert_eq!(onset, 1280);
    }

    #[test]
    fn onset_mid_batch_detection() {
        // Lookback = silence, batch = [silence(640) | speech(640)].
        // Onset should be at sample 640 within the batch.
        let lookback: VecDeque<f32> = silence(1280).into();
        let mut batch = silence(640);
        batch.extend(speech(640));
        let onset = find_speech_onset(&lookback, &batch, 1280, ONSET_ENERGY_THRESHOLD, 256);
        // search_start = 0; onset should be at 640 (first energetic 256-window after silence)
        // The window aligned at 512..768 would be mixed; 640..896 is mostly speech.
        // We just verify it's earlier than batch_start and ≤ 1280+640.
        assert!(onset <= 1280 + 640);
        assert!(onset >= 512); // silence occupies first 640 samples
    }
}
