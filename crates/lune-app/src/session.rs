use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use lune_asr::{IsotonicCalibrator, OrtForcedAligner};
use tokio::sync::Notify;
use tokio::sync::{Mutex, RwLock, mpsc};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AdaptiveVadConfig, AudioConfig, AudioStream, VadSegmenter, VadSegmenterConfig};
use lune_commands::VoiceCommandRouter;
use lune_core::Result;
use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSegment, AsrSessionConfig};
use lune_core::config::{PipelineConfig, StylePreset, UserSettings, builtin_registry};
use lune_core::error::Error;
use lune_core::llm::TextPostProcessor;
use lune_core::session::{DictationSession, SessionEvent, SessionStatus};
use lune_dict::DictionaryManager;
use lune_persist::Database;

/// Configuration for the session pipeline.
///
/// The `pipeline` field holds fully resolved per-backend settings (VAD
/// timing, confidence thresholds, session limits). Construct it via
/// `AsrRegistry::pipeline_config().with_overrides(&user_settings.pipeline)`.
#[derive(Clone)]
pub struct SessionManagerConfig {
    pub asr_config: AsrSessionConfig,
    pub asr_backend_kind: AsrBackendKind,
    /// Fully resolved pipeline config (backend defaults + user overrides).
    pub pipeline: PipelineConfig,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            asr_config: AsrSessionConfig::default(),
            asr_backend_kind: AsrBackendKind::OrtQwenAsr,
            pipeline: PipelineConfig::default(),
        }
    }
}

struct ActiveHandle {
    control_tx: mpsc::UnboundedSender<SessionControl>,
    session_id: Uuid,
}

enum SessionControl {
    Stop,
    Cancel,
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
    idle_notify: Arc<Notify>,
    /// Optional post-utterance forced aligner for word-order correction.
    /// When `Some`, spawned as a blocking task after each silence flush.
    aligner: Option<Arc<OrtForcedAligner>>,
    /// Persisted adaptive VAD state across sessions.
    adaptive_vad_state: Arc<Mutex<Option<AdaptiveVadConfig>>>,
    /// Path to persist adaptive VAD state across app restarts.
    vad_state_path: Option<PathBuf>,
    /// Isotonic calibrator applied to per-word ASR confidences before the
    /// LLM gate. Defaults to identity so behaviour matches the raw-threshold
    /// path when no calibration data is available.
    calibrator: Arc<IsotonicCalibrator>,
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
            idle_notify: Arc::new(Notify::new()),
            aligner: None,
            adaptive_vad_state: Arc::new(Mutex::new(None)),
            vad_state_path: None,
            calibrator: Arc::new(IsotonicCalibrator::identity()),
        }
    }

    /// Attach an isotonic confidence calibrator. Loaded at session start and
    /// applied to every segment before the LLM gate — downstream code
    /// continues to compare `wc.confidence < threshold`, but `wc.confidence`
    /// is now `P(correct | raw)` instead of raw softmax output.
    pub fn with_calibrator(mut self, calibrator: Arc<IsotonicCalibrator>) -> Self {
        info!(
            breakpoints = calibrator.breakpoint_count(),
            "calibrator attached to session manager"
        );
        self.calibrator = calibrator;
        self
    }

    /// Load a calibrator from disk, or use identity if the file is missing
    /// or corrupt. Mirrors `with_vad_state_path` semantics: non-fatal on any
    /// failure — we never block a dictation session because the calibrator
    /// failed to load.
    pub fn with_calibrator_path(mut self, path: PathBuf) -> Self {
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(json) => match serde_json::from_str::<IsotonicCalibrator>(&json) {
                    Ok(cal) => {
                        info!(
                            path = %path.display(),
                            breakpoints = cal.breakpoint_count(),
                            "loaded isotonic calibrator from disk"
                        );
                        self.calibrator = Arc::new(cal);
                    }
                    Err(e) => {
                        warn!(error = %e, path = %path.display(),
                              "corrupt calibrator file, falling back to identity");
                    }
                },
                Err(e) => {
                    warn!(error = %e, path = %path.display(),
                          "failed to read calibrator file, using identity");
                }
            }
        }
        self
    }

    /// Attach a pre-built forced aligner to this session manager.
    /// When set, the aligner runs after each silence flush.
    pub fn with_aligner(mut self, aligner: Arc<OrtForcedAligner>) -> Self {
        self.aligner = Some(aligner);
        self
    }

    /// Set the file path for persisting adaptive VAD state across restarts.
    /// If the file exists and contains valid JSON, the state is loaded immediately.
    pub fn with_vad_state_path(mut self, path: PathBuf) -> Self {
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(json) => match serde_json::from_str::<AdaptiveVadConfig>(&json) {
                    Ok(state) => {
                        info!(state = %state.debug_state(), path = %path.display(), "loaded adaptive VAD state from disk");
                        self.adaptive_vad_state = Arc::new(Mutex::new(Some(state)));
                    }
                    Err(e) => {
                        warn!(error = %e, path = %path.display(), "corrupt adaptive VAD file, starting fresh");
                    }
                },
                Err(e) => {
                    warn!(error = %e, path = %path.display(), "failed to read adaptive VAD file");
                }
            }
        }
        self.vad_state_path = Some(path);
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
        let (control_tx, control_rx) = mpsc::unbounded_channel();

        let current_style_id = {
            let settings = self.settings.read().await;
            settings.default_style.clone()
        };
        let style_preset = builtin_registry()
            .get(&current_style_id)
            .unwrap_or_else(|| {
                builtin_registry()
                    .get("casual")
                    .expect("builtin 'casual' preset must exist")
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
        let idle_notify = Arc::clone(&self.idle_notify);
        let aligner = self.aligner.as_ref().map(Arc::clone);
        let settings = Arc::clone(&self.settings);
        let adaptive_vad_state = Arc::clone(&self.adaptive_vad_state);
        let vad_state_path = self.vad_state_path.clone();
        let calibrator = Arc::clone(&self.calibrator);

        tokio::task::spawn(run_session(
            session_id,
            asr,
            router,
            post_processor,
            dict,
            persist,
            config,
            style_preset,
            event_tx,
            control_rx,
            active_arc,
            idle_notify,
            audio_override,
            aligner,
            settings,
            adaptive_vad_state,
            vad_state_path,
            calibrator,
        ));

        *lock = Some(ActiveHandle {
            control_tx,
            session_id,
        });
        info!(%session_id, "session started");
        Ok(session_id)
    }

    pub async fn stop(&self) {
        let lock = self.active.lock().await;
        if let Some(handle) = lock.as_ref() {
            debug!(session_id = %handle.session_id, "sending stop signal");
            let _ = handle.control_tx.send(SessionControl::Stop);
        }
    }

    pub async fn stop_and_wait(&self) {
        let notified = {
            let lock = self.active.lock().await;
            match lock.as_ref() {
                Some(handle) => {
                    debug!(
                        session_id = %handle.session_id,
                        "sending stop signal and waiting for session shutdown"
                    );
                    // Register the Notified future BEFORE sending Stop to avoid
                    // a TOCTOU race where notify_waiters fires between lock release
                    // and the first poll of the Notified future.
                    let notified = self.idle_notify.notified();
                    let mut boxed = Box::pin(notified);
                    boxed.as_mut().enable();
                    let _ = handle.control_tx.send(SessionControl::Stop);
                    Some(boxed)
                }
                None => None,
            }
        };

        if let Some(notified) = notified {
            notified.await;
        }
    }

    #[inline]
    pub async fn cancel(&self) {
        let lock = self.active.lock().await;
        if let Some(handle) = lock.as_ref() {
            debug!(session_id = %handle.session_id, "sending cancel signal");
            let _ = handle.control_tx.send(SessionControl::Cancel);
        }
    }

    #[inline]
    pub async fn active_session_id(&self) -> Option<Uuid> {
        self.active.lock().await.as_ref().map(|h| h.session_id)
    }
}

// ── Pipeline task ─────────────────────────────────────────────────────────────

fn append_segments(
    all_segments: &mut Vec<AsrSegment>,
    joined_text: &mut String,
    new_segments: Vec<AsrSegment>,
    calibrator: &IsotonicCalibrator,
) -> bool {
    // Benchmark-only knobs for ablation runs. Not part of user-facing config.
    //
    // - `LUNE_CALIBRATOR=0`  skip calibrator remap (equivalent to identity)
    // - `LUNE_RUNNER_UP=0`   clear `alternate` fields so `mark_uncertain`
    //                        falls back to legacy `[?word?]` markers
    //
    // Read once per call; env lookups are cheap and this path runs at
    // segment boundary (O(segments_per_utterance) calls per session).
    let skip_calibration = std::env::var("LUNE_CALIBRATOR").is_ok_and(|v| v == "0");
    let strip_runner_up = std::env::var("LUNE_RUNNER_UP").is_ok_and(|v| v == "0");

    let mut appended = false;
    for mut seg in new_segments {
        // Remap raw softmax confidences into calibrated P(correct | raw).
        // Downstream gate comparisons (`wc.confidence < threshold`) continue
        // to work unchanged — the scale of `wc.confidence` has moved from
        // raw softmax space to probability-of-correctness space. For the
        // identity calibrator this is a no-op.
        if !skip_calibration {
            for wc in &mut seg.word_confidences {
                wc.confidence = calibrator.calibrate(wc.confidence);
            }
            if let Some(min) = seg.min_token_confidence.as_mut() {
                *min = calibrator.calibrate(*min);
            }
            if let Some(conf) = seg.confidence.as_mut() {
                *conf = calibrator.calibrate(*conf);
            }
        }
        if strip_runner_up {
            for wc in &mut seg.word_confidences {
                wc.alternates.clear();
            }
        }

        // Build the N-best sentence hypotheses on the post-calibration
        // confidence scale so ranking uses P(correct|raw) — the real
        // likelihood order. If calibration is skipped (identity), the
        // ranking still works since relative ordering is preserved.
        //
        // Threshold matches the marking threshold so only genuinely
        // uncertain words contribute alternate hypotheses.
        let mark_threshold = 0.85;
        let k_max = 5;
        seg.build_hypotheses(mark_threshold, k_max);

        let trimmed = seg.text.trim();
        if !trimmed.is_empty() {
            if !joined_text.is_empty() {
                joined_text.push(' ');
            }
            joined_text.push_str(trimmed);
            appended = true;
        }
        all_segments.push(seg);
    }
    appended
}

fn prepare_text_for_llm(
    segments: &[AsrSegment],
    source_text: &str,
    segment_text: &str,
    threshold: f32,
    lang_tag: &str,
    contextual: bool,
) -> (String, bool) {
    if source_text == segment_text {
        return if contextual {
            lune_llm::prepare_input_contextual(segments, threshold, lang_tag)
        } else {
            lune_llm::prepare_input(segments, threshold, lang_tag)
        };
    }

    // Remap word confidences onto rewritten text (after command stripping / FA reorder).
    // Collect all (word_lower, confidence) pairs, match each source word to the first
    // unconsumed entry so markers survive reordering and word removal.
    let mut conf_pool: HashMap<String, VecDeque<f32>> = HashMap::new();
    for wc in segments.iter().flat_map(|seg| seg.word_confidences.iter()) {
        conf_pool
            .entry(wc.word.to_lowercase())
            .or_default()
            .push_back(wc.confidence);
    }

    let marked: String = source_text
        .split_whitespace()
        .map(|word| {
            let lower = word.to_lowercase();
            if let Some(confidences) = conf_pool.get_mut(&lower)
                && let Some(confidence) = confidences.pop_front()
                && confidence < threshold
            {
                return format!("[?{word}?]");
            }
            word.to_owned()
        })
        .collect::<Vec<_>>()
        .join(" ");

    // Only remove fillers that are marked uncertain ([?uh?]) — bare confident
    // fillers are left for the LLM to handle. This prevents stripping real words
    // that the ASR misheard as fillers (e.g. quick "you" → "uh").
    let cleaned = match lune_llm::fillers::remove_fillers_marked_only(&marked, lang_tag) {
        Ok(text) => text,
        Err(_) => marked,
    };
    let has_uncertain = lune_llm::has_uncertainty_markers(&cleaned);
    (cleaned, has_uncertain)
}

/// Fold every segment's pre-built N-best hypothesis list into a single
/// aggregate hypothesis list for the whole utterance.
///
/// Strategy: cartesian product of per-segment hypotheses is prohibitively
/// large (for 3 segments × 5 hypotheses each, 125 combinations). Instead,
/// enumerate "swap one segment at a time" — the aggregate list contains
/// the joined argmax plus one aggregate per segment where that segment
/// contributes an alternate and all others stay at their argmax. This
/// is the same Hamming-≤1 principle the per-segment builder uses, applied
/// at the utterance level.
///
/// Deduplicates case-insensitively and caps at 5 entries total to keep
/// the prompt compact.
fn collect_nbest(segments: &[AsrSegment]) -> Vec<String> {
    // Must have at least one segment with ≥2 hypotheses for n-best to
    // be meaningful. Otherwise return empty and let the caller fall
    // through to the marker path.
    let any_multi = segments.iter().any(|s| s.hypotheses.len() >= 2);
    if !any_multi {
        return Vec::new();
    }

    // Utterance-level argmax = each segment's argmax joined.
    let argmaxes: Vec<&str> = segments
        .iter()
        .map(|s| {
            s.hypotheses
                .first()
                .map(String::as_str)
                .unwrap_or(s.text.as_str())
        })
        .collect();
    let argmax_sentence = argmaxes.join(" ");

    let mut result: Vec<String> = vec![argmax_sentence.clone()];
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(argmax_sentence.to_ascii_lowercase());

    // One aggregate per (segment, alternate) combination, in segment order.
    for (seg_idx, seg) in segments.iter().enumerate() {
        for alt in seg.hypotheses.iter().skip(1) {
            if result.len() >= 5 {
                break;
            }
            let mut pieces: Vec<&str> = argmaxes.clone();
            pieces[seg_idx] = alt.as_str();
            let aggregate = pieces.join(" ");
            let key = aggregate.to_ascii_lowercase();
            if seen.insert(key) {
                result.push(aggregate);
            }
        }
        if result.len() >= 5 {
            break;
        }
    }

    result
}

fn drain_processing_controls(
    control_rx: &mut mpsc::UnboundedReceiver<SessionControl>,
    cancelled: &mut bool,
) {
    while let Ok(control) = control_rx.try_recv() {
        if matches!(control, SessionControl::Stop | SessionControl::Cancel) {
            *cancelled = true;
        }
    }
}

#[inline]
async fn clear_active(active_arc: &Arc<Mutex<Option<ActiveHandle>>>, idle_notify: &Notify) {
    active_arc.lock().await.take();
    idle_notify.notify_waiters();
}

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
    mut control_rx: mpsc::UnboundedReceiver<SessionControl>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: Arc<Notify>,
    audio_override: Option<AudioStream>,
    aligner: Option<Arc<OrtForcedAligner>>,
    settings: Arc<RwLock<UserSettings>>,
    adaptive_vad_state: Arc<Mutex<Option<AdaptiveVadConfig>>>,
    vad_state_path: Option<PathBuf>,
    calibrator: Arc<IsotonicCalibrator>,
) {
    let mut status = SessionStatus::Completed;

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
                clear_active(&active_arc, &idle_notify).await;
                return;
            }
        }
    };

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
            clear_active(&active_arc, &idle_notify).await;
            return;
        }
    };

    let vad_config = VadSegmenterConfig {
        speech_threshold: config.pipeline.speech_threshold,
        min_speech_ms: config.pipeline.min_speech_ms,
        sentence_close_ms: config.pipeline.sentence_close_ms,
        hard_close_ms: config.pipeline.hard_close_ms,
        max_segment_ms: config.pipeline.max_segment_ms,
        adaptive: config.pipeline.adaptive_vad,
    };
    let mut segmenter = if config.pipeline.adaptive_vad {
        // Restore learned state from previous session if available
        let prev_state = adaptive_vad_state.lock().await.take();
        if let Some(state) = prev_state {
            debug!(%session_id, state = %state.debug_state(), "restoring adaptive VAD state");
            VadSegmenter::new_with_adaptive(vad_config, state)
        } else {
            VadSegmenter::new(vad_config)
        }
    } else {
        VadSegmenter::new(vad_config)
    };
    let _ = event_tx.send(SessionEvent::Started).await;

    let mut all_audio = aligner.as_ref().map(|_| Vec::with_capacity(16_000 * 30));
    let mut segments = Vec::with_capacity(asr.segment_capacity_hint());
    let mut raw_text = String::with_capacity(256);
    let mut speech_samples: usize = 0;
    let mut first_asr_error: Option<String> = None;
    let mut saw_asr_success = false;
    let mut cancelled = false;

    let llm_enabled = config.pipeline.word_confidence_threshold.is_some();
    let flush_silence = vec![0.0f32; 16_000];
    let mut progressively_delivered = false;
    let mut audio_poll_tick = tokio::time::interval(Duration::from_millis(10));
    audio_poll_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut next_level_emit = tokio::time::Instant::now();
    let level_emit_interval = Duration::from_millis(67); // ~15 fps
    let mut peak_rms: f32 = 0.0;
    let mut next_adaptive_log = if config.pipeline.adaptive_vad {
        Some(tokio::time::Instant::now() + Duration::from_secs(30))
    } else {
        None
    };

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

            _ = audio_poll_tick.tick() => {
                let mut saw_audio = false;
                while let Some(chunk) = audio_stream.try_next_chunk() {
                    saw_audio = true;
                    let rms = {
                        let n = chunk.samples.len() as f32;
                        if n > 0.0 {
                            let sum_sq: f32 = chunk.samples.iter().map(|&s| s * s).sum();
                            (sum_sq / n).sqrt()
                        } else {
                            0.0
                        }
                    };
                    if rms > peak_rms { peak_rms = rms; }
                    let (boundaries, in_speech) = segmenter.push(&chunk.samples);

                    // Always push audio to ASR (including silence) — the encoder
                    // expects continuous audio with natural pause structure.
                    // VAD is used only for speech-time accounting and session limits.
                    if in_speech {
                        speech_samples += chunk.samples.len();
                    }
                    if let Some(audio) = &mut all_audio {
                        // Cap at max_speech_samples to prevent unbounded growth
                        // during mostly-silent sessions with aligner enabled.
                        if audio.len() < config.pipeline.max_speech_samples {
                            audio.extend_from_slice(&chunk.samples);
                        }
                    }

                    if let Err(e) = asr.push_audio(&asr_session, &chunk.samples).await {
                        let msg = e.to_string();
                        warn!(error = %msg, "push_audio error during streaming ASR");
                        first_asr_error.get_or_insert(msg);
                        continue;
                    }

                    match asr.poll_results(&asr_session).await {
                        Ok(new_segments) => {
                            if append_segments(&mut segments, &mut raw_text, new_segments, &calibrator) {
                                saw_asr_success = true;
                                let _ = event_tx.try_send(SessionEvent::PartialText);
                            }
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            warn!(error = %msg, "poll_results error during streaming ASR");
                            first_asr_error.get_or_insert(msg);
                        }
                    }

                    // Progressive inference: flush ASR at each VAD segment boundary.
                    // When LLM correction is off, emit text immediately per-sentence.
                    // When LLM is on, still flush per-segment (reduces final poll_final
                    // work) but hold text until LLM processes the full transcript.
                    if !boundaries.is_empty() {
                        for boundary in &boundaries {
                            debug!(
                                start_ms = boundary.start_ms,
                                end_ms = boundary.end_ms,
                                avg_prob = format!("{:.2}", boundary.avg_prob),
                                "speech segment boundary → flushing ASR"
                            );
                        }

                        let prev_text_len = raw_text.len();
                        let _ = asr.push_audio(&asr_session, &flush_silence).await;
                        match asr.poll_final(&asr_session).await {
                            Ok(new_segments) => {
                                if append_segments(&mut segments, &mut raw_text, new_segments, &calibrator) {
                                    saw_asr_success = true;
                                    // Deliver only the NEW text from this segment
                                    if !llm_enabled {
                                        let new_text = raw_text[prev_text_len..].trim_start();
                                        if !new_text.is_empty() {
                                            let _ = event_tx.try_send(
                                                SessionEvent::FinalizedText(new_text.to_owned()),
                                            );
                                            progressively_delivered = true;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let msg = e.to_string();
                                warn!(error = %msg, "poll_final error at segment boundary");
                                first_asr_error.get_or_insert(msg);
                            }
                        }
                        if let Err(e) = asr.reset_context(&asr_session).await {
                            warn!(error = %e, "reset_context error at segment boundary");
                        }
                    }

                    if speech_samples >= config.pipeline.max_speech_samples {
                        info!(
                            %session_id,
                            speech_secs = speech_samples / 16_000,
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

                // Periodic adaptive VAD debug log
                if let Some(ref mut deadline) = next_adaptive_log
                    && tokio::time::Instant::now() >= *deadline
                {
                    if let Some(state) = segmenter.debug_adaptive_state() {
                        debug!(%session_id, adaptive_vad = %state, "adaptive VAD state");
                    }
                    *deadline = tokio::time::Instant::now() + Duration::from_secs(30);
                }

                if !saw_audio && audio_stream.is_offline() {
                    debug!(%session_id, "offline audio exhausted");
                    break 'capture;
                }
            }
        }
    }

    // Flush any remaining speech from the segmenter
    if let Some(boundary) = segmenter.finish() {
        debug!(
            start_ms = boundary.start_ms,
            end_ms = boundary.end_ms,
            avg_prob = format!("{:.2}", boundary.avg_prob),
            "final speech segment boundary"
        );
    }

    // Persist adaptive VAD state for the next session (in-memory + disk)
    if let Some(state) = segmenter.take_adaptive() {
        debug!(%session_id, state = %state.debug_state(), "saving adaptive VAD state");
        if let Some(ref path) = vad_state_path {
            match serde_json::to_string(&state) {
                Ok(json) => {
                    if let Err(e) = std::fs::write(path, &json) {
                        warn!(error = %e, path = %path.display(), "failed to persist adaptive VAD to disk");
                    }
                }
                Err(e) => warn!(error = %e, "failed to serialize adaptive VAD state"),
            }
        }
        *adaptive_vad_state.lock().await = Some(state);
    }

    if cancelled {
        if let Err(e) = asr.reset_context(&asr_session).await {
            warn!(error = %e, "reset_context error during cancel");
        }
    } else {
        let _ = event_tx.send(SessionEvent::Processing).await;
        let silence_flush = vec![0.0f32; 16_000];
        if let Err(e) = asr.push_audio(&asr_session, &silence_flush).await {
            let msg = e.to_string();
            warn!(error = %msg, "push_audio (silence flush) error");
            first_asr_error.get_or_insert(msg);
        }
        match asr.poll_final(&asr_session).await {
            Ok(new_segments) => {
                if append_segments(&mut segments, &mut raw_text, new_segments, &calibrator) {
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

    if cancelled {
        finalize_db(
            session_id,
            &persist,
            SessionStatus::Cancelled,
            Some(Utc::now()),
            None,
        )
        .await;
        let _ = event_tx.send(SessionEvent::Stopped).await;
        clear_active(&active_arc, &idle_notify).await;
        return;
    }

    if speech_samples == 0 {
        if let Some(error) = first_asr_error.take().filter(|_| !saw_asr_success) {
            status = SessionStatus::Failed(error.clone());
            let _ = event_tx.send(SessionEvent::Error(error)).await;
        } else {
            info!(%session_id, "no speech detected, ending session");
        }
        finalize_db(session_id, &persist, status, Some(Utc::now()), None).await;
        let _ = event_tx.send(SessionEvent::Stopped).await;
        clear_active(&active_arc, &idle_notify).await;
        return;
    }

    if segments.is_empty() || raw_text.is_empty() {
        if let Some(error) = first_asr_error.take().filter(|_| !saw_asr_success) {
            status = SessionStatus::Failed(error.clone());
            let _ = event_tx.send(SessionEvent::Error(error)).await;
            finalize_db(session_id, &persist, status, Some(Utc::now()), None).await;
        } else {
            // No speech detected — treat as cancelled (don't pollute history)
            info!(%session_id, "no speech detected, discarding empty session");
            finalize_db(
                session_id,
                &persist,
                SessionStatus::Cancelled,
                Some(Utc::now()),
                None,
            )
            .await;
        }
        let _ = event_tx.send(SessionEvent::Stopped).await;
        clear_active(&active_arc, &idle_notify).await;
        return;
    }

    let aligned_text = if let (Some(al), Some(audio)) = (&aligner, all_audio) {
        let al = Arc::clone(al);
        let text = raw_text.clone();
        match tokio::task::spawn_blocking(move || al.align(&audio, &text)).await {
            Ok(Ok(words)) => {
                let reordered = words
                    .iter()
                    .map(|w| w.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                if reordered != raw_text {
                    debug!(%session_id, "forced aligner reordered text");
                    let _ = event_tx.try_send(SessionEvent::CorrectedText(reordered.clone()));
                    reordered
                } else {
                    raw_text.clone()
                }
            }
            Ok(Err(e)) => {
                warn!(error = %e, "forced aligner failed, using raw text");
                raw_text.clone()
            }
            Err(e) => {
                warn!(error = %e, "forced aligner task join error");
                raw_text.clone()
            }
        }
    } else {
        raw_text.clone()
    };

    info!(%session_id, "[ASR] {aligned_text}");

    let route = router.route_full(&aligned_text);
    let mut style_preset = style_preset;
    for cmd in &route.commands {
        if let lune_core::command::VoiceCommand::SwitchStyle(style_id) = cmd
            && let Some(next_preset) = builtin_registry().get(style_id)
        {
            style_preset = next_preset;
            info!(
                %session_id,
                style = %style_preset.id,
                "switched style preset via voice command"
            );
        }
        let payload = serde_json::to_string(cmd).unwrap_or_else(|_| format!("{cmd:?}"));
        let _ = event_tx.try_send(SessionEvent::CommandDetected(payload));
    }
    let text_after_commands = route.remaining.trim().to_owned();
    if text_after_commands.is_empty() {
        info!(%session_id, "all text was voice commands");
        finalize_db(session_id, &persist, status, Some(Utc::now()), None).await;
        let _ = event_tx.send(SessionEvent::Stopped).await;
        clear_active(&active_arc, &idle_notify).await;
        return;
    }

    let dictionary_entries = {
        let guard = dict.lock().await;
        guard
            .top_relevant(&text_after_commands, 20)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()
    };

    let lang = settings.read().await.language.clone();
    let lang_tag = lang.split(['-', '_']).next().unwrap_or("en");

    // Bimodal adaptive LLM gate (arXiv:2509.25048 — confidence-guided
    // correction). The LLM correction pass runs only when the segment
    // lands in the productive middle band: not so clean that correction
    // is wasted work, not so broken that the corrector will overcorrect.
    //
    // - skip-when-clean: every word ≥ llm_skip_threshold → bypass
    // - skip-when-broken: any segment's min_token_confidence below
    //                      llm_skip_noisy_threshold → bypass
    //
    // The gate is "adaptive" in that the decision is driven by live,
    // per-segment metrics (every utterance is classified individually
    // from its own calibrated confidence distribution) rather than a
    // static cutoff applied to the prepared prompt.
    let skip_clean = config.pipeline.llm_skip_threshold.is_some_and(|skip| {
        !segments.is_empty() && segments.iter().all(|s| s.is_clean_for_llm(skip))
    });
    let skip_broken = config
        .pipeline
        .llm_skip_noisy_threshold
        .is_some_and(|broken| {
            !segments.is_empty() && segments.iter().any(|s| s.is_broken_for_llm(broken))
        });

    let (prepared, has_uncertain) = if skip_clean || skip_broken {
        debug!(
            %session_id,
            reason = if skip_clean { "clean" } else { "broken" },
            clean = config.pipeline.llm_skip_threshold.unwrap_or(1.0),
            broken = config.pipeline.llm_skip_noisy_threshold.unwrap_or(0.0),
            "[LLM] bimodal gate: bypassing correction"
        );
        (text_after_commands.clone(), false)
    } else {
        match config.pipeline.word_confidence_threshold {
            Some(threshold) => prepare_text_for_llm(
                &segments,
                &text_after_commands,
                &raw_text,
                threshold,
                lang_tag,
                true, // use contextual confidence
            ),
            None => {
                // No confidence data (e.g. OpenAI backend) — skip LLM, pass through raw text.
                (text_after_commands.clone(), false)
            }
        }
    };

    drain_processing_controls(&mut control_rx, &mut cancelled);
    let post_llm_text = if cancelled {
        info!(%session_id, "session cancelled during processing, skipping LLM");
        String::new()
    } else if !has_uncertain {
        info!(%session_id, "[LLM] no change");
        prepared
    } else {
        // Collect the aggregate N-best hypothesis list across segments.
        // When ≥2 distinct hypotheses exist, the backend routes to the
        // ProGRes-style rescoring path (strictly additive, worst-case
        // returns the argmax). Otherwise we fall through to the existing
        // marker correction path on `raw_text`.
        let nbest_enabled = std::env::var("LUNE_NBEST").is_ok_and(|v| v != "0");
        let hypotheses: Vec<String> = if nbest_enabled {
            collect_nbest(&segments)
        } else {
            Vec::new()
        };
        if !hypotheses.is_empty() {
            info!(
                %session_id,
                count = hypotheses.len(),
                "[LLM] n-best input: {:?}",
                hypotheses
            );
        } else {
            info!(%session_id, "[LLM] marker input: {prepared}");
        }
        let input = lune_core::llm::PostProcessInput {
            raw_text: prepared.clone(),
            style_preset: Arc::clone(&style_preset),
            app_category: None,
            dictionary: dictionary_entries.clone(),
            context: None,
            language: lang_tag.into(),
            hypotheses,
        };

        match post_processor.transform(&input).await {
            Ok(output) => {
                drain_processing_controls(&mut control_rx, &mut cancelled);
                if cancelled {
                    info!(%session_id, "session cancelled during LLM processing");
                    String::new()
                } else {
                    info!(%session_id, "[LLM] {}", output.cleaned_text);
                    output.cleaned_text
                }
            }
            Err(e) => {
                warn!(error = %e, "LLM correction failed, using prepared text");
                prepared
            }
        }
    };

    drain_processing_controls(&mut control_rx, &mut cancelled);
    if cancelled {
        finalize_db(
            session_id,
            &persist,
            SessionStatus::Cancelled,
            Some(Utc::now()),
            None,
        )
        .await;
        let _ = event_tx.send(SessionEvent::Stopped).await;
        clear_active(&active_arc, &idle_notify).await;
        return;
    }

    let final_text = {
        let guard = dict.lock().await;
        guard.apply(&post_llm_text)
    };

    // When we already delivered text progressively per-segment (LLM off),
    // don't re-paste — emit CorrectedText as a UI signal with the
    // dictionary-applied version. Otherwise emit FinalizedText to deliver.
    if progressively_delivered {
        let _ = event_tx
            .send(SessionEvent::CorrectedText(final_text.clone()))
            .await;
    } else {
        let _ = event_tx
            .send(SessionEvent::FinalizedText(final_text.clone()))
            .await;
    }

    let transcript = if final_text.is_empty() {
        None
    } else {
        Some(final_text)
    };
    finalize_db(session_id, &persist, status, Some(Utc::now()), transcript).await;

    let _ = event_tx.send(SessionEvent::Stopped).await;
    clear_active(&active_arc, &idle_notify).await;
    info!(%session_id, "session stopped");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::time::Duration;

    use lune_audio::AudioStream;
    use lune_core::asr::{AsrSegment, AsrSessionConfig, WordConfidence};
    use lune_core::error::Error as CoreError;
    use lune_core::llm::{PostProcessInput, PostProcessOutput};
    use tokio::sync::{Notify, mpsc};

    struct ScriptedAsr {
        responses: Mutex<VecDeque<Result<Vec<AsrSegment>>>>,
        configs: Mutex<Vec<AsrSessionConfig>>,
    }

    #[async_trait::async_trait]
    impl AsrBackend for ScriptedAsr {
        type SessionHandle = ();

        async fn start_session(&self, config: AsrSessionConfig) -> Result<()> {
            self.configs.lock().await.push(config);
            Ok(())
        }
        async fn push_audio(&self, _: &(), _: &[f32]) -> Result<()> {
            Ok(())
        }
        async fn poll_results(&self, _: &()) -> Result<Vec<AsrSegment>> {
            self.responses
                .lock()
                .await
                .pop_front()
                .unwrap_or_else(|| Ok(vec![]))
        }
        async fn poll_final(&self, _: &()) -> Result<Vec<AsrSegment>> {
            self.responses
                .lock()
                .await
                .pop_front()
                .unwrap_or_else(|| Ok(vec![]))
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

    struct BlockingPostProcessor {
        entered: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[async_trait::async_trait]
    impl TextPostProcessor for BlockingPostProcessor {
        async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
            self.entered.notify_waiters();
            self.release.notified().await;
            Ok(PostProcessOutput {
                cleaned_text: input.raw_text.clone(),
                metadata: None,
            })
        }
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn speech_samples() -> Vec<f32> {
        let wav = workspace_root().join("test_fixtures/librispeech_clean.wav");
        let reader = hound::WavReader::open(wav).expect("open speech fixture");
        let spec = reader.spec();
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

    fn speech_stream() -> AudioStream {
        AudioStream::from_samples(&speech_samples(), AudioConfig::default()).unwrap()
    }

    fn silent_stream(duration_secs: usize) -> AudioStream {
        let samples = vec![0.0f32; 16_000 * duration_secs];
        AudioStream::from_samples(&samples, AudioConfig::default()).unwrap()
    }

    fn make_segment(text: &str, uncertain: bool) -> AsrSegment {
        AsrSegment {
            id: 1,
            text: text.into(),
            confidence: Some(if uncertain { 0.5 } else { 0.95 }),
            min_token_confidence: Some(if uncertain { 0.4 } else { 0.9 }),
            word_confidences: if uncertain {
                vec![
                    WordConfidence::new("helo", 0.4),
                    WordConfidence::new("world", 0.95),
                ]
            } else {
                vec![]
            },
            hypotheses: Vec::new(),
            is_final: true,
            start_ms: Some(0),
            end_ms: Some(500),
        }
    }

    fn scripted_asr(responses: Vec<Result<Vec<AsrSegment>>>) -> ScriptedAsr {
        ScriptedAsr {
            responses: Mutex::new(VecDeque::from(responses)),
            configs: Mutex::new(vec![]),
        }
    }

    fn make_manager<A: AsrBackend + 'static>(
        asr: A,
        post_processor: Arc<dyn TextPostProcessor>,
    ) -> (SessionManager<A>, Arc<Mutex<Database>>) {
        let db = lune_persist::Database::open_in_memory().unwrap();
        let dict =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        let settings = Arc::new(RwLock::new(lune_core::config::UserSettings::default()));
        let db = Arc::new(Mutex::new(db));

        let manager = SessionManager::new(
            Arc::new(asr),
            Arc::new(VoiceCommandRouter::new()),
            post_processor,
            Arc::new(Mutex::new(dict)),
            Arc::clone(&db),
            SessionManagerConfig::default(),
            settings,
        );
        (manager, db)
    }

    fn local_set() -> tokio::task::LocalSet {
        tokio::task::LocalSet::new()
    }

    async fn collect_events(mut rx: mpsc::Receiver<SessionEvent>) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        while let Ok(Some(event)) = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            let terminal = matches!(event, SessionEvent::Stopped);
            events.push(event);
            if terminal {
                break;
            }
        }
        events
    }

    #[tokio::test]
    async fn start_returns_session_id() {
        local_set()
            .run_until(async {
                let (mgr, _) = make_manager(scripted_asr(vec![]), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(16);
                let id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();
                assert!(!id.is_nil());
                let _ = collect_events(rx).await;
            })
            .await;
    }

    #[tokio::test]
    async fn double_start_errors() {
        local_set()
            .run_until(async {
                let (mgr, _) = make_manager(scripted_asr(vec![]), Arc::new(EchoPostProcessor));
                let (tx1, _rx1) = mpsc::channel(16);
                let (tx2, _rx2) = mpsc::channel(16);
                mgr.start_with_audio(tx1, speech_stream()).await.unwrap();
                let err = mgr.start_with_audio(tx2, speech_stream()).await;
                assert!(err.is_err());
                mgr.stop().await;
            })
            .await;
    }

    #[tokio::test]
    async fn stop_clears_active() {
        local_set()
            .run_until(async {
                let (mgr, _) = make_manager(scripted_asr(vec![]), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(16);
                mgr.start_with_audio(tx, speech_stream()).await.unwrap();
                mgr.stop().await;
                let _ = collect_events(rx).await;
                assert!(mgr.active_session_id().await.is_none());
            })
            .await;
    }

    #[tokio::test]
    async fn stop_and_wait_finalizes_session() {
        local_set()
            .run_until(async {
                let (mgr, db) = make_manager(scripted_asr(vec![]), Arc::new(EchoPostProcessor));
                let (tx, _rx) = mpsc::channel(16);
                let session_id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                mgr.stop_and_wait().await;

                assert!(mgr.active_session_id().await.is_none());
                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert!(session.ended_at.is_some());
                assert!(matches!(session.status, SessionStatus::Completed));
            })
            .await;
    }

    #[tokio::test]
    async fn rewritten_text_uses_source_text_for_llm() {
        // Segment text has "new line" voice command; source_text has it stripped.
        // Word confidences match the segment text — "helo" is low-conf (ASR error).
        let segments = vec![AsrSegment {
            id: 1,
            text: "helo new line world".into(),
            confidence: Some(0.5),
            min_token_confidence: Some(0.4),
            word_confidences: vec![
                WordConfidence::new("helo", 0.4),
                WordConfidence::new("new", 0.95),
                WordConfidence::new("line", 0.95),
                WordConfidence::new("world", 0.95),
            ],
            hypotheses: Vec::new(),
            is_final: true,
            start_ms: Some(0),
            end_ms: Some(500),
        }];
        let (prepared, has_uncertain) = prepare_text_for_llm(
            &segments,
            "helo world",
            "helo new line world",
            0.65,
            "en",
            true,
        );
        assert_eq!(prepared, "[?helo?] world");
        assert!(has_uncertain);
    }

    #[tokio::test]
    async fn final_text_strips_voice_commands() {
        local_set()
            .run_until(async {
                let responses = vec![Ok(vec![make_segment("hello new line world", false)])];
                let (mgr, db) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(32);
                let session_id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                let events = collect_events(rx).await;
                let finalized: Vec<&str> = events
                    .iter()
                    .filter_map(|event| match event {
                        SessionEvent::FinalizedText(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(finalized.last().copied(), Some("hello world"));

                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert_eq!(session.transcript.as_deref(), Some("hello world"));
            })
            .await;
    }

    #[tokio::test]
    async fn asr_errors_mark_session_failed() {
        local_set()
            .run_until(async {
                let responses = vec![
                    Err(CoreError::Asr("boom".into())),
                    Err(CoreError::Asr("boom".into())),
                ];
                let (mgr, db) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(32);
                let session_id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                let events = collect_events(rx).await;
                assert!(events.iter().any(
                    |event| matches!(event, SessionEvent::Error(msg) if msg.contains("boom"))
                ));
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, SessionEvent::FinalizedText(_)))
                );

                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert!(
                    matches!(session.status, SessionStatus::Failed(ref msg) if msg.contains("boom"))
                );
            })
            .await;
    }

    #[tokio::test]
    async fn asr_errors_without_detected_speech_still_fail_session() {
        local_set()
            .run_until(async {
                let responses = vec![
                    Err(CoreError::Asr("boom".into())),
                    Err(CoreError::Asr("boom".into())),
                ];
                let (mgr, db) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(32);
                let session_id = mgr.start_with_audio(tx, silent_stream(1)).await.unwrap();

                let events = collect_events(rx).await;
                assert!(events.iter().any(
                    |event| matches!(event, SessionEvent::Error(msg) if msg.contains("boom"))
                ));

                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert!(
                    matches!(session.status, SessionStatus::Failed(ref msg) if msg.contains("boom"))
                );
            })
            .await;
    }

    #[tokio::test]
    async fn cancel_during_llm_skips_final_output() {
        local_set()
            .run_until(async {
                let entered = Arc::new(Notify::new());
                let release = Arc::new(Notify::new());
                let post = Arc::new(BlockingPostProcessor {
                    entered: Arc::clone(&entered),
                    release: Arc::clone(&release),
                });
                let responses = vec![Ok(vec![make_segment("helo world", true)])];
                let (mgr, db) = make_manager(scripted_asr(responses), post);
                let (tx, rx) = mpsc::channel(32);
                let session_id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                tokio::time::timeout(Duration::from_secs(5), entered.notified())
                    .await
                    .expect("LLM should start");
                mgr.cancel().await;
                release.notify_waiters();

                let events = collect_events(rx).await;
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, SessionEvent::FinalizedText(_)))
                );

                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert!(matches!(session.status, SessionStatus::Cancelled));
                assert_eq!(session.transcript, None);
            })
            .await;
    }
}
