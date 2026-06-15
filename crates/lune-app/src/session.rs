use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::Notify;
use tokio::sync::{Mutex, RwLock, mpsc};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AudioStream, AudioStreamHandle};
use lune_commands::VoiceCommandRouter;
use lune_core::Result;
use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSegment, AsrSessionConfig};
use lune_core::config::{PipelineConfig, StylePreset, UserSettings, builtin_registry};
use lune_core::error::Error;
use lune_core::llm::TextPostProcessor;
use lune_core::session::{DictationSession, SessionEvent, SessionStatus};
use lune_dict::DictionaryManager;
use lune_dict::corrector::LexicalGate;
use lune_persist::{Database, PersistResult};

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
            asr_backend_kind: AsrBackendKind::Qwen3Asr,
            pipeline: PipelineConfig::default(),
        }
    }
}

struct ActiveHandle {
    control_tx: mpsc::UnboundedSender<SessionControl>,
    session_id: Uuid,
}

pub(crate) enum SessionControl {
    Stop,
    Cancel,
}

/// How the session should obtain audio.
pub(crate) enum SessionAudio {
    /// Open a fresh `AudioStream` inside the session task (default path).
    Open,
    /// Consume from a stream owned by the caller (tests/file replay).
    Owned(AudioStream),
    /// Consume from a long-lived pre-warmed stream's ring buffer.
    Shared(AudioStreamHandle),
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
    heuristic: Arc<LexicalGate>,
    config: SessionManagerConfig,
    settings: Arc<RwLock<UserSettings>>,
    active: Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: Arc<Notify>,
}

impl<A: AsrBackend + 'static> SessionManager<A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        asr: Arc<A>,
        router: Arc<VoiceCommandRouter>,
        post_processor: Arc<dyn TextPostProcessor>,
        dict: Arc<Mutex<DictionaryManager>>,
        persist: Arc<Mutex<Database>>,
        heuristic: Arc<LexicalGate>,
        config: SessionManagerConfig,
        settings: Arc<RwLock<UserSettings>>,
    ) -> Self {
        Self {
            asr,
            router,
            post_processor,
            dict,
            persist,
            heuristic,
            config,
            settings,
            active: Arc::new(Mutex::new(None)),
            idle_notify: Arc::new(Notify::new()),
        }
    }

    /// Start a new session. Returns `Err` if one is already active.
    pub async fn start(&self, event_tx: mpsc::Sender<SessionEvent>) -> Result<Uuid> {
        self.start_inner(event_tx, SessionAudio::Open).await
    }

    /// Start a session with a pre-built audio stream (e.g. from a file).
    pub async fn start_with_audio(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        audio: AudioStream,
    ) -> Result<Uuid> {
        self.start_inner(event_tx, SessionAudio::Owned(audio)).await
    }

    /// Start a session reading from a pre-warmed shared audio stream.
    /// The handle's ring is drained on entry so stale audio captured
    /// during the idle period is discarded.
    pub async fn start_with_handle(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        handle: AudioStreamHandle,
    ) -> Result<Uuid> {
        self.start_inner(event_tx, SessionAudio::Shared(handle))
            .await
    }

    async fn start_inner(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        audio_override: SessionAudio,
    ) -> Result<Uuid> {
        // Resolve style preset BEFORE touching the active lock — a missing
        // style is a config error, not a session-state error, and we don't
        // want to mark the session active just to fail validation.
        let (current_style_id, custom_styles) = {
            let settings = self.settings.read().await;
            (
                settings.default_style.clone(),
                settings.custom_styles.clone(),
            )
        };
        let style_preset = builtin_registry()
            .get(&current_style_id)
            .or_else(|| {
                custom_styles
                    .iter()
                    .find(|s| s.id == current_style_id)
                    .map(|s| Arc::new(s.clone()))
            })
            .or_else(|| builtin_registry().get("minimal"))
            .ok_or_else(|| Error::config("no usable style preset found"))?;

        // Hold the active lock across persist + spawn and only assign
        // `*lock = Some(...)` at the very end. Cancellation safety
        // relies on the `MutexGuard` auto-dropping on early return:
        // if `start_inner` is wrapped in `tokio::time::timeout` and
        // cancelled at any await below, the slot stays `None` and a
        // subsequent start succeeds. Style resolution is intentionally
        // outside the lock so a config error returns without
        // contending with concurrent stop/cancel callers.
        let mut lock = self.active.lock().await;
        if lock.is_some() {
            return Err(Error::session("session already active"));
        }

        let session_id = Uuid::new_v4();
        let (control_tx, control_rx) = mpsc::unbounded_channel();

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
            let insert_result = self.persist.lock().await.insert_session(&session);
            notify_persist_failure("failed to persist session start", insert_result, &event_tx)
                .await;
        }

        let asr = Arc::clone(&self.asr);
        let router = Arc::clone(&self.router);
        let post_processor = Arc::clone(&self.post_processor);
        let dict = Arc::clone(&self.dict);
        let persist = Arc::clone(&self.persist);
        let config = self.config.clone();
        let active_arc = Arc::clone(&self.active);
        let idle_notify = Arc::clone(&self.idle_notify);
        let settings = Arc::clone(&self.settings);
        let heuristic = Arc::clone(&self.heuristic);
        let active_arc_guard = Arc::clone(&active_arc);
        let idle_notify_guard = Arc::clone(&idle_notify);
        let event_tx_guard = event_tx.clone();
        let handle = tokio::task::spawn(run_session(
            session_id,
            asr,
            router,
            post_processor,
            dict,
            persist,
            heuristic,
            config,
            style_preset,
            event_tx,
            control_rx,
            active_arc,
            idle_notify,
            audio_override,
            settings,
        ));
        tokio::task::spawn(async move {
            if handle.await.is_err() {
                tracing::error!(%session_id, "session task panicked — forcing cleanup");
                if let Err(e) = event_tx_guard
                    .send(SessionEvent::Error("internal session panic".into()))
                    .await
                {
                    warn!(error = %e, %session_id, "could not deliver panic error to listener");
                }
                clear_active(&active_arc_guard, &idle_notify_guard).await;
            }
        });

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
            if let Err(e) = handle.control_tx.send(SessionControl::Stop) {
                debug!(error = %e, session_id = %handle.session_id, "stop signal not delivered — session task already exited");
            }
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
                    if let Err(e) = handle.control_tx.send(SessionControl::Stop) {
                        debug!(error = %e, session_id = %handle.session_id, "stop signal not delivered — session task already exited");
                    }
                    Some(boxed)
                }
                None => None,
            }
        };

        if let Some(notified) = notified {
            notified.await;
        }
    }

    pub async fn cancel(&self) {
        let lock = self.active.lock().await;
        if let Some(handle) = lock.as_ref() {
            debug!(session_id = %handle.session_id, "sending cancel signal");
            if let Err(e) = handle.control_tx.send(SessionControl::Cancel) {
                debug!(error = %e, session_id = %handle.session_id, "cancel signal not delivered — session task already exited");
            }
        }
    }

    pub async fn active_session_id(&self) -> Option<Uuid> {
        self.active.lock().await.as_ref().map(|h| h.session_id)
    }
}

// ── Pipeline task ─────────────────────────────────────────────────────────────

// Drain any control signals that arrived during post-VAD processing.
//
// Stop is intentionally ignored here: recording has already ended, so a
// Stop request just means "wrap up" — letting processing finish and
// delivering the transcript is the user-visible win. Only Cancel
// discards the partial result.
fn drain_processing_controls(
    control_rx: &mut mpsc::UnboundedReceiver<SessionControl>,
    cancelled: &mut bool,
) {
    while let Ok(control) = control_rx.try_recv() {
        if matches!(control, SessionControl::Cancel) {
            *cancelled = true;
        }
    }
}

async fn clear_active(active_arc: &Arc<Mutex<Option<ActiveHandle>>>, idle_notify: &Notify) {
    active_arc.lock().await.take();
    idle_notify.notify_waiters();
}

/// Log a persist failure and surface it to the listener as a
/// `SessionEvent::Warning`. Returns the inner value on success, `None` on
/// failure — so the caller can keep going without taking the session down.
///
/// Persist writes during a session (initial insert, finalize update) are
/// not on the critical path: the user's transcript still reaches the
/// active app. Silently swallowing the error, though, hides history loss
/// from anything watching the event stream.
async fn notify_persist_failure<T>(
    label: &str,
    result: PersistResult<T>,
    event_tx: &mpsc::Sender<SessionEvent>,
) -> Option<T> {
    match result {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(error = %e, label, "persist write failed");
            let _ = event_tx
                .send(SessionEvent::Warning(format!("{label}: {e}")))
                .await;
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_session<A: AsrBackend + 'static>(
    session_id: Uuid,
    asr: Arc<A>,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    heuristic: Arc<LexicalGate>,
    config: SessionManagerConfig,
    style_preset: Arc<StylePreset>,
    event_tx: mpsc::Sender<SessionEvent>,
    mut control_rx: mpsc::UnboundedReceiver<SessionControl>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: Arc<Notify>,
    audio_override: SessionAudio,
    _settings: Arc<RwLock<UserSettings>>,
) {
    let vad_pcen_enabled = _settings.read().await.vad_pcen;

    let capture = match crate::capture::run_capture(
        session_id,
        &*asr,
        config.asr_config.clone(),
        audio_override,
        &config.pipeline,
        vad_pcen_enabled,
        asr.segment_capacity_hint(),
        &event_tx,
        &mut control_rx,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            fail_init(
                &e,
                session_id,
                &event_tx,
                &persist,
                &active_arc,
                &idle_notify,
            )
            .await;
            return;
        }
    };
    let crate::capture::CaptureResult {
        segments,
        raw_text,
        speech_samples,
        cancelled,
        mut first_asr_error,
        saw_asr_success,
        mut status,
    } = capture;

    if cancelled {
        stop_session(
            session_id,
            SessionStatus::Cancelled,
            &event_tx,
            &persist,
            &active_arc,
            &idle_notify,
        )
        .await;
        return;
    }

    if speech_samples == 0 || segments.is_empty() || raw_text.is_empty() {
        if let Some(error) = first_asr_error.take().filter(|_| !saw_asr_success) {
            status = SessionStatus::Failed(error.clone());
            let _ = event_tx.send(SessionEvent::Error(error)).await;
        } else if speech_samples > 0 {
            info!(%session_id, "speech detected but ASR produced no output, discarding");
            status = SessionStatus::Cancelled;
        } else {
            info!(%session_id, "no speech detected, ending session");
        }
        stop_session(
            session_id,
            status,
            &event_tx,
            &persist,
            &active_arc,
            &idle_notify,
        )
        .await;
        return;
    }

    process_and_deliver(
        session_id,
        raw_text,
        segments,
        status,
        router,
        post_processor,
        dict,
        persist,
        heuristic,
        config,
        style_preset,
        event_tx,
        control_rx,
        active_arc,
        idle_notify,
        _settings,
        cancelled,
    )
    .await;
}

/// Post-capture pipeline: voice command routing, heuristic correction,
/// optional LLM polishing, and final delivery.
///
/// Called once per session after the `'capture` loop has produced
/// `(raw_text, segments)`. All early-exit paths (empty after voice
/// commands, mid-LLM cancellation) finalize the DB and clear active
/// state before returning, so the caller has nothing to clean up.
///
/// `cancelled` carries the cancel signal observed during capture; this
/// function may flip it to true if a later signal arrives via
/// `control_rx`.
#[allow(clippy::too_many_arguments)]
async fn process_and_deliver(
    session_id: Uuid,
    raw_text: String,
    segments: Vec<AsrSegment>,
    status: SessionStatus,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    heuristic: Arc<LexicalGate>,
    config: SessionManagerConfig,
    mut style_preset: Arc<StylePreset>,
    event_tx: mpsc::Sender<SessionEvent>,
    mut control_rx: mpsc::UnboundedReceiver<SessionControl>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: Arc<Notify>,
    settings: Arc<RwLock<UserSettings>>,
    mut cancelled: bool,
) {
    debug!(%session_id, "[ASR] output ready, len={}", raw_text.len());

    let route = router.route_full(&raw_text);
    for cmd in &route.commands {
        if let lune_core::command::VoiceCommand::SwitchStyle(style_id) = cmd {
            let next_preset =
                builtin_registry()
                    .get(style_id)
                    .or_else(|| match settings.try_read() {
                        Ok(s) => s
                            .custom_styles
                            .iter()
                            .find(|cs| cs.id.as_str() == style_id.as_str())
                            .map(|cs| Arc::new(cs.clone())),
                        Err(_) => {
                            tracing::warn!(
                                %session_id,
                                style = %style_id,
                                "settings locked during style switch, skipping custom lookup"
                            );
                            None
                        }
                    });
            if let Some(next_preset) = next_preset {
                style_preset = next_preset;
                info!(
                    %session_id,
                    style = %style_preset.id,
                    "switched style preset via voice command"
                );
            }
        }
        let payload = serde_json::to_string(cmd).unwrap_or_else(|_| format!("{cmd:?}"));
        let _ = event_tx.try_send(SessionEvent::CommandDetected(payload));
    }
    let text_after_commands = route.remaining.trim().to_owned();
    if text_after_commands.is_empty() {
        info!(%session_id, "all text was voice commands");
        finalize_db(
            session_id,
            Arc::clone(&persist),
            status,
            Some(Utc::now()),
            None,
        )
        .await;
        let _ = event_tx.send(SessionEvent::Stopped).await;
        clear_active(&active_arc, &idle_notify).await;
        return;
    }

    let pipeline_t0 = tokio::time::Instant::now();
    // ── Step 1: Dictionary scan + Heuristic correction ─────────────────
    let (dictionary_entries, dict_replacements) = {
        let guard = dict.lock().await;
        let mut entries = guard
            .top_relevant(&text_after_commands, 20)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        // Soften Title-case-only replacements of ordinary words ("Dev",
        // "Server") before they reach the LLM prompt and input seed — the
        // forced capital is wrong mid-sentence, and the sentence-initial
        // capitalizer restores it where it belongs. The heuristic corrector
        // applies the same guard internally.
        for entry in &mut entries {
            if let Some(lower) = lune_dict::corrector::soften_forced_title_case(&entry.replacement)
            {
                entry.replacement = lower;
            }
        }
        let replacements = guard.scan(&text_after_commands);
        (entries, replacements)
    };
    let word_risk_scores: Vec<(&str, f32)> = segments
        .iter()
        .flat_map(|s| s.word_confidences.iter())
        .map(|wc| (wc.word.as_str(), wc.word_error_risk))
        .collect();

    let detected_lang = segments
        .first()
        .and_then(|s| s.detected_language.as_deref())
        .and_then(lune_core::lang::display_name)
        .unwrap_or("English");
    let heuristic_flag_risk = config
        .pipeline
        .word_confidence_threshold
        .map(|threshold| 1.0 - threshold)
        .unwrap_or(lune_dict::corrector::HWCE_FLAG_THRESHOLD);
    let heuristic_result = heuristic.correct_with_risk_threshold(
        &text_after_commands,
        &word_risk_scores,
        &dict_replacements,
        detected_lang,
        heuristic_flag_risk,
    );
    // Apply deterministic punctuation (leading capital, terminal period) and
    // detect whether residual punctuation is ambiguous. The returned text is
    // both the fast-path output and the LLM input.
    // De-fuse words the ASR ran together inconsistently ("MakeSure" while "make
    // sure" also appears) before punctuation/casing, so a split sentence-start
    // capitalizes correctly. English only — the p_cap/freq guards are English.
    let corrected = if detected_lang == "English" {
        lune_dict::punctuation::defuse_inconsistent(&heuristic_result.corrected_text)
    } else {
        heuristic_result.corrected_text.clone()
    };
    let punctuation = lune_dict::punctuation::normalize(&corrected);
    // Downcase mid-sentence common-word capitals (the "Make Sure" ASR artifact,
    // and any over-capitalization the heuristic left) on the always-on path, so
    // casing is normalized once whether or not the gated LLM pass runs. English
    // only — p_cap is an English table and other languages capitalize nouns.
    let text_after_heuristic = if detected_lang == "English" {
        lune_dict::punctuation::decase_common_words(&punctuation.text)
    } else {
        punctuation.text
    };

    let unresolved_count = heuristic_result.unresolved_words.len();
    let dropped_risks = heuristic_result.dropped_risk_entries;

    // Words the corrector could neither resolve nor confidently preserve —
    // shown in the gate UI and the reason the correction pass runs. They are
    // deliberately NOT fed to the LLM prompt as hints: risk-based hint lists
    // flagged ordinary words ("am", "our", "a") and biased the model into
    // wrong edits and hint-echo punctuation.
    let unresolved_words: Vec<(String, usize)> = heuristic_result
        .unresolved_words
        .iter()
        .map(|ws| (ws.word.clone(), ws.position))
        .collect();

    // Gate on three signals: (1) residual heuristic uncertainty — a word the
    // corrector could neither resolve nor confidently preserve; (2) ambiguous
    // punctuation (run-on, interrogative without an end mark); (3) the
    // confidence band — when the weakest word's confidence falls below
    // `llm_skip_threshold`, the correction pass runs even if the heuristic
    // resolved everything, since trust-resolved words can still be homophone
    // mishears ("no"/"know"). Skip-when-broken vetoes correction when the
    // transcript is shaky across the board — correction degrades quality in
    // that regime (arXiv:2509.25048).
    // Styling is orthogonal to correction: a clean transcript still gets the
    // style pass when the preset asks for it, so a predicted correction no-op
    // only skips the correction pass, never styling.
    let wants_styling = style_preset.needs_styling();
    let crate::gate::GateOutcome {
        needs_correction,
        needs_llm,
        conf_attention,
        too_broken,
        min_conf,
        avg_conf,
    } = crate::gate::decide(crate::gate::GateInputs {
        pipeline: &config.pipeline,
        all_resolved: heuristic_result.all_resolved,
        needs_punct: punctuation.needs_llm,
        wants_styling,
        word_risk_scores: &word_risk_scores,
        word_risk_positions: &heuristic_result.word_risk_positions,
        word_count: text_after_commands.split_whitespace().count(),
    });

    info!(%session_id, "[ASR] raw: {raw_text}");
    info!(
        %session_id,
        lang = detected_lang,
        unresolved = unresolved_count,
        dropped_risks,
        needs_llm,
        needs_correction,
        conf_attention,
        too_broken,
        wants_styling,
        needs_punct = punctuation.needs_llm,
        min_conf,
        avg_conf,
        "bypass gate"
    );

    let _ = event_tx
        .send(SessionEvent::GateDecision {
            used_llm: needs_llm,
            lang: detected_lang.to_owned(),
            unresolved: unresolved_words,
        })
        .await;

    drain_processing_controls(&mut control_rx, &mut cancelled);
    let post_llm_text = if cancelled {
        info!(%session_id, "session cancelled during processing, skipping LLM");
        String::new()
    } else if !needs_llm {
        info!(%session_id, "ASR confidence high — bypassing LLM");
        text_after_heuristic.clone()
    } else {
        info!(%session_id, "[LLM] input: {text_after_heuristic}");
        let llm_t0 = tokio::time::Instant::now();
        let timeout_secs = config.pipeline.llm_timeout_secs;
        let input = lune_core::llm::PostProcessInput {
            raw_text: text_after_heuristic.clone(),
            style_preset: Arc::clone(&style_preset),
            app_category: None,
            dictionary: dictionary_entries.clone(),
            language: detected_lang.into(),
            correction_needed: needs_correction,
        };
        let llm_result = if timeout_secs > 0 {
            tokio::time::timeout(
                Duration::from_secs(timeout_secs as u64),
                post_processor.transform(&input),
            )
            .await
        } else {
            Ok(post_processor.transform(&input).await)
        };
        match llm_result {
            Ok(Ok(output)) => {
                debug!(%session_id, llm_ms = llm_t0.elapsed().as_millis() as u64, "[LLM] output ready, len={}", output.cleaned_text.len());
                // Deterministic backstop: any sentence the model left
                // uncapitalized (or that the merge reverted to input casing)
                // still gets its sentence-initial capital.
                lune_dict::punctuation::capitalize_sentence_starts(&output.cleaned_text)
            }
            Ok(Err(e)) => {
                warn!(error = %e, llm_ms = llm_t0.elapsed().as_millis() as u64, "LLM correction failed, using text as-is");
                text_after_heuristic
            }
            Err(_) => {
                warn!(
                    llm_ms = llm_t0.elapsed().as_millis() as u64,
                    timeout_secs, "LLM timeout ({timeout_secs}s) exceeded, using text as-is"
                );
                text_after_heuristic
            }
        }
    };

    drain_processing_controls(&mut control_rx, &mut cancelled);
    if cancelled {
        stop_session(
            session_id,
            SessionStatus::Cancelled,
            &event_tx,
            &persist,
            &active_arc,
            &idle_notify,
        )
        .await;
        return;
    }

    info!(%session_id, pipeline_ms = pipeline_t0.elapsed().as_millis() as u64, "post-processing complete");

    let _ = event_tx
        .send(SessionEvent::FinalizedText(post_llm_text.clone()))
        .await;

    let transcript = if post_llm_text.is_empty() {
        None
    } else {
        Some(post_llm_text)
    };

    // Record (raw ASR, final delivered) as labelled data for the homophone
    // calibrator. Best-effort — a write failure never affects delivery.
    if let Some(final_text) = transcript.as_deref() {
        let result = persist
            .lock()
            .await
            .insert_correction_pair(session_id, &raw_text, final_text);
        if let Err(e) = result {
            warn!(error = %e, %session_id, "failed to persist correction pair");
        }
    }

    // Await the DB finalize inline so a runtime shutdown between here and
    // the next tick can't drop the transcript. The lock is short-held and
    // sqlite writes are sync, so worst-case added latency is ~10ms.
    let ended_at = Some(Utc::now());
    finalize_db(session_id, persist, status, ended_at, transcript).await;

    let _ = event_tx.send(SessionEvent::Stopped).await;
    clear_active(&active_arc, &idle_notify).await;
    info!(%session_id, "session stopped");
}

#[cold]
async fn finalize_db(
    session_id: Uuid,
    persist: Arc<Mutex<Database>>,
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

#[cold]
async fn fail_init(
    error: impl std::fmt::Display,
    session_id: Uuid,
    event_tx: &mpsc::Sender<SessionEvent>,
    persist: &Arc<Mutex<Database>>,
    active_arc: &Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: &Notify,
) {
    let msg = error.to_string();
    let _ = event_tx.send(SessionEvent::Error(msg.clone())).await;
    finalize_db(
        session_id,
        Arc::clone(persist),
        SessionStatus::Failed(msg),
        None,
        None,
    )
    .await;
    clear_active(active_arc, idle_notify).await;
}

#[cold]
async fn stop_session(
    session_id: Uuid,
    status: SessionStatus,
    event_tx: &mpsc::Sender<SessionEvent>,
    persist: &Arc<Mutex<Database>>,
    active_arc: &Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: &Notify,
) {
    finalize_db(
        session_id,
        Arc::clone(persist),
        status,
        Some(Utc::now()),
        None,
    )
    .await;
    let _ = event_tx.send(SessionEvent::Stopped).await;
    clear_active(active_arc, idle_notify).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::time::Duration;

    use lune_audio::{AudioConfig, AudioStream};
    use lune_core::asr::{AsrSegment, AsrSessionConfig, WordConfidence};
    use lune_core::error::Error as CoreError;
    use lune_core::llm::{PostProcessInput, PostProcessOutput};
    use lune_persist::PersistError;
    use tokio::sync::{Notify, mpsc};

    #[tokio::test]
    async fn notify_persist_failure_emits_warning_with_label_and_error() {
        let (tx, mut rx) = mpsc::channel(2);
        let err: PersistResult<()> = Err(PersistError::NotFound("xyz".into()));
        let out = notify_persist_failure("save snapshot", err, &tx).await;
        assert!(out.is_none(), "failure must collapse to None");
        let event = rx.try_recv().expect("expected a Warning event");
        match event {
            SessionEvent::Warning(msg) => {
                assert!(msg.contains("save snapshot"), "label missing: {msg}");
                assert!(msg.contains("xyz"), "error text missing: {msg}");
            }
            other => panic!("expected Warning, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn notify_persist_failure_passes_value_through_on_ok() {
        let (tx, mut rx) = mpsc::channel(2);
        let ok: PersistResult<u32> = Ok(42);
        let out = notify_persist_failure("save snapshot", ok, &tx).await;
        assert_eq!(out, Some(42));
        assert!(
            rx.try_recv().is_err(),
            "no event should fire on the happy path",
        );
    }

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
        let wav = workspace_root().join("fixtures/audio/librispeech_clean.wav");
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
            confidence: Some(if uncertain { 0.6 } else { 0.95 }),
            min_token_confidence: Some(if uncertain { 0.55 } else { 0.9 }),

            word_confidences: vec![],
            is_final: true,
            start_ms: Some(0),
            end_ms: Some(500),
            detected_language: None,
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
        lune_dict::lexicon::warm();
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
            Arc::new(LexicalGate::new()),
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

    /// Regression: when `Stop` arrives while the `'capture` loop is
    /// actively pumping audio (not at the very start, not after the
    /// stream has exhausted), the loop must break out via the `biased`
    /// select in well under the audio's own duration.
    ///
    /// Uses a 10-second silent stream — without an early break, the
    /// session would run for ~10s. We assert it finalizes in under 2s.
    /// Silent audio means no segments are produced, so the session
    /// takes the "no speech detected" path and terminates as
    /// `Completed` with no transcript.
    #[tokio::test]
    async fn stop_breaks_capture_loop_promptly() {
        local_set()
            .run_until(async {
                let (mgr, db) = make_manager(scripted_asr(vec![]), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(64);
                let session_id = mgr.start_with_audio(tx, silent_stream(10)).await.unwrap();

                // Yield long enough for the capture loop to start
                // pumping ticks, then issue Stop while it's mid-loop.
                tokio::time::sleep(Duration::from_millis(100)).await;

                let stop_t0 = tokio::time::Instant::now();
                mgr.stop_and_wait().await;
                let stop_elapsed = stop_t0.elapsed();

                assert!(
                    stop_elapsed < Duration::from_secs(2),
                    "stop_and_wait took {stop_elapsed:?}; the biased \
                     select arm for control_rx must short-circuit \
                     audio_poll_tick — otherwise we'd wait for the \
                     10s silent stream to exhaust"
                );

                assert!(mgr.active_session_id().await.is_none());

                let events = collect_events(rx).await;
                assert!(
                    events.iter().any(|e| matches!(e, SessionEvent::Started)),
                    "expected Started event before Stop arrived"
                );
                assert!(
                    events.iter().any(|e| matches!(e, SessionEvent::Stopped)),
                    "expected Stopped terminal event"
                );
                assert!(
                    !events
                        .iter()
                        .any(|e| matches!(e, SessionEvent::FinalizedText(_))),
                    "no speech was produced; FinalizedText should not fire"
                );

                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert!(session.ended_at.is_some(), "ended_at must be set");
                assert!(
                    matches!(session.status, SessionStatus::Completed),
                    "graceful Stop must yield Completed (got {:?})",
                    session.status
                );
                assert_eq!(session.transcript, None);
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
    async fn non_exact_command_words_pass_through() {
        local_set()
            .run_until(async {
                // "new line" mid-utterance is dictation, not a command: commands
                // match only when they are the whole utterance.
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
                assert_eq!(finalized.last().copied(), Some("Hello new line world."));

                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert_eq!(session.transcript.as_deref(), Some("Hello new line world."));
            })
            .await;
    }

    #[tokio::test]
    async fn asr_errors_mark_session_failed() {
        local_set()
            .run_until(async {
                let responses = vec![Err(CoreError::asr("boom")), Err(CoreError::asr("boom"))];
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
                let responses = vec![Err(CoreError::asr("boom")), Err(CoreError::asr("boom"))];
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
                let mut segment = make_segment("qxvtm world", true);
                segment.word_confidences = vec![WordConfidence {
                    word: "qxvtm".into(),
                    confidence: 0.2,
                    word_error_risk: 0.9,
                }];
                let responses = vec![Ok(vec![segment])];
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

    /// Regression: a `Stop` arriving during post-VAD processing means
    /// "wrap up and deliver the partial transcript" — it must NOT mark
    /// the session as cancelled. An earlier version matched both
    /// `Stop | Cancel`, dropping transcripts when the user released
    /// the hotkey while the LLM/post-processing was still running.
    #[test]
    fn drain_processing_controls_ignores_stop() {
        let (tx, mut rx) = mpsc::unbounded_channel::<SessionControl>();
        tx.send(SessionControl::Stop).unwrap();
        tx.send(SessionControl::Stop).unwrap();
        let mut cancelled = false;
        drain_processing_controls(&mut rx, &mut cancelled);
        assert!(
            !cancelled,
            "Stop during post-VAD processing must leave cancelled = false"
        );
    }

    /// Regression: `Cancel` during processing must still flip the
    /// cancelled flag even when interleaved with `Stop` signals.
    #[test]
    fn drain_processing_controls_marks_cancel() {
        let (tx, mut rx) = mpsc::unbounded_channel::<SessionControl>();
        tx.send(SessionControl::Stop).unwrap();
        tx.send(SessionControl::Cancel).unwrap();
        tx.send(SessionControl::Stop).unwrap();
        let mut cancelled = false;
        drain_processing_controls(&mut rx, &mut cancelled);
        assert!(
            cancelled,
            "Cancel during processing must flip cancelled to true"
        );
    }

    /// Regression: `start_inner` must remain cancellation-safe. If the
    /// future is dropped while it is parked on the persist write (e.g.
    /// wrapped in `tokio::time::timeout`), the active slot must not
    /// stay populated — otherwise no future session can ever start and
    /// `SessionManager` is permanently wedged.
    #[tokio::test]
    async fn start_is_cancellation_safe_during_persist() {
        local_set()
            .run_until(async {
                let (mgr, db) = make_manager(scripted_asr(vec![]), Arc::new(EchoPostProcessor));

                // Hold the persist lock so the persist write inside start_inner
                // parks. The exact phase doesn't matter — we just need a
                // suspension point that the timeout can land in.
                let persist_guard = db.lock().await;

                let (tx, _rx) = mpsc::channel(16);
                let start_fut = mgr.start_with_audio(tx, speech_stream());
                let timed = tokio::time::timeout(Duration::from_millis(50), start_fut).await;
                assert!(
                    timed.is_err(),
                    "expected start_inner to be cancelled by timeout"
                );

                drop(persist_guard);

                assert!(
                    mgr.active_session_id().await.is_none(),
                    "cancelled start left active = Some; SessionManager is wedged"
                );

                let (tx2, rx2) = mpsc::channel(16);
                let id = mgr
                    .start_with_audio(tx2, speech_stream())
                    .await
                    .expect("subsequent start should succeed after cancellation");
                assert!(!id.is_nil());
                mgr.stop_and_wait().await;
                let _ = collect_events(rx2).await;
            })
            .await;
    }
}
