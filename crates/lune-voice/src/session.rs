use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::Notify;
use tokio::sync::{Mutex, RwLock, mpsc};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AudioStream, AudioStreamHandle};
use lune_core::Result;
use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSegment, AsrSessionConfig};
use lune_core::config::{PipelineConfig, StylePreset, UserSettings, builtin_registry};
use lune_core::error::Error;
use lune_core::llm::TextPostProcessor;
use lune_core::session::{DictationSession, SessionEvent, SessionStatus};
use lune_dict::DictionaryManager;
use lune_dict::commands::VoiceCommandRouter;
use lune_dict::corrector::LexicalGate;
use lune_persist::{Database, PersistResult};

use crate::redact::Transcript;

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
    /// Whether a dictation may leave anything behind on disk.
    ///
    /// `false` is what the "Don't save history" retention setting selects, and
    /// it means exactly that: no `sessions` row, no transcript, no correction
    /// pair. Retention that only prunes *later* would still write every
    /// utterance first, which is not what the setting promises.
    pub persist_history: bool,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            asr_config: AsrSessionConfig::default(),
            asr_backend_kind: AsrBackendKind::WhisperCpp,
            pipeline: PipelineConfig::default(),
            persist_history: true,
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

/// What the session does with the transcript once capture ends.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionKind {
    /// Full dictation: commands, snippets, correction, styling, delivery,
    /// history.
    Dictate,
    /// Transcribe-only: emits the *raw* ASR text (the exact string snippet
    /// matching runs on) as `FinalizedText` and leaves no database trace.
    /// The runtime routes it back to the UI instead of delivering it — used
    /// by the snippet trigger recorder.
    CaptureRaw,
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
        self.start_inner(event_tx, SessionAudio::Open, SessionKind::Dictate)
            .await
    }

    /// Start a session with a pre-built audio stream (e.g. from a file).
    pub async fn start_with_audio(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        audio: AudioStream,
    ) -> Result<Uuid> {
        self.start_inner(event_tx, SessionAudio::Owned(audio), SessionKind::Dictate)
            .await
    }

    /// Start a session reading from a pre-warmed shared audio stream.
    /// The handle's ring is drained on entry so stale audio captured
    /// during the idle period is discarded.
    pub async fn start_with_handle(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        handle: AudioStreamHandle,
    ) -> Result<Uuid> {
        self.start_inner(event_tx, SessionAudio::Shared(handle), SessionKind::Dictate)
            .await
    }

    /// Start a capture-only session (see [`SessionKind::CaptureRaw`]): records
    /// and transcribes like dictation but delivers nothing, persists nothing,
    /// and emits the raw transcript as `FinalizedText`.
    pub async fn start_capture(&self, event_tx: mpsc::Sender<SessionEvent>) -> Result<Uuid> {
        self.start_inner(event_tx, SessionAudio::Open, SessionKind::CaptureRaw)
            .await
    }

    /// Capture-only session on a pre-warmed shared audio stream.
    pub async fn start_capture_with_handle(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        handle: AudioStreamHandle,
    ) -> Result<Uuid> {
        self.start_inner(
            event_tx,
            SessionAudio::Shared(handle),
            SessionKind::CaptureRaw,
        )
        .await
    }

    /// Capture-only session with a pre-built audio stream (tests, benches).
    pub async fn start_capture_with_audio(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        audio: AudioStream,
    ) -> Result<Uuid> {
        self.start_inner(
            event_tx,
            SessionAudio::Owned(audio),
            SessionKind::CaptureRaw,
        )
        .await
    }

    async fn start_inner(
        &self,
        event_tx: mpsc::Sender<SessionEvent>,
        audio_override: SessionAudio,
        kind: SessionKind,
    ) -> Result<Uuid> {
        // Resolve style preset BEFORE touching the active lock — a missing
        // style is a config error, not a session-state error, and we don't
        // want to mark the session active just to fail validation.
        let current_style_id = if kind == SessionKind::CaptureRaw {
            None // capture never styles — skip the lookup entirely
        } else {
            self.settings.read().await.default_style.clone()
        };
        // `None` means no styling — the raw (heuristically-formatted) transcript
        // is delivered as-is. A selected id that no longer resolves (e.g. a
        // deleted custom style) also degrades to no styling rather than erroring.
        // Custom styles live in the database (the source of truth); builtins are
        // the static registry.
        let style_preset: Option<Arc<StylePreset>> = match current_style_id {
            None => None,
            Some(id) => match builtin_registry().get(&id) {
                Some(preset) => Some(preset),
                None => {
                    let loaded = {
                        let mut db = self.persist.lock().await;
                        blocking_db(|| db.get_custom_style(&id))
                    };
                    match loaded {
                        Ok(found) => found.map(|s| Arc::new(s.preset)),
                        Err(e) => {
                            // A genuine DB/deserialize error is not the same as a
                            // missing style — surface it, then degrade to no styling
                            // (optional) rather than failing the whole session.
                            tracing::warn!(
                                style = %id,
                                error = %e,
                                "failed to load custom style from database; delivering unstyled transcript"
                            );
                            None
                        }
                    }
                }
            },
        };

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

        // Capture-only sessions leave no history: no row at start, nothing to
        // finalize at the end. Neither does any session when the user turned
        // history off.
        let persists = kind == SessionKind::Dictate && self.config.persist_history;
        if persists {
            let session = DictationSession {
                id: session_id,
                started_at: Utc::now(),
                ended_at: None,
                style: style_preset.as_ref().map(|p| p.id.clone()),
                backend: self.config.asr_backend_kind,
                status: SessionStatus::Active,
                transcript: None,
            };
            let insert_result = {
                let mut db = self.persist.lock().await;
                blocking_db(|| db.insert_session(&session))
            };
            notify_persist_failure("failed to persist session start", insert_result, &event_tx)
                .await;
        }

        // Only a session that inserted a row above has one the panic
        // supervisor needs to close.
        let panicked_row_to_close = persists.then(|| Arc::clone(&self.persist));
        let asr = Arc::clone(&self.asr);
        let active_arc = Arc::clone(&self.active);
        let idle_notify = Arc::clone(&self.idle_notify);
        let active_arc_guard = Arc::clone(&active_arc);
        let idle_notify_guard = Arc::clone(&idle_notify);
        let event_tx_guard = event_tx.clone();
        let deps = SessionDeps {
            session_id,
            router: Arc::clone(&self.router),
            post_processor: Arc::clone(&self.post_processor),
            dict: Arc::clone(&self.dict),
            persist: Arc::clone(&self.persist),
            heuristic: Arc::clone(&self.heuristic),
            config: self.config.clone(),
            event_tx,
            active_arc,
            idle_notify,
            settings: Arc::clone(&self.settings),
        };
        let handle = tokio::task::spawn(run_session(
            deps,
            asr,
            style_preset,
            control_rx,
            audio_override,
            kind,
        ));
        tokio::task::spawn(async move {
            if handle.await.is_err() {
                tracing::error!(%session_id, "session task panicked — forcing cleanup");
                let error_dropped = !send_bounded(
                    &event_tx_guard,
                    SessionEvent::Error("internal session panic".into()),
                )
                .await;
                let _ =
                    send_bounded(&event_tx_guard, SessionEvent::Stopped { error_dropped }).await;
                clear_active(&active_arc_guard, &idle_notify_guard).await;
                if let Some(persist) = panicked_row_to_close {
                    finalize_panicked_session(session_id, persist).await;
                }
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

/// Upper bound on a session event send. The channel is generously sized
/// (`mpsc::channel(256)`) and a session emits a handful of events, so this
/// only fires if the consumer stopped polling entirely.
const EVENT_SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// Variant name for logging.
///
/// Deliberately not `Debug`: `FinalizedText`, `CorrectedText`, `Warning`,
/// `Error` and `CommandDetected` all carry text that is redacted by default
/// (see `redact.rs`), and a dropped-event log line must not be the one place
/// a transcript leaks.
fn event_name(event: &SessionEvent) -> &'static str {
    match event {
        SessionEvent::Started => "Started",
        SessionEvent::PartialText => "PartialText",
        SessionEvent::FinalizedText { .. } => "FinalizedText",
        SessionEvent::CorrectedText(_) => "CorrectedText",
        SessionEvent::Processing => "Processing",
        SessionEvent::AudioLevel(_) => "AudioLevel",
        SessionEvent::CommandDetected { .. } => "CommandDetected",
        SessionEvent::GateDecision { .. } => "GateDecision",
        SessionEvent::Error(_) => "Error",
        SessionEvent::Warning(_) => "Warning",
        SessionEvent::Stopped { .. } => "Stopped",
        // `SessionEvent` is `#[non_exhaustive]`; a label is not worth a
        // breaking match on every new variant.
        _ => "other",
    }
}

/// Send a session event, bounded in time. Every send from a session task
/// must go through here.
///
/// The event channel is bounded. A stuck consumer that lets the buffer fill
/// blocks the sender indefinitely, so the session task stops polling for
/// cancellation and `clear_active` never runs — permanently wedging the "one
/// session at a time" invariant, with the mic still open. Giving up on an
/// event is strictly better than never clearing the session.
///
/// Returns whether the event was delivered. A dropped `Error` must be
/// reported on the following `Stopped` (`error_dropped: true`), or the
/// listener sees only a clean stop and reports the failed session as success.
#[must_use]
pub(crate) async fn send_bounded(
    event_tx: &mpsc::Sender<SessionEvent>,
    event: SessionEvent,
) -> bool {
    let name = event_name(&event);
    match tokio::time::timeout(EVENT_SEND_TIMEOUT, event_tx.send(event)).await {
        Ok(Ok(())) => true,
        // Listener gone (shutdown) — nothing to report to.
        Ok(Err(_)) => false,
        Err(_) => {
            warn!(
                event = name,
                "session event consumer stalled; event dropped so the session can proceed"
            );
            false
        }
    }
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
            let _ = send_bounded(event_tx, SessionEvent::Warning(format!("{label}: {e}"))).await;
            None
        }
    }
}

/// Per-session shared handles threaded through the session task. Bundling
/// them keeps `run_session`/`process_and_deliver` from each carrying a dozen
/// positional args — and a new dependency becomes one field, not two threaded
/// params.
struct SessionDeps {
    session_id: Uuid,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    heuristic: Arc<LexicalGate>,
    config: SessionManagerConfig,
    event_tx: mpsc::Sender<SessionEvent>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: Arc<Notify>,
    settings: Arc<RwLock<UserSettings>>,
}

async fn run_session<A: AsrBackend + 'static>(
    deps: SessionDeps,
    asr: Arc<A>,
    style_preset: Option<Arc<StylePreset>>,
    mut control_rx: mpsc::UnboundedReceiver<SessionControl>,
    audio_override: SessionAudio,
    kind: SessionKind,
) {
    let session_id = deps.session_id;
    let (input_device, vad_pcen) = {
        let settings = deps.settings.read().await;
        (settings.input_device.clone(), settings.vad_pcen)
    };

    let capture = match crate::capture::run_capture(
        session_id,
        &*asr,
        deps.config.asr_config.clone(),
        audio_override,
        &deps.config.pipeline,
        vad_pcen,
        input_device,
        asr.segment_capacity_hint(),
        &deps.event_tx,
        &mut control_rx,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            if kind == SessionKind::CaptureRaw {
                // No DB row exists for a capture — end without finalize.
                let error_dropped = !send_bounded(&deps.event_tx, SessionEvent::Error(e)).await;
                let _ = send_bounded(&deps.event_tx, SessionEvent::Stopped { error_dropped }).await;
                clear_active(&deps.active_arc, &deps.idle_notify).await;
                return;
            }
            fail_init(
                &e,
                session_id,
                &deps.event_tx,
                &deps.persist,
                deps.config.persist_history,
                &deps.active_arc,
                &deps.idle_notify,
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
        if kind == SessionKind::CaptureRaw {
            let _ = send_bounded(&deps.event_tx, SessionEvent::stopped()).await;
            clear_active(&deps.active_arc, &deps.idle_notify).await;
            return;
        }
        stop_session(
            session_id,
            SessionStatus::Cancelled,
            &deps.event_tx,
            &deps.persist,
            deps.config.persist_history,
            &deps.active_arc,
            &deps.idle_notify,
        )
        .await;
        return;
    }

    // Capture-only sessions end here: the raw transcript — the exact text the
    // snippet matcher will compare against at expansion time — goes back to
    // the caller as `FinalizedText` (routed to the UI, never delivered), with
    // no correction, no commands/snippet expansion, and no history. Empty
    // text (silence) is emitted too so the UI can say "didn't catch that".
    if kind == SessionKind::CaptureRaw {
        let mut error_dropped = false;
        if let Some(error) = first_asr_error.take().filter(|_| !saw_asr_success) {
            error_dropped = !send_bounded(&deps.event_tx, SessionEvent::Error(error)).await;
        } else {
            info!(%session_id, "capture session finished, raw len={}", raw_text.len());
            let _ = send_bounded(
                &deps.event_tx,
                SessionEvent::FinalizedText {
                    session_id,
                    text: raw_text.trim().to_owned(),
                },
            )
            .await;
        }
        let _ = send_bounded(&deps.event_tx, SessionEvent::Stopped { error_dropped }).await;
        clear_active(&deps.active_arc, &deps.idle_notify).await;
        return;
    }

    if segments.is_empty() || raw_text.is_empty() {
        if let Some(error) = first_asr_error.take().filter(|_| !saw_asr_success) {
            status = SessionStatus::Failed(error.clone());
            let _ = send_bounded(&deps.event_tx, SessionEvent::Error(error)).await;
        } else {
            // whisper.cpp's VAD gates speech at decode time, so an empty result
            // means no speech was found (silence/noise) — a graceful no-op, not
            // a failure. (Raw captured-sample count is no longer a speech signal.)
            info!(
                %session_id,
                captured_samples = speech_samples,
                "no speech detected, ending session"
            );
        }
        stop_session(
            session_id,
            status,
            &deps.event_tx,
            &deps.persist,
            deps.config.persist_history,
            &deps.active_arc,
            &deps.idle_notify,
        )
        .await;
        return;
    }

    process_and_deliver(
        deps,
        raw_text,
        segments,
        status,
        style_preset,
        control_rx,
        cancelled,
    )
    .await;
}

/// Output of the deterministic correction + formatting stage.
struct Shaped {
    /// Heuristic-corrected, punctuated, de-cased text — the fast-path output
    /// and the styling-LLM input.
    text_after_heuristic: String,
    detected_lang: &'static str,
    /// Whether the styling LLM pass should run. The LLM does styling only, so
    /// this is simply "a non-default style is selected" — no correction gating.
    needs_llm: bool,
    /// Words the corrector could neither resolve nor confidently preserve.
    unresolved_words: Vec<(String, usize)>,
    /// Risk entries that didn't align to a text token (diagnostic).
    dropped_risks: usize,
}

/// Pure and synchronous: heuristic correction, then English-only de-fusing,
/// punctuation/casing normalization, and the gate that decides whether the LLM
/// pass runs. The dictionary scan (an async lock) stays with the caller, which
/// passes the pre-scanned `dict_replacements` in.
fn correct_and_format(
    text: &str,
    segments: &[AsrSegment],
    dict_replacements: &[lune_dict::DictMatch],
    heuristic: &LexicalGate,
    pipeline: &PipelineConfig,
    fmt: &lune_core::config::FormatToggles,
    wants_styling: bool,
) -> Shaped {
    let word_risk_scores: Vec<(&str, f32, f32)> = segments
        .iter()
        .flat_map(|s| s.word_confidences.iter())
        .map(|wc| (wc.word.as_str(), wc.word_error_risk, wc.mean_confidence))
        .collect();

    let detected_lang = segments
        .first()
        .and_then(|s| s.detected_language.as_deref())
        .and_then(lune_core::lang::display_name)
        .unwrap_or("English");
    let heuristic_flag_risk = pipeline
        .word_confidence_threshold
        .map(|threshold| 1.0 - threshold)
        .unwrap_or(lune_dict::corrector::HWCE_FLAG_THRESHOLD);
    let heuristic_result = heuristic.correct_with_risk_threshold(
        text,
        &word_risk_scores,
        dict_replacements,
        detected_lang,
        heuristic_flag_risk,
        fmt.brand_casing,
    );
    // De-fuse words the ASR ran together inconsistently before punctuation /
    // casing, so a split sentence-start capitalizes correctly. English only —
    // the p_cap/freq guards are English.
    let corrected = if detected_lang == "English" {
        // De-fuse run-together words, then drop hesitation fillers ("um"/"uh")
        // before punctuation/casing — English-only, so the multilingual ASR and
        // languages where "um" is a real word are untouched.
        let defused = lune_dict::formatting::defuse_inconsistent(&heuristic_result.corrected_text);
        lune_dict::formatting::strip_fillers(&defused)
    } else {
        heuristic_result.corrected_text.clone()
    };
    // Whisper sometimes wraps a whole read-aloud utterance in quotes; the user
    // dictated the content, not the quote marks. Strip a single wrapping pair
    // BEFORE punctuation normalization — otherwise normalize appends terminal
    // punctuation OUTSIDE the closing quote (`"…".`), and the strip (which
    // requires the quote to be the final char) can no longer detect the pair.
    let corrected = lune_dict::formatting::strip_wrapping_quotes(&corrected);
    let punctuation = lune_dict::formatting::normalize_with(&corrected, fmt);
    // Downcase mid-sentence common-word capitals on the always-on path so casing
    // is normalized once whether or not the gated LLM pass runs. English only.
    let text_after_heuristic = if detected_lang == "English" {
        lune_dict::formatting::decase_common_words(&punctuation.text)
    } else {
        punctuation.text
    };

    // Words the corrector could neither resolve nor confidently preserve —
    // surfaced in the gate UI and the reason the correction pass runs. They are
    // deliberately NOT fed to the LLM prompt as hints.
    let unresolved_words: Vec<(String, usize)> = heuristic_result
        .unresolved_words
        .iter()
        .map(|ws| (ws.word.clone(), ws.position))
        .collect();

    // The LLM does styling only (no correction), so the pass runs iff the user
    // selected a non-default style — no confidence / punctuation / correction
    // gating. The per-word risk scores still drive the deterministic heuristic
    // above; they no longer route the LLM.
    Shaped {
        text_after_heuristic,
        detected_lang,
        needs_llm: wants_styling,
        unresolved_words,
        dropped_risks: heuristic_result.dropped_risk_entries,
    }
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
async fn process_and_deliver(
    deps: SessionDeps,
    raw_text: String,
    segments: Vec<AsrSegment>,
    status: SessionStatus,
    mut style_preset: Option<Arc<StylePreset>>,
    mut control_rx: mpsc::UnboundedReceiver<SessionControl>,
    mut cancelled: bool,
) {
    let SessionDeps {
        session_id,
        router,
        post_processor,
        dict,
        persist,
        heuristic,
        config,
        event_tx,
        active_arc,
        idle_notify,
        // Custom styles now resolve from the DB (`persist`), not settings.
        settings: _,
    } = deps;
    debug!(%session_id, "[ASR] output ready, len={}", raw_text.len());

    let route = router.route_full(&raw_text);
    // `insert <name>` is handled below by the snippet expander (and falls back
    // to ordinary dictation when nothing matches), so it is not re-emitted as
    // a CommandDetected event for the runtime to log and skip.
    let snippet_name = route.commands.iter().find_map(|cmd| match cmd {
        lune_core::command::VoiceCommand::InsertSnippet(name) => Some(name.clone()),
        _ => None,
    });
    for cmd in &route.commands {
        if matches!(cmd, lune_core::command::VoiceCommand::InsertSnippet(_)) {
            continue;
        }
        if let lune_core::command::VoiceCommand::SwitchStyle(style_id) = cmd {
            let next_preset = match builtin_registry().get(style_id) {
                Some(preset) => Some(preset),
                None => match persist.try_lock() {
                    Ok(mut db) => match db.get_custom_style(style_id) {
                        Ok(found) => found.map(|cs| Arc::new(cs.preset)),
                        Err(e) => {
                            tracing::warn!(
                                %session_id,
                                style = %style_id,
                                error = %e,
                                "failed to load custom style during voice-command style switch"
                            );
                            None
                        }
                    },
                    Err(_) => {
                        tracing::warn!(
                            %session_id,
                            style = %style_id,
                            "database busy during style switch, skipping custom lookup"
                        );
                        None
                    }
                },
            };
            if let Some(next_preset) = next_preset {
                info!(
                    %session_id,
                    style = %next_preset.id,
                    "switched style preset via voice command"
                );
                style_preset = Some(next_preset);
            }
        }
        let payload = serde_json::to_string(cmd).unwrap_or_else(|_| format!("{cmd:?}"));
        let _ = event_tx.try_send(SessionEvent::CommandDetected {
            session_id,
            payload,
        });
    }
    // ── Voice snippets ──────────────────────────────────────────────────
    // Two spoken forms expand a user-defined template: the parametric
    // `insert <name>` command, or the bare trigger phrase as the whole
    // utterance. The rendered body is delivered verbatim — no correction, no
    // styling — because the template already is exactly the text the user
    // wants typed. An `insert <name>` that names no snippet is put back as
    // ordinary dictation ("insert the disk…" must still type) instead of
    // consuming the utterance and typing nothing.
    let mut snippet_fallback_raw = false;
    if snippet_name.is_some() || route.commands.is_empty() {
        let snippets = {
            let mut db = persist.lock().await;
            blocking_db(|| db.list_snippets())
        }
        .unwrap_or_else(|e| {
            warn!(error = %e, %session_id, "failed to load snippets");
            Vec::new()
        });
        let matched = match &snippet_name {
            Some(name) => lune_dict::snippets::resolve_name(&snippets, name),
            None => lune_dict::snippets::match_utterance(&snippets, route.remaining.trim()),
        };
        if let Some(snippet) = matched {
            info!(
                %session_id,
                snippet = %snippet.id,
                trigger = %snippet.trigger_phrase,
                "expanding voice snippet"
            );
            let rendered = render_snippet_body(&snippet.body).await;
            drain_processing_controls(&mut control_rx, &mut cancelled);
            if cancelled {
                stop_session(
                    session_id,
                    SessionStatus::Cancelled,
                    &event_tx,
                    &persist,
                    config.persist_history,
                    &active_arc,
                    &idle_notify,
                )
                .await;
                return;
            }
            deliver_final_text(
                session_id,
                &raw_text,
                rendered,
                status,
                &event_tx,
                persist,
                config.persist_history,
                &active_arc,
                &idle_notify,
            )
            .await;
            return;
        }
        snippet_fallback_raw = snippet_name.is_some();
    }

    let text_after_commands = if snippet_fallback_raw {
        raw_text.trim().to_owned()
    } else {
        route.remaining.trim().to_owned()
    };
    if text_after_commands.is_empty() {
        info!(%session_id, "all text was voice commands");
        finalize_db(
            session_id,
            Arc::clone(&persist),
            config.persist_history,
            status,
            Some(Utc::now()),
            None,
        )
        .await;
        let _ = send_bounded(&event_tx, SessionEvent::stopped()).await;
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
    let fmt = style_preset
        .as_ref()
        .map(|p| p.formatting)
        .unwrap_or_default();
    let wants_styling = style_preset.is_some();
    // Deterministic correction + formatting + bypass-gate, as one span so a
    // single dictation request shows its shaping cost end-to-end.
    let Shaped {
        text_after_heuristic,
        detected_lang,
        needs_llm,
        unresolved_words,
        dropped_risks,
    } = {
        let _span = tracing::info_span!("correct_format", %session_id).entered();
        correct_and_format(
            &text_after_commands,
            &segments,
            &dict_replacements,
            &heuristic,
            &config.pipeline,
            &fmt,
            wants_styling,
        )
    };
    let unresolved_count = unresolved_words.len();

    info!(%session_id, raw = %Transcript::new(&raw_text), "[ASR] raw");
    info!(
        %session_id,
        lang = detected_lang,
        unresolved = unresolved_count,
        dropped_risks,
        needs_llm,
        "style gate"
    );

    let _ = send_bounded(
        &event_tx,
        SessionEvent::GateDecision {
            used_llm: needs_llm,
            lang: detected_lang.to_owned(),
            unresolved: unresolved_words,
        },
    )
    .await;

    drain_processing_controls(&mut control_rx, &mut cancelled);
    let post_llm_text = if cancelled {
        info!(%session_id, "session cancelled during processing, skipping LLM");
        String::new()
    } else if !needs_llm {
        info!(%session_id, "no style selected — skipping styling LLM");
        text_after_heuristic.clone()
    } else {
        info!(%session_id, input = %Transcript::new(&text_after_heuristic), "[LLM] input");
        let llm_t0 = tokio::time::Instant::now();
        let timeout_secs = config.pipeline.llm_timeout_secs;
        let input = lune_core::llm::PostProcessInput {
            raw_text: text_after_heuristic.clone(),
            style_preset: Arc::clone(
                style_preset
                    .as_ref()
                    .expect("needs_llm implies a selected style preset"),
            ),
            app_category: None,
            dictionary: dictionary_entries.clone(),
            language: detected_lang.into(),
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
                // still gets its sentence-initial capital (when the style enables it).
                if fmt.auto_capitalization {
                    lune_dict::formatting::capitalize_sentence_starts(&output.cleaned_text)
                } else {
                    output.cleaned_text
                }
            }
            Ok(Err(e)) => {
                warn!(error = %e, llm_ms = llm_t0.elapsed().as_millis() as u64, "LLM styling failed, using text as-is");
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
            config.persist_history,
            &active_arc,
            &idle_notify,
        )
        .await;
        return;
    }

    info!(%session_id, pipeline_ms = pipeline_t0.elapsed().as_millis() as u64, "post-processing complete");

    deliver_final_text(
        session_id,
        &raw_text,
        post_llm_text,
        status,
        &event_tx,
        persist,
        config.persist_history,
        &active_arc,
        &idle_notify,
    )
    .await;
}

/// Emit the finalized text, persist the (raw → final) pair and transcript,
/// and close the session — shared by the normal pipeline exit and the voice
/// snippet fast path (which skips correction and styling entirely).
#[allow(clippy::too_many_arguments)] // the session-teardown bundle plus the two texts
async fn deliver_final_text(
    session_id: Uuid,
    raw_text: &str,
    final_text: String,
    status: SessionStatus,
    event_tx: &mpsc::Sender<SessionEvent>,
    persist: Arc<Mutex<Database>>,
    persist_history: bool,
    active_arc: &Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: &Arc<Notify>,
) {
    let _ = send_bounded(
        event_tx,
        SessionEvent::FinalizedText {
            session_id,
            text: final_text.clone(),
        },
    )
    .await;

    let transcript = if final_text.is_empty() {
        None
    } else {
        Some(final_text)
    };

    // Record (raw ASR, final delivered) as labelled data. Best-effort — a
    // write failure never affects delivery. Skipped entirely when history is
    // off: this pair is the whole transcript twice over.
    if let Some(final_text) = transcript.as_deref().filter(|_| persist_history) {
        let mut db = persist.lock().await;
        if let Err(e) = blocking_db(|| db.insert_correction_pair(session_id, raw_text, final_text))
        {
            warn!(error = %e, %session_id, "failed to persist correction pair");
        }
    }

    // Await the DB finalize inline so a runtime shutdown between here and
    // the next tick can't drop the transcript. The lock is short-held and
    // sqlite writes are sync, so worst-case added latency is ~10ms.
    let ended_at = Some(Utc::now());
    finalize_db(
        session_id,
        persist,
        persist_history,
        status,
        ended_at,
        transcript,
    )
    .await;

    let _ = send_bounded(event_tx, SessionEvent::stopped()).await;
    clear_active(active_arc, idle_notify).await;
    info!(%session_id, "session stopped");
}

/// Resolve placeholder values and render a snippet body. Date and time come
/// from the local clock; `{clipboard}` triggers a clipboard read only when
/// the body references it (the read shells out and can take a moment).
async fn render_snippet_body(body: &str) -> String {
    use lune_dict::snippets::{PlaceholderValues, needs_clipboard, render_body};

    let clipboard = if needs_clipboard(body) {
        read_clipboard_text().await
    } else {
        None
    };
    let now = chrono::Local::now();
    let values = PlaceholderValues {
        date: now.format("%B %-d, %Y").to_string(),
        time: now.format("%H:%M").to_string(),
        datetime: now.format("%B %-d, %Y at %H:%M").to_string(),
        clipboard,
    };
    render_body(body, &values)
}

/// Best-effort clipboard text for `{clipboard}` placeholders. The read shells
/// out to `wl-paste`/`xclip`, so it runs on the blocking pool.
#[cfg(unix)]
async fn read_clipboard_text() -> Option<String> {
    tokio::task::spawn_blocking(crate::uinput_typer::clipboard_text)
        .await
        .ok()
        .flatten()
}

#[cfg(not(unix))]
async fn read_clipboard_text() -> Option<String> {
    None
}

/// Run a synchronous SQLite call without parking an async worker.
///
/// `ARCHITECTURE.md` states persistence never stalls the runtime, and the Tauri
/// command layer keeps that promise with `spawn_blocking`. The session
/// lifecycle cannot use the same tool: `Database` lives behind a shared
/// `tokio::sync::Mutex`, and the guard cannot cross into a spawned task.
///
/// `block_in_place` fits instead. It needs no `Send` bound, spawns nothing, and
/// — being synchronous — adds no cancellation point, so `start_inner`'s
/// cancellation safety is exactly what it was. Writes are small (a session row
/// is ~1.4 KB), but a WAL autocheckpoint can land on any of them and turn one
/// into a multi-megabyte copy plus two fsyncs, which is not something to do on
/// a worker thread while a dictation is starting.
///
/// A current-thread runtime has no worker to hand off to, so the closure runs
/// inline — that is the test harness, where it is what we want anyway.
fn blocking_db<T>(f: impl FnOnce() -> T) -> T {
    use tokio::runtime::{Handle, RuntimeFlavor};
    match Handle::try_current() {
        Ok(h) if h.runtime_flavor() == RuntimeFlavor::MultiThread => tokio::task::block_in_place(f),
        _ => f(),
    }
}

/// Close out the history row of a session whose task panicked.
///
/// The row was inserted `Active` with no `ended_at` at session start, and the
/// panicking task never reached `finalize_db`. Without this it stays that way
/// for good: invisible to the running app, and a permanently in-progress entry
/// in the History tab. The transcript column is left alone — a panic after the
/// transcript was stored must not erase it.
///
/// Only an unfinished row is touched. A panic can also land *after*
/// `finalize_db` has run (both `deliver_final_text` and `stop_session` do more
/// work afterwards), and downgrading a session that actually succeeded would
/// rewrite real history.
#[cold]
async fn finalize_panicked_session(session_id: Uuid, persist: Arc<Mutex<Database>>) {
    let mut db = persist.lock().await;
    match blocking_db(|| db.get_session(session_id)) {
        Ok(Some(mut session)) if session.ended_at.is_none() => {
            session.status = SessionStatus::Failed("internal session panic".into());
            session.ended_at = Some(Utc::now());
            if let Err(e) = blocking_db(|| db.update_session(&session)) {
                warn!(error = %e, %session_id, "could not close the panicked session's row");
            }
        }
        // Already finalized — the session completed and the panic came later.
        Ok(Some(_)) => {}
        // The start-of-session insert failed too; nothing to close.
        Ok(None) => {}
        Err(e) => warn!(error = %e, %session_id, "DB error closing the panicked session's row"),
    }
}

#[cold]
async fn finalize_db(
    session_id: Uuid,
    persist: Arc<Mutex<Database>>,
    persist_history: bool,
    status: SessionStatus,
    ended_at: Option<chrono::DateTime<Utc>>,
    transcript: Option<String>,
) {
    // No row was ever inserted for this session, so there is nothing to close
    // — and re-creating one here would defeat the retention setting.
    if !persist_history {
        return;
    }
    let mut db = persist.lock().await;
    match blocking_db(|| db.get_session(session_id)) {
        Ok(Some(mut s)) => {
            s.status = status;
            s.ended_at = ended_at;
            s.transcript = transcript;
            if let Err(e) = blocking_db(|| db.update_session(&s)) {
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
    persist_history: bool,
    active_arc: &Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: &Notify,
) {
    let msg = error.to_string();
    let error_dropped = !send_bounded(event_tx, SessionEvent::Error(msg.clone())).await;
    finalize_db(
        session_id,
        Arc::clone(persist),
        persist_history,
        SessionStatus::Failed(msg),
        None,
        None,
    )
    .await;
    // Every other terminal path emits `Stopped`; the init-failure path must
    // too, or runtime.rs never reaches its `SessionStopped` arm for a mic
    // that was busy or failed to open.
    let _ = send_bounded(event_tx, SessionEvent::Stopped { error_dropped }).await;
    clear_active(active_arc, idle_notify).await;
}

#[cold]
async fn stop_session(
    session_id: Uuid,
    status: SessionStatus,
    event_tx: &mpsc::Sender<SessionEvent>,
    persist: &Arc<Mutex<Database>>,
    persist_history: bool,
    active_arc: &Arc<Mutex<Option<ActiveHandle>>>,
    idle_notify: &Notify,
) {
    finalize_db(
        session_id,
        Arc::clone(persist),
        persist_history,
        status,
        Some(Utc::now()),
        None,
    )
    .await;
    let _ = send_bounded(event_tx, SessionEvent::stopped()).await;
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

    /// Panics where a real backend would decode. Stands in for any bug that
    /// unwinds the session task.
    struct PanickingAsr;

    #[async_trait::async_trait]
    impl AsrBackend for PanickingAsr {
        type SessionHandle = ();

        async fn start_session(&self, _: AsrSessionConfig) -> Result<()> {
            Ok(())
        }
        async fn push_audio(&self, _: &(), _: &[f32]) -> Result<()> {
            Ok(())
        }
        async fn poll_final(&self, _: &()) -> Result<Vec<AsrSegment>> {
            panic!("decode blew up");
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
        make_manager_with_settings(
            asr,
            post_processor,
            lune_core::config::UserSettings::default(),
        )
    }

    /// Like [`make_manager`] but with caller-supplied settings — used by tests
    /// that need the styling LLM to run (it runs iff a style is selected, i.e.
    /// `default_style` is `Some`).
    fn make_manager_with_settings<A: AsrBackend + 'static>(
        asr: A,
        post_processor: Arc<dyn TextPostProcessor>,
        user_settings: lune_core::config::UserSettings,
    ) -> (SessionManager<A>, Arc<Mutex<Database>>) {
        lune_dict::lexicon::warm();
        let db = lune_persist::Database::open_in_memory().unwrap();
        let dict =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        let settings = Arc::new(RwLock::new(user_settings));
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

    /// A stalled listener that never drains loses the `Error`. `send_bounded`
    /// has to say so, or the following `Stopped` reads as a clean session.
    #[tokio::test(start_paused = true)]
    async fn stalled_listener_reports_a_dropped_terminal_error() {
        let (tx, _rx) = mpsc::channel::<SessionEvent>(1);
        assert!(
            send_bounded(&tx, SessionEvent::Started).await,
            "fills queue"
        );

        let delivered = send_bounded(&tx, SessionEvent::Error("boom".into())).await;
        assert!(
            !delivered,
            "a dropped error must be reported as undelivered"
        );
    }

    async fn collect_events(mut rx: mpsc::Receiver<SessionEvent>) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        while let Ok(Some(event)) = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            let terminal = matches!(event, SessionEvent::Stopped { .. });
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

    /// The decode window (capture ends → transcript delivered) is seconds of
    /// inference, so the pill must show "processing" the whole time. That
    /// depends on `SessionEvent::Processing` being emitted *before* the final
    /// output — a regression here would leave the UI stuck on "listening"
    /// through the decode. `capture.rs` emits `Processing` right before
    /// `poll_final`.
    #[tokio::test]
    async fn processing_is_emitted_before_final_output() {
        local_set()
            .run_until(async {
                let (mgr, _) = make_manager(
                    scripted_asr(vec![Ok(vec![make_segment("hello", false)])]),
                    Arc::new(EchoPostProcessor),
                );
                let (tx, rx) = mpsc::channel(16);
                mgr.start_with_audio(tx, speech_stream()).await.unwrap();
                mgr.stop().await;
                let events = collect_events(rx).await;
                let processing_idx = events
                    .iter()
                    .position(|e| matches!(e, SessionEvent::Processing));
                let final_idx = events
                    .iter()
                    .position(|e| matches!(e, SessionEvent::FinalizedText { .. }));
                assert!(processing_idx.is_some(), "Processing must be emitted");
                assert!(final_idx.is_some(), "FinalizedText must be emitted");
                assert!(
                    processing_idx.unwrap() < final_idx.unwrap(),
                    "Processing must precede FinalizedText; events: {events:?}"
                );
            })
            .await;
    }

    /// A panic in the session task must not strand the app. The supervisor has
    /// to report it, release the "one session at a time" slot, and close the
    /// history row that `start` opened — otherwise the user gets a dead hotkey
    /// and a permanently in-progress History entry.
    ///
    /// This is the only test that exercises a real unwind, and it only means
    /// anything because the release profile unwinds too (see `[profile.release]`
    /// in the workspace `Cargo.toml`). The panic backtrace in the test output
    /// is expected.
    #[tokio::test]
    async fn a_panicked_session_is_reported_and_its_row_closed() {
        local_set()
            .run_until(async {
                let (mgr, db) = make_manager(PanickingAsr, Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(16);
                let id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();
                let events = collect_events(rx).await;

                assert!(
                    events.iter().any(|e| matches!(e, SessionEvent::Error(_))),
                    "the panic must be reported, not swallowed; events: {events:?}"
                );
                assert!(
                    events
                        .iter()
                        .any(|e| matches!(e, SessionEvent::Stopped { .. })),
                    "the listener must still see a terminal event; events: {events:?}"
                );

                // Cleanup runs after the events are sent, so wait for it.
                let closed = tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        let row = db.lock().await.get_session(id).ok().flatten();
                        if let Some(row) = row.filter(|r| r.ended_at.is_some()) {
                            return row;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .expect("the panicked session's history row must be closed");

                assert!(
                    matches!(closed.status, SessionStatus::Failed(_)),
                    "a panicked session must not be left Active; got {:?}",
                    closed.status
                );
                assert!(
                    mgr.active.lock().await.is_none(),
                    "the session slot must be released so the next hotkey works"
                );
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
                    events
                        .iter()
                        .any(|e| matches!(e, SessionEvent::Stopped { .. })),
                    "expected Stopped terminal event"
                );
                assert!(
                    !events
                        .iter()
                        .any(|e| matches!(e, SessionEvent::FinalizedText { .. })),
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
                        SessionEvent::FinalizedText { text, .. } => Some(text.as_str()),
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
    async fn insert_snippet_command_expands_template() {
        local_set()
            .run_until(async {
                let responses = vec![Ok(vec![make_segment("Insert greeting.", false)])];
                let (mgr, db) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                db.lock()
                    .await
                    .insert_snippet("greet", "greeting", "Hello from Lune!\nSecond line.")
                    .unwrap();
                let (tx, rx) = mpsc::channel(32);
                let session_id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                let events = collect_events(rx).await;
                let finalized: Vec<&str> = events
                    .iter()
                    .filter_map(|event| match event {
                        SessionEvent::FinalizedText { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                // The template is delivered verbatim — multiline, no correction,
                // no terminal-period formatting.
                assert_eq!(
                    finalized.last().copied(),
                    Some("Hello from Lune!\nSecond line.")
                );

                let session = db.lock().await.get_session(session_id).unwrap().unwrap();
                assert_eq!(
                    session.transcript.as_deref(),
                    Some("Hello from Lune!\nSecond line.")
                );
            })
            .await;
    }

    #[tokio::test]
    async fn bare_trigger_expands_snippet_with_placeholders() {
        local_set()
            .run_until(async {
                let responses = vec![Ok(vec![make_segment("Meeting header!", false)])];
                let (mgr, db) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                db.lock()
                    .await
                    .insert_snippet("mh", "meeting header", "## Meeting — {date}")
                    .unwrap();
                let (tx, rx) = mpsc::channel(32);
                let _ = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                let events = collect_events(rx).await;
                let text = events
                    .iter()
                    .find_map(|event| match event {
                        SessionEvent::FinalizedText { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .expect("snippet delivered");
                let expected =
                    format!("## Meeting — {}", chrono::Local::now().format("%B %-d, %Y"));
                assert_eq!(text, expected);
            })
            .await;
    }

    #[tokio::test]
    async fn capture_session_returns_raw_text_without_side_effects() {
        local_set()
            .run_until(async {
                let responses = vec![Ok(vec![make_segment("Signature.", false)])];
                let (mgr, db) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                // A matching snippet must NOT expand during capture — capture
                // returns the raw phrase itself for the trigger recorder.
                db.lock()
                    .await
                    .insert_snippet("sig", "signature", "Best regards,\nDayton")
                    .unwrap();
                let (tx, rx) = mpsc::channel(32);
                let session_id = mgr
                    .start_capture_with_audio(tx, speech_stream())
                    .await
                    .unwrap();

                let events = collect_events(rx).await;
                let text = events
                    .iter()
                    .find_map(|event| match event {
                        SessionEvent::FinalizedText { text, .. } => Some(text.clone()),
                        _ => None,
                    })
                    .expect("capture emits raw text");
                assert_eq!(text, "Signature.");

                // Capture leaves no history: no session row was ever written.
                assert!(db.lock().await.get_session(session_id).unwrap().is_none());
            })
            .await;
    }

    #[tokio::test]
    async fn insert_with_unknown_name_falls_back_to_dictation() {
        local_set()
            .run_until(async {
                // Regression: "insert …" used to be consumed as a command and
                // type nothing. With no matching snippet it must dictate.
                let responses = vec![Ok(vec![make_segment(
                    "insert the disk into the drive",
                    false,
                )])];
                let (mgr, _db) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(32);
                let _ = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                let events = collect_events(rx).await;
                let finalized: Vec<&str> = events
                    .iter()
                    .filter_map(|event| match event {
                        SessionEvent::FinalizedText { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    finalized.last().copied(),
                    Some("Insert the disk into the drive.")
                );
            })
            .await;
    }

    #[test]
    fn correct_and_format_capitalizes_and_punctuates_clean_english() {
        // The deterministic shaping stage: a clean lowercase utterance gets a
        // leading capital and terminal period, detected language defaults to
        // English, and nothing is left unresolved.
        lune_dict::lexicon::warm();
        let heuristic = LexicalGate::new();
        let pipeline = PipelineConfig::default();
        let fmt = lune_core::config::FormatToggles::default();
        let segments = vec![make_segment("hello world", false)];

        let shaped = correct_and_format(
            "hello world",
            &segments,
            &[] as &[lune_dict::DictMatch],
            &heuristic,
            &pipeline,
            &fmt,
            false,
        );

        assert_eq!(shaped.text_after_heuristic, "Hello world.");
        assert_eq!(shaped.detected_lang, "English");
        assert!(shaped.unresolved_words.is_empty());
    }

    #[tokio::test]
    async fn command_detected_carries_session_id() {
        local_set()
            .run_until(async {
                // A whole-utterance voice command emits CommandDetected; it must
                // carry the session id so the delivery path stays correlated.
                let responses = vec![Ok(vec![make_segment("switch to formal", false)])];
                let (mgr, _) = make_manager(scripted_asr(responses), Arc::new(EchoPostProcessor));
                let (tx, rx) = mpsc::channel(32);
                let session_id = mgr.start_with_audio(tx, speech_stream()).await.unwrap();

                let events = collect_events(rx).await;
                let detected = events.iter().find_map(|e| match e {
                    SessionEvent::CommandDetected {
                        session_id: id,
                        payload,
                    } => Some((*id, payload.clone())),
                    _ => None,
                });
                let (evt_session_id, _payload) =
                    detected.expect("a whole-utterance command must emit CommandDetected");
                assert_eq!(
                    evt_session_id, session_id,
                    "CommandDetected must carry the originating session id"
                );
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
                        .any(|event| matches!(event, SessionEvent::FinalizedText { .. }))
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
                    mean_confidence: 0.2,
                }];
                let responses = vec![Ok(vec![segment])];
                // The styling LLM runs only when a style is selected; setting
                // "casual" enables it, so the cancel-during-LLM path is exercised.
                let settings = lune_core::config::UserSettings {
                    default_style: Some("casual".into()),
                    ..Default::default()
                };
                let (mgr, db) = make_manager_with_settings(scripted_asr(responses), post, settings);
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
                        .any(|event| matches!(event, SessionEvent::FinalizedText { .. }))
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
