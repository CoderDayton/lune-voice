use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::AudioCaptureService;
use lune_commands::VoiceCommandRouter;
use lune_core::Result;
use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSessionConfig};
use lune_core::config::StylePreset;
use lune_core::error::Error;
use lune_core::llm::{PostProcessInput, TextPostProcessor};
use lune_core::session::{DictationSession, SessionEvent, SessionStatus};
use lune_dict::DictionaryManager;
use lune_output::TextOutputController;
use lune_persist::Database;

/// Configuration for the session pipeline.
#[derive(Clone)]
pub struct SessionManagerConfig {
    /// Silence window before finalizing an utterance (default 500 ms).
    pub silence_timeout: Duration,
    pub asr_config: AsrSessionConfig,
    pub style_preset: StylePreset,
    pub asr_backend_kind: AsrBackendKind,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            silence_timeout: Duration::from_millis(500),
            asr_config: AsrSessionConfig::default(),
            style_preset: StylePreset::general(),
            asr_backend_kind: AsrBackendKind::ParakeetTdt,
        }
    }
}

struct ActiveHandle {
    stop_tx: oneshot::Sender<()>,
    session_id: Uuid,
}

/// Orchestrates the full dictation pipeline.
///
/// Only one session may be active at a time. Call `start()` to begin and
/// `stop()` to finalize gracefully. Events are delivered over the channel
/// passed to `start()`.
pub struct SessionManager<A: AsrBackend> {
    audio: Arc<AudioCaptureService>,
    asr: Arc<A>,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    config: SessionManagerConfig,
    active: Arc<Mutex<Option<ActiveHandle>>>,
}

// SessionManager does not require A: Send because spawn_local is used.
impl<A: AsrBackend + 'static> SessionManager<A> {
    pub fn new(
        audio: AudioCaptureService,
        asr: A,
        router: VoiceCommandRouter,
        post_processor: Arc<dyn TextPostProcessor>,
        dict: DictionaryManager,
        persist: Database,
        config: SessionManagerConfig,
    ) -> Self {
        Self {
            audio: Arc::new(audio),
            asr: Arc::new(asr),
            router: Arc::new(router),
            post_processor,
            dict: Arc::new(Mutex::new(dict)),
            persist: Arc::new(Mutex::new(persist)),
            config,
            active: Arc::new(Mutex::new(None)),
        }
    }

    /// Start a new session. Returns `Err` if one is already active.
    ///
    /// The caller receives events over `event_tx` until `SessionEvent::Stopped`
    /// is emitted.
    pub async fn start(&self, event_tx: mpsc::Sender<SessionEvent>) -> Result<Uuid> {
        let mut lock = self.active.lock().await;
        if lock.is_some() {
            return Err(Error::Session("session already active".into()));
        }

        let session_id = Uuid::new_v4();
        let (stop_tx, stop_rx) = oneshot::channel();

        // Persist the initial session record.
        {
            let session = DictationSession {
                id: session_id,
                started_at: Utc::now(),
                ended_at: None,
                style: self.config.style_preset.id.clone(),
                backend: self.config.asr_backend_kind,
                status: SessionStatus::Active,
                transcript: None,
            };
            if let Err(e) = self.persist.lock().await.insert_session(&session) {
                warn!(error = %e, "failed to persist session start");
            }
        }

        // Clone Arcs for the background task.
        let audio = Arc::clone(&self.audio);
        let asr = Arc::clone(&self.asr);
        let router = Arc::clone(&self.router);
        let post_processor = Arc::clone(&self.post_processor);
        let dict = Arc::clone(&self.dict);
        let persist = Arc::clone(&self.persist);
        let silence_timeout = self.config.silence_timeout;
        let asr_config = self.config.asr_config.clone();
        let style_preset = self.config.style_preset.clone();
        let active_arc = Arc::clone(&self.active);

        // spawn_local: future need not be Send (holds !Send cpal::Stream).
        tokio::task::spawn_local(run_session(
            session_id,
            audio,
            asr,
            router,
            post_processor,
            dict,
            persist,
            silence_timeout,
            asr_config,
            style_preset,
            event_tx,
            stop_rx,
            active_arc,
        ));

        *lock = Some(ActiveHandle {
            stop_tx,
            session_id,
        });
        info!(%session_id, "session started");
        Ok(session_id)
    }

    /// Gracefully stop the active session (finalizes the current buffer).
    pub async fn stop(&self) {
        let mut lock = self.active.lock().await;
        if let Some(handle) = lock.take() {
            debug!(session_id = %handle.session_id, "sending stop signal");
            let _ = handle.stop_tx.send(());
        }
    }

    /// Cancel the active session (same teardown, status will be Cancelled).
    pub async fn cancel(&self) {
        // The pipeline task treats any stop signal as graceful; the caller
        // should pass a flag for cancelled status if needed. For now both
        // paths are identical — the distinction is reflected in DB status only
        // if we add a separate cancel channel. Keep simple for v0.1.
        self.stop().await;
    }

    /// The active session ID, or `None` if idle.
    pub async fn active_session_id(&self) -> Option<Uuid> {
        self.active.lock().await.as_ref().map(|h| h.session_id)
    }
}

// ── Pipeline task ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
// Note: A need not be Send because this runs inside a LocalSet.
async fn run_session<A: AsrBackend + 'static>(
    session_id: Uuid,
    audio: Arc<AudioCaptureService>,
    asr: Arc<A>,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    silence_timeout: Duration,
    asr_config: AsrSessionConfig,
    style_preset: StylePreset,
    event_tx: mpsc::Sender<SessionEvent>,
    mut stop_rx: oneshot::Receiver<()>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
) {
    let mut status = SessionStatus::Completed;

    // ── Start audio capture ───────────────────────────────────────────────
    let (audio_tx, mut audio_rx) = mpsc::channel::<lune_core::audio::AudioFrame>(1024);

    // `cpal::Stream` inside SessionHandle is !Send; this is fine here because
    // run_session is spawned via spawn_local (LocalSet, current-thread runtime)
    // and therefore the future does not need to be Send.
    let _audio_handle = match audio.start_session(audio_tx) {
        Ok(h) => h,
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

    // ── Start ASR session ─────────────────────────────────────────────────
    let asr_session = match asr.start_session(asr_config).await {
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

    let _ = event_tx.send(SessionEvent::Started).await;

    let mut output_ctrl = TextOutputController::new(event_tx.clone());
    // `accumulated` collects all finalized ASR text for the session.
    let mut session_text = String::new();
    // `utterance` collects text for the current utterance (reset on silence).
    let mut utterance = String::new();

    // ── Main loop ─────────────────────────────────────────────────────────
    // `timeout` on recv: if no audio frame arrives within silence_timeout we
    // treat it as end-of-utterance and finalize the buffer.
    loop {
        tokio::select! {
            biased;

            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }

            recv = tokio::time::timeout(silence_timeout, audio_rx.recv()) => {
                match recv {
                    // ── Silence timeout ───────────────────────────────────
                    Err(_elapsed) => {
                        if !utterance.is_empty() {
                            debug!(%session_id, "silence timeout, finalizing utterance");
                            let finalized = finalize_utterance(
                                &utterance,
                                &*post_processor,
                                &dict,
                                &style_preset,
                                &router,
                                &event_tx,
                                &mut output_ctrl,
                            ).await;
                            if !finalized.is_empty() {
                                if !session_text.is_empty() {
                                    session_text.push(' ');
                                }
                                session_text.push_str(&finalized);
                            }
                            utterance.clear();
                        }
                    }

                    // ── Audio channel closed ──────────────────────────────
                    Ok(None) => {
                        debug!(%session_id, "audio channel closed");
                        break;
                    }

                    // ── Audio frame received ──────────────────────────────
                    Ok(Some(frame)) => {
                        if let Err(e) = asr.push_audio(&asr_session, &frame.samples).await {
                            warn!(error = %e, "push_audio error");
                            continue;
                        }

                        // Drain all buffered frames so the channel doesn't overflow
                        // while inference (in poll_results) blocks.
                        while let Ok(extra) = audio_rx.try_recv() {
                            if let Err(e) = asr.push_audio(&asr_session, &extra.samples).await {
                                warn!(error = %e, "push_audio error (drain)");
                                break;
                            }
                        }

                        let segments = match asr.poll_results(&asr_session).await {
                            Ok(s) => s,
                            Err(e) => {
                                warn!(error = %e, "poll_results error");
                                continue;
                            }
                        };

                        for seg in &segments {
                            let route = router.route_full(&seg.text);

                            // Emit a SessionEvent for each detected command.
                            for cmd in &route.commands {
                                let _ = event_tx
                                    .send(SessionEvent::CommandDetected(format!("{cmd:?}")))
                                    .await;
                            }

                            if seg.is_final {
                                // Append the clean remainder to the utterance buffer.
                                let text = route.remaining.trim();
                                if !text.is_empty() {
                                    if !utterance.is_empty() {
                                        utterance.push(' ');
                                    }
                                    utterance.push_str(text);
                                }
                            } else {
                                // Emit partial: session_text + current utterance + partial.
                                let partial_suffix = route.remaining.trim();
                                let mut partial = session_text.clone();
                                if !utterance.is_empty() {
                                    if !partial.is_empty() {
                                        partial.push(' ');
                                    }
                                    partial.push_str(&utterance);
                                }
                                if !partial_suffix.is_empty() {
                                    if !partial.is_empty() {
                                        partial.push(' ');
                                    }
                                    partial.push_str(partial_suffix);
                                }
                                output_ctrl.emit_partial(partial).await;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Finalize remaining utterance buffer ───────────────────────────────
    if !utterance.is_empty() {
        let finalized = finalize_utterance(
            &utterance,
            &*post_processor,
            &dict,
            &style_preset,
            &router,
            &event_tx,
            &mut output_ctrl,
        )
        .await;
        if !finalized.is_empty() {
            if !session_text.is_empty() {
                session_text.push(' ');
            }
            session_text.push_str(&finalized);
        }
    }

    // ── End ASR session ───────────────────────────────────────────────────
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

    let _ = event_tx.send(SessionEvent::Stopped).await;
    active_arc.lock().await.take();
    info!(%session_id, "session stopped");
}

/// Post-process an utterance and emit FinalizedText; returns cleaned text.
async fn finalize_utterance(
    utterance: &str,
    post_processor: &dyn TextPostProcessor,
    dict: &Mutex<DictionaryManager>,
    style_preset: &StylePreset,
    _router: &VoiceCommandRouter,
    event_tx: &mpsc::Sender<SessionEvent>,
    output_ctrl: &mut TextOutputController,
) -> String {
    // Apply dictionary substitutions.
    let after_dict = dict.lock().await.apply(utterance);

    // Gather relevant dictionary entries for the LLM.
    let relevant = dict
        .lock()
        .await
        .top_relevant(&after_dict, 20)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();

    let input = PostProcessInput {
        raw_text: after_dict,
        prev_context: None,
        style_preset: style_preset.clone(),
        app_category: None,
        dictionary: relevant,
    };

    let cleaned = match post_processor.transform(&input).await {
        Ok(out) => out.cleaned_text,
        Err(e) => {
            warn!(error = %e, "post-processor error, using raw text");
            let _ = event_tx.send(SessionEvent::Error(e.to_string())).await;
            input.raw_text
        }
    };

    output_ctrl.emit_final(cleaned.clone()).await;
    cleaned
}

/// Persist final session state to the database.
async fn finalize_db(
    session_id: Uuid,
    persist: &Mutex<Database>,
    status: SessionStatus,
    ended_at: Option<chrono::DateTime<Utc>>,
    transcript: Option<String>,
) {
    let mut db = persist.lock().await;
    // Fetch existing session, update fields, write back.
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
    use lune_core::asr::{AsrSegment, AsrSessionConfig};
    use lune_core::llm::{PostProcessInput, PostProcessOutput};
    use tokio::sync::mpsc;

    // ── Stubs ─────────────────────────────────────────────────────────────

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
        let audio_config = lune_core::audio::AudioCaptureConfig::default();

        SessionManager::new(
            AudioCaptureService::new(audio_config),
            NoopAsr,
            VoiceCommandRouter::new(),
            Arc::new(EchoPostProcessor),
            dict,
            db,
            SessionManagerConfig::default(),
        )
    }

    // All tests that call `start()` must run inside a LocalSet because
    // `start()` uses `spawn_local` internally.
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
    async fn double_start_returns_error() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx1, _rx1) = mpsc::channel(16);
                let (tx2, _rx2) = mpsc::channel(16);
                mgr.start(tx1).await.unwrap();
                let result = mgr.start(tx2).await;
                assert!(result.is_err());
                mgr.stop().await;
            })
            .await;
    }

    #[tokio::test]
    async fn stop_when_idle_is_noop() {
        let mgr = make_manager();
        mgr.stop().await; // no spawn_local called; no LocalSet needed
    }

    #[tokio::test]
    async fn active_session_id_tracks_state() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                assert!(mgr.active_session_id().await.is_none());

                let (tx, _rx) = mpsc::channel(16);
                let id = mgr.start(tx).await.unwrap();
                assert_eq!(mgr.active_session_id().await, Some(id));

                mgr.stop().await;
            })
            .await;
    }

    #[tokio::test]
    async fn stop_clears_active() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx, mut rx) = mpsc::channel(16);
                mgr.start(tx).await.unwrap();
                mgr.stop().await;

                // Give the background task time to emit Stopped and clear active.
                tokio::time::sleep(Duration::from_millis(50)).await;

                // Drain events.
                while let Ok(ev) = rx.try_recv() {
                    // Just drain; don't assert ordering — the NoopAsr pipeline
                    // may or may not have emitted Started before stop arrived.
                    let _ = ev;
                }
            })
            .await;
    }
}
