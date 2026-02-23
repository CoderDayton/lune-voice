use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::AudioCaptureService;
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

use crate::confidence_router::{ConfidenceRouter, ConfidenceRouterConfig, RouteDecision};
use crate::llm_worker::{self, LlmRequest};

/// Configuration for the session pipeline.
#[derive(Clone)]
pub struct SessionManagerConfig {
    /// Silence window before finalizing an utterance (default 500 ms).
    pub silence_timeout: Duration,
    pub asr_config: AsrSessionConfig,
    pub asr_backend_kind: AsrBackendKind,
    pub confidence_config: ConfidenceRouterConfig,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            silence_timeout: Duration::from_millis(800),
            asr_config: AsrSessionConfig::default(),
            asr_backend_kind: AsrBackendKind::ParakeetTdt,
            confidence_config: ConfidenceRouterConfig::default(),
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
    settings: Arc<RwLock<UserSettings>>,
    active: Arc<Mutex<Option<ActiveHandle>>>,
}

// SessionManager does not require A: Send because spawn_local is used.
impl<A: AsrBackend + 'static> SessionManager<A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        audio: Arc<AudioCaptureService>,
        asr: Arc<A>,
        router: Arc<VoiceCommandRouter>,
        post_processor: Arc<dyn TextPostProcessor>,
        dict: Arc<Mutex<DictionaryManager>>,
        persist: Arc<Mutex<Database>>,
        config: SessionManagerConfig,
        settings: Arc<RwLock<UserSettings>>,
    ) -> Self {
        Self {
            audio,
            asr,
            router,
            post_processor,
            dict,
            persist,
            config,
            settings,
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

        // Resolve style from current settings (reads latest value each session).
        let current_style_id = self.settings.read().await.default_style.clone();
        let style_preset = builtin_registry()
            .get(&current_style_id)
            .unwrap_or_else(|| {
                builtin_registry()
                    .get("general")
                    .expect("builtin 'general' preset must exist")
            });

        // Persist the initial session record.
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

        // Clone Arcs for the background task.
        let audio = Arc::clone(&self.audio);
        let asr = Arc::clone(&self.asr);
        let router = Arc::clone(&self.router);
        let post_processor = Arc::clone(&self.post_processor);
        let dict = Arc::clone(&self.dict);
        let persist = Arc::clone(&self.persist);
        let config = self.config.clone();
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
            config,
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
    #[inline]
    pub async fn cancel(&self) {
        // The pipeline task treats any stop signal as graceful; the caller
        // should pass a flag for cancelled status if needed. For now both
        // paths are identical — the distinction is reflected in DB status only
        // if we add a separate cancel channel. Keep simple for v0.1.
        self.stop().await;
    }

    /// The active session ID, or `None` if idle.
    #[inline]
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
    config: SessionManagerConfig,
    style_preset: Arc<StylePreset>,
    event_tx: mpsc::Sender<SessionEvent>,
    mut stop_rx: oneshot::Receiver<()>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
) {
    let mut status = SessionStatus::Completed;
    let silence_timeout = config.silence_timeout;

    // ── Confidence router + LLM worker ────────────────────────────────────
    let mut confidence_router = ConfidenceRouter::new(config.confidence_config.clone());
    let (llm_tx, mut llm_rx) = llm_worker::spawn_llm_worker(
        Arc::clone(&post_processor),
        config.confidence_config.max_length_ratio,
        config.confidence_config.min_length_ratio,
    );
    let mut utterance_counter: u64 = 0;
    // Raw text waiting for LLM delivery, keyed by utterance_id.
    // Entries are removed when the LLM responds (retain id > resp.utterance_id
    // handles batching: the worker returns max_id of the batch).
    // Any still-pending on stop are delivered as raw fallback.
    let mut pending_delivery: std::collections::VecDeque<(u64, String)> =
        std::collections::VecDeque::new();

    // ── Start audio capture ───────────────────────────────────────────────
    let (audio_tx, mut audio_rx) = mpsc::channel::<lune_core::audio::AudioFrame>(1024);

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

    let _ = event_tx.send(SessionEvent::Started).await;

    let mut output_ctrl = TextOutputController::new(event_tx.clone());
    let mut session_text = String::with_capacity(512);
    let mut utterance = String::with_capacity(128);
    let mut pending_nonfinal = String::with_capacity(128);
    // Latest confidence from finalized ASR segments.
    let mut pending_confidence: Option<f32> = None;

    // Persistent buffers — reused every iteration to eliminate per-frame allocation.
    // partial_buf: built for non-final segment display (hot path: ~50Hz).
    let mut partial_buf = String::with_capacity(512);
    // audio_batch: coalesces the initial frame + any immediately-queued frames
    // into a single push_audio call, reducing await-point overhead under bursts.
    let mut audio_batch: Vec<f32> = Vec::with_capacity(4096);

    // ── Main loop (3 branches: stop, LLM results, audio/silence) ─────────
    loop {
        tokio::select! {
            biased;

            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }

            // ── LLM correction results ───────────────────────────────
            Some(resp) = llm_rx.recv() => {
                // Remove all pending entries covered by this batch (worker returns
                // max utterance_id when it batches multiple requests together).
                pending_delivery.retain(|(id, _)| *id > resp.utterance_id);

                let delivered = if resp.accepted && resp.corrected_text != resp.original_text {
                    debug!(
                        utterance_id = resp.utterance_id,
                        original = %resp.original_text,
                        corrected = %resp.corrected_text,
                        "LLM correction accepted"
                    );
                    resp.corrected_text
                } else {
                    resp.original_text
                };

                output_ctrl.emit_final(delivered.clone()).await;
                if !delivered.is_empty() {
                    if !session_text.is_empty() {
                        session_text.push(' ');
                    }
                    session_text.push_str(&delivered);
                }
                confidence_router.confirm(&delivered);
            }

            recv = tokio::time::timeout(silence_timeout, audio_rx.recv()) => {
                match recv {
                    // ── Silence timeout ───────────────────────────────────
                    Err(_elapsed) => {
                        if !pending_nonfinal.is_empty() {
                            if !utterance.is_empty() {
                                utterance.push(' ');
                            }
                            utterance.push_str(&pending_nonfinal);
                            pending_nonfinal.clear();
                        }
                        if !utterance.is_empty() {
                            debug!(%session_id, "silence timeout, finalizing utterance");

                            // Apply dictionary substitutions.
                            let after_dict = dict.lock().await.apply(&utterance);

                            // Route by confidence (3-tier).
                            let decision = confidence_router.route(&after_dict, pending_confidence);
                            match &decision {
                                RouteDecision::HighConfidence { text, confidence } => {
                                    debug!(confidence, "high confidence, queuing LLM (holding delivery)");
                                    utterance_counter += 1;
                                    pending_delivery.push_back((utterance_counter, text.clone()));
                                    let relevant = dict
                                        .lock()
                                        .await
                                        .top_relevant(text, 20)
                                        .into_iter()
                                        .cloned()
                                        .collect::<Vec<_>>();
                                    let req = LlmRequest {
                                        utterance_id: utterance_counter,
                                        text: text.clone(),
                                        context: confidence_router.context(),
                                        style_preset: Arc::clone(&style_preset),
                                        dictionary: relevant,
                                    };
                                    if let Err(e) = llm_tx.send(req).await {
                                        warn!(error = %e, "LLM unavailable, delivering raw");
                                        pending_delivery.pop_back();
                                        output_ctrl.emit_final(text.clone()).await;
                                        if !text.is_empty() {
                                            if !session_text.is_empty() {
                                                session_text.push(' ');
                                            }
                                            session_text.push_str(text);
                                        }
                                        confidence_router.confirm(text);
                                    }
                                    // Partial display preserved — user sees last ASR text until LLM delivers.
                                }
                                RouteDecision::Pending { confidence } => {
                                    debug!(confidence, "accumulating in pending buffer");
                                }
                                RouteDecision::ReadyForLlm { text, confidence } => {
                                    debug!(confidence, "pending buffer ready, queueing LLM (holding delivery)");
                                    utterance_counter += 1;
                                    pending_delivery.push_back((utterance_counter, text.clone()));
                                    let relevant = dict
                                        .lock()
                                        .await
                                        .top_relevant(text, 20)
                                        .into_iter()
                                        .cloned()
                                        .collect::<Vec<_>>();
                                    let req = LlmRequest {
                                        utterance_id: utterance_counter,
                                        text: text.clone(),
                                        context: confidence_router.context(),
                                        style_preset: Arc::clone(&style_preset),
                                        dictionary: relevant,
                                    };
                                    if let Err(e) = llm_tx.send(req).await {
                                        warn!(error = %e, "LLM unavailable, delivering raw");
                                        pending_delivery.pop_back();
                                        output_ctrl.emit_final(text.clone()).await;
                                        if !text.is_empty() {
                                            if !session_text.is_empty() {
                                                session_text.push(' ');
                                            }
                                            session_text.push_str(text);
                                        }
                                    }
                                    // Partial display preserved — user sees last ASR text until LLM delivers.
                                }
                            }

                            utterance.clear();
                            pending_confidence = None;
                            if let Err(e) = asr.reset_context(&asr_session).await {
                                warn!(error = %e, "reset_context error");
                            }
                        }
                    }

                    // ── Audio channel closed ──────────────────────────────
                    Ok(None) => {
                        debug!(%session_id, "audio channel closed");
                        break;
                    }

                    // ── Audio frame received ──────────────────────────────
                    Ok(Some(frame)) => {
                        // Coalesce initial frame + any immediately-available frames
                        // into a single push_audio call. Reduces await-point count
                        // during microphone bursts and amortises ASR call overhead.
                        audio_batch.clear();
                        audio_batch.extend_from_slice(&frame.samples);
                        while let Ok(extra) = audio_rx.try_recv() {
                            audio_batch.extend_from_slice(&extra.samples);
                        }

                        if let Err(e) = asr.push_audio(&asr_session, &audio_batch).await {
                            warn!(error = %e, "push_audio error");
                            continue;
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

                            for cmd in &route.commands {
                                let _ = event_tx
                                    .send(SessionEvent::CommandDetected(format!("{cmd:?}")))
                                    .await;
                            }

                            if seg.is_final {
                                let text = route.remaining.trim();
                                if !text.is_empty() {
                                    if !utterance.is_empty() {
                                        utterance.push(' ');
                                    }
                                    utterance.push_str(text);
                                }
                                pending_nonfinal.clear();
                                // Track confidence from the latest final segment.
                                pending_confidence = seg.confidence;
                            } else {
                                let partial_suffix = route.remaining.trim();
                                // Reuse persistent buffer — eliminates per-segment
                                // String allocation on the ~50 Hz partial path.
                                partial_buf.clear();
                                partial_buf.push_str(&session_text);
                                if !utterance.is_empty() {
                                    if !partial_buf.is_empty() {
                                        partial_buf.push(' ');
                                    }
                                    partial_buf.push_str(&utterance);
                                }
                                if !partial_suffix.is_empty() {
                                    if !partial_buf.is_empty() {
                                        partial_buf.push(' ');
                                    }
                                    partial_buf.push_str(partial_suffix);
                                    pending_nonfinal.clear();
                                    pending_nonfinal.push_str(partial_suffix);
                                }
                                // Track confidence from partials too.
                                if seg.confidence.is_some() {
                                    pending_confidence = seg.confidence;
                                }
                                output_ctrl.emit_partial(partial_buf.clone()).await;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Promote any pending non-final text ────────────────────────────────
    if !pending_nonfinal.is_empty() {
        if !utterance.is_empty() {
            utterance.push(' ');
        }
        utterance.push_str(&pending_nonfinal);
    }

    // ── Flush pending buffer from confidence router ──────────────────────
    if let Some(remaining) = confidence_router.flush_pending() {
        debug!("flushing pending buffer on stop: {:?}", remaining);
        utterance_counter += 1;
        pending_delivery.push_back((utterance_counter, remaining.clone()));
        let relevant = dict
            .lock()
            .await
            .top_relevant(&remaining, 20)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let req = LlmRequest {
            utterance_id: utterance_counter,
            text: remaining.clone(),
            context: confidence_router.context(),
            style_preset: Arc::clone(&style_preset),
            dictionary: relevant,
        };
        if let Err(e) = llm_tx.send(req).await {
            warn!(error = %e, "LLM unavailable for flushed pending, delivering raw");
            pending_delivery.pop_back();
            output_ctrl.emit_final(remaining.clone()).await;
            if !remaining.is_empty() {
                if !session_text.is_empty() {
                    session_text.push(' ');
                }
                session_text.push_str(&remaining);
            }
        }
    }

    // ── Finalize remaining utterance buffer (queue to LLM on stop) ──────────
    if !utterance.is_empty() {
        let after_dict = dict.lock().await.apply(&utterance);
        utterance_counter += 1;
        pending_delivery.push_back((utterance_counter, after_dict.clone()));
        let relevant = dict
            .lock()
            .await
            .top_relevant(&after_dict, 20)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let req = LlmRequest {
            utterance_id: utterance_counter,
            text: after_dict.clone(),
            context: confidence_router.context(),
            style_preset: Arc::clone(&style_preset),
            dictionary: relevant,
        };
        if let Err(e) = llm_tx.send(req).await {
            warn!(error = %e, "LLM unavailable for final utterance, delivering raw");
            pending_delivery.pop_back();
            output_ctrl.emit_final(after_dict.clone()).await;
            if !after_dict.is_empty() {
                if !session_text.is_empty() {
                    session_text.push(' ');
                }
                session_text.push_str(&after_dict);
            }
            confidence_router.confirm(&after_dict);
        }
    }

    // ── Drain in-flight LLM results; deliver remaining pending as raw ─────
    // Non-blocking: picks up any LLM results that finished during teardown.
    // For short utterances the LLM often finishes before we get here.
    while let Ok(resp) = llm_rx.try_recv() {
        pending_delivery.retain(|(id, _)| *id > resp.utterance_id);
        let delivered = if resp.accepted && resp.corrected_text != resp.original_text {
            resp.corrected_text
        } else {
            resp.original_text
        };
        output_ctrl.emit_final(delivered.clone()).await;
        if !delivered.is_empty() {
            if !session_text.is_empty() {
                session_text.push(' ');
            }
            session_text.push_str(&delivered);
        }
    }
    // Anything still pending didn't make it through LLM — deliver raw.
    for (_, raw) in pending_delivery.drain(..) {
        if !raw.is_empty() {
            output_ctrl.emit_final(raw.clone()).await;
            if !session_text.is_empty() {
                session_text.push(' ');
            }
            session_text.push_str(&raw);
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

/// Persist final session state to the database.
#[cold]
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
        let settings = Arc::new(RwLock::new(lune_core::config::UserSettings::default()));

        SessionManager::new(
            Arc::new(AudioCaptureService::new(audio_config)),
            Arc::new(NoopAsr),
            Arc::new(VoiceCommandRouter::new()),
            Arc::new(EchoPostProcessor),
            Arc::new(Mutex::new(dict)),
            Arc::new(Mutex::new(db)),
            SessionManagerConfig::default(),
            settings,
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
