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

use crate::llm_worker::{self, LlmRequest};

/// Configuration for the session pipeline.
#[derive(Clone)]
pub struct SessionManagerConfig {
    /// Silence window before finalizing an utterance (default 800 ms).
    pub silence_timeout: Duration,
    pub asr_config: AsrSessionConfig,
    pub asr_backend_kind: AsrBackendKind,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            silence_timeout: Duration::from_millis(800),
            asr_config: AsrSessionConfig::default(),
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
    settings: Arc<RwLock<UserSettings>>,
    active: Arc<Mutex<Option<ActiveHandle>>>,
}

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
    pub async fn start(&self, event_tx: mpsc::Sender<SessionEvent>) -> Result<Uuid> {
        let mut lock = self.active.lock().await;
        if lock.is_some() {
            return Err(Error::Session("session already active".into()));
        }

        let session_id = Uuid::new_v4();
        let (stop_tx, stop_rx) = oneshot::channel();

        let current_style_id = self.settings.read().await.default_style.clone();
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

        let audio = Arc::clone(&self.audio);
        let asr = Arc::clone(&self.asr);
        let router = Arc::clone(&self.router);
        let post_processor = Arc::clone(&self.post_processor);
        let dict = Arc::clone(&self.dict);
        let persist = Arc::clone(&self.persist);
        let config = self.config.clone();
        let active_arc = Arc::clone(&self.active);

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

    let (llm_tx, mut llm_rx, mut llm_stream_rx) =
        llm_worker::spawn_llm_worker(Arc::clone(&post_processor));

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

    // Accumulates streamed chunks; flushed to session_text on LlmResponse.
    let mut llm_partial_buf = String::with_capacity(256);

    // Reused each audio frame to avoid allocation on the ~50Hz partial path.
    let mut partial_buf = String::with_capacity(512);
    let mut audio_batch: Vec<f32> = Vec::with_capacity(4096);

    // ── Main loop ─────────────────────────────────────────────────────────
    loop {
        tokio::select! {
            biased;

            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }

            // ── LLM streaming chunks → paste immediately ──────────────
            Some(chunk) = llm_stream_rx.recv() => {
                llm_partial_buf.push_str(&chunk.text);
                output_ctrl.emit_final(chunk.text).await;
            }

            // ── LLM complete → commit to transcript ──────────────────
            Some(_resp) = llm_rx.recv() => {
                if !llm_partial_buf.is_empty() {
                    if !session_text.is_empty() {
                        session_text.push(' ');
                    }
                    session_text.push_str(&llm_partial_buf);
                    llm_partial_buf.clear();
                }
            }

            recv = tokio::time::timeout(silence_timeout, audio_rx.recv()) => {
                match recv {
                    // ── Silence timeout → send utterance to LLM ──────────
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

                            let (after_dict, relevant) = {
                                let guard = dict.lock().await;
                                let applied = guard.apply(&utterance);
                                let rel = guard
                                    .top_relevant(&applied, 20)
                                    .into_iter()
                                    .cloned()
                                    .collect::<Vec<_>>();
                                (applied, rel)
                            };

                            let req = LlmRequest {
                                text: after_dict,
                                style_preset: Arc::clone(&style_preset),
                                dictionary: relevant,
                            };
                            if let Err(e) = llm_tx.send(req).await {
                                warn!(error = %e, "LLM worker unavailable");
                            }

                            utterance.clear();
                            if let Err(e) = asr.reset_context(&asr_session).await {
                                warn!(error = %e, "reset_context error");
                            }
                        }
                    }

                    Ok(None) => {
                        debug!(%session_id, "audio channel closed");
                        break;
                    }

                    Ok(Some(frame)) => {
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
                            } else {
                                let partial_suffix = route.remaining.trim();
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
                                output_ctrl.emit_partial(partial_buf.clone()).await;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Teardown ──────────────────────────────────────────────────────────

    if !pending_nonfinal.is_empty() {
        if !utterance.is_empty() {
            utterance.push(' ');
        }
        utterance.push_str(&pending_nonfinal);
    }

    // Send any remaining utterance to LLM.
    if !utterance.is_empty() {
        let (after_dict, relevant) = {
            let guard = dict.lock().await;
            let applied = guard.apply(&utterance);
            let rel = guard
                .top_relevant(&applied, 20)
                .into_iter()
                .cloned()
                .collect::<Vec<_>>();
            (applied, rel)
        };
        let req = LlmRequest {
            text: after_dict,
            style_preset: Arc::clone(&style_preset),
            dictionary: relevant,
        };
        if let Err(e) = llm_tx.send(req).await {
            warn!(error = %e, "LLM unavailable for final utterance");
        }
    }

    // Drain any chunks that arrived during teardown.
    while let Ok(chunk) = llm_stream_rx.try_recv() {
        llm_partial_buf.push_str(&chunk.text);
        output_ctrl.emit_final(chunk.text).await;
    }
    if !llm_partial_buf.is_empty() {
        if !session_text.is_empty() {
            session_text.push(' ');
        }
        session_text.push_str(&llm_partial_buf);
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

    let _ = event_tx.send(SessionEvent::Stopped).await;
    active_arc.lock().await.take();
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
        mgr.stop().await;
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

                tokio::time::sleep(Duration::from_millis(50)).await;

                while let Ok(ev) = rx.try_recv() {
                    let _ = ev;
                }
            })
            .await;
    }
}
