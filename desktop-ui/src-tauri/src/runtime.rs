use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::Instant;

use lune_app::calibration::{CalibrationFrame, CalibrationSession};
use lune_app::{DeliveryService, SessionManager, SessionManagerConfig};
#[cfg(unix)]
use lune_app::{WindowTarget, capture_focus_target};
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::asr::AsrSessionConfig;
use lune_core::command::VoiceCommand;
use lune_core::config::{ActivationMode, LlmBackendConfig, PillVisibility, UserSettings};
use lune_core::llm::LlmService;
use lune_core::session::SessionEvent;
use lune_dict::DictionaryManager;
use lune_dict::commands::VoiceCommandRouter;
use lune_llm::{NullPostProcessor, OpenAiLlmBackend, OpenAiLlmConfig};
use lune_persist::{AppPaths, Database, SettingsStore};
use tauri::{AppHandle, Emitter, ipc::Channel};
use tokio::sync::{Mutex, RwLock, mpsc};
#[cfg(unix)]
use tracing::info;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::dto::{CalibrationDiagnosticsDto, CalibrationStreamEvent, SessionStreamEvent};

const LOCAL_LLM_IDLE_TTL: Duration = Duration::from_secs(8);
const HOLD_WARMUP_GRACE: Duration = Duration::from_millis(250);
const TOGGLE_DEBOUNCE: Duration = Duration::from_millis(150);
const CANCEL_DEBOUNCE: Duration = Duration::from_millis(80);

pub struct AppRuntime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    app: AppHandle,
    pipeline: Mutex<LivePipeline>,
    status: RwLock<String>,
    session_stream: RwLock<Option<SessionStreamSlot>>,
    session_stream_seq: AtomicU64,
    /// Shared settings — updated on reload, read for dictation_mode at delivery time.
    settings: Arc<RwLock<UserSettings>>,
    /// Shared LLM service — used by dictation pipeline and on-demand tasks.
    llm_service: RwLock<Arc<dyn LlmService>>,
    /// Background thread that owns the uinput/clipboard handle.
    delivery: DeliveryService,
    /// Debounce guard: timestamp of last toggle/start/stop action.
    last_toggle_at: Mutex<Instant>,
    /// Separate debounce for cancel so it isn't blocked by a recent toggle.
    last_cancel_at: Mutex<Instant>,
    /// Active microphone-calibration session (mutually exclusive with dictation).
    calibration: Mutex<Option<CalibrationSession>>,
    /// Long-lived microphone capture stream when `mic_prewarm` is enabled.
    /// Sessions consume via `AudioStreamHandle` so the cpal::Stream stays
    /// open across hotkey presses and the first chunk arrives ~70 ms sooner.
    prewarmed_audio: Mutex<Option<AudioStream>>,
}

struct LivePipeline {
    paths: AppPaths,
    manager: Option<Arc<SessionManager<AsrRegistry>>>,
    local_llm: Option<Arc<LazyLocalLlm>>,
    active: bool,
    session_id: Option<String>,
    reload_required: bool,
    #[cfg(unix)]
    focus_target: Option<WindowTarget>,
}

struct SessionStreamSlot {
    id: u64,
    channel: Channel<SessionStreamEvent>,
}

impl AppRuntime {
    pub async fn new(app: AppHandle) -> Result<Self, String> {
        let paths = resolve_paths()?;
        let settings = load_settings(&paths)?;
        init_ort_runtime();

        let models_ready = lune_persist::has_required_models(
            &paths.models_dir(),
            &settings.asr_backend,
            &settings.llm_backend,
        );

        let (pipeline, llm_service, settings) = if models_ready {
            build_pipeline(paths.clone(), settings).await?
        } else {
            info!("models not yet downloaded — deferring pipeline build");
            let settings = Arc::new(RwLock::new(settings));
            let null_svc: Arc<dyn LlmService> = Arc::new(NullPostProcessor::new());
            let pipeline = LivePipeline {
                paths,
                manager: None,
                local_llm: None,
                active: false,
                session_id: None,
                reload_required: true,
                #[cfg(unix)]
                focus_target: None,
            };
            (pipeline, null_svc, settings)
        };

        let terminal_mode = settings.read().await.terminal_mode;
        let delivery = DeliveryService::new(terminal_mode)?;

        let runtime = Self {
            inner: Arc::new(RuntimeInner {
                app,
                pipeline: Mutex::new(pipeline),
                status: RwLock::new("ready".into()),
                session_stream: RwLock::new(None),
                session_stream_seq: AtomicU64::new(0),
                settings: Arc::clone(&settings),
                llm_service: RwLock::new(llm_service),
                delivery,
                last_toggle_at: Mutex::new(Instant::now() - TOGGLE_DEBOUNCE),
                last_cancel_at: Mutex::new(Instant::now() - CANCEL_DEBOUNCE),
                calibration: Mutex::new(None),
                prewarmed_audio: Mutex::new(None),
            }),
        };

        runtime.sync_mic_prewarm(None).await;
        Ok(runtime)
    }

    /// Reconcile the long-lived audio stream with the desired `mic_prewarm`
    /// state. Pass `None` to read the current in-memory setting (suitable for
    /// startup); pass `Some(value)` after a settings save when in-memory
    /// settings may be stale.
    pub async fn sync_mic_prewarm(&self, want: Option<bool>) {
        let want = match want {
            Some(w) => w,
            None => self.inner.settings.read().await.mic_prewarm,
        };
        let mut slot = self.inner.prewarmed_audio.lock().await;
        match (want, slot.is_some()) {
            (true, false) => match AudioStream::new(AudioConfig::default()) {
                Ok(s) => {
                    debug!("mic prewarm: capture stream opened");
                    *slot = Some(s);
                }
                Err(e) => warn!("mic prewarm: failed to open capture stream: {e}"),
            },
            (false, true) => {
                debug!("mic prewarm: closing capture stream");
                *slot = None;
            }
            _ => {}
        }
    }

    async fn prewarmed_handle(&self) -> Option<lune_audio::AudioStreamHandle> {
        let slot = self.inner.prewarmed_audio.lock().await;
        slot.as_ref().map(|s| s.handle())
    }

    /// Return the persisted `start_minimized` preference. Read at app
    /// startup to decide whether to hide the main window on launch.
    pub async fn start_minimized(&self) -> bool {
        self.inner.settings.read().await.start_minimized
    }

    pub async fn service_status(&self) -> String {
        self.inner.status.read().await.clone()
    }

    pub fn app_handle(&self) -> &AppHandle {
        &self.inner.app
    }

    pub async fn llm_service(&self) -> Arc<dyn LlmService> {
        Arc::clone(&*self.inner.llm_service.read().await)
    }

    pub async fn set_session_stream(
        &self,
        channel: Channel<SessionStreamEvent>,
    ) -> Result<u64, String> {
        let id = self
            .inner
            .session_stream_seq
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        {
            let mut slot = self.inner.session_stream.write().await;
            *slot = Some(SessionStreamSlot { id, channel });
        }

        self.push_session_event(SessionStreamEvent::ServiceStatus {
            status: self.service_status().await,
        })
        .await;

        let active_session_id = {
            let pipeline = self.inner.pipeline.lock().await;
            pipeline.session_id.clone()
        };

        if let Some(session_id) = active_session_id {
            self.push_session_event(SessionStreamEvent::SessionStarted { session_id })
                .await;
        }

        Ok(id)
    }

    pub async fn clear_session_stream(&self, subscription_id: u64) {
        let mut slot = self.inner.session_stream.write().await;
        if slot
            .as_ref()
            .is_some_and(|current| current.id == subscription_id)
        {
            *slot = None;
        }
    }

    pub async fn calibrate_start(
        &self,
        channel: Channel<CalibrationStreamEvent>,
    ) -> Result<(), String> {
        {
            let pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return Err("cannot calibrate while a dictation session is active".into());
            }
        }
        {
            let mut slot = self.inner.calibration.lock().await;
            if slot.is_some() {
                return Err("a calibration session is already running".into());
            }
            let (tx, rx) = mpsc::channel::<CalibrationFrame>(64);
            let session = CalibrationSession::start(tx);
            *slot = Some(session);
            tokio::spawn(forward_calibration(rx, channel));
        }
        Ok(())
    }

    pub async fn calibrate_advance(&self) -> Result<(), String> {
        let slot = self.inner.calibration.lock().await;
        let session = slot
            .as_ref()
            .ok_or_else(|| "no active calibration session".to_string())?;
        session.advance_to_speech();
        Ok(())
    }

    pub async fn calibrate_cancel(&self) {
        // Drop notifies the cancel signal and aborts the spawned task.
        let _ = self.inner.calibration.lock().await.take();
    }

    pub async fn mark_reload_required(&self) {
        {
            let mut pipeline = self.inner.pipeline.lock().await;
            pipeline.reload_required = true;
        }
    }

    pub async fn eager_rebuild(&self) -> Result<(), String> {
        let paths = {
            let pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return Ok(());
            }
            pipeline.paths.clone()
        };
        let settings = load_settings(&paths)?;
        let (rebuilt, new_llm_service, new_settings) = build_pipeline(paths, settings)
            .await
            .map_err(|e| format!("eager rebuild: {e}"))?;
        let mut pipeline = self.inner.pipeline.lock().await;
        if !pipeline.active {
            *pipeline = rebuilt;
            *self.inner.llm_service.write().await = new_llm_service;
            let new_settings_val = new_settings.read().await.clone();
            *self.inner.settings.write().await = new_settings_val;
            tracing::info!("pipeline pre-warmed");
        }
        Ok(())
    }

    pub async fn start_dictation(&self) -> Result<String, String> {
        if !self.inner.delivery.is_alive() {
            return Err("text delivery service is unavailable — cannot start dictation".into());
        }

        // Our own typed/pasted output may still be in flight; starting now
        // would let those synthetic keystrokes (or a focused UI button they
        // re-activate) toggle sessions in a feedback loop.
        if self.inner.delivery.is_busy() {
            debug!("start_dictation suppressed — text delivery in progress");
            return Err("text delivery in progress".into());
        }

        if self.inner.session_stream.read().await.is_none() {
            warn!(
                "starting dictation with no IPC stream subscriber — UI will not reflect session state"
            );
        }

        // Optimistic UI: tell the pill we're listening before doing session work.
        // Reverted on failure so the pill doesn't stick in "listening" state.
        self.publish_status("listening").await;

        // Capture focus off the start path: the value is only consumed at
        // delivery time (many seconds later), but the underlying Wayland /
        // subprocess probes are synchronous and block the tokio worker for
        // ~400–500 ms on COSMIC. Letting them race the audio bring-up
        // unblocks the first VAD chunk by the same amount.
        #[cfg(unix)]
        let focus_capture = tokio::task::spawn_blocking(capture_focus_target);

        let manager = match self.prepare_manager_for_start().await {
            Ok(m) => m,
            Err(e) => {
                self.publish_status("ready").await;
                return Err(e);
            }
        };
        let (event_tx, event_rx) = mpsc::channel(256);
        let prewarm = self.prewarmed_handle().await;
        let start_result = match prewarm {
            Some(handle) => manager.start_with_handle(event_tx, handle).await,
            None => manager.start(event_tx).await,
        };
        let session_id = match start_result {
            Ok(session_id) => session_id,
            Err(error) => {
                self.set_pipeline_session(None, false).await;
                self.publish_status("error").await;
                return Err(format!("start dictation: {error}"));
            }
        };

        let session_id = session_id.to_string();
        self.set_pipeline_session(Some(session_id.clone()), true)
            .await;
        self.schedule_local_llm_warmup().await;

        self.push_session_event(SessionStreamEvent::SessionStarted {
            session_id: session_id.clone(),
        })
        .await;
        #[cfg(unix)]
        {
            let focus_target = focus_capture.await.ok();
            debug!(?focus_target, "captured focus target for session");
            self.inner.pipeline.lock().await.focus_target = focus_target;
        }

        self.spawn_event_forwarder(event_rx);

        Ok(session_id)
    }

    pub async fn stop_dictation(&self) -> Result<(), String> {
        if let Some(manager) = self.active_manager().await {
            manager.stop().await;
        }
        Ok(())
    }

    pub async fn stop_and_wait_dictation(&self) -> Result<(), String> {
        if let Some(manager) = self.active_manager().await {
            manager.stop_and_wait().await;
        }
        Ok(())
    }

    pub async fn is_active(&self) -> bool {
        self.inner.pipeline.lock().await.active
    }

    pub async fn cancel_dictation(&self) -> Result<(), String> {
        {
            let mut last = self.inner.last_cancel_at.lock().await;
            if last.elapsed() < CANCEL_DEBOUNCE {
                debug!("cancel_dictation debounced");
                return Ok(());
            }
            *last = Instant::now();
        }
        if let Some(manager) = self.active_manager().await {
            manager.cancel().await;
        }
        Ok(())
    }

    pub async fn toggle_dictation(&self) -> Result<Option<String>, String> {
        {
            let mut last = self.inner.last_toggle_at.lock().await;
            if last.elapsed() < TOGGLE_DEBOUNCE {
                debug!("toggle_dictation debounced");
                return Ok(None);
            }
            *last = Instant::now();
        }

        let is_active = self.inner.pipeline.lock().await.active;

        if is_active {
            self.stop_dictation().await?;
            Ok(None)
        } else {
            if self.inner.delivery.is_busy() {
                debug!("toggle_dictation suppressed — text delivery in progress");
                return Ok(None);
            }
            self.start_dictation().await.map(Some)
        }
    }

    fn spawn_event_forwarder(&self, mut event_rx: mpsc::Receiver<SessionEvent>) {
        let app = self.inner.app.clone();
        let runtime = self.clone();

        tauri::async_runtime::spawn(async move {
            let mut saw_error = false;
            let mut saw_text = false;
            while let Some(event) = event_rx.recv().await {
                match event {
                    SessionEvent::Started => {
                        runtime.publish_status("listening").await;
                    }
                    SessionEvent::PartialText => {
                        runtime
                            .push_session_event(SessionStreamEvent::PartialText)
                            .await;
                    }
                    SessionEvent::Processing => {
                        runtime.publish_status("processing").await;
                        runtime
                            .push_session_event(SessionStreamEvent::Processing)
                            .await;
                    }
                    SessionEvent::FinalizedText { session_id, text } => {
                        saw_text = true;
                        let mode = runtime.inner.settings.read().await.dictation_mode;
                        let delivered;
                        #[cfg(unix)]
                        {
                            let target = runtime.inner.pipeline.lock().await.focus_target.clone();
                            delivered = if let Some(ref t) = target {
                                runtime
                                    .inner
                                    .delivery
                                    .deliver_to(Some(session_id), &text, mode, t)
                            } else {
                                runtime
                                    .inner
                                    .delivery
                                    .deliver(Some(session_id), &text, mode)
                            };
                        }
                        #[cfg(not(unix))]
                        {
                            delivered =
                                runtime
                                    .inner
                                    .delivery
                                    .deliver(Some(session_id), &text, mode);
                        }
                        if !delivered {
                            // Non-fatal: surface as a warning toast, not a
                            // SessionError. SessionError flips serviceStatus to
                            // 'error' and is then immediately overwritten by the
                            // FinalizedText below (serviceStatus 'ready'), which
                            // made the status badge flash red for one frame.
                            runtime
                                .push_session_event(SessionStreamEvent::SessionWarning {
                                    message: "Text delivery failed — output service is unavailable"
                                        .into(),
                                })
                                .await;
                        }
                        runtime
                            .push_session_event(SessionStreamEvent::FinalizedText { text })
                            .await;
                    }
                    SessionEvent::CorrectedText(text) => {
                        // Corrected text is a UI signal; the finalized text event
                        // remains the authoritative delivery payload.
                        runtime
                            .push_session_event(SessionStreamEvent::CorrectedText { text })
                            .await;
                    }
                    SessionEvent::AudioLevel(level) => {
                        // Send to main window via session stream
                        runtime
                            .push_session_event(SessionStreamEvent::AudioLevel { level })
                            .await;
                        // Send to pill overlay via targeted event (avoids needing
                        // a second session stream subscription that would overwrite
                        // the main window's channel)
                        let _ = app.emit_to("pill", "lune://audio-level", level);
                    }
                    SessionEvent::CommandDetected {
                        session_id,
                        payload,
                    } => {
                        runtime.handle_voice_command(session_id, &payload).await;
                        runtime
                            .push_session_event(SessionStreamEvent::CommandDetected { payload })
                            .await;
                    }
                    SessionEvent::Error(message) => {
                        saw_error = true;
                        // Log full error internally; send sanitized version to frontend.
                        warn!(error = %message, "session error");
                        runtime.finish_session(false).await;
                        runtime.publish_status("error").await;
                        runtime
                            .push_session_event(SessionStreamEvent::SessionError {
                                message: sanitize_error_for_frontend(&message),
                            })
                            .await;
                    }
                    SessionEvent::Warning(message) => {
                        warn!(warning = %message, "session warning");
                        runtime
                            .push_session_event(SessionStreamEvent::SessionWarning {
                                message: sanitize_error_for_frontend(&message),
                            })
                            .await;
                    }
                    SessionEvent::Stopped => {
                        runtime.finish_session(saw_text).await;
                        if !saw_error {
                            runtime.publish_status("ready").await;
                        }
                        runtime
                            .push_session_event(SessionStreamEvent::SessionStopped)
                            .await;
                        // Only reload history if the session produced text —
                        // empty/cancelled sessions shouldn't trigger UI churn
                        if saw_text && let Err(e) = app.emit("lune://history-changed", ()) {
                            warn!(error = %e, "failed to emit history-changed event");
                        }
                    }
                    _ => {}
                }
            }
        });
    }

    /// Dispatch a `VoiceCommand` payload string received from the pipeline.
    ///
    /// - `NewLine` / `NewParagraph` / `ListItem` → deliver the corresponding
    ///   whitespace/punctuation text.
    /// - `StopDictation` → stop the active session.
    /// - Other commands (e.g. `SwitchStyle`, `InsertSnippet`, `ScratchThat`) →
    ///   logged and skipped; higher-level handling belongs in the frontend or a
    ///   future command handler.
    async fn handle_voice_command(&self, session_id: Uuid, payload: &str) {
        let cmd: VoiceCommand = match serde_json::from_str(payload) {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, payload, "CommandDetected: failed to parse VoiceCommand JSON");
                return;
            }
        };

        let mode = self.inner.settings.read().await.dictation_mode;

        #[cfg(unix)]
        let focus_target = self.inner.pipeline.lock().await.focus_target.clone();

        match cmd {
            VoiceCommand::NewLine => {
                debug!("CommandDetected: NewLine → delivering \\n");
                #[cfg(unix)]
                if let Some(ref t) = focus_target {
                    self.inner
                        .delivery
                        .deliver_to(Some(session_id), "\n", mode, t);
                } else {
                    self.inner.delivery.deliver(Some(session_id), "\n", mode);
                }
                #[cfg(not(unix))]
                self.inner.delivery.deliver(Some(session_id), "\n", mode);
            }
            VoiceCommand::NewParagraph => {
                debug!("CommandDetected: NewParagraph → delivering \\n\\n");
                #[cfg(unix)]
                if let Some(ref t) = focus_target {
                    self.inner
                        .delivery
                        .deliver_to(Some(session_id), "\n\n", mode, t);
                } else {
                    self.inner.delivery.deliver(Some(session_id), "\n\n", mode);
                }
                #[cfg(not(unix))]
                self.inner.delivery.deliver(Some(session_id), "\n\n", mode);
            }
            VoiceCommand::ListItem => {
                debug!("CommandDetected: ListItem → delivering \\n- ");
                #[cfg(unix)]
                if let Some(ref t) = focus_target {
                    self.inner
                        .delivery
                        .deliver_to(Some(session_id), "\n- ", mode, t);
                } else {
                    self.inner.delivery.deliver(Some(session_id), "\n- ", mode);
                }
                #[cfg(not(unix))]
                self.inner.delivery.deliver(Some(session_id), "\n- ", mode);
            }
            VoiceCommand::StopDictation => {
                debug!("CommandDetected: StopDictation → stopping session");
                if let Some(manager) = self.active_manager().await {
                    manager.stop().await;
                }
            }
            other => {
                debug!(cmd = ?other, "CommandDetected: no delivery action for this command");
            }
        }
    }

    async fn finish_session(&self, produced_output: bool) {
        // Free the active flag and notify the UI BEFORE doing any LLM
        // bookkeeping. A second toggle press during the brief window where
        // the LLM is being unloaded would otherwise see active=true and
        // try to stop an already-stopping session (no-op), forcing the
        // user to press a third time.
        self.set_pipeline_session(None, false).await;
        if produced_output {
            self.schedule_local_llm_unload().await;
        } else {
            // Run eviction in the background so finish_session returns
            // immediately to the event-forwarder loop.
            let runtime = self.clone();
            tauri::async_runtime::spawn(async move {
                runtime.evict_local_llm_now().await;
            });
        }
    }

    async fn evict_local_llm_now(&self) {
        let local_llm = {
            let pipeline = self.inner.pipeline.lock().await;
            pipeline.local_llm.clone()
        };
        if let Some(local_llm) = local_llm {
            tracing::info!(target: "vram", "session produced no output — evicting LLM immediately");
            local_llm.unload().await;
        }
    }

    async fn publish_status(&self, status: &str) {
        *self.inner.status.write().await = status.to_owned();
        self.push_session_event(SessionStreamEvent::ServiceStatus {
            status: status.to_owned(),
        })
        .await;
        // Notify the pill overlay window of the status change.
        if let Err(e) = self
            .inner
            .app
            .emit_to("pill", "lune://pill-state", status.to_owned())
        {
            tracing::debug!(error = %e, "pill state emit skipped (pill window may not exist)");
        }
        // Direct pill show/hide bypasses the frontend IPC roundtrip so the
        // window reacts in the same main-thread tick as the status emission.
        let vis = self.inner.settings.read().await.pill_visibility;
        match status {
            "listening" | "processing" if vis != PillVisibility::Never => {
                crate::pill::show_pill_direct(&self.inner.app);
            }
            "ready" | "error" if vis == PillVisibility::WhenDictating => {
                crate::pill::hide_pill_direct(&self.inner.app);
            }
            _ => {}
        }
    }

    async fn push_session_event(&self, event: SessionStreamEvent) {
        let failed = {
            let slot = self.inner.session_stream.read().await;
            match slot.as_ref() {
                Some(slot) => {
                    if let Err(e) = slot.channel.send(event) {
                        warn!(
                            error = %e,
                            "session stream channel send failed — frontend may have disconnected"
                        );
                        Some(slot.id)
                    } else {
                        None
                    }
                }
                None => None,
            }
        };

        if let Some(id) = failed {
            let mut slot = self.inner.session_stream.write().await;
            if slot.as_ref().is_some_and(|current| current.id == id) {
                *slot = None;
            }
        }
    }

    async fn active_manager(&self) -> Option<Arc<SessionManager<AsrRegistry>>> {
        let pipeline = self.inner.pipeline.lock().await;
        if pipeline.active {
            pipeline.manager.as_ref().map(Arc::clone)
        } else {
            None
        }
    }

    async fn prepare_manager_for_start(&self) -> Result<Arc<SessionManager<AsrRegistry>>, String> {
        // Phase 1: Check if reload needed (short lock).
        let reload_paths = {
            let pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return Err("dictation session already active".into());
            }
            // Rebuild when settings changed OR pipeline was never built (models were missing at boot)
            if pipeline.reload_required || pipeline.manager.is_none() {
                Some(pipeline.paths.clone())
            } else {
                None
            }
        };

        // Phase 2: If reload needed, rebuild pipeline outside lock, then
        // re-acquire and atomically swap + activate in one critical section.
        if let Some(paths) = reload_paths {
            let settings = load_settings(&paths)?;
            let (rebuilt, new_llm_service, new_settings) = build_pipeline(paths, settings)
                .await
                .map_err(|e| format!("Pipeline build failed: {e}"))?;
            let mut pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return Err("dictation session became active during reload".into());
            }
            *pipeline = rebuilt;
            pipeline.active = true;
            *self.inner.llm_service.write().await = new_llm_service;
            let new_settings_val = new_settings.read().await.clone();
            let mut settings = self.inner.settings.write().await;
            *settings = new_settings_val;
            let mgr = pipeline
                .manager
                .as_ref()
                .ok_or("build_pipeline succeeded but manager is None")?;
            return Ok(Arc::clone(mgr));
        }

        // Phase 3: No reload — just activate.
        let mut pipeline = self.inner.pipeline.lock().await;
        if pipeline.active {
            return Err("dictation session already active".into());
        }
        if pipeline.manager.is_none() {
            return Err("Models not downloaded. Please complete model setup first.".into());
        }
        pipeline.active = true;
        let mgr = pipeline
            .manager
            .as_ref()
            .ok_or("pipeline manager is None")?;
        Ok(Arc::clone(mgr))
    }

    async fn set_pipeline_session(&self, session_id: Option<String>, active: bool) {
        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.session_id = session_id;
        pipeline.active = active;
    }

    async fn schedule_local_llm_warmup(&self) {
        let (local_llm, activation_mode) = {
            let pipeline = self.inner.pipeline.lock().await;
            let settings = self.inner.settings.read().await;
            (pipeline.local_llm.clone(), settings.activation_mode)
        };
        let Some(local_llm) = local_llm else {
            return;
        };
        local_llm.cancel_scheduled_unload();

        let runtime = self.clone();
        tauri::async_runtime::spawn(async move {
            if activation_mode == ActivationMode::Hold {
                // Avoid loading a multi-GB local LLM for accidental key taps.
                tokio::time::sleep(HOLD_WARMUP_GRACE).await;
                let active = runtime.inner.pipeline.lock().await.active;
                if !active {
                    return;
                }
            }
            local_llm.warm_up().await;
        });
    }

    async fn schedule_local_llm_unload(&self) {
        let local_llm = {
            let pipeline = self.inner.pipeline.lock().await;
            pipeline.local_llm.clone()
        };
        if let Some(local_llm) = local_llm {
            local_llm.schedule_unload_after(LOCAL_LLM_IDLE_TTL).await;
        }
    }
}

impl Clone for AppRuntime {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

async fn build_pipeline(
    paths: AppPaths,
    settings: UserSettings,
) -> Result<(LivePipeline, Arc<dyn LlmService>, Arc<RwLock<UserSettings>>), String> {
    let asr = AsrRegistry::from_config(&settings.asr_backend, settings.device, &paths.models_dir())
        .map_err(|e| format!("build asr backend: {e}"))?;
    let (llm_service, local_llm) = build_llm_service(&settings, &paths.models_dir()).await?;
    let post_processor = Arc::new(lune_llm::CorrectionPipeline::new(Arc::clone(&llm_service)));
    let router = Arc::new(VoiceCommandRouter::new());
    let dict = Arc::new(Mutex::new(
        DictionaryManager::new(
            Database::open(&paths.database_file()).map_err(|e| format!("open dict db: {e}"))?,
        )
        .map_err(|e| format!("init dictionary manager: {e}"))?,
    ));
    let db = Arc::new(Mutex::new(
        Database::open(&paths.database_file()).map_err(|e| format!("open session db: {e}"))?,
    ));
    // Wire dictionary replacement strings into the ASR decoder as boost phrases.
    {
        let pipeline = asr.pipeline_config().with_overrides(&settings.pipeline);
        if pipeline.word_boost_enabled {
            let phrases: Vec<String> = dict
                .lock()
                .await
                .entries()
                .iter()
                .map(|e| e.replacement.clone())
                .collect();
            asr.set_boost_phrases(&phrases, pipeline.word_boost_score);
        }
    }

    let settings = Arc::new(RwLock::new(settings.clone()));

    let config = SessionManagerConfig {
        asr_backend_kind: asr.kind(),
        asr_config: AsrSessionConfig {
            // dictation_language is the user-facing ASR language; "auto"
            // means let the model detect. UI `language` is unrelated.
            language: {
                let lang = settings.read().await.dictation_language.clone();
                if lang == "auto" || lang.is_empty() {
                    None
                } else {
                    Some(lang)
                }
            },
            ..Default::default()
        },
        pipeline: asr
            .pipeline_config()
            .with_overrides(&settings.read().await.pipeline),
    };

    let heuristic = {
        let mut ngram = lune_dict::ngram::NGramModel::load_bundled().unwrap_or_else(|e| {
            tracing::warn!("Failed to load bundled ngram model: {e}, using empty");
            lune_dict::ngram::NGramModel::new(lune_dict::ngram::ScoringConfig::trigram_default())
                .unwrap()
        });
        if let Ok(sessions) = db.lock().await.list_sessions(500) {
            let transcripts: Vec<&str> = sessions
                .iter()
                .filter(|s| matches!(s.status, lune_core::session::SessionStatus::Completed))
                .filter_map(|s| s.transcript.as_deref())
                .collect();
            ngram.train_sentences(transcripts);
        }
        // Self-evaluating homophone calibration: replay the resolver over the
        // user's own (raw → final) history and conformally bound over-correction
        // per confusion set (Angelopoulos et al., arXiv:2208.02814). Logged now;
        // consumed once the resolver is wired into the correction path.
        match db.lock().await.list_correction_pairs(1000) {
            Ok(pairs) if !pairs.is_empty() => {
                let cal = lune_dict::homophone_calib::calibrate(
                    &ngram,
                    &pairs,
                    lune_dict::homophone_calib::DEFAULT_ALPHA,
                    1,
                );
                tracing::info!(
                    "homophone calibration ({} pairs):\n{}",
                    pairs.len(),
                    lune_dict::homophone_calib::summarize(&cal),
                );
            }
            Ok(_) => tracing::debug!("homophone calibration: no correction pairs yet"),
            Err(e) => tracing::warn!(error = %e, "homophone calibration: could not load pairs"),
        }
        // The trained ngram above still feeds homophone calibration; the
        // lexical gate no longer holds it.
        Arc::new(lune_dict::corrector::LexicalGate::new())
    };

    let manager = SessionManager::new(
        Arc::new(asr),
        router,
        post_processor,
        dict,
        db,
        heuristic,
        config,
        Arc::clone(&settings),
    );

    let pipeline = LivePipeline {
        paths,
        manager: Some(Arc::new(manager)),
        local_llm,
        active: false,
        session_id: None,
        reload_required: false,
        #[cfg(unix)]
        focus_target: None,
    };

    Ok((pipeline, llm_service, settings))
}

async fn forward_calibration(
    mut rx: mpsc::Receiver<CalibrationFrame>,
    channel: Channel<CalibrationStreamEvent>,
) {
    while let Some(frame) = rx.recv().await {
        let event = match frame {
            CalibrationFrame::PhaseChanged(phase) => CalibrationStreamEvent::PhaseChanged {
                phase: phase.as_str().to_owned(),
            },
            CalibrationFrame::Frame {
                phase,
                vad_score,
                energy,
            } => CalibrationStreamEvent::Frame {
                phase: phase.as_str().to_owned(),
                vad_score,
                energy,
            },
            CalibrationFrame::Result(r) => CalibrationStreamEvent::Result {
                speech_threshold: r.speech_threshold,
                speech_threshold_end: r.speech_threshold_end,
                diagnostics: CalibrationDiagnosticsDto {
                    noise_p95: r.noise_p95,
                    speech_p10: r.speech_p10,
                    pause_p20: r.pause_p20,
                },
            },
            CalibrationFrame::Error(message) => CalibrationStreamEvent::Error { message },
        };
        if let Err(e) = channel.send(event) {
            warn!(error = %e, "calibration channel send failed; dropping subscription");
            break;
        }
    }
}

fn resolve_paths() -> Result<AppPaths, String> {
    let paths = AppPaths::resolve().ok_or("failed to resolve app paths")?;
    paths
        .ensure_dirs()
        .map_err(|e| format!("ensure app dirs: {e}"))?;
    Ok(paths)
}

fn load_settings(paths: &AppPaths) -> Result<UserSettings, String> {
    let mut settings = SettingsStore::new(paths.settings_file())
        .load()
        .map_err(|e| format!("load settings: {e}"))?;
    settings.migrate_pipeline_fields();
    Ok(settings)
}

fn init_ort_runtime() {
    // CPU-only global init. CUDA EP is registered per-session by
    // build_ort_session in lune-asr — keeps VAD and other lightweight
    // ORT consumers on CPU automatically.
    let _ = ort::init().with_logger(quiet_ort_logger()).commit();
}

fn quiet_ort_logger() -> ort::logging::LoggerFunction {
    Arc::new(|level, _category, id, location, message| {
        if level >= ort::logging::LogLevel::Error {
            tracing::error!(
                target: "ort::logging",
                ort_id = %id,
                ort_location = %location,
                "{message}"
            );
        }
    })
}

async fn build_llm_service(
    settings: &UserSettings,
    models_dir: &std::path::Path,
) -> Result<(Arc<dyn LlmService>, Option<Arc<LazyLocalLlm>>), String> {
    match &settings.llm_backend {
        LlmBackendConfig::Local { .. } => {
            let style_dir = models_dir.join(lune_core::models::DIR_STYLE_LLM);
            if style_dir.join("tokenizer.json").exists() {
                // Decode budget 512: the corrector must re-emit the whole
                // transcript, and 128 tokens (~90 words) truncated every
                // longer dictation's correction — the validator then threw
                // the truncated output away. Greedy decode stops at EOS, so
                // short utterances never pay for the larger cap. Matches the
                // ASR decoder's own 512-token budget.
                let llm = Arc::new(LazyLocalLlm::new(
                    style_dir,
                    settings.device.use_gpu(cfg!(feature = "cuda")),
                    settings.device.strict_cuda(),
                    512,
                ));
                Ok((llm.clone() as Arc<dyn LlmService>, Some(llm)))
            } else {
                tracing::info!("style LLM not downloaded yet, using passthrough");
                Ok((
                    Arc::new(NullPostProcessor::new()) as Arc<dyn LlmService>,
                    None,
                ))
            }
        }
        LlmBackendConfig::OpenAiCompatible {
            base_url,
            api_key,
            model,
            max_tokens,
        } => {
            let backend = Arc::new(OpenAiLlmBackend::new(OpenAiLlmConfig {
                base_url: base_url.to_string(),
                api_key: api_key.expose().to_owned(),
                model: model.clone(),
                max_tokens: *max_tokens,
                temperature: 0.2,
            }));
            Ok((backend as Arc<dyn LlmService>, None))
        }
        _ => Ok((
            Arc::new(NullPostProcessor::new()) as Arc<dyn LlmService>,
            None,
        )),
    }
}

struct LazyLocalLlm {
    model_dir: std::path::PathBuf,
    use_gpu: bool,
    cuda_required: bool,
    max_new_tokens: usize,
    generation: AtomicU64,
    unload_epoch: AtomicU64,
    state: Mutex<LazyLlmState>,
}

enum LazyLlmState {
    Empty,
    Loading(u64),
    Ready(Arc<lune_llm::backend::ort_llm::OrtStyleLlm>),
}

impl LazyLocalLlm {
    fn new(
        model_dir: std::path::PathBuf,
        use_gpu: bool,
        cuda_required: bool,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            model_dir,
            use_gpu,
            cuda_required,
            max_new_tokens,
            generation: AtomicU64::new(0),
            unload_epoch: AtomicU64::new(0),
            state: Mutex::new(LazyLlmState::Empty),
        }
    }

    async fn warm_up(&self) {
        let generation = self.generation.load(Ordering::SeqCst);
        {
            let mut state = self.state.lock().await;
            match *state {
                LazyLlmState::Ready(_) | LazyLlmState::Loading(_) => return,
                LazyLlmState::Empty => *state = LazyLlmState::Loading(generation),
            }
        }

        let warmup_t0 = std::time::Instant::now();
        let model_dir = self.model_dir.clone();
        let use_gpu = self.use_gpu;
        let cuda_required = self.cuda_required;
        let max_new_tokens = self.max_new_tokens;
        let loaded = tokio::task::spawn_blocking(move || {
            lune_llm::backend::ort_llm::OrtStyleLlm::new(
                lune_llm::backend::ort_llm::OrtStyleLlmConfig {
                    model_dir,
                    use_gpu,
                    cuda_required,
                    max_new_tokens,
                },
            )
            .map(Arc::new)
            .map_err(|e| e.to_string())
        })
        .await;

        let mut state = self.state.lock().await;
        let still_current = self.generation.load(Ordering::SeqCst) == generation
            && matches!(*state, LazyLlmState::Loading(g) if g == generation);
        if !still_current {
            *state = LazyLlmState::Empty;
            return;
        }

        match loaded {
            Ok(Ok(llm)) => {
                let weights_mib = std::fs::read_dir(&self.model_dir)
                    .ok()
                    .map(|rd| {
                        rd.flatten()
                            .filter(|e| {
                                e.file_name().to_str().is_some_and(|n| {
                                    n.ends_with(".onnx")
                                        || n.ends_with(".onnx_data")
                                        || n.ends_with(".opt")
                                })
                            })
                            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                            .sum::<u64>() as f64
                            / (1024.0 * 1024.0)
                    })
                    .unwrap_or(0.0);
                tracing::info!(
                    target: "vram",
                    device = if self.use_gpu { "cuda" } else { "cpu" },
                    weights_mib = format_args!("{weights_mib:.1}"),
                    warmup_ms = warmup_t0.elapsed().as_millis() as u64,
                    "LLM session loaded"
                );
                *state = LazyLlmState::Ready(llm);
            }
            Ok(Err(error)) => {
                tracing::warn!(%error, "local style LLM warm-up failed, using passthrough");
                *state = LazyLlmState::Empty;
            }
            Err(error) => {
                tracing::warn!(%error, "local style LLM warm-up task failed, using passthrough");
                *state = LazyLlmState::Empty;
            }
        }
    }

    async fn unload(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        let mut state = self.state.lock().await;
        if matches!(&*state, LazyLlmState::Ready(_) | LazyLlmState::Loading(_)) {
            tracing::info!("local style LLM unloaded");
        }
        *state = LazyLlmState::Empty;
    }

    fn cancel_scheduled_unload(&self) {
        self.unload_epoch.fetch_add(1, Ordering::SeqCst);
    }

    async fn schedule_unload_after(self: Arc<Self>, ttl: Duration) {
        let should_schedule = {
            let state = self.state.lock().await;
            matches!(&*state, LazyLlmState::Ready(_) | LazyLlmState::Loading(_))
        };
        if !should_schedule {
            return;
        }

        let epoch = self.unload_epoch.fetch_add(1, Ordering::SeqCst) + 1;
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(ttl).await;
            if self.unload_epoch.load(Ordering::SeqCst) == epoch {
                self.unload().await;
            }
        });
    }
}

#[async_trait::async_trait]
impl LlmService for LazyLocalLlm {
    async fn complete(&self, system: &str, user: &str) -> lune_core::Result<String> {
        let llm = {
            let state = self.state.lock().await;
            match &*state {
                LazyLlmState::Ready(llm) => Some(Arc::clone(llm)),
                LazyLlmState::Empty | LazyLlmState::Loading(_) => None,
            }
        };
        match llm {
            Some(llm) => llm.complete(system, user).await,
            None => Err(lune_core::Error::llm("local LLM not ready")),
        }
    }
}

/// Strip file paths and internal details from error messages before sending to
/// the frontend.  Keeps the first clause (the human-readable part) and drops
/// anything after a colon that looks like a path or stack context.
fn sanitize_error_for_frontend(raw: &str) -> String {
    // Truncate very long errors.
    let capped = if raw.chars().count() > 512 {
        raw.chars().take(512).collect::<String>()
    } else {
        raw.to_owned()
    };

    // Strip absolute file paths (e.g. "/home/user/.local/share/…").
    static PATH_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(/[a-zA-Z0-9._\-]+){3,}").expect("path regex")
    });
    PATH_RE.replace_all(&capped, "[path]").into_owned()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use lune_core::llm::LlmService;

    use super::{LazyLocalLlm, sanitize_error_for_frontend};

    #[test]
    fn sanitize_error_handles_non_ascii_without_panicking() {
        let raw = format!("é{}", "/tmp/example/path/segment".repeat(80));
        let sanitized = sanitize_error_for_frontend(&raw);
        assert!(!sanitized.is_empty());
    }

    #[tokio::test]
    async fn lazy_local_llm_errors_when_not_warmed() {
        let llm = LazyLocalLlm::new(std::path::PathBuf::from("/missing/model"), false, false, 8);
        let result = llm.complete("system", "user").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn lazy_local_llm_ttl_unloads_in_flight_warmup() {
        let llm = Arc::new(LazyLocalLlm::new(
            std::path::PathBuf::from("/missing/model"),
            false,
            false,
            8,
        ));
        *llm.state.lock().await = super::LazyLlmState::Loading(0);

        Arc::clone(&llm)
            .schedule_unload_after(Duration::from_millis(1))
            .await;
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert!(matches!(
            *llm.state.lock().await,
            super::LazyLlmState::Empty
        ));
    }

    #[tokio::test]
    async fn lazy_local_llm_ttl_can_be_cancelled() {
        let llm = Arc::new(LazyLocalLlm::new(
            std::path::PathBuf::from("/missing/model"),
            false,
            false,
            8,
        ));
        *llm.state.lock().await = super::LazyLlmState::Loading(0);

        Arc::clone(&llm)
            .schedule_unload_after(Duration::from_millis(1))
            .await;
        llm.cancel_scheduled_unload();
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert!(matches!(
            *llm.state.lock().await,
            super::LazyLlmState::Loading(_)
        ));
    }
}
