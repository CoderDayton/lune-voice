use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lune_app::{AppCommand, DeliveryService, SessionManager, SessionManagerConfig};
use lune_asr::{AsrRegistry, OrtForcedAligner};
use lune_commands::VoiceCommandRouter;
use lune_core::asr::AsrSessionConfig;
use lune_core::command::VoiceCommand;
use lune_core::config::{LlmBackendConfig, UserSettings};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_dict::DictionaryManager;
use lune_llm::{NullPostProcessor, OpenAiLlmBackend, OpenAiLlmConfig};
use lune_persist::{AppPaths, Database, SettingsStore, resolve_local_model_paths};
use tauri::{AppHandle, Emitter, ipc::Channel};
use tokio::sync::{Mutex, RwLock, mpsc};
#[cfg(unix)]
use tracing::info;
use tracing::{debug, warn};

use crate::dto::SessionStreamEvent;

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
    /// Background thread that owns the uinput/clipboard handle.
    delivery: DeliveryService,
    /// Watch sender for activation mode — kept alive so the hotkey listener
    /// receives runtime changes when the user toggles Toggle↔Hold in settings.
    #[cfg(unix)]
    activation_mode_tx: tokio::sync::watch::Sender<lune_core::config::ActivationMode>,
    /// Hotkey command sender — dropping this signals the command loop to exit.
    #[cfg(unix)]
    _hotkey_cmd_tx: Option<mpsc::Sender<AppCommand>>,
    /// Hotkey listener thread handle — joined on shutdown for clean cleanup.
    #[cfg(unix)]
    _hotkey_thread: Option<std::thread::JoinHandle<()>>,
}

struct LivePipeline {
    paths: AppPaths,
    manager: Option<Arc<SessionManager<AsrRegistry>>>,
    active: bool,
    session_id: Option<String>,
    reload_required: bool,
}

struct SessionStreamSlot {
    id: u64,
    channel: Channel<SessionStreamEvent>,
}

impl AppRuntime {
    pub async fn new(app: AppHandle) -> Result<Self, String> {
        let paths = resolve_paths()?;
        let mut settings = load_settings(&paths)?;
        resolve_local_model_paths(&paths.models_dir(), &mut settings);
        init_ort_runtime();

        // Check model readiness *before* attempting pipeline build.
        // If models are present, build errors are fatal (e.g. corrupt DB,
        // bad config). If models are absent, boot with an empty pipeline
        // and let the user download via the setup UI.
        let models_ready = lune_persist::check_model_status(
            &paths.models_dir(),
            &settings.asr_backend,
            &settings.llm_backend,
        )
        .iter()
        .all(|s| s.ready);

        let (pipeline, settings) = if models_ready {
            build_pipeline(paths.clone(), settings).await?
        } else {
            info!("models not yet downloaded — deferring pipeline build");
            let settings = Arc::new(RwLock::new(settings));
            let pipeline = LivePipeline {
                paths,
                manager: None,
                active: false,
                session_id: None,
                reload_required: true,
            };
            (pipeline, settings)
        };

        let terminal_mode = settings.read().await.terminal_mode;
        let delivery = DeliveryService::new(terminal_mode);

        // ── Hotkey listener (evdev) ──────────────────────────────────────
        // Create channels and spawn the listener before building RuntimeInner
        // so the handle and sender can be stored for clean shutdown.
        #[cfg(unix)]
        let (activation_mode_tx, hotkey_cmd_tx, hotkey_thread, hotkey_cmd_rx) = {
            let mode = settings.read().await.activation_mode;
            let (am_tx, am_rx) = tokio::sync::watch::channel(mode);
            let hotkey_config = settings.read().await.hotkey.clone();
            let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(64);

            let handle = lune_app::hotkey_listener::spawn(hotkey_config, am_rx, cmd_tx.clone());

            match &handle {
                Some(_) => info!("hotkey listener started (evdev)"),
                None => warn!("hotkey listener failed to start — global shortcuts disabled"),
            }

            (am_tx, Some(cmd_tx), handle, Some(cmd_rx))
        };

        let runtime = Self {
            inner: Arc::new(RuntimeInner {
                app,
                pipeline: Mutex::new(pipeline),
                status: RwLock::new("ready".into()),
                session_stream: RwLock::new(None),
                session_stream_seq: AtomicU64::new(0),
                settings: Arc::clone(&settings),
                delivery,
                #[cfg(unix)]
                activation_mode_tx,
                #[cfg(unix)]
                _hotkey_cmd_tx: hotkey_cmd_tx,
                #[cfg(unix)]
                _hotkey_thread: hotkey_thread,
            }),
        };

        // Spawn the async command forwarder if the listener started.
        #[cfg(unix)]
        if let Some(cmd_rx) = hotkey_cmd_rx
            && runtime.inner._hotkey_thread.is_some()
        {
            let rt = runtime.clone();
            tauri::async_runtime::spawn(async move {
                Self::run_hotkey_command_loop(rt, cmd_rx).await;
            });
        }

        Ok(runtime)
    }

    /// Receive hotkey commands and dispatch to dictation methods.
    #[cfg(unix)]
    async fn run_hotkey_command_loop(runtime: AppRuntime, mut cmd_rx: mpsc::Receiver<AppCommand>) {
        while let Some(cmd) = cmd_rx.recv().await {
            let result = match cmd {
                AppCommand::ToggleDictation => {
                    debug!("hotkey: toggle dictation");
                    runtime.toggle_dictation().await.map(|_| ())
                }
                AppCommand::StartDictation => {
                    debug!("hotkey: start dictation");
                    runtime.start_dictation().await.map(|_| ())
                }
                AppCommand::StopDictation => {
                    debug!("hotkey: stop dictation");
                    runtime.stop_dictation().await
                }
                AppCommand::CancelDictation => {
                    debug!("hotkey: cancel dictation");
                    runtime.cancel_dictation().await
                }
                AppCommand::Quit => break,
            };
            if let Err(e) = result {
                warn!(error = %e, "hotkey command failed");
            }
        }
        info!("hotkey command loop exited");
    }

    pub async fn service_status(&self) -> String {
        self.inner.status.read().await.clone()
    }

    pub fn app_handle(&self) -> &AppHandle {
        &self.inner.app
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

    pub async fn mark_reload_required(&self) {
        {
            let mut pipeline = self.inner.pipeline.lock().await;
            pipeline.reload_required = true;
        }

        // Propagate activation mode to the hotkey listener so Toggle↔Hold
        // changes take effect without restarting the app.
        #[cfg(unix)]
        {
            let mode = self.inner.settings.read().await.activation_mode;
            let _ = self.inner.activation_mode_tx.send(mode);
        }
    }

    pub async fn start_dictation(&self) -> Result<String, String> {
        let manager = self.prepare_manager_for_start().await?;

        let (event_tx, event_rx) = mpsc::channel(256);
        let session_id = match manager.start(event_tx).await {
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

        self.push_session_event(SessionStreamEvent::SessionStarted {
            session_id: session_id.clone(),
        })
        .await;
        self.publish_status("listening").await;
        self.spawn_event_forwarder(event_rx);

        Ok(session_id)
    }

    pub async fn stop_dictation(&self) -> Result<(), String> {
        if let Some(manager) = self.active_manager().await {
            manager.stop().await;
        }
        Ok(())
    }

    pub async fn cancel_dictation(&self) -> Result<(), String> {
        if let Some(manager) = self.active_manager().await {
            manager.cancel().await;
        }
        Ok(())
    }

    pub async fn toggle_dictation(&self) -> Result<Option<String>, String> {
        let is_active = self.inner.pipeline.lock().await.active;

        if is_active {
            self.stop_dictation().await?;
            Ok(None)
        } else {
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
                    SessionEvent::FinalizedText(text) => {
                        saw_text = true;
                        let mode = runtime.inner.settings.read().await.dictation_mode;
                        runtime.inner.delivery.deliver(&text, mode);
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
                    SessionEvent::CommandDetected(ref payload) => {
                        runtime.handle_voice_command(payload).await;
                        runtime
                            .push_session_event(SessionStreamEvent::CommandDetected {
                                payload: payload.clone(),
                            })
                            .await;
                    }
                    SessionEvent::Error(message) => {
                        saw_error = true;
                        // Log full error internally; send sanitized version to frontend.
                        warn!(error = %message, "session error");
                        runtime.finish_session().await;
                        runtime.publish_status("error").await;
                        runtime
                            .push_session_event(SessionStreamEvent::SessionError {
                                message: sanitize_error_for_frontend(&message),
                            })
                            .await;
                    }
                    SessionEvent::Stopped => {
                        runtime.finish_session().await;
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
    async fn handle_voice_command(&self, payload: &str) {
        let cmd: VoiceCommand = match serde_json::from_str(payload) {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, payload, "CommandDetected: failed to parse VoiceCommand JSON");
                return;
            }
        };

        let mode = self.inner.settings.read().await.dictation_mode;

        match cmd {
            VoiceCommand::NewLine => {
                debug!("CommandDetected: NewLine → delivering \\n");
                self.inner.delivery.deliver("\n", mode);
            }
            VoiceCommand::NewParagraph => {
                debug!("CommandDetected: NewParagraph → delivering \\n\\n");
                self.inner.delivery.deliver("\n\n", mode);
            }
            VoiceCommand::ListItem => {
                debug!("CommandDetected: ListItem → delivering \\n- ");
                self.inner.delivery.deliver("\n- ", mode);
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

    async fn finish_session(&self) {
        self.set_pipeline_session(None, false).await;
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
            let mut settings = load_settings(&paths)?;
            resolve_local_model_paths(&paths.models_dir(), &mut settings);
            let (rebuilt, new_settings) = build_pipeline(paths, settings)
                .await
                .map_err(|e| format!("Pipeline build failed: {e}"))?;
            let mut pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return Err("dictation session became active during reload".into());
            }
            *pipeline = rebuilt;
            pipeline.active = true;
            // Propagate new settings into the shared arc so delivery uses the
            // latest dictation_mode and terminal_mode after a reload.
            let new_settings_val = new_settings.read().await.clone();
            #[cfg(unix)]
            let _ = self
                .inner
                .activation_mode_tx
                .send(new_settings_val.activation_mode);
            let mut settings = self.inner.settings.write().await;
            *settings = new_settings_val;
            let mgr = pipeline
                .manager
                .as_ref()
                .expect("build_pipeline succeeded but manager is None");
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
        Ok(Arc::clone(
            pipeline
                .manager
                .as_ref()
                .expect("manager confirmed Some at line above"),
        ))
    }

    async fn set_pipeline_session(&self, session_id: Option<String>, active: bool) {
        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.session_id = session_id;
        pipeline.active = active;
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
) -> Result<(LivePipeline, Arc<RwLock<UserSettings>>), String> {
    let asr = AsrRegistry::from_config(&settings.asr_backend, settings.device)
        .map_err(|e| format!("build asr backend: {e}"))?;
    let aligner = build_aligner(&asr, &settings)?;
    let post_processor = build_post_processor(&settings).await?;
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
    let settings = Arc::new(RwLock::new(settings.clone()));

    let config = SessionManagerConfig {
        asr_backend_kind: asr.kind(),
        asr_config: AsrSessionConfig {
            language: Some(settings.read().await.language.clone()),
            ..Default::default()
        },
        pipeline: asr
            .pipeline_config()
            .with_overrides(&settings.read().await.pipeline),
    };

    let mut manager = SessionManager::new(
        Arc::new(asr),
        router,
        post_processor,
        dict,
        db,
        config,
        Arc::clone(&settings),
    )
    .with_vad_state_path(paths.adaptive_vad_file());

    if let Some(aligner) = aligner {
        manager = manager.with_aligner(aligner);
    }

    let pipeline = LivePipeline {
        paths,
        manager: Some(Arc::new(manager)),
        active: false,
        session_id: None,
        reload_required: false,
    };

    Ok((pipeline, settings))
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
    let _ = ort::init().commit();
}

fn build_aligner(
    asr: &AsrRegistry,
    settings: &UserSettings,
) -> Result<Option<Arc<OrtForcedAligner>>, String> {
    match (&settings.aligner, asr.as_ort_qwen()) {
        (Some(cfg), Some(backend)) if cfg.enabled => backend
            .make_forced_aligner(&cfg.model_dir)
            .map(|aligner| {
                let aligner = Arc::new(aligner);
                backend.attach_aligner(Arc::clone(&aligner));
                Some(aligner)
            })
            .map_err(|e| format!("build forced aligner: {e}")),
        _ => Ok(None),
    }
}

async fn build_post_processor(
    settings: &UserSettings,
) -> Result<Arc<dyn TextPostProcessor>, String> {
    match &settings.llm_backend {
        // Local LLM-based correction is gone — the Nemotron cross-encoder
        // reranker handles all local correction at the session-manager
        // layer. Return a passthrough so the trait slot is filled.
        LlmBackendConfig::LlamaCppLocal { .. } => {
            Ok(Arc::new(NullPostProcessor::new()) as Arc<dyn TextPostProcessor>)
        }
        LlmBackendConfig::OpenAiCompatible {
            base_url,
            api_key,
            model,
            max_tokens,
        } => {
            let backend = OpenAiLlmBackend::new(OpenAiLlmConfig {
                base_url: base_url.to_string(),
                api_key: api_key.expose().to_owned(),
                model: model.clone(),
                max_tokens: *max_tokens,
                temperature: 0.2,
            });
            Ok(Arc::new(backend) as Arc<dyn TextPostProcessor>)
        }
        _ => Err("unsupported LLM backend type in desktop UI runtime".into()),
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
    use super::sanitize_error_for_frontend;

    #[test]
    fn sanitize_error_handles_non_ascii_without_panicking() {
        let raw = format!("é{}", "/tmp/example/path/segment".repeat(80));
        let sanitized = sanitize_error_for_frontend(&raw);
        assert!(!sanitized.is_empty());
    }
}
