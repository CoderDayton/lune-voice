use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::Instant;

use lune_app::{DeliveryService, SessionManager, SessionManagerConfig};
#[cfg(unix)]
use lune_app::{WindowTarget, capture_focus_target};
use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::asr::AsrSessionConfig;
use lune_core::command::VoiceCommand;
use lune_core::config::{
    ActivationMode, DictationMode, LlmBackendConfig, PillVisibility, UserSettings,
};
use lune_core::llm::LlmService;
use lune_core::session::SessionEvent;
use lune_dict::DictionaryManager;
use lune_dict::commands::VoiceCommandRouter;
use lune_llm::{NullPostProcessor, OpenAiLlmBackend, OpenAiLlmConfig};
use lune_persist::{AppPaths, Database, SettingsStore};
use tauri::{AppHandle, Emitter, Manager, ipc::Channel};
use tokio::sync::{Mutex, Notify, RwLock, mpsc};
#[cfg(unix)]
use tracing::info;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::dto::SessionStreamEvent;

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
    /// Long-lived microphone capture stream when `mic_prewarm` is enabled.
    /// Sessions consume via `AudioStreamHandle` so the cpal::Stream stays
    /// open across hotkey presses and the first chunk arrives ~70 ms sooner.
    prewarmed_audio: Mutex<Option<PrewarmedStream>>,
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

/// A prewarmed capture stream paired with the device it was opened on, so a
/// later device-preference change can be detected and the stream rebuilt live.
struct PrewarmedStream {
    stream: AudioStream,
    device: Option<String>,
}

impl AppRuntime {
    pub async fn new(app: AppHandle) -> Result<Self, String> {
        let paths = resolve_paths()?;
        // One-time on boot: lift any plaintext API keys still sitting in
        // settings.toml into the OS keyring. Non-fatal — a failure just leaves
        // the keys where they are.
        if let Err(e) = SettingsStore::new(paths.settings_file()).migrate_secrets_to_keyring() {
            tracing::warn!(error = %e, "API key keyring migration failed; keys remain in settings.toml");
        }
        // One-time on boot: move any legacy custom styles out of settings.toml
        // into the database, the source of truth for user-created styles. Fatal
        // on failure *when legacy styles are present*: they are no longer a
        // `UserSettings` field, so continuing past a failed migration would let
        // the next settings save rewrite the file without them and lose them
        // permanently. With nothing to migrate it stays harmless.
        {
            let store = SettingsStore::new(paths.settings_file());
            match Database::open(&paths.database_file()) {
                Ok(mut db) => store
                    .migrate_custom_styles_to_db(&mut db)
                    .map_err(|e| format!("custom-styles migration failed: {e}"))?,
                Err(e) => {
                    // Couldn't open the DB. Only fatal when legacy styles remain
                    // in the TOML (at risk of a clobbering save). If that check
                    // itself can't read the file, assume the worst and refuse.
                    if store.has_legacy_custom_styles().unwrap_or(true) {
                        return Err(format!(
                            "cannot migrate custom styles into the database (open failed): {e}"
                        ));
                    }
                    tracing::warn!(error = %e, "open DB for custom-styles migration failed; no legacy styles to migrate, continuing");
                }
            }
        }
        let settings = load_settings(&paths)?;

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
                prewarmed_audio: Mutex::new(None),
            }),
        };

        runtime.sync_mic_prewarm(None, None).await;

        Ok(runtime)
    }

    /// Push saved sound-cue settings into `inner.settings` immediately so the
    /// next dictation honors them. The optimistic "listening" status (and its
    /// Start cue) fires before `prepare_manager_for_start` refreshes
    /// `inner.settings` on rebuild, so without this a toggle/volume change
    /// would be off by one dictation.
    pub async fn sync_sound_cues(&self, enabled: bool, volume: f32) {
        let mut settings = self.inner.settings.write().await;
        settings.sound_cues_enabled = enabled;
        settings.sound_cue_volume = volume.clamp(0.0, 1.0);
    }

    /// Reconcile the long-lived audio stream with the desired `mic_prewarm`
    /// state and input device. For each argument, `None` reads the current
    /// in-memory setting (suitable for startup); a `Some(_)` override is used
    /// after a settings save, when `inner.settings` is stale until the next
    /// pipeline rebuild. `device` is `Option<Option<String>>`: the outer
    /// `None` reads the setting, `Some(d)` overrides with device `d` (where
    /// the inner `None` means the system default).
    pub async fn sync_mic_prewarm(&self, want: Option<bool>, device: Option<Option<String>>) {
        let (want, input_device) = {
            let settings = self.inner.settings.read().await;
            (
                want.unwrap_or(settings.mic_prewarm),
                device.unwrap_or_else(|| settings.input_device.clone()),
            )
        };
        let mut slot = self.inner.prewarmed_audio.lock().await;
        let open = slot.as_ref().map(|p| p.device.as_deref());
        let action = reconcile_prewarm(open, want, input_device.as_deref());
        match action {
            PrewarmAction::Keep => {}
            PrewarmAction::Close => {
                debug!("mic prewarm: closing capture stream");
                *slot = None;
            }
            PrewarmAction::Open | PrewarmAction::Reopen => {
                if action == PrewarmAction::Reopen {
                    debug!("mic prewarm: input device changed, rebuilding capture stream");
                }
                // Drop any existing stream first so its device handle is freed
                // before we open the new one.
                *slot = None;
                let config = AudioConfig {
                    input_device: input_device.clone(),
                    ..AudioConfig::default()
                };
                match AudioStream::new(config) {
                    Ok(stream) => {
                        debug!("mic prewarm: capture stream opened");
                        *slot = Some(PrewarmedStream {
                            stream,
                            device: input_device,
                        });
                    }
                    Err(e) => warn!("mic prewarm: failed to open capture stream: {e}"),
                }
            }
        }
    }

    async fn prewarmed_handle(&self) -> Option<lune_audio::AudioStreamHandle> {
        let slot = self.inner.prewarmed_audio.lock().await;
        slot.as_ref().map(|p| p.stream.handle())
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

        // No subscriber is expected — not a bug — when the main window was
        // never created (`start_minimized`/tray-only operation has no UI to
        // reflect state into). Only warn when the window exists but hasn't
        // subscribed yet, which is the real race worth flagging.
        if self.inner.session_stream.read().await.is_none()
            && self.inner.app.get_webview_window("main").is_some()
        {
            warn!(
                "starting dictation with no IPC stream subscriber — UI will not reflect session state"
            );
        }

        // Optimistic UI: tell the pill we're listening before doing session work.
        // Reverted on failure so the pill doesn't stick in "listening" state.
        self.publish_status("listening").await;

        // Anchor the pre-roll to the key-press: drain the hot capture ring NOW,
        // before the (possibly slow) manager prep, keeping ~480 ms of lead-in.
        // Audio spoken during warmup then accumulates in the ring and still
        // reaches the recognizer, so a long startup no longer eats the first
        // words. Draining inside the capture task instead would anchor the
        // window to post-warmup time and discard that lead-in.
        let prewarmed = self.prewarmed_handle().await;
        if let Some(handle) = &prewarmed {
            handle.drain_keep_last(6);
        }

        // Capture focus off the start path: the value is only consumed at
        // delivery time (many seconds later), but the underlying Wayland /
        // subprocess probes are synchronous and block the tokio worker for
        // ~400–500 ms on COSMIC. Letting them race the audio bring-up
        // unblocks the first VAD chunk by the same amount.
        #[cfg(unix)]
        let focus_capture = tokio::task::spawn_blocking(capture_focus_target);

        // Snapshot the clipboard now, off the paste hot path: on the uinput
        // paste backend the restore-after-paste read can block up to 2s on a
        // slow clipboard owner, so priming here overlaps it with recording +
        // ASR (mirrors the focus capture above). Only the clipboard delivery
        // path borrows the clipboard, so skip the wl-paste round trip for
        // per-key typing that keeps no copy; the rare non-ASCII fallback there
        // still snapshots synchronously. No-op anyway on backends that never
        // touch the clipboard (IBus, macOS/Windows).
        {
            let (mode, copy) = {
                let s = self.inner.settings.read().await;
                (s.dictation_mode, s.copy_to_clipboard)
            };
            if copy || !matches!(mode, DictationMode::TypeInPlace) {
                self.inner.delivery.prime_clipboard();
            }
        }

        let manager = match self.prepare_manager_for_start().await {
            Ok(m) => m,
            Err(e) => {
                self.publish_status("ready").await;
                return Err(e);
            }
        };
        let (event_tx, event_rx) = mpsc::channel(256);
        let start_result = match prewarmed {
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
                        let (mode, copy) = {
                            let s = runtime.inner.settings.read().await;
                            (s.dictation_mode, s.copy_to_clipboard)
                        };
                        let delivered;
                        #[cfg(unix)]
                        {
                            let target = runtime.inner.pipeline.lock().await.focus_target.clone();
                            delivered = if let Some(ref t) = target {
                                runtime.inner.delivery.deliver_to(
                                    Some(session_id),
                                    &text,
                                    mode,
                                    copy,
                                    t,
                                )
                            } else {
                                runtime
                                    .inner
                                    .delivery
                                    .deliver(Some(session_id), &text, mode, copy)
                            };
                        }
                        #[cfg(not(unix))]
                        {
                            delivered =
                                runtime
                                    .inner
                                    .delivery
                                    .deliver(Some(session_id), &text, mode, copy);
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
                        .deliver_to(Some(session_id), "\n", mode, false, t);
                } else {
                    self.inner
                        .delivery
                        .deliver(Some(session_id), "\n", mode, false);
                }
                #[cfg(not(unix))]
                self.inner
                    .delivery
                    .deliver(Some(session_id), "\n", mode, false);
            }
            VoiceCommand::NewParagraph => {
                debug!("CommandDetected: NewParagraph → delivering \\n\\n");
                #[cfg(unix)]
                if let Some(ref t) = focus_target {
                    self.inner
                        .delivery
                        .deliver_to(Some(session_id), "\n\n", mode, false, t);
                } else {
                    self.inner
                        .delivery
                        .deliver(Some(session_id), "\n\n", mode, false);
                }
                #[cfg(not(unix))]
                self.inner
                    .delivery
                    .deliver(Some(session_id), "\n\n", mode, false);
            }
            VoiceCommand::ListItem => {
                debug!("CommandDetected: ListItem → delivering \\n- ");
                #[cfg(unix)]
                if let Some(ref t) = focus_target {
                    self.inner
                        .delivery
                        .deliver_to(Some(session_id), "\n- ", mode, false, t);
                } else {
                    self.inner
                        .delivery
                        .deliver(Some(session_id), "\n- ", mode, false);
                }
                #[cfg(not(unix))]
                self.inner
                    .delivery
                    .deliver(Some(session_id), "\n- ", mode, false);
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

    /// Play a dictation sound cue if cues are enabled, at the saved volume.
    async fn play_cue(&self, cue: lune_app::sound::Cue) {
        let (enabled, volume) = {
            let s = self.inner.settings.read().await;
            (s.sound_cues_enabled, s.sound_cue_volume)
        };
        if enabled {
            lune_app::sound::play(cue, volume);
        }
    }

    async fn publish_status(&self, status: &str) {
        let prev = std::mem::replace(&mut *self.inner.status.write().await, status.to_owned());

        // Sound cues play from the backend — the only surface guaranteed alive
        // (the main window is destroyed on close / absent when minimized, and
        // the pill may not exist).
        if let Some(cue) = cue_for_transition(&prev, status) {
            self.play_cue(cue).await;
        }

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

    /// Rebuild the pipeline now when a settings change is pending, so callers
    /// that consume `llm_service()` outside a dictation session (style example
    /// generation) don't run against a stale service — e.g. the Null
    /// passthrough kept from before the styling model was downloaded. Skipped
    /// while a session is active; the lazy rebuild on the next dictation start
    /// covers that case. A failed rebuild is logged, not fatal: the styling
    /// path's own readiness checks produce the user-facing error.
    async fn reload_pipeline_if_required(&self) {
        let reload_paths = {
            let pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return;
            }
            if pipeline.reload_required || pipeline.manager.is_none() {
                Some(pipeline.paths.clone())
            } else {
                None
            }
        };
        let Some(paths) = reload_paths else {
            return;
        };
        let settings = match load_settings(&paths) {
            Ok(s) => s,
            Err(error) => {
                warn!(%error, "styles: settings load failed; keeping current pipeline");
                return;
            }
        };
        match build_pipeline(paths, settings).await {
            Ok((rebuilt, new_llm_service, new_settings)) => {
                let mut pipeline = self.inner.pipeline.lock().await;
                if pipeline.active {
                    // A session started while we rebuilt; drop this rebuild and
                    // let the next dictation start redo it.
                    return;
                }
                *pipeline = rebuilt;
                *self.inner.llm_service.write().await = new_llm_service;
                let new_settings_val = new_settings.read().await.clone();
                *self.inner.settings.write().await = new_settings_val;
            }
            Err(error) => {
                warn!(%error, "styles: pipeline rebuild failed; keeping current pipeline");
            }
        }
    }

    /// Ensure the styling engine can actually serve a generation request:
    /// apply any pending pipeline reload, then load the local model into
    /// memory and wait for it (no-op for the remote provider, whose readiness
    /// is configuration-only). Returns a user-facing message on failure.
    /// Configuration-level checks (`styling_llm_readiness`) are the caller's
    /// preflight — this validates initialization, not setup.
    pub async fn ensure_styling_llm_ready(&self) -> Result<(), String> {
        self.reload_pipeline_if_required().await;
        let local_llm = { self.inner.pipeline.lock().await.local_llm.clone() };
        if let Some(local_llm) = local_llm {
            local_llm.ensure_ready().await?;
        }
        Ok(())
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

    pub(crate) async fn schedule_local_llm_unload(&self) {
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
    // Whisper model load takes seconds on the larger models; keep it off the
    // async workers so IPC and the UI stay responsive during startup/reload.
    let asr = {
        let backend_config = settings.asr_backend.clone();
        let device = settings.device;
        let models_dir = paths.models_dir();
        tokio::task::spawn_blocking(move || {
            AsrRegistry::from_config(&backend_config, device, &models_dir)
        })
        .await
        .map_err(|e| format!("asr build task: {e}"))?
        .map_err(|e| format!("build asr backend: {e}"))?
    };
    let (llm_service, local_llm) = build_llm_service(&settings, &paths.models_dir()).await?;
    let style_pipeline = Arc::new(lune_llm::StylePipeline::new(Arc::clone(&llm_service)));
    let post_processor: Arc<dyn lune_core::llm::TextPostProcessor> = style_pipeline;
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
    // Resolve backend defaults + user TOML overrides once, and surface any
    // invariant violations — a bad override otherwise degrades sessions
    // silently (e.g. an instant wall-clock deadline).
    let pipeline_config = asr.pipeline_config().with_overrides(&settings.pipeline);
    for issue in pipeline_config.validate() {
        tracing::warn!(issue, "pipeline configuration violates invariant");
    }
    // Wire dictionary replacement strings into the ASR decoder as boost phrases.
    {
        let pipeline = &pipeline_config;
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
        pipeline: pipeline_config,
    };

    let heuristic = Arc::new(lune_dict::corrector::LexicalGate::new());

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

/// Find the first `*.gguf` file in `dir`, if the style model has been
/// downloaded. The styling LLM ships as a single GGUF in `DIR_STYLE_LLM`.
fn find_gguf(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) == Some("gguf") {
            Some(path)
        } else {
            None
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
            match find_gguf(&style_dir) {
                Some(model_path) => {
                    // Decode budget 512: matches the ASR decoder's own
                    // 512-token budget. Greedy decode stops at EOS, so short
                    // utterances never pay for the larger cap.
                    let llm = Arc::new(LazyLocalLlm::new(
                        model_path,
                        settings.device.use_gpu(cfg!(feature = "llama-cuda")),
                        512,
                    ));
                    Ok((llm.clone() as Arc<dyn LlmService>, Some(llm)))
                }
                None => {
                    tracing::info!("style LLM not downloaded yet, using passthrough");
                    Ok((
                        Arc::new(NullPostProcessor::new()) as Arc<dyn LlmService>,
                        None,
                    ))
                }
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
    model_path: std::path::PathBuf,
    use_gpu: bool,
    max_new_tokens: usize,
    generation: AtomicU64,
    unload_epoch: AtomicU64,
    state: Mutex<LazyLlmState>,
    /// Signaled on every transition out of `Loading`, so `complete()` can
    /// wait out an in-flight warm-up instead of failing a race it didn't
    /// need to lose.
    state_changed: Notify,
}

enum LazyLlmState {
    Empty,
    Loading(u64),
    Ready(Arc<lune_llm::backend::llama_cpp::LlamaCppBackend>),
}

impl LazyLocalLlm {
    fn new(model_path: std::path::PathBuf, use_gpu: bool, max_new_tokens: usize) -> Self {
        Self {
            model_path,
            use_gpu,
            max_new_tokens,
            generation: AtomicU64::new(0),
            unload_epoch: AtomicU64::new(0),
            state: Mutex::new(LazyLlmState::Empty),
            state_changed: Notify::new(),
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
        let model_path = self.model_path.clone();
        let use_gpu = self.use_gpu;
        let max_new_tokens = self.max_new_tokens;
        let loaded = tokio::task::spawn_blocking(move || {
            lune_llm::backend::llama_cpp::LlamaCppBackend::new(
                lune_llm::backend::llama_cpp::LlamaCppConfig {
                    model_path,
                    use_gpu,
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
            drop(state);
            self.state_changed.notify_waiters();
            return;
        }

        match loaded {
            Ok(Ok(llm)) => {
                let weights_mib = std::fs::metadata(&self.model_path)
                    .map(|m| m.len() as f64 / (1024.0 * 1024.0))
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
        drop(state);
        self.state_changed.notify_waiters();
    }

    /// Load the model if needed and wait until it can serve. Unlike `warm_up`
    /// (fire-and-forget on dictation start, returns without waiting for an
    /// in-flight load), this gives the styles-page validation path a
    /// definitive ready/failed answer before example generation is allowed.
    async fn ensure_ready(&self) -> Result<(), String> {
        self.cancel_scheduled_unload();
        let mut attempted = false;
        loop {
            // Register before reading state so a transition between the read
            // and the await can't be missed.
            let notified = self.state_changed.notified();
            let loading = {
                let state = self.state.lock().await;
                match &*state {
                    LazyLlmState::Ready(_) => return Ok(()),
                    LazyLlmState::Loading(_) => true,
                    LazyLlmState::Empty => false,
                }
            };
            if loading {
                notified.await;
            } else if attempted {
                // warm_up ran and the state fell back to Empty — the load
                // failed (details already logged by warm_up).
                return Err("The styling model failed to load. Check the application logs.".into());
            } else {
                attempted = true;
                self.warm_up().await;
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
        drop(state);
        self.state_changed.notify_waiters();
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
        // A short utterance can finish transcription before a warm-up
        // triggered at session start has loaded the model. Wait out an
        // in-flight load rather than failing the race — the outer
        // `llm_timeout_secs` in session.rs still bounds the total wait.
        loop {
            let notified = self.state_changed.notified();
            let llm = {
                let state = self.state.lock().await;
                match &*state {
                    LazyLlmState::Ready(llm) => Some(Arc::clone(llm)),
                    LazyLlmState::Empty => {
                        return Err(lune_core::Error::llm("local LLM not ready"));
                    }
                    LazyLlmState::Loading(_) => None,
                }
            };
            match llm {
                Some(llm) => return llm.complete(system, user).await,
                None => notified.await,
            }
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

/// What to do with the long-lived prewarm capture stream when reconciling the
/// desired state against what's currently open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrewarmAction {
    /// Leave the current state untouched.
    Keep,
    /// Open a fresh stream on the desired device.
    Open,
    /// Tear down the open stream.
    Close,
    /// Close and reopen on the desired device — the device changed under us.
    Reopen,
}

/// Decide the prewarm stream action. `open` describes the currently open
/// stream: `None` = nothing open; `Some(dev)` = open on `dev`, where `dev` is
/// `None` for the system default. Comparing the open device against `desired`
/// is what makes a device change rebuild the stream live (no restart needed).
fn reconcile_prewarm(
    open: Option<Option<&str>>,
    want: bool,
    desired: Option<&str>,
) -> PrewarmAction {
    match (want, open) {
        (false, Some(_)) => PrewarmAction::Close,
        (false, None) => PrewarmAction::Keep,
        (true, None) => PrewarmAction::Open,
        (true, Some(current)) if current == desired => PrewarmAction::Keep,
        (true, Some(_)) => PrewarmAction::Reopen,
    }
}

/// Map a service-status transition to the sound cue that should play, or
/// `None`. Dedups on the previous status so each cue fires once — "listening"
/// is published both optimistically and again on `SessionEvent::Started`.
///
/// The Stop cue fires on `listening -> processing`, i.e. the moment recording
/// ends. `SessionEvent::Processing` is emitted once by the session for every
/// way capture can end — key release, toggle-off, max-speech cutoff — and
/// never on cancel, so this single transition covers them all. It also lands
/// well before ASR/LLM/delivery, so the cue is immediate, not "after the text
/// appears" (which is the later "ready" transition).
fn cue_for_transition(prev: &str, status: &str) -> Option<lune_app::sound::Cue> {
    use lune_app::sound::Cue;
    match status {
        "listening" if prev != "listening" => Some(Cue::Start),
        "processing" if prev == "listening" => Some(Cue::Stop),
        "error" if prev != "error" => Some(Cue::Error),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use lune_core::llm::LlmService;

    use super::{
        LazyLocalLlm, PrewarmAction, cue_for_transition, reconcile_prewarm,
        sanitize_error_for_frontend,
    };

    #[test]
    fn start_cue_fires_once_despite_double_listening() {
        use lune_app::sound::Cue;
        // Optimistic publish enters "listening" from idle.
        assert_eq!(cue_for_transition("ready", "listening"), Some(Cue::Start));
        // SessionEvent::Started republishes "listening" — must not replay.
        assert_eq!(cue_for_transition("listening", "listening"), None);
    }

    #[test]
    fn stop_cue_fires_when_recording_ends() {
        use lune_app::sound::Cue;
        // "processing" = capture ended (release, toggle-off, or max cutoff),
        // emitted once by the session and never on cancel.
        assert_eq!(
            cue_for_transition("listening", "processing"),
            Some(Cue::Stop)
        );
    }

    #[test]
    fn ready_status_emits_no_cue() {
        // The Stop cue fires on `processing`, not the post-delivery "ready".
        assert_eq!(cue_for_transition("processing", "ready"), None);
        assert_eq!(cue_for_transition("listening", "ready"), None);
        assert_eq!(cue_for_transition("ready", "ready"), None);
    }

    #[test]
    fn error_cue_fires_once() {
        use lune_app::sound::Cue;
        assert_eq!(cue_for_transition("processing", "error"), Some(Cue::Error));
        assert_eq!(cue_for_transition("error", "error"), None);
    }

    #[test]
    fn no_cue_for_unknown_status_or_stray_processing() {
        assert_eq!(cue_for_transition("ready", "calibrating"), None);
        // "processing" only cues when it directly follows "listening".
        assert_eq!(cue_for_transition("ready", "processing"), None);
    }

    /// The follow-up behavior: when the chosen device changes while prewarm
    /// stays enabled, the open stream must be rebuilt (Reopen), not left on the
    /// stale device.
    #[test]
    fn reconcile_reopens_when_device_changes_while_enabled() {
        let action =
            reconcile_prewarm(Some(Some("Built-in Microphone")), true, Some("USB Headset"));
        assert_eq!(action, PrewarmAction::Reopen);
    }

    #[test]
    fn reconcile_keeps_when_device_unchanged() {
        assert_eq!(
            reconcile_prewarm(Some(Some("USB Headset")), true, Some("USB Headset")),
            PrewarmAction::Keep
        );
        // Default ↔ default is also unchanged.
        assert_eq!(
            reconcile_prewarm(Some(None), true, None),
            PrewarmAction::Keep
        );
    }

    #[test]
    fn reconcile_opens_when_enabled_and_nothing_open() {
        assert_eq!(
            reconcile_prewarm(None, true, Some("USB Headset")),
            PrewarmAction::Open
        );
    }

    #[test]
    fn reconcile_closes_when_disabled_with_stream_open() {
        assert_eq!(
            reconcile_prewarm(Some(None), false, Some("USB Headset")),
            PrewarmAction::Close
        );
    }

    #[test]
    fn reconcile_keeps_when_disabled_and_nothing_open() {
        assert_eq!(reconcile_prewarm(None, false, None), PrewarmAction::Keep);
    }

    #[test]
    fn sanitize_error_handles_non_ascii_without_panicking() {
        let raw = format!("é{}", "/tmp/example/path/segment".repeat(80));
        let sanitized = sanitize_error_for_frontend(&raw);
        assert!(!sanitized.is_empty());
    }

    #[tokio::test]
    async fn lazy_local_llm_errors_when_not_warmed() {
        let llm = LazyLocalLlm::new(std::path::PathBuf::from("/missing/model"), false, 8);
        let result = llm.complete("system", "user").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn lazy_local_llm_ttl_unloads_in_flight_warmup() {
        let llm = Arc::new(LazyLocalLlm::new(
            std::path::PathBuf::from("/missing/model"),
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
    async fn complete_waits_for_in_flight_warmup_instead_of_failing_fast() {
        let llm = Arc::new(LazyLocalLlm::new(
            std::path::PathBuf::from("/missing/model"),
            false,
            8,
        ));
        *llm.state.lock().await = super::LazyLlmState::Loading(0);

        let waiting = Arc::clone(&llm);
        let handle = tokio::spawn(async move { waiting.complete("system", "user").await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !handle.is_finished(),
            "complete() must wait out an in-flight warm-up instead of failing fast"
        );

        llm.unload().await;
        let result = handle.await.unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn ensure_ready_reports_failed_load_instead_of_hanging() {
        // A bad model path (or a stub build) makes warm_up fall back to Empty;
        // ensure_ready must surface that as a definitive error so the styles
        // page can show it, rather than spinning or reporting ready.
        let llm = LazyLocalLlm::new(std::path::PathBuf::from("/missing/model"), false, 8);
        let err = llm.ensure_ready().await.unwrap_err();
        assert!(err.contains("failed to load"), "got: {err}");
    }

    #[tokio::test]
    async fn lazy_local_llm_ttl_can_be_cancelled() {
        let llm = Arc::new(LazyLocalLlm::new(
            std::path::PathBuf::from("/missing/model"),
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
