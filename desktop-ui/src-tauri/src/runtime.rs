// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::Instant;

use lune_asr::AsrRegistry;
use lune_audio::{AudioConfig, AudioStream};
use lune_core::asr::AsrSessionConfig;
use lune_core::command::VoiceCommand;
use lune_core::config::{
    AsrBackendConfig, LlmBackendConfig, PillVisibility, PrivacyMode, UserSettings,
};
use lune_core::llm::LlmService;
use lune_core::session::SessionEvent;
use lune_dict::UserDictionary;
use lune_dict::commands::VoiceCommandRouter;
use lune_llm::{NullPostProcessor, OpenAiLlmBackend, OpenAiLlmConfig};
use lune_persist::{AppPaths, Database, SettingsStore};
use lune_voice::{
    DeliveryHealth, DeliveryService, SessionAudio, SessionKind, SessionManager,
    SessionManagerConfig,
};
use lune_voice::{WindowTarget, capture_focus_target};
use tauri::{AppHandle, Emitter, Manager, ipc::Channel};
use tokio::sync::{Mutex, RwLock, mpsc};
use tracing::{debug, info, warn};

use crate::dto::SessionStreamEvent;
use crate::persistence::load_settings_checked;

const TOGGLE_DEBOUNCE: Duration = Duration::from_millis(150);
const CANCEL_DEBOUNCE: Duration = Duration::from_millis(80);
/// Shown when a session reported failure but its error text never arrived.
const DROPPED_ERROR_MESSAGE: &str = "Dictation failed, but the error details were lost.";

pub struct AppRuntime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    app: AppHandle,
    pipeline: Mutex<LivePipeline>,
    /// Serializes `prepare_manager_for_start`'s check-then-rebuild, which
    /// cannot hold `pipeline` across the (slow) `build_pipeline` call.
    start_gate: Mutex<()>,
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
    /// When a dictation last started or ended — the clock the idle model-unload
    /// watchdog measures against. Reset in `set_pipeline_session`, which every
    /// session transition passes through.
    last_activity: Mutex<Instant>,
    /// Long-lived microphone capture stream, always open while the app runs.
    /// Sessions consume via `AudioStreamHandle` so the cpal::Stream stays
    /// open across hotkey presses and the first chunk arrives ~70 ms sooner.
    prewarmed_audio: Mutex<Option<PrewarmedStream>>,
    /// Serializes `sync_capture_device`'s decide-clear-open-install sequence.
    /// Separate from `prewarmed_audio` so the slot stays readable while an
    /// open is in flight: two concurrent syncs (two quick settings saves)
    /// would otherwise each decide to open and hold the same exclusive
    /// capture device at once, and one of them fails.
    capture_open: Mutex<()>,
    /// Set when a capture stream opened on the system default because the
    /// configured microphone was missing, and cleared by the next dictation
    /// that reports it. One toast per fallback, not one per dictation: the
    /// condition persists until the device comes back.
    capture_fallback_unreported: std::sync::atomic::AtomicBool,
}

/// What the dictation pipeline is doing right now.
///
/// `Loading` exists because it is a real state with a real duration: building
/// the pipeline loads the whisper model, which takes seconds on the larger
/// ones. Collapsing it into `Idle` (as a single `active: bool` does) makes
/// every stop and cancel arriving in that window a silent no-op — a
/// push-to-talk release lands on nothing, and the session then starts with the
/// key already back up and nothing left to end it.
/// Whether moving from `current` to `next` invalidates the loaded ASR backend.
///
/// Compared by hand rather than with `PartialEq`: the remote variant carries an
/// API key, and secrets are deliberately not comparable. The remote backend is
/// rebuilt unconditionally — it loads no model, so nothing is saved by keeping
/// it, and its key may have changed underneath us.
fn asr_backend_needs_reload(current: &AsrBackendConfig, next: &AsrBackendConfig) -> bool {
    match (current, next) {
        (AsrBackendConfig::WhisperCpp { model: a }, AsrBackendConfig::WhisperCpp { model: b }) => {
            a != b
        }
        (
            AsrBackendConfig::ParakeetCpp { model: a },
            AsrBackendConfig::ParakeetCpp { model: b },
        ) => a != b,
        _ => true,
    }
}

/// A start refused because one is already running or loading.
const ALREADY_ACTIVE: &str = "dictation session already active";

/// A start that abandoned itself because a stop or cancel arrived while it was
/// still loading. Not a failure — the user got what they asked for.
const START_ABORTED: &str = "dictation start cancelled before it began";

/// How a start failure that happened before any session existed is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartFailureReport {
    /// A stop or cancel arrived during the load. Walk the pill back to
    /// "ready" and say nothing else.
    Abandoned,
    /// Prep failed. The hotkey already played the start cue, so without an
    /// error status and a toast the user gets a beep, silence, and no way to
    /// learn the cause.
    Error,
}

fn classify_start_failure(error: &str) -> StartFailureReport {
    if error == START_ABORTED {
        StartFailureReport::Abandoned
    } else {
        StartFailureReport::Error
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineState {
    Idle,
    /// A start is in flight. No `SessionManager` is running yet, so a stop or
    /// cancel here cannot be forwarded — it is recorded as
    /// [`LivePipeline::abort_pending_start`] instead and consumed by the start.
    Loading,
    Active,
}

/// How much of the pipeline a pending change actually invalidates.
///
/// Ordered: a heavier scope always wins when two changes queue up before the
/// next rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReloadScope {
    None,
    /// Everything built *around* the ASR backend — dictionary, styles, styling
    /// service, pipeline config. The loaded model is reused as-is, because
    /// none of these reach into it.
    Config,
    /// The ASR backend itself. Only a change to the backend, the device policy
    /// or the models directory needs this, and only this costs a model load.
    Full,
}

struct LivePipeline {
    paths: AppPaths,
    manager: Option<Arc<SessionManager<AsrRegistry>>>,
    /// The loaded ASR backend, kept so a `Config`-scope reload can hand the
    /// same one to the new `SessionManager` instead of loading it again.
    asr: Option<Arc<AsrRegistry>>,
    state: PipelineState,
    /// Set when a stop or cancel arrives during `Loading`. The in-flight start
    /// consumes it and abandons itself rather than opening a mic nobody asked
    /// to keep open.
    abort_pending_start: bool,
    session_id: Option<String>,
    reload: ReloadScope,
    /// The window focused when this dictation started, captured off the start
    /// path and consumed at delivery.
    focus_target: Option<WindowTarget>,
}

impl LivePipeline {
    /// Whether a start is in flight or a session is running. Both mean "do not
    /// begin another one, and do not rebuild underneath it".
    fn is_busy(&self) -> bool {
        self.state != PipelineState::Idle
    }

    /// Plan the next rebuild: the backend to carry forward into it, and
    /// whatever the pending reload invalidated.
    ///
    /// A [`ReloadScope::Full`] takes the loaded model *out of the pipeline
    /// here*, before the caller loads its replacement. Leaving it in place
    /// keeps it resident across the load — two whole models in device memory
    /// at the peak, which OOMs a card sized for exactly one, and reports as
    /// "the new model failed to load" rather than "you ran out".
    ///
    /// Both fields have to go: `manager` holds a second `Arc` to the same
    /// backend, so taking `asr` alone frees nothing. `is_busy()` is the
    /// caller's guarantee that no running session holds a third.
    ///
    /// This is why a failed rebuild has no old model to fall back on. Keeping
    /// one in reserve is exactly the double residency being fixed; the pending
    /// scope survives, so the next start retries the rebuild.
    fn take_reload_plan(&mut self) -> (Option<Arc<AsrRegistry>>, ReleasedBackend) {
        match self.reload {
            ReloadScope::Full => (
                None,
                ReleasedBackend {
                    manager: self.manager.take(),
                    asr: self.asr.take(),
                },
            ),
            ReloadScope::None | ReloadScope::Config => {
                (self.asr.clone(), ReleasedBackend::default())
            }
        }
    }

    /// Claim the pipeline for a start. `false` means one is already running or
    /// loading and the caller must not proceed.
    fn begin_start(&mut self) -> bool {
        if self.is_busy() {
            return false;
        }
        self.state = PipelineState::Loading;
        self.abort_pending_start = false;
        true
    }

    /// Record a stop or cancel that arrived before the start finished loading.
    /// `true` means it was recorded here, so there is no session to forward it
    /// to; the start will abandon itself.
    fn note_stop_during_start(&mut self) -> bool {
        if self.state == PipelineState::Loading {
            self.abort_pending_start = true;
            true
        } else {
            false
        }
    }

    /// Complete a start. `false` means a stop or cancel landed while loading,
    /// so the pipeline returns to `Idle` and no session is opened.
    fn finish_start(&mut self) -> bool {
        if self.abort_pending_start {
            self.abort_pending_start = false;
            self.state = PipelineState::Idle;
            return false;
        }
        self.state = PipelineState::Active;
        true
    }
}

/// Pipeline handles that a rebuild took out of service, on their way to being
/// dropped. Holding the last `Arc` to an [`AsrRegistry`] means holding the
/// model's device memory, so this exists to make the release an explicit step
/// rather than an incidental drop.
#[derive(Default)]
#[must_use = "dropping this inline stalls the async worker — pass it to `release_backend`"]
struct ReleasedBackend {
    manager: Option<Arc<SessionManager<AsrRegistry>>>,
    asr: Option<Arc<AsrRegistry>>,
}

impl ReleasedBackend {
    /// Strip a built-but-unwanted pipeline down to the parts worth releasing.
    /// A rebuild that loses a race is discarded whole, and the model it just
    /// loaded goes with it.
    fn from_discarded(pipeline: &mut LivePipeline) -> Self {
        Self {
            manager: pipeline.manager.take(),
            asr: pipeline.asr.take(),
        }
    }

    fn is_empty(&self) -> bool {
        self.manager.is_none() && self.asr.is_none()
    }
}

/// Drop released pipeline handles off the async workers, and wait for it.
///
/// `WhisperContext`'s destructor frees the model's weights. On CUDA that is a
/// `cudaFree` of gigabytes and it blocks — the same reason the matching *load*
/// runs on `spawn_blocking` rather than inline.
///
/// Awaited rather than detached because [`LivePipeline::take_reload_plan`]
/// depends on it: releasing the old model before loading the new one only
/// avoids double residency if the free has finished by the time the load
/// starts.
async fn release_backend(released: ReleasedBackend) {
    if released.is_empty() {
        return;
    }
    if tokio::task::spawn_blocking(move || drop(released))
        .await
        .is_err()
    {
        warn!("backend release task failed; device memory may not have been reclaimed");
    }
}

struct SessionStreamSlot {
    id: u64,
    channel: Channel<SessionStreamEvent>,
}

/// How long to wait for a microphone to open before giving up on it.
///
/// `AudioStream::new` is synchronous all the way down to ALSA/PipeWire and has
/// no bound of its own: a hung sound server, or a USB mic that enumerates but
/// never answers, blocks its caller forever. On the startup path that caller
/// is `AppRuntime::new`, which Tauri's `setup()` runs inside `block_on` before
/// the event loop exists — so an unbounded open means no window, no tray and
/// no hotkey, ever.
const CAPTURE_OPEN_TIMEOUT: Duration = Duration::from_secs(5);

/// How long to wait for the delivery thread to report what it typed.
///
/// Typing is paced per key and long text legitimately takes seconds, so this
/// is generous. It bounds a driver that never answers at all, not slow typing.
const DELIVERY_OUTCOME_TIMEOUT: Duration = Duration::from_secs(120);

/// The warning for a dictation that did not reach the focused app.
fn undelivered_message(on_clipboard: bool) -> &'static str {
    if on_clipboard {
        "Dictation was not typed into the app — it is on the clipboard, paste it with Ctrl+V"
    } else {
        "Dictation was not delivered — the clipboard is unavailable and \
         the text cannot be typed safely"
    }
}

/// How long a dictation start waits for delivery to finish starting.
///
/// Covers the hotkey being pressed in the moments after launch, where the
/// driver is a few milliseconds from ready. Deliberately short: a start that
/// is still pending after this is waiting on something the user has to act on
/// — the polkit prompt for uinput access — and an error that says so beats a
/// hotkey that appears to do nothing.
const DELIVERY_START_GRACE: Duration = Duration::from_secs(2);

/// Open a capture stream without blocking the async runtime, and without
/// waiting on the driver indefinitely.
///
/// A timed-out open leaves its blocking-pool thread parked in the driver until
/// the driver returns. That costs one pool thread; the alternative costs the
/// caller, which on the startup path is the whole app.
async fn open_capture_bounded(config: AudioConfig) -> Result<AudioStream, String> {
    let open = tokio::task::spawn_blocking(move || AudioStream::new(config));
    match tokio::time::timeout(CAPTURE_OPEN_TIMEOUT, open).await {
        Ok(Ok(Ok(stream))) => Ok(stream),
        Ok(Ok(Err(e))) => Err(e.to_string()),
        Ok(Err(e)) => Err(format!("capture open task failed: {e}")),
        Err(_) => Err(format!(
            "the microphone did not open within {}s",
            CAPTURE_OPEN_TIMEOUT.as_secs()
        )),
    }
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
        let mut loaded = load_settings_checked(&paths)?;
        crate::persistence::migrate_input_device(&paths, &mut loaded).await;
        let settings = loaded.settings.clone();

        // Retention runs once per launch, not once per window load: with
        // `start_minimized` and a hotkey-only workflow the main window may
        // never open, and on Linux the pill is drawn natively rather than in a
        // webview, so nothing else would ever trigger it.
        match Database::open(&paths.database_file()) {
            Ok(mut db) => {
                crate::persistence::enforce_retention(&mut db, &settings, loaded.trust);
                // A previous run that was killed outright left its dictation
                // row open; nothing in that process survived to close it.
                match db.close_orphaned_sessions() {
                    Ok(0) => {}
                    Ok(closed) => {
                        tracing::info!(closed, "closed dictations left open by a previous run");
                    }
                    Err(e) => tracing::warn!(error = %e, "closing orphaned sessions failed"),
                }
            }
            Err(e) => tracing::warn!(error = %e, "open DB for startup maintenance failed"),
        }

        let models_ready =
            lune_persist::has_required_models(&paths.models_dir(), &settings.asr_backend);

        let (pipeline, llm_service, settings) =
            initial_pipeline(paths, settings, models_ready).await;

        let terminal_mode = settings.read().await.terminal_mode;
        let delivery = DeliveryService::new(terminal_mode)?;

        let runtime = Self {
            inner: Arc::new(RuntimeInner {
                app,
                pipeline: Mutex::new(pipeline),
                start_gate: Mutex::new(()),
                status: RwLock::new("ready".into()),
                session_stream: RwLock::new(None),
                session_stream_seq: AtomicU64::new(0),
                settings: Arc::clone(&settings),
                llm_service: RwLock::new(llm_service),
                delivery,
                last_toggle_at: Mutex::new(Instant::now() - TOGGLE_DEBOUNCE),
                last_cancel_at: Mutex::new(Instant::now() - CANCEL_DEBOUNCE),
                last_activity: Mutex::new(Instant::now()),
                prewarmed_audio: Mutex::new(None),
                capture_open: Mutex::new(()),
                capture_fallback_unreported: std::sync::atomic::AtomicBool::new(false),
            }),
        };

        runtime.sync_capture_device(None).await;
        runtime.spawn_idle_unload_watchdog();

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

    /// Push the saved idle-unload timeout into `inner.settings` immediately.
    ///
    /// The watchdog reads it from there on every tick, and `inner.settings` is
    /// otherwise stale until the next pipeline rebuild — so without this,
    /// switching the feature on would not take effect until the dictation
    /// after next. Same reason and same shape as [`Self::sync_sound_cues`].
    pub async fn sync_idle_unload(&self, minutes: Option<u32>) {
        self.inner.settings.write().await.model_idle_unload_minutes = minutes;
    }

    /// Reconcile the long-lived capture stream with the desired input device.
    ///
    /// The stream itself is unconditional — it is what makes the start path
    /// record from the instant the hotkey is pressed, across a model reload
    /// included — so this only ever opens or rebuilds one, never closes it.
    ///
    /// `device` is `Option<Option<String>>`: the outer `None` reads the current
    /// setting (suitable for startup), `Some(d)` overrides with device `d`
    /// (where the inner `None` means the system default). The override exists
    /// because `inner.settings` is stale after a save until the next pipeline
    /// rebuild.
    pub async fn sync_capture_device(&self, device: Option<Option<String>>) {
        let input_device = match device {
            Some(d) => d,
            None => self.inner.settings.read().await.input_device.clone(),
        };
        // One sync at a time. The open itself is serialized here rather than
        // under `prewarmed_audio`, which is deliberately released across it:
        // holding *that* mutex over the open makes every later
        // `prewarmed_handle()` await forever — every future dictation, plus
        // the settings-save IPC that reaches here.
        let _opening = self.inner.capture_open.lock().await;
        // Decide under the slot lock, open outside it.
        let action = {
            let slot = self.inner.prewarmed_audio.lock().await;
            let open = slot.as_ref().map(|p| p.device.as_deref());
            let healthy = slot.as_ref().is_none_or(|p| !p.stream.has_failed());
            if !healthy {
                warn!("mic prewarm: the open capture stream reported an error; rebuilding it");
            }
            reconcile_prewarm(open, input_device.as_deref(), healthy)
        };
        match action {
            PrewarmAction::Keep => return,
            PrewarmAction::Reopen => {
                debug!("mic prewarm: input device changed, rebuilding capture stream");
            }
            PrewarmAction::Open => {}
        }

        // Drop any existing stream first so its device handle is freed before
        // we open the new one. Until the new one lands, the start path falls
        // back to `open_capture_for_start`.
        *self.inner.prewarmed_audio.lock().await = None;

        let config = AudioConfig {
            input_device: input_device.clone(),
            ..AudioConfig::default()
        };
        match open_capture_bounded(config).await {
            Ok(stream) => {
                debug!("mic prewarm: capture stream opened");
                self.note_capture_fallback(stream.fell_back());
                // `capture_open` is still held, so no other sync can have
                // installed a stream between the clear above and this store.
                *self.inner.prewarmed_audio.lock().await = Some(PrewarmedStream {
                    stream,
                    device: input_device,
                });
            }
            Err(e) => warn!("mic prewarm: failed to open capture stream: {e}"),
        }
    }

    async fn prewarmed_handle(&self) -> Option<lune_audio::AudioStreamHandle> {
        let mut slot = self.inner.prewarmed_audio.lock().await;
        // A stream cpal has reported an error on still hands out handles, but
        // they never produce a chunk — and nothing rebuilds the stream until
        // the next settings save. Drop it here so the start path falls back to
        // `open_capture_for_start` and the dictation actually records, instead
        // of running the 5 s starvation watchdog on a dead ring.
        //
        // Dropped rather than merely skipped: a dead stream still owns its
        // device handle and capture thread, and the fallback is about to open
        // the same device. Leaving it in place would keep that handle for the
        // life of the process (nothing rebuilds it until a settings save) and
        // make every later start race it for an exclusive capture device.
        if slot.as_ref().is_some_and(|p| p.stream.has_failed()) {
            warn!("mic prewarm: the open capture stream reported an error; dropping it");
            *slot = None;
        }
        slot.as_ref().map(|p| p.stream.handle())
    }

    /// Open a capture stream for one session, for the case where the
    /// long-lived stream is missing — its device failed to open, or was
    /// unplugged and has not been rebuilt yet.
    ///
    /// Called on the start path *before* the pipeline is prepared, so its ring
    /// (≈10 s) is already filling while a model loads. The session would
    /// otherwise open the mic itself inside `run_capture` — after the load —
    /// and everything said in the meantime would be gone.
    ///
    /// `None` on failure, which restores the old behaviour exactly: the
    /// session opens its own stream and reports the error the same way it
    /// always did. Nothing here is worth failing a dictation over.
    async fn open_capture_for_start(&self) -> Option<AudioStream> {
        let input_device = self.inner.settings.read().await.input_device.clone();
        let config = AudioConfig {
            input_device,
            ..AudioConfig::default()
        };
        match open_capture_bounded(config).await {
            Ok(stream) => {
                self.note_capture_fallback(stream.fell_back());
                Some(stream)
            }
            Err(e) => {
                debug!("start: could not pre-open capture stream ({e}); session will open its own");
                None
            }
        }
    }

    /// Return the persisted `start_minimized` preference. Read at app
    /// startup to decide whether to hide the main window on launch.
    pub async fn start_minimized(&self) -> bool {
        self.inner.settings.read().await.start_minimized
    }

    /// Return the persisted `check_updates_on_launch` preference. Read at app
    /// startup to decide whether to contact the release endpoint at all.
    pub async fn check_updates_on_launch(&self) -> bool {
        self.inner.settings.read().await.check_updates_on_launch
    }

    /// Return the persisted pill-visibility preference. Read at app startup to
    /// decide whether to create the pill overlay window at all (Never = none).
    pub async fn pill_visibility(&self) -> PillVisibility {
        self.inner.settings.read().await.pill_visibility
    }

    /// Immediately map or unmap the pill to match a (possibly just-changed)
    /// visibility setting against the live service status — so toggling the
    /// setting takes effect at once instead of waiting for the next status
    /// transition (i.e. the next dictation). Mirrors the show/hide decision in
    /// `publish_status`. Takes `vis` explicitly because `inner.settings` stays
    /// stale after a save until the next pipeline rebuild.
    pub async fn reconcile_pill_visibility(&self, vis: PillVisibility) {
        let status = self.inner.status.read().await.clone();
        if crate::pill::should_show_pill(vis, &status) {
            crate::pill::show_pill_direct(&self.inner.app);
        } else {
            crate::pill::hide_pill_direct(&self.inner.app);
        }
    }

    /// Return the persisted pill position, if the user has dragged it. Read at
    /// startup to seed the overlay's placement before its first show.
    pub async fn pill_position(&self) -> Option<lune_core::config::PillPosition> {
        self.inner.settings.read().await.pill_position
    }

    /// The pill overlay config derived from persisted settings. Read at startup
    /// to seed the overlay's geometry/placement before the window is created.
    pub async fn pill_overlay_config(&self) -> crate::pill::PillOverlayConfig {
        crate::pill::PillOverlayConfig::from_settings(&*self.inner.settings.read().await)
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

    async fn request_reload(&self, scope: ReloadScope) {
        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.reload = pipeline.reload.max(scope);
    }

    /// A dictionary, snippet or style edit. None of these reach into the ASR
    /// model, so the loaded backend is carried forward and the next dictation
    /// starts without a model load.
    pub async fn mark_config_changed(&self) {
        self.request_reload(ReloadScope::Config).await;
    }

    /// A settings save. Only a change to the transcription backend or the
    /// device policy invalidates the loaded model; everything else on the
    /// settings page is config-scope.
    pub async fn mark_settings_changed(&self, next: &UserSettings) {
        let scope = {
            let current = self.inner.settings.read().await;
            if next.device != current.device
                || asr_backend_needs_reload(&current.asr_backend, &next.asr_backend)
            {
                ReloadScope::Full
            } else {
                ReloadScope::Config
            }
        };
        self.request_reload(scope).await;
    }

    pub async fn eager_rebuild(&self) -> Result<(), String> {
        // Same gate as `prepare_manager_for_start` and `reload_pipeline_if_required`:
        // this is the third path that rebuilds outside the `pipeline` lock, so
        // without it a post-download rebuild can run concurrently with a start's
        // build (two model loads) or with an idle unload — which would then be
        // silently undone when this installs its freshly built pipeline.
        let _start_guard = self.inner.start_gate.lock().await;

        let (paths, reuse_asr, released) = {
            let mut pipeline = self.inner.pipeline.lock().await;
            if pipeline.is_busy() {
                return Ok(());
            }
            let (reuse_asr, released) = pipeline.take_reload_plan();
            (pipeline.paths.clone(), reuse_asr, released)
        };
        release_backend(released).await;
        let settings = load_settings(&paths)?;
        let (mut rebuilt, new_llm_service, new_settings) =
            build_pipeline(paths, settings, reuse_asr)
                .await
                .map_err(|e| format!("eager rebuild: {e}"))?;
        let mut pipeline = self.inner.pipeline.lock().await;
        if pipeline.is_busy() {
            // A session claimed the pipeline while we built. This rebuild is
            // dead, and it is holding a freshly loaded model.
            let discarded = ReleasedBackend::from_discarded(&mut rebuilt);
            drop(pipeline);
            release_backend(discarded).await;
            return Ok(());
        }
        *pipeline = rebuilt;
        *self.inner.llm_service.write().await = new_llm_service;
        let new_settings_val = new_settings.read().await.clone();
        *self.inner.settings.write().await = new_settings_val;
        tracing::info!("pipeline pre-warmed");
        Ok(())
    }

    /// The delivery health, giving a still-starting thread a moment to settle.
    async fn delivery_health_settled(&self) -> DeliveryHealth {
        let deadline = Instant::now() + DELIVERY_START_GRACE;
        loop {
            let health = self.inner.delivery.health();
            if health != DeliveryHealth::Starting || Instant::now() >= deadline {
                return health;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    pub async fn start_dictation(&self) -> Result<String, String> {
        match self.delivery_health_settled().await {
            DeliveryHealth::Ready => {}
            DeliveryHealth::Starting => {
                return Err(
                    "Text delivery is still starting. If a permission prompt is \
                            open, finish it and press the hotkey again."
                        .into(),
                );
            }
            DeliveryHealth::Stopped => {
                return Err("text delivery service is unavailable — cannot start dictation".into());
            }
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
        // Nothing prewarmed: open the mic *here*, ahead of the model load,
        // rather than letting the session open it afterwards. Both paths then
        // buffer into a ring from the moment the key was pressed, so a reload
        // — an idle unload's especially, at ~160 ms warm and ~750 ms cold — is
        // spent recording rather than deaf.
        let fallback_stream = match prewarmed {
            Some(_) => None,
            None => self.open_capture_for_start().await,
        };

        // Capture focus off the start path: the value is only consumed at
        // delivery time (many seconds later), but the underlying Wayland /
        // subprocess probes are synchronous and block the tokio worker for
        // ~400–500 ms on COSMIC. Letting them race the audio bring-up
        // unblocks the first VAD chunk by the same amount.
        let focus_capture = tokio::task::spawn_blocking(capture_focus_target);

        // Snapshot the clipboard now, off the paste hot path: on the uinput
        // paste backend the restore-after-paste read can block up to 2s on a
        // slow clipboard owner, so priming here overlaps it with recording +
        // ASR (mirrors the focus capture above). Unconditional, because every
        // delivery mode can end up pasting — per-key typing routes anything
        // long, non-ASCII or multi-line to the clipboard, and taking the
        // snapshot at that point would put the stall right back on the
        // delivery path. No-op on backends that never touch the clipboard
        // (IBus, macOS/Windows).
        self.inner.delivery.prime_clipboard();

        let manager = match self.prepare_manager_for_start().await {
            Ok(m) => m,
            Err(e) => {
                self.report_start_failure(&e).await;
                return Err(e);
            }
        };
        let (event_tx, event_rx) = mpsc::channel(256);
        let audio = Self::session_audio(prewarmed, fallback_stream);
        let start_result = manager.start(event_tx, audio, SessionKind::Dictate).await;
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

        self.report_capture_fallback().await;
        self.push_session_event(SessionStreamEvent::SessionStarted {
            session_id: session_id.clone(),
        })
        .await;
        let focus_target = focus_capture.await.ok();
        debug!(?focus_target, "captured focus target for session");
        self.inner.pipeline.lock().await.focus_target = focus_target;

        self.spawn_event_forwarder(event_rx, false);

        Ok(session_id)
    }

    /// Where a starting session reads audio: the pre-warmed ring when one is
    /// live, else the stream opened for this start, else the session opens
    /// its own.
    fn session_audio(
        prewarmed: Option<lune_audio::AudioStreamHandle>,
        fallback: Option<AudioStream>,
    ) -> SessionAudio {
        match (prewarmed, fallback) {
            (Some(handle), _) => SessionAudio::Shared(handle),
            (None, Some(stream)) => SessionAudio::Owned(stream),
            (None, None) => SessionAudio::Open,
        }
    }

    /// Start a capture-only session for the snippet trigger recorder. Speech
    /// is transcribed exactly like dictation, but the *raw* transcript is
    /// streamed back to the UI as `CaptureText` — nothing is typed into the
    /// focused app and no history row is written. Stopped/cancelled through
    /// the ordinary `stop_dictation` / `cancel_dictation` paths.
    pub async fn start_trigger_capture(&self) -> Result<String, String> {
        // A previous dictation's synthetic keystrokes could re-activate the
        // capture button; refuse to start until delivery has settled (same
        // feedback-loop guard as start_dictation).
        if self.inner.delivery.is_busy() {
            debug!("start_trigger_capture suppressed — text delivery in progress");
            return Err("text delivery in progress".into());
        }

        self.publish_status("listening").await;

        let prewarmed = self.prewarmed_handle().await;
        if let Some(handle) = &prewarmed {
            handle.drain_keep_last(6);
        }
        // Same reason as `start_dictation`: get a ring filling before the
        // (possibly reloading) pipeline prep, not after it.
        let fallback_stream = match prewarmed {
            Some(_) => None,
            None => self.open_capture_for_start().await,
        };

        let manager = match self.prepare_manager_for_start().await {
            Ok(m) => m,
            Err(e) => {
                self.report_start_failure(&e).await;
                return Err(e);
            }
        };
        let (event_tx, event_rx) = mpsc::channel(256);
        let audio = Self::session_audio(prewarmed, fallback_stream);
        let start_result = manager
            .start(event_tx, audio, SessionKind::CaptureRaw)
            .await;
        let session_id = match start_result {
            Ok(session_id) => session_id,
            Err(error) => {
                self.set_pipeline_session(None, false).await;
                self.publish_status("error").await;
                return Err(format!("start trigger capture: {error}"));
            }
        };

        let session_id = session_id.to_string();
        self.set_pipeline_session(Some(session_id.clone()), true)
            .await;
        self.report_capture_fallback().await;
        self.push_session_event(SessionStreamEvent::SessionStarted {
            session_id: session_id.clone(),
        })
        .await;
        self.spawn_event_forwarder(event_rx, true);

        Ok(session_id)
    }

    pub async fn stop_dictation(&self) -> Result<(), String> {
        if self.defer_stop_to_pending_start().await {
            return Ok(());
        }
        if let Some(manager) = self.active_manager().await {
            manager.stop().await;
        }
        Ok(())
    }

    pub async fn stop_and_wait_dictation(&self) -> Result<(), String> {
        if self.defer_stop_to_pending_start().await {
            return Ok(());
        }
        if let Some(manager) = self.active_manager().await {
            manager.stop_and_wait().await;
        }
        Ok(())
    }

    pub async fn is_active(&self) -> bool {
        self.inner.pipeline.lock().await.state == PipelineState::Active
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
        if self.defer_stop_to_pending_start().await {
            return Ok(());
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

        // `Loading` counts as "on": a toggle press during the model load is
        // the user turning it back off, not asking for a second session.
        let is_on = self.inner.pipeline.lock().await.is_busy();

        if is_on {
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

    /// `capture` marks a capture-only session: `FinalizedText` is routed to
    /// the UI as `CaptureText` instead of being delivered, and no
    /// history-changed invalidation fires (nothing was persisted).
    fn spawn_event_forwarder(&self, mut event_rx: mpsc::Receiver<SessionEvent>, capture: bool) {
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
                    SessionEvent::Processing => {
                        runtime.publish_status("processing").await;
                        runtime
                            .push_session_event(SessionStreamEvent::Processing)
                            .await;
                    }
                    SessionEvent::FinalizedText { session_id, text } => {
                        saw_text = true;
                        if capture {
                            runtime
                                .push_session_event(SessionStreamEvent::CaptureText { text })
                                .await;
                            continue;
                        }
                        let (mode, copy) = {
                            let s = runtime.inner.settings.read().await;
                            (s.dictation_mode, s.copy_to_clipboard)
                        };
                        let target = runtime.inner.pipeline.lock().await.focus_target.clone();
                        let outcome_rx = match target {
                            Some(ref t) => runtime.inner.delivery.deliver_to_tracked(
                                Some(session_id),
                                &text,
                                mode,
                                copy,
                                t,
                            ),
                            None => runtime.inner.delivery.deliver_tracked(
                                Some(session_id),
                                &text,
                                mode,
                                copy,
                            ),
                        };
                        match outcome_rx {
                            // Enqueue failed — the delivery thread is gone, so
                            // nothing will ever be typed.
                            None => {
                                // Non-fatal: surface as a warning toast, not a
                                // SessionError. SessionError flips serviceStatus to
                                // 'error' and is then immediately overwritten by the
                                // FinalizedText below (serviceStatus 'ready'), which
                                // made the status badge flash red for one frame.
                                runtime
                                    .push_session_event(SessionStreamEvent::SessionWarning {
                                        message:
                                            "Text delivery failed — output service is unavailable"
                                                .into(),
                                    })
                                    .await;
                            }
                            // Queued. Typing is paced and can take seconds, so
                            // wait for the real outcome off this loop and warn
                            // only if the driver actually dropped the text
                            // (clipboard tooling missing and the text is unsafe
                            // to type key-by-key — see `fallback_to_keys`).
                            Some(rx) => {
                                let runtime = runtime.clone();
                                tauri::async_runtime::spawn(async move {
                                    let outcome = tauri::async_runtime::spawn_blocking(move || {
                                        // Bounded: a driver wedged in the
                                        // kernel or on the clipboard never
                                        // sends, and an unbounded recv leaks
                                        // one blocking-pool thread per
                                        // dictation until the pool's 512-thread
                                        // cap stalls all blocking work.
                                        rx.recv_timeout(DELIVERY_OUTCOME_TIMEOUT).ok()
                                    })
                                    .await
                                    .ok()
                                    .flatten();
                                    if let Some(outcome) = outcome {
                                        if !outcome.delivered {
                                            runtime
                                                .push_session_event(
                                                    SessionStreamEvent::SessionWarning {
                                                        message: undelivered_message(
                                                            outcome.clipboard_has_text,
                                                        )
                                                        .into(),
                                                    },
                                                )
                                                .await;
                                        }
                                    }
                                });
                            }
                        }
                        runtime
                            .push_session_event(SessionStreamEvent::FinalizedText { text })
                            .await;
                    }
                    SessionEvent::AudioLevel(level) => {
                        // Send to main window via session stream
                        runtime
                            .push_session_event(SessionStreamEvent::AudioLevel { level })
                            .await;
                        // And into the pill overlay, which reads it separately
                        // from the main window's session stream.
                        crate::pill::publish_audio_level(&app, level);
                    }
                    SessionEvent::CommandDetected {
                        session_id: _,
                        payload,
                    } => {
                        runtime.handle_voice_command(&payload).await;
                        runtime
                            .push_session_event(SessionStreamEvent::CommandDetected { payload })
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
                    SessionEvent::Warning(message) => {
                        warn!(warning = %message, "session warning");
                        runtime
                            .push_session_event(SessionStreamEvent::SessionWarning {
                                message: sanitize_error_for_frontend(&message),
                            })
                            .await;
                    }
                    SessionEvent::Stopped { error_dropped } => {
                        runtime.finish_session().await;
                        // The session failed but its `Error` never reached us
                        // (this consumer stalled past the send timeout).
                        // Reporting "ready" here would show a failed dictation
                        // as a clean one, so surface a generic failure instead.
                        if error_dropped && !saw_error {
                            saw_error = true;
                            warn!("session ended in error but the error event was dropped");
                            runtime.publish_status("error").await;
                            runtime
                                .push_session_event(SessionStreamEvent::SessionError {
                                    message: DROPPED_ERROR_MESSAGE.to_owned(),
                                })
                                .await;
                        }
                        if !saw_error {
                            runtime.publish_status("ready").await;
                        }
                        runtime
                            .push_session_event(SessionStreamEvent::SessionStopped)
                            .await;
                        // Only reload history if the session produced text —
                        // empty/cancelled sessions shouldn't trigger UI churn.
                        // Capture sessions never persist, so never invalidate.
                        if saw_text
                            && !capture
                            && let Err(e) = app.emit("lune://history-changed", ())
                        {
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
    /// Nothing is acted on here. Both commands are applied in the pipeline
    /// itself (`session.rs`): `SwitchStyle` swaps the style preset,
    /// `InsertSnippet` expands a snippet. The payload is still parsed so a
    /// malformed one is reported rather than silently dropped.
    async fn handle_voice_command(&self, payload: &str) {
        match serde_json::from_str::<VoiceCommand>(payload) {
            Ok(cmd) => debug!(?cmd, "CommandDetected: no runtime action for this command"),
            Err(e) => {
                warn!(error = %e, payload, "CommandDetected: failed to parse VoiceCommand JSON");
            }
        }
    }

    async fn finish_session(&self) {
        self.set_pipeline_session(None, false).await;
    }

    /// Play a dictation sound cue if cues are enabled, at the saved volume.
    async fn play_cue(&self, cue: lune_voice::sound::Cue) {
        let (enabled, volume) = {
            let s = self.inner.settings.read().await;
            (s.sound_cues_enabled, s.sound_cue_volume)
        };
        if enabled {
            lune_voice::sound::play(cue, volume);
        }
    }

    /// Record whether the capture stream we just opened is on the configured
    /// microphone. A fallback arms one toast; a clean open disarms it, so the
    /// device coming back is not reported as a problem.
    fn note_capture_fallback(&self, fell_back: bool) {
        self.inner
            .capture_fallback_unreported
            .store(fell_back, Ordering::Release);
    }

    /// Tell the user once that the dictation is running on the wrong mic.
    ///
    /// Without this the configured USB headset can be unplugged and every
    /// later dictation comes from the laptop mic — degraded transcripts,
    /// permanently, with nothing on screen saying why.
    async fn report_capture_fallback(&self) {
        if self
            .inner
            .capture_fallback_unreported
            .swap(false, Ordering::AcqRel)
        {
            self.push_session_event(SessionStreamEvent::SessionWarning {
                message: "Your selected microphone was not found — recording from the system \
                          default instead."
                    .into(),
            })
            .await;
        }
    }

    /// Undo the optimistic "listening" after prep failed, and tell the user
    /// when it was a failure rather than their own stop.
    ///
    /// Only touches the pill when nothing is actually running: losing the race
    /// to another start means a session *is* live, and publishing over it
    /// would make the UI lie.
    async fn report_start_failure(&self, error: &str) {
        let report = classify_start_failure(error);
        if report == StartFailureReport::Abandoned {
            debug!("dictation start abandoned — stop arrived during the pipeline load");
        }
        if self.inner.pipeline.lock().await.is_busy() {
            return;
        }
        match report {
            StartFailureReport::Abandoned => self.publish_status("ready").await,
            StartFailureReport::Error => {
                warn!(error = %error, "dictation start failed");
                self.publish_status("error").await;
                self.push_session_event(SessionStreamEvent::SessionError {
                    message: sanitize_error_for_frontend(error),
                })
                .await;
            }
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
        crate::pill::publish_status(&self.inner.app, status);
        // Pill visibility is decided entirely here — the single source of truth,
        // from the live status + the setting — then the window is mapped/unmapped
        // directly (bypassing the frontend IPC roundtrip for instant response).
        // The frontend only renders content, so the mapped window and its content
        // can't desync.
        let vis = self.inner.settings.read().await.pill_visibility;
        if crate::pill::should_show_pill(vis, status) {
            crate::pill::show_pill_direct(&self.inner.app);
        } else {
            crate::pill::hide_pill_direct(&self.inner.app);
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
        if pipeline.state == PipelineState::Active {
            pipeline.manager.as_ref().map(Arc::clone)
        } else {
            None
        }
    }

    async fn prepare_manager_for_start(&self) -> Result<Arc<SessionManager<AsrRegistry>>, String> {
        // Serialize the whole reload+activate section. The `pipeline` mutex is
        // released across `build_pipeline` (seconds of model loading), so two
        // concurrent starts — IPC from either window, or a Hold-mode key-down,
        // none of which are mutually exclusive on their own — could both pass
        // the Phase 1 check and both load the model. On CUDA that is two
        // concurrent allocations of the same model and can OOM the GPU. The
        // loser now waits here and takes the winner's freshly-built pipeline.
        let _start_guard = self.inner.start_gate.lock().await;

        // Phase 1: claim the pipeline and decide whether a rebuild is needed
        // (short lock). Entering `Loading` here is what makes a stop or cancel
        // arriving during the rebuild land somewhere instead of on the floor.
        let reload_plan = {
            let mut pipeline = self.inner.pipeline.lock().await;
            if !pipeline.begin_start() {
                return Err(ALREADY_ACTIVE.into());
            }
            let needs_rebuild = pipeline.reload != ReloadScope::None || pipeline.manager.is_none();
            needs_rebuild.then(|| {
                let (reuse_asr, released) = pipeline.take_reload_plan();
                (pipeline.paths.clone(), reuse_asr, released)
            })
        };

        // Phase 2: If a rebuild is needed, do it outside the lock, then
        // re-acquire and atomically swap + activate in one critical section.
        if let Some((paths, reuse_asr, released)) = reload_plan {
            // Before the load, not after: a model change would otherwise hold
            // the outgoing model's device memory for the whole of the new
            // model's allocation.
            release_backend(released).await;
            let settings = match load_settings(&paths) {
                Ok(s) => s,
                Err(e) => {
                    self.abandon_pending_start().await;
                    return Err(e);
                }
            };
            let (rebuilt, new_llm_service, new_settings) =
                match build_pipeline(paths, settings, reuse_asr).await {
                    Ok(v) => v,
                    Err(e) => {
                        self.abandon_pending_start().await;
                        return Err(format!("Pipeline build failed: {e}"));
                    }
                };
            let mut pipeline = self.inner.pipeline.lock().await;
            // The freshly built pipeline is installed either way: the work is
            // done, and the next start must not repeat it. `rebuilt` starts
            // `Idle`, so carry the pending abort across the swap.
            let aborted = pipeline.abort_pending_start;
            *pipeline = rebuilt;
            pipeline.abort_pending_start = aborted;
            *self.inner.llm_service.write().await = new_llm_service;
            let new_settings_val = new_settings.read().await.clone();
            *self.inner.settings.write().await = new_settings_val;
            // The user let go of the key (or cancelled) while the model was
            // loading. Honour that rather than open a mic they already asked to
            // close.
            if !pipeline.finish_start() {
                return Err(START_ABORTED.into());
            }
            // `finish_start` already moved to `Active`, so a bail-out here has
            // to walk it back: a pipeline left `Active` with no manager refuses
            // every later start and has nothing to forward a stop to.
            let Some(mgr) = pipeline.manager.as_ref() else {
                pipeline.state = PipelineState::Idle;
                return Err("build_pipeline succeeded but manager is None".into());
            };
            return Ok(Arc::clone(mgr));
        }

        // Phase 3: No rebuild — activate the pipeline already in hand.
        let mut pipeline = self.inner.pipeline.lock().await;
        if pipeline.manager.is_none() {
            pipeline.state = PipelineState::Idle;
            pipeline.abort_pending_start = false;
            return Err("Models not downloaded. Please complete model setup first.".into());
        }
        if !pipeline.finish_start() {
            return Err(START_ABORTED.into());
        }
        let Some(mgr) = pipeline.manager.as_ref() else {
            pipeline.state = PipelineState::Idle;
            return Err("pipeline manager is None".into());
        };
        Ok(Arc::clone(mgr))
    }

    /// Drop out of `Loading` after a start that never got off the ground.
    async fn abandon_pending_start(&self) {
        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.state = PipelineState::Idle;
        pipeline.abort_pending_start = false;
    }

    /// Record a stop or cancel that arrived while a start was still loading.
    ///
    /// Returns `true` when it was recorded, which tells the caller there is no
    /// running session to forward the request to — the start will abandon
    /// itself when it next looks.
    async fn defer_stop_to_pending_start(&self) -> bool {
        self.inner.pipeline.lock().await.note_stop_during_start()
    }

    /// Rebuild the pipeline now when a settings change is pending, so callers
    /// that consume `llm_service()` outside a dictation session (style example
    /// generation) don't run against a stale service — e.g. the Null
    /// passthrough kept from before the styling model was downloaded. Skipped
    /// while a session is active; the lazy rebuild on the next dictation start
    /// covers that case. A failed rebuild is logged, not fatal: the styling
    /// path's own readiness checks produce the user-facing error.
    async fn reload_pipeline_if_required(&self) {
        // Same gate as `prepare_manager_for_start`: this is the second path
        // that rebuilds outside the `pipeline` lock, so without it a style
        // example generation racing a dictation start still produces two
        // concurrent `build_pipeline` calls (two GPU model allocations).
        let _start_guard = self.inner.start_gate.lock().await;

        let reload_plan = {
            let mut pipeline = self.inner.pipeline.lock().await;
            if pipeline.is_busy() {
                return;
            }
            if pipeline.reload != ReloadScope::None || pipeline.manager.is_none() {
                let (reuse_asr, released) = pipeline.take_reload_plan();
                Some((pipeline.paths.clone(), reuse_asr, released))
            } else {
                None
            }
        };
        let Some((paths, reuse_asr, released)) = reload_plan else {
            return;
        };
        release_backend(released).await;
        let settings = match load_settings(&paths) {
            Ok(s) => s,
            Err(error) => {
                warn!(%error, "styles: settings load failed; keeping current pipeline");
                return;
            }
        };
        match build_pipeline(paths, settings, reuse_asr).await {
            Ok((mut rebuilt, new_llm_service, new_settings)) => {
                let mut pipeline = self.inner.pipeline.lock().await;
                if pipeline.is_busy() {
                    // A session started while we rebuilt; drop this rebuild and
                    // let the next dictation start redo it. It is holding a
                    // freshly loaded model, so release it off the workers.
                    let discarded = ReleasedBackend::from_discarded(&mut rebuilt);
                    drop(pipeline);
                    release_backend(discarded).await;
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
    /// apply any pending pipeline reload. The remote provider's readiness is
    /// configuration-only, so this is a reload check, not an initialization
    /// wait. Configuration-level checks (`styling_llm_readiness`) are the
    /// caller's preflight.
    pub async fn ensure_styling_llm_ready(&self) -> Result<(), String> {
        self.reload_pipeline_if_required().await;
        Ok(())
    }

    async fn set_pipeline_session(&self, session_id: Option<String>, active: bool) {
        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.session_id = session_id;
        pipeline.state = if active {
            PipelineState::Active
        } else {
            PipelineState::Idle
        };
        pipeline.abort_pending_start = false;
        // Every session start and end passes through here, so this is the one
        // place the idle clock has to be reset — no call site can forget it.
        *self.inner.last_activity.lock().await = Instant::now();
    }

    /// Free the loaded ASR model after a long enough gap between dictations,
    /// giving back the process's whole model footprint (514 MB on the default
    /// Medium) until the next hotkey press reloads it.
    ///
    /// A polling watchdog rather than a timer armed at each session end: the
    /// timeout can be changed in the settings UI and the model can be loaded or
    /// released by half a dozen other paths, and re-reading that state on a
    /// tick is simpler — and has fewer ways to be wrong — than keeping a timer
    /// in sync with all of them. [`idle_unload_decision`] holds the policy and
    /// picks each sleep, so the loop below is only a clock.
    fn spawn_idle_unload_watchdog(&self) {
        let runtime = self.clone();
        tauri::async_runtime::spawn(async move {
            loop {
                let timeout = runtime
                    .inner
                    .settings
                    .read()
                    .await
                    .model_idle_unload_minutes
                    // Zero means off, not "unload the moment a dictation ends".
                    // `apply_settings_dto` normalizes it on the save path, but a
                    // hand-edited settings.toml never passes through that — and
                    // a zero timeout here would free the model after every
                    // utterance and pay a reload on the next one.
                    .filter(|&m| m > 0)
                    .map(|m| Duration::from_secs(u64::from(m) * 60));

                let (busy, loaded) = {
                    let pipeline = runtime.inner.pipeline.lock().await;
                    (pipeline.is_busy(), pipeline.asr.is_some())
                };
                // Belt and braces alongside the reset in `set_pipeline_session`:
                // that one is exact, this one guarantees a long dictation can
                // never age into its own unload deadline.
                if busy {
                    *runtime.inner.last_activity.lock().await = Instant::now();
                }
                let idle_for = runtime.inner.last_activity.lock().await.elapsed();

                match idle_unload_decision(timeout, idle_for, busy, loaded) {
                    IdleAction::Wait(d) => tokio::time::sleep(d).await,
                    IdleAction::Unload => runtime.unload_idle_model().await,
                }
            }
        });
    }

    /// Take the loaded model out of the pipeline and free it.
    ///
    /// Reuses the reload machinery rather than adding a second teardown path:
    /// a [`ReloadScope::Full`] is exactly "the loaded backend is no longer
    /// valid", which is what an idle unload means. Leaving the scope set is
    /// what makes the next `prepare_manager_for_start` rebuild — and it is
    /// deliberately not cleared here, so a start that races this unload
    /// reloads rather than running against a half-torn-down pipeline.
    ///
    /// The busy re-check under the lock is the race that matters: the watchdog
    /// samples `busy` before sleeping, and a dictation can start in between.
    async fn unload_idle_model(&self) {
        // Same gate the rebuild paths take, so this cannot interleave with a
        // `build_pipeline` and free a model that is being installed.
        let _start_guard = self.inner.start_gate.lock().await;

        let released = {
            let mut pipeline = self.inner.pipeline.lock().await;
            if pipeline.is_busy() || pipeline.asr.is_none() {
                return;
            }
            pipeline.reload = pipeline.reload.max(ReloadScope::Full);
            let (_, released) = pipeline.take_reload_plan();
            released
        };
        release_backend(released).await;
        info!("ASR model unloaded after idle timeout; the next dictation reloads it");
    }
}

impl Clone for AppRuntime {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// The pipeline the app starts with: built now when the models are on disk,
/// otherwise left for the first dictation to build.
///
/// Infallible on purpose. Startup runs inside Tauri's `setup`, which panics
/// on an error, so a pipeline that cannot build (a truncated model, remote
/// transcription while remote services are off, no GPU memory) would stop
/// the app from ever opening — and with it the settings that fix the cause.
/// A failed build is deferred instead; the next dictation retries it and
/// reports the reason.
async fn initial_pipeline(
    paths: AppPaths,
    settings: UserSettings,
    models_ready: bool,
) -> (LivePipeline, Arc<dyn LlmService>, Arc<RwLock<UserSettings>>) {
    if !models_ready {
        info!("models not yet downloaded — deferring pipeline build");
        return deferred_pipeline(paths, settings);
    }
    match build_pipeline(paths.clone(), settings.clone(), None).await {
        Ok(built) => built,
        Err(error) => {
            warn!(%error, "pipeline build failed at startup; deferring it to the first dictation");
            deferred_pipeline(paths, settings)
        }
    }
}

/// A pipeline with nothing loaded and a full rebuild pending, so the next
/// start builds it.
fn deferred_pipeline(
    paths: AppPaths,
    settings: UserSettings,
) -> (LivePipeline, Arc<dyn LlmService>, Arc<RwLock<UserSettings>>) {
    let null_svc: Arc<dyn LlmService> = Arc::new(NullPostProcessor::new());
    let pipeline = LivePipeline {
        paths,
        manager: None,
        asr: None,
        state: PipelineState::Idle,
        abort_pending_start: false,
        session_id: None,
        reload: ReloadScope::Full,
        focus_target: None,
    };
    (pipeline, null_svc, Arc::new(RwLock::new(settings)))
}

/// Build a fresh pipeline.
///
/// `reuse_asr` carries the already-loaded backend forward for a
/// [`ReloadScope::Config`] rebuild. Everything else here is cheap; the model
/// load is not, so a dictionary or style edit must never trigger one.
async fn build_pipeline(
    paths: AppPaths,
    settings: UserSettings,
    reuse_asr: Option<Arc<AsrRegistry>>,
) -> Result<(LivePipeline, Arc<dyn LlmService>, Arc<RwLock<UserSettings>>), String> {
    // `LocalOnly` is the default, so refusing here is the difference between a
    // stated preference and an enforced one: a remote backend would ship every
    // utterance's audio off the machine. Styling degrades to passthrough for
    // the same reason (`build_llm_service`), but transcription has no local
    // fallback to degrade to, so this fails the build with a fixable message.
    if remote_backends_denied(settings.privacy_mode)
        && matches!(
            settings.asr_backend,
            AsrBackendConfig::OpenAiCompatible { .. }
        )
    {
        return Err("remote services are turned off, but transcription is set to use one. Switch transcription to an on-device engine (Whisper or Parakeet), or turn on Allow remote services in Settings → General.".into());
    }
    // Whisper model load takes seconds on the larger models; keep it off the
    // async workers so IPC and the UI stay responsive during startup/reload.
    let asr: Arc<AsrRegistry> = if let Some(asr) = reuse_asr {
        asr
    } else {
        let backend_config = settings.asr_backend.clone();
        let device = settings.device;
        let models_dir = paths.models_dir();
        // Runtime, not `cfg!`: the CUDA backend is a downloadable ggml plugin,
        // so a `whisper-cpp` build can still have a usable GPU by the time this
        // runs. `lib.rs` loads the plugins before the first pipeline build.
        let cuda_available = lune_asr::gpu_backend::cuda_available(paths.data_dir());
        let built = tokio::task::spawn_blocking(move || {
            AsrRegistry::from_config(&backend_config, device, &models_dir, cuda_available)
        })
        .await
        .map_err(|e| format!("asr build task: {e}"))?
        .map_err(|e| format!("build asr backend: {e}"))?;
        Arc::new(built)
    };
    let llm_service = build_llm_service(&settings).await?;
    let style_pipeline = Arc::new(lune_llm::StylePipeline::new(Arc::clone(&llm_service)));
    let post_processor: Arc<dyn lune_core::llm::TextPostProcessor> = style_pipeline;
    let router = Arc::new(VoiceCommandRouter::new());
    let mut session_db =
        Database::open(&paths.database_file()).map_err(|e| format!("open session db: {e}"))?;
    let dict = Arc::new(UserDictionary::new(
        session_db
            .list_dictionary_entries()
            .map_err(|e| format!("load dictionary: {e}"))?,
    ));
    let db = Arc::new(Mutex::new(session_db));
    // Resolve backend defaults + user TOML overrides once, then repair any
    // invariant violation rather than running on it. An out-of-range override
    // otherwise degrades every session silently — an instant wall-clock
    // deadline, one syllable per dictation, a multi-gigabyte audio buffer.
    let (pipeline_config, repairs) = asr
        .pipeline_config()
        .with_overrides(&settings.pipeline)
        .clamp();
    for repair in repairs {
        tracing::warn!(repair, "pipeline override was out of range; corrected");
    }
    // "Don't save history" has to mean "write nothing" rather than "prune
    // nothing", and "keep forever" still records — see
    // `UserSettings::history_retention`. Read before `settings` is wrapped.
    let persist_history = settings.history_retention().records();
    let settings = Arc::new(RwLock::new(settings.clone()));

    let config = SessionManagerConfig {
        persist_history,
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
        Arc::clone(&asr),
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
        asr: Some(asr),
        state: PipelineState::Idle,
        abort_pending_start: false,
        session_id: None,
        reload: ReloadScope::None,
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
    Ok(load_settings_checked(paths)?.settings)
}

/// Whether `privacy_mode` forbids sending audio or text to a remote service.
///
/// Only `LocalOnly` denies. Under `AllowRemote` a remote service still runs
/// only once it is configured.
fn remote_backends_denied(mode: PrivacyMode) -> bool {
    matches!(mode, PrivacyMode::LocalOnly)
}

async fn build_llm_service(settings: &UserSettings) -> Result<Arc<dyn LlmService>, String> {
    match &settings.llm_backend {
        LlmBackendConfig::OpenAiCompatible {
            base_url,
            api_key,
            model,
            max_tokens,
        } => {
            // Local-only means local-only: styling is optional, so a remote
            // provider is dropped to passthrough rather than failing the app.
            if remote_backends_denied(settings.privacy_mode) {
                tracing::info!("privacy mode is local-only; styling stays off (passthrough)");
                return Ok(Arc::new(NullPostProcessor::new()) as Arc<dyn LlmService>);
            }
            // Selecting a style on the Styles page flips `llm_enabled` on, and
            // with the local engine gone that always resolves to this remote
            // provider — defaulted to api.openai.com. Building a live backend
            // from a half-filled provider would POST every transcript to a
            // third party the user never configured (and get a 401 back), so
            // an unconfigured provider degrades to passthrough instead.
            //
            // Deferring to the same readiness check the UI shows, rather than
            // repeating its conditions: a copy that drifts would have the app
            // reporting styling as unconfigured while quietly using it.
            if let Err(reason) = lune_persist::styling_llm_readiness(&settings.llm_backend) {
                tracing::info!(reason = ?reason, "styling provider is not ready, using passthrough");
                return Ok(Arc::new(NullPostProcessor::new()) as Arc<dyn LlmService>);
            }
            // Styling is optional, so a backend that refuses to build (e.g. a
            // plaintext public endpoint) must degrade to passthrough, not fail
            // the whole app at startup — the readiness check above already
            // routes unconfigured providers here, and this is the same path.
            let backend = match OpenAiLlmBackend::new(OpenAiLlmConfig {
                base_url: base_url.to_string(),
                api_key: api_key.clone(),
                model: model.clone(),
                max_tokens: *max_tokens,
                temperature: 0.2,
            }) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::warn!(error = %e, "styling LLM refused; using passthrough");
                    return Ok(Arc::new(NullPostProcessor::new()) as Arc<dyn LlmService>);
                }
            };
            Ok(backend as Arc<dyn LlmService>)
        }
        _ => Ok(Arc::new(NullPostProcessor::new()) as Arc<dyn LlmService>),
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
    /// Close and reopen on the desired device — the device changed under us.
    Reopen,
}

/// Decide the prewarm stream action. `open` describes the currently open
/// stream: `None` = nothing open; `Some(dev)` = open on `dev`, where `dev` is
/// `None` for the system default. Comparing the open device against `desired`
/// is what makes a device change rebuild the stream live (no restart needed).
///
/// There is no "close" outcome: the capture stream is always wanted. It is what
/// lets the start path record from the moment the hotkey is pressed, including
/// across a model reload, so tearing it down would only reintroduce the lost
/// first words it exists to prevent.
fn reconcile_prewarm(
    open: Option<Option<&str>>,
    desired: Option<&str>,
    healthy: bool,
) -> PrewarmAction {
    match open {
        None => PrewarmAction::Open,
        // A stream on the right device is still useless if cpal has reported an
        // error on it — the usual cause is the device being unplugged, and the
        // name it was opened under does not change when that happens.
        Some(current) if current == desired && healthy => PrewarmAction::Keep,
        Some(_) => PrewarmAction::Reopen,
    }
}

/// What the idle-unload watchdog should do on this tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdleAction {
    /// Free the loaded model now.
    Unload,
    /// Nothing to do yet; look again after this long. Never zero, so a caller
    /// looping on it cannot spin.
    Wait(Duration),
}

/// Longest a watchdog tick will sleep before re-reading its inputs. The timeout
/// and the loaded/busy state can all change while it waits, so even a 12-hour
/// setting is re-examined on this cadence rather than committing to one long
/// sleep it would have to be woken out of.
const IDLE_POLL_CEILING: Duration = Duration::from_secs(60);

/// Decide whether the idle model-unload should fire.
///
/// Pure, so the policy is testable without a Tauri app handle, a model, or a
/// real clock — the watchdog around it only sleeps and calls the unload.
///
/// `timeout` is `None` when the feature is off. `loaded` is false when there is
/// nothing to free, which is both the already-unloaded case and the
/// never-built one. `busy` covers a dictation that is running *or* still
/// loading: unloading under either would free a model in active use.
fn idle_unload_decision(
    timeout: Option<Duration>,
    idle_for: Duration,
    busy: bool,
    loaded: bool,
) -> IdleAction {
    // A disabled or already-satisfied watchdog still ticks, because the setting
    // can be switched on and a model can be loaded between two ticks.
    let Some(timeout) = timeout.filter(|_| loaded && !busy) else {
        return IdleAction::Wait(IDLE_POLL_CEILING);
    };
    match timeout.checked_sub(idle_for) {
        // `checked_sub` yields `Some(0)` exactly at the deadline, which is due.
        Some(remaining) if !remaining.is_zero() => {
            IdleAction::Wait(remaining.min(IDLE_POLL_CEILING))
        }
        _ => IdleAction::Unload,
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
fn cue_for_transition(prev: &str, status: &str) -> Option<lune_voice::sound::Cue> {
    use lune_voice::sound::Cue;
    match status {
        "listening" if prev != "listening" => Some(Cue::Start),
        "processing" if prev == "listening" => Some(Cue::Stop),
        "error" if prev != "error" => Some(Cue::Error),
        _ => None,
    }
}

#[cfg(test)]
mod idle_unload_tests {
    use super::{IDLE_POLL_CEILING, IdleAction, idle_unload_decision};
    use std::time::Duration;

    const HALF_HOUR: Duration = Duration::from_secs(30 * 60);

    /// Loaded, idle, and past the deadline — the case the feature exists for.
    #[test]
    fn fires_once_the_timeout_has_elapsed() {
        assert_eq!(
            idle_unload_decision(
                Some(HALF_HOUR),
                HALF_HOUR + Duration::from_secs(1),
                false,
                true
            ),
            IdleAction::Unload
        );
    }

    /// Exactly at the deadline counts as elapsed, not as "one more tick".
    #[test]
    fn fires_exactly_on_the_deadline() {
        assert_eq!(
            idle_unload_decision(Some(HALF_HOUR), HALF_HOUR, false, true),
            IdleAction::Unload
        );
    }

    /// Short of the deadline the watchdog waits precisely the remainder, so a
    /// 5-minute timeout does not have to wait out a 60-second poll interval.
    #[test]
    fn waits_the_exact_remainder_when_it_fits_under_the_ceiling() {
        let timeout = Duration::from_secs(90);
        assert_eq!(
            idle_unload_decision(Some(timeout), Duration::from_secs(45), false, true),
            IdleAction::Wait(Duration::from_secs(45))
        );
    }

    /// A long remainder is capped, so the timeout and the loaded/busy state are
    /// re-read regularly instead of being committed to for half an hour.
    #[test]
    fn caps_a_long_wait_at_the_poll_ceiling() {
        assert_eq!(
            idle_unload_decision(Some(HALF_HOUR), Duration::ZERO, false, true),
            IdleAction::Wait(IDLE_POLL_CEILING)
        );
    }

    /// Off means off, however long the machine has been idle.
    #[test]
    fn never_unloads_when_the_setting_is_off() {
        assert_eq!(
            idle_unload_decision(None, Duration::from_secs(86_400), false, true),
            IdleAction::Wait(IDLE_POLL_CEILING)
        );
    }

    /// The important safety property: a dictation that is running or still
    /// loading must never have its model freed underneath it, no matter how
    /// stale `idle_for` looks.
    #[test]
    fn never_unloads_while_busy() {
        assert_eq!(
            idle_unload_decision(Some(HALF_HOUR), HALF_HOUR * 10, true, true),
            IdleAction::Wait(IDLE_POLL_CEILING)
        );
    }

    /// Nothing loaded means nothing to free — and re-firing would keep
    /// re-entering the unload path forever.
    #[test]
    fn does_not_fire_when_no_model_is_loaded() {
        assert_eq!(
            idle_unload_decision(Some(HALF_HOUR), HALF_HOUR * 10, false, false),
            IdleAction::Wait(IDLE_POLL_CEILING)
        );
    }

    /// A zero timeout is degenerate but must not produce a zero-length wait —
    /// a caller looping on `Wait` would spin the CPU.
    #[test]
    fn a_wait_is_never_zero_length() {
        for idle in [Duration::ZERO, Duration::from_secs(1), HALF_HOUR] {
            for timeout in [Duration::ZERO, Duration::from_secs(1), HALF_HOUR] {
                if let IdleAction::Wait(d) = idle_unload_decision(Some(timeout), idle, false, true)
                {
                    assert!(
                        !d.is_zero(),
                        "zero wait for timeout={timeout:?} idle={idle:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Arc, AsrRegistry, LivePipeline, PipelineState, PrewarmAction, ReloadScope, START_ABORTED,
        StartFailureReport, asr_backend_needs_reload, classify_start_failure, cue_for_transition,
        reconcile_prewarm, release_backend, remote_backends_denied, sanitize_error_for_frontend,
        undelivered_message,
    };

    /// A dictation that sits on the clipboard is recoverable, and the user has
    /// to be told to paste it — blaming an unavailable clipboard would be wrong.
    #[test]
    fn an_undelivered_dictation_says_whether_it_is_on_the_clipboard() {
        assert!(undelivered_message(true).contains("clipboard"));
        assert!(undelivered_message(true).contains("paste"));
        assert!(!undelivered_message(false).contains("paste"));
    }

    /// The hotkey has already played the start cue by the time prep runs. A
    /// real prep failure has to reach the user; an abort is what they asked
    /// for and must stay silent.
    #[test]
    fn a_failed_start_is_reported_and_an_aborted_one_is_not() {
        assert_eq!(
            classify_start_failure(START_ABORTED),
            StartFailureReport::Abandoned
        );
        assert_eq!(
            classify_start_failure(
                "whisper model file is too small: most likely a partial download; \
                 delete it and download the model again"
            ),
            StartFailureReport::Error
        );
        assert_eq!(
            classify_start_failure("Models not downloaded. Please complete model setup first."),
            StartFailureReport::Error
        );
    }
    use lune_asr::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend};
    use lune_core::config::{AsrBackendConfig, PrivacyMode, WhisperModel};

    /// A pipeline with no manager and no backend loaded — enough to drive the
    /// start/stop state machine, which is the part that had the bug.
    fn idle_pipeline() -> LivePipeline {
        LivePipeline {
            paths: lune_persist::AppPaths::resolve().expect("resolve XDG paths"),
            manager: None,
            asr: None,
            state: PipelineState::Idle,
            abort_pending_start: false,
            session_id: None,
            reload: ReloadScope::None,
            focus_target: None,
        }
    }

    #[test]
    fn a_stop_during_the_pipeline_load_abandons_the_start() {
        // The push-to-talk case: key down begins a start that has to load a
        // model, key up arrives seconds later, before any session exists.
        // Dropping it there leaves the mic open with nothing left to close it.
        let mut pipeline = idle_pipeline();
        assert!(pipeline.begin_start());
        assert_eq!(pipeline.state, PipelineState::Loading);

        assert!(
            pipeline.note_stop_during_start(),
            "a stop while loading must be recorded, not discarded"
        );
        assert!(
            !pipeline.finish_start(),
            "the start must abandon itself rather than open a session"
        );
        assert_eq!(pipeline.state, PipelineState::Idle);
        assert!(!pipeline.abort_pending_start, "the flag must not linger");
    }

    #[test]
    fn an_undisturbed_start_becomes_active() {
        let mut pipeline = idle_pipeline();
        assert!(pipeline.begin_start());
        assert!(pipeline.finish_start());
        assert_eq!(pipeline.state, PipelineState::Active);
    }

    #[test]
    fn a_second_start_is_refused_while_one_is_loading() {
        let mut pipeline = idle_pipeline();
        assert!(pipeline.begin_start());
        assert!(
            !pipeline.begin_start(),
            "Loading is busy — a racing start must not claim the pipeline too"
        );
    }

    #[test]
    fn a_stop_outside_a_pending_start_is_not_swallowed() {
        // With a session actually running the stop has somewhere to go, so it
        // must be forwarded rather than absorbed as a pending-start abort.
        let mut pipeline = idle_pipeline();
        pipeline.begin_start();
        pipeline.finish_start();
        assert!(!pipeline.note_stop_during_start());
        assert_eq!(pipeline.state, PipelineState::Active);

        let mut idle = idle_pipeline();
        assert!(!idle.note_stop_during_start());
    }

    #[test]
    fn a_heavier_reload_scope_wins_when_changes_queue_up() {
        assert_eq!(
            ReloadScope::None.max(ReloadScope::Config),
            ReloadScope::Config
        );
        assert_eq!(
            ReloadScope::Config.max(ReloadScope::Full),
            ReloadScope::Full
        );
        assert_eq!(
            ReloadScope::Full.max(ReloadScope::Config),
            ReloadScope::Full
        );
    }

    /// A loaded backend that costs nothing to build. The remote variant has no
    /// model behind it, but `take_reload_plan` and `release_backend` only ever
    /// move the `Arc` around — which variant it wraps is immaterial to them.
    fn loaded_backend() -> Arc<AsrRegistry> {
        let backend = OpenAiCompatibleAsrBackend::new(OpenAiAsrConfig::default())
            .expect("the default remote config builds an HTTP client, no network");
        Arc::new(AsrRegistry::OpenAi(backend))
    }

    #[test]
    fn a_full_reload_takes_the_loaded_backend_out_of_the_pipeline() {
        let mut pipeline = idle_pipeline();
        let probe = loaded_backend();
        pipeline.asr = Some(Arc::clone(&probe));
        pipeline.reload = ReloadScope::Full;

        let (reuse, released) = pipeline.take_reload_plan();
        assert!(
            reuse.is_none(),
            "a backend change must force a fresh model load"
        );
        assert!(
            pipeline.asr.is_none(),
            "the old model must leave the pipeline before the replacement is \
             loaded — holding it across the load puts two whole models in VRAM"
        );

        drop(released);
        assert_eq!(
            Arc::strong_count(&probe),
            1,
            "the released handle must be the last reference, or nothing is freed"
        );
    }

    #[test]
    fn a_config_reload_keeps_the_loaded_backend_in_place() {
        let mut pipeline = idle_pipeline();
        let probe = loaded_backend();
        pipeline.asr = Some(Arc::clone(&probe));
        pipeline.reload = ReloadScope::Config;

        let (reuse, released) = pipeline.take_reload_plan();
        assert!(
            reuse.is_some(),
            "a dictionary or style edit must reuse the loaded model, not reload it"
        );
        assert!(pipeline.asr.is_some(), "and must leave it in the pipeline");
        assert!(
            released.is_empty(),
            "nothing was invalidated, so nothing may be released"
        );
    }

    #[tokio::test]
    async fn release_backend_frees_the_model_before_it_returns() {
        // `take_reload_plan` releases the old model so the *next* load has the
        // memory back. That only holds if the free has actually completed by
        // the time this returns — a detached drop would race the new load.
        let probe = loaded_backend();
        let mut pipeline = idle_pipeline();
        pipeline.asr = Some(Arc::clone(&probe));
        pipeline.reload = ReloadScope::Full;
        let (_, released) = pipeline.take_reload_plan();

        release_backend(released).await;
        assert_eq!(
            Arc::strong_count(&probe),
            1,
            "the model must be freed before the caller loads its replacement"
        );
    }

    #[test]
    fn only_a_different_whisper_model_forces_a_reload() {
        let small = AsrBackendConfig::WhisperCpp {
            model: WhisperModel::Small,
        };
        let medium = AsrBackendConfig::WhisperCpp {
            model: WhisperModel::Medium,
        };
        // A dictionary or style edit leaves the backend untouched — reloading
        // ~874 MB of weights for it is the cost this check exists to avoid.
        assert!(!asr_backend_needs_reload(&small, &small));
        assert!(asr_backend_needs_reload(&small, &medium));
    }

    #[test]
    fn switching_between_local_engines_reloads_and_staying_does_not() {
        let whisper = AsrBackendConfig::WhisperCpp {
            model: WhisperModel::Small,
        };
        let parakeet = AsrBackendConfig::ParakeetCpp {
            model: lune_core::config::ParakeetModel::TdtV3Q8,
        };
        assert!(!asr_backend_needs_reload(&parakeet, &parakeet));
        let smaller = AsrBackendConfig::ParakeetCpp {
            model: lune_core::config::ParakeetModel::TdtV3Q4K,
        };
        assert!(asr_backend_needs_reload(&parakeet, &smaller));
        assert!(asr_backend_needs_reload(&whisper, &parakeet));
        assert!(asr_backend_needs_reload(&parakeet, &whisper));
    }

    /// Startup runs inside Tauri's `setup`, which panics on any error. A
    /// pipeline that cannot build — here, remote transcription saved while
    /// remote services are off — must leave the app running with the build
    /// deferred, so the next dictation reports the reason instead.
    #[tokio::test]
    async fn a_pipeline_that_cannot_build_does_not_stop_the_app_starting() {
        let root = std::env::temp_dir().join("lune-startup-build-failure");
        let paths = lune_persist::AppPaths::from_dirs(root.join("config"), root.join("data"));
        let settings = lune_core::config::UserSettings {
            privacy_mode: PrivacyMode::LocalOnly,
            asr_backend: AsrBackendConfig::OpenAiCompatible {
                base_url: "https://api.example.test/v1".parse().expect("url"),
                api_key: lune_core::config::RedactedString::new("key"),
                model: "whisper-1".into(),
                streaming: lune_core::config::StreamingMode::HttpPolling,
            },
            ..Default::default()
        };

        let (pipeline, _, _) = super::initial_pipeline(paths, settings, true).await;
        assert!(pipeline.manager.is_none());
        assert_eq!(pipeline.reload, ReloadScope::Full);
        assert_eq!(pipeline.state, PipelineState::Idle);
    }

    #[test]
    fn only_local_only_denies_remote_backends() {
        assert!(remote_backends_denied(PrivacyMode::LocalOnly));
        assert!(!remote_backends_denied(PrivacyMode::AllowRemote));
    }

    #[test]
    fn start_cue_fires_once_despite_double_listening() {
        use lune_voice::sound::Cue;
        // Optimistic publish enters "listening" from idle.
        assert_eq!(cue_for_transition("ready", "listening"), Some(Cue::Start));
        // SessionEvent::Started republishes "listening" — must not replay.
        assert_eq!(cue_for_transition("listening", "listening"), None);
    }

    #[test]
    fn stop_cue_fires_when_recording_ends() {
        use lune_voice::sound::Cue;
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
        use lune_voice::sound::Cue;
        assert_eq!(cue_for_transition("processing", "error"), Some(Cue::Error));
        assert_eq!(cue_for_transition("error", "error"), None);
    }

    #[test]
    fn no_cue_for_unknown_status_or_stray_processing() {
        assert_eq!(cue_for_transition("ready", "calibrating"), None);
        // "processing" only cues when it directly follows "listening".
        assert_eq!(cue_for_transition("ready", "processing"), None);
    }

    /// The follow-up behavior: when the chosen device changes, the open stream
    /// must be rebuilt (Reopen), not left on the stale device.
    #[test]
    fn reconcile_reopens_when_device_changes() {
        let action =
            reconcile_prewarm(Some(Some("Built-in Microphone")), Some("USB Headset"), true);
        assert_eq!(action, PrewarmAction::Reopen);
    }

    #[test]
    fn reconcile_rebuilds_a_stream_the_device_died_under() {
        // Unplug and replug: the device name is unchanged, so name comparison
        // alone keeps the dead stream and every later dictation burns the 5 s
        // starvation watchdog until an unrelated setting is touched.
        assert_eq!(
            reconcile_prewarm(Some(Some("USB Headset")), Some("USB Headset"), false),
            PrewarmAction::Reopen
        );
        assert_eq!(
            reconcile_prewarm(Some(None), None, false),
            PrewarmAction::Reopen
        );
    }

    #[test]
    fn reconcile_keeps_when_device_unchanged() {
        assert_eq!(
            reconcile_prewarm(Some(Some("USB Headset")), Some("USB Headset"), true),
            PrewarmAction::Keep
        );
        // Default ↔ default is also unchanged.
        assert_eq!(
            reconcile_prewarm(Some(None), None, true),
            PrewarmAction::Keep
        );
    }

    /// The capture stream is unconditional, so "nothing open" always means
    /// open one — there is no longer a state in which it stays closed.
    #[test]
    fn reconcile_always_opens_when_nothing_is_open() {
        assert_eq!(
            reconcile_prewarm(None, Some("USB Headset"), true),
            PrewarmAction::Open
        );
        assert_eq!(reconcile_prewarm(None, None, true), PrewarmAction::Open);
    }

    #[test]
    fn sanitize_error_handles_non_ascii_without_panicking() {
        let raw = format!("é{}", "/tmp/example/path/segment".repeat(80));
        let sanitized = sanitize_error_for_frontend(&raw);
        assert!(!sanitized.is_empty());
    }
}
