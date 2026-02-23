mod cli;
mod dbus;
mod delivery;
mod events;
mod factory;

use std::sync::Arc;

use tokio::sync::{RwLock, mpsc, watch};
use tracing_subscriber::{EnvFilter, fmt};

use lune_core::config::DictationMode;

use lune_app::{SessionManager, SessionManagerConfig};

use crate::events::{AppCommand, AppStateUpdate};

fn main() {
    let mut args = std::env::args_os();
    args.next(); // skip argv[0]

    let first = args.next();
    let first_str = first.as_deref().and_then(|s| s.to_str());

    match first_str {
        Some("toggle" | "start" | "stop" | "cancel" | "status" | "quit" | "mode" | "settings") => {
            let collected: Vec<String> = std::env::args().skip(1).collect();
            cli::run_cli(&collected);
        }
        Some("daemon") | None => run_daemon(),
        Some(other) => {
            eprintln!("unknown subcommand: {other}");
            eprintln!(
                "usage: lune-voice [toggle|start|stop|cancel|status|mode|settings|quit|daemon]"
            );
            std::process::exit(1);
        }
    }
}

fn run_daemon() {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    tracing::info!("lune-voice daemon starting");

    // ── Resolve paths, load settings, ensure models ──────────────────────

    let paths = lune_persist::AppPaths::resolve().expect("resolve app paths");
    paths.ensure_dirs().expect("create app directories");

    let settings_store = Arc::new(lune_persist::SettingsStore::new(paths.settings_file()));
    let mut settings = match settings_store.load() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "failed to parse settings, using defaults (pre-release migration)");
            lune_core::config::UserSettings::default()
        }
    };

    let models =
        lune_persist::ensure_default_models(&paths.models_dir()).expect("ensure models downloaded");

    factory::patch_model_paths(&mut settings, &models);

    tracing::info!(
        asr_dir = %models.asr_model_dir.display(),
        llm_path = %models.llm_model_path.display(),
        "models ready"
    );

    // ── Print keybinding hint on first run ────────────────────────────────

    eprintln!();
    eprintln!("  Bind keyboard shortcuts in COSMIC Settings → Keyboard → Custom Shortcuts:");
    eprintln!("    Toggle mode:  Ctrl+Space  →  lune-voice toggle");
    eprintln!("    Hold mode:    Ctrl+Space (press)  →  lune-voice start");
    eprintln!("                  Ctrl+Space (release) →  lune-voice stop");
    eprintln!("    (optional)    Escape  →  lune-voice cancel");
    eprintln!();

    // ── Build tokio current_thread runtime ────────────────────────────────
    // current_thread avoids the overhead of multi-thread work-stealing for
    // this single-threaded event loop; all hot paths stay on one core.

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let local = tokio::task::LocalSet::new();

    // ── Phase 1: D-Bus connection OUTSIDE LocalSet ───────────────────────
    // zbus internally calls tokio::spawn, which panics inside LocalSet.

    // Channel depth 64 is generous for human-speed commands; avoids alloc churn.
    let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(64);
    // Pre-allocate capacity for the longest state string ("listening" = 9 chars)
    let shared_state = Arc::new(RwLock::new(String::with_capacity(16)));

    // Initialise to "idle" without a second allocation
    {
        let mut guard = rt.block_on(shared_state.write());
        guard.push_str("idle");
    }

    let (dictation_mode_tx, dictation_mode_rx) = watch::channel(settings.dictation_mode);
    let (activation_mode_tx, activation_mode_rx) = watch::channel(settings.activation_mode);

    let settings = Arc::new(RwLock::new(settings));

    let dbus_iface = dbus::DaemonInterface::new(
        cmd_tx,
        Arc::clone(&shared_state),
        Arc::clone(&settings),
        dictation_mode_tx,
        activation_mode_tx,
        Arc::clone(&settings_store),
        rt.handle().clone(),
    );

    let dbus_conn = rt
        .block_on(async { dbus::build_connection(dbus_iface).await })
        .expect("build D-Bus connection");

    tracing::info!("D-Bus service registered on session bus");

    // ── Phase 2: pipeline inside LocalSet ────────────────────────────────

    local.block_on(&rt, async move {
        run_pipeline(
            settings,
            paths.database_file(),
            cmd_rx,
            shared_state,
            &dbus_conn,
            dictation_mode_rx,
            activation_mode_rx,
        )
        .await;
    });

    tracing::info!("lune-voice stopped");
}

async fn run_pipeline(
    settings: Arc<RwLock<lune_core::config::UserSettings>>,
    db_path: std::path::PathBuf,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
    shared_state: Arc<RwLock<String>>,
    dbus_conn: &zbus::Connection,
    dictation_mode_rx: watch::Receiver<DictationMode>,
    _activation_mode_rx: watch::Receiver<lune_core::config::ActivationMode>,
) {
    // Snapshot settings for backend construction (model loading is one-time).
    let snap = settings.read().await.clone();

    // ── Build pipeline components ────────────────────────────────────────

    tracing::info!("building ASR backend...");
    let asr = match factory::build_asr_backend(&snap) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(error = %e, "failed to build ASR backend");
            return;
        }
    };

    tracing::info!("ASR backend ready, building LLM post-processor...");
    let post_processor = match factory::build_post_processor(&snap) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "failed to build LLM post-processor");
            return;
        }
    };

    tracing::info!("LLM post-processor ready");
    let audio = Arc::new(lune_audio::AudioCaptureService::new(
        lune_core::audio::AudioCaptureConfig::default(),
    ));
    let router = Arc::new(lune_commands::VoiceCommandRouter::new());

    tracing::info!("opening database...");
    factory::ensure_db_path(&db_path).expect("ensure DB path");
    let db = Arc::new(tokio::sync::Mutex::new(
        lune_persist::Database::open(&db_path).expect("open database"),
    ));

    let dict_db = lune_persist::Database::open_in_memory().expect("open dict database");
    let dict = Arc::new(tokio::sync::Mutex::new(
        lune_dict::DictionaryManager::new(dict_db).expect("init dictionary"),
    ));

    let asr_backend_kind = match snap.asr_backend {
        lune_core::config::AsrBackendConfig::OpenAiCompatible { .. } => {
            lune_core::asr::AsrBackendKind::OpenAiCompatible
        }
        _ => lune_core::asr::AsrBackendKind::ParakeetTdt,
    };

    let config = SessionManagerConfig {
        asr_backend_kind,
        ..Default::default()
    };

    let manager = SessionManager::new(
        audio,
        Arc::new(asr),
        router,
        post_processor,
        dict,
        db,
        config,
        Arc::clone(&settings),
    );

    let delivery_service = delivery::DeliveryService::new();

    tracing::info!("pipeline ready, waiting for commands");

    // ── Command loop ─────────────────────────────────────────────────────

    loop {
        let Some(cmd) = cmd_rx.recv().await else {
            tracing::info!("command channel closed, shutting down");
            manager.stop().await;
            break;
        };

        match cmd {
            AppCommand::ToggleDictation => {
                if manager.active_session_id().await.is_some() {
                    stop_dictation(&manager, &shared_state, dbus_conn).await;
                } else {
                    start_dictation(
                        &manager,
                        &shared_state,
                        dbus_conn,
                        dictation_mode_rx.clone(),
                        delivery_service.clone(),
                    )
                    .await;
                }
            }
            AppCommand::StartDictation => {
                if manager.active_session_id().await.is_none() {
                    start_dictation(
                        &manager,
                        &shared_state,
                        dbus_conn,
                        dictation_mode_rx.clone(),
                        delivery_service.clone(),
                    )
                    .await;
                }
            }
            AppCommand::StopDictation => {
                if manager.active_session_id().await.is_some() {
                    stop_dictation(&manager, &shared_state, dbus_conn).await;
                }
            }
            AppCommand::CancelDictation => {
                tracing::info!("cancelling dictation");
                manager.cancel().await;
            }
            AppCommand::Quit => {
                tracing::info!("quit command received");
                manager.stop().await;
                update_state(&shared_state, dbus_conn, AppStateUpdate::Idle).await;
                break;
            }
        }
    }
}

async fn start_dictation<A: lune_core::asr::AsrBackend + 'static>(
    manager: &SessionManager<A>,
    shared_state: &Arc<RwLock<String>>,
    dbus_conn: &zbus::Connection,
    dictation_mode_rx: watch::Receiver<DictationMode>,
    delivery_service: delivery::DeliveryService,
) {
    tracing::info!("starting dictation");
    // Depth 64: matches command channel; avoids back-pressure stalls on bursts
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);

    let state_ref = Arc::clone(shared_state);
    let conn_clone = dbus_conn.clone();
    tokio::task::spawn_local(async move {
        while let Some(event) = event_rx.recv().await {
            // Fast-path: skip work for high-frequency PartialText without lock
            let update = match event {
                lune_core::session::SessionEvent::Started => AppStateUpdate::Listening,
                lune_core::session::SessionEvent::PartialText(_) => AppStateUpdate::PartialText,
                lune_core::session::SessionEvent::FinalizedText(t) => {
                    // `borrow()` is lock-free on watch channels — cheap hot path
                    let mode = *dictation_mode_rx.borrow();
                    delivery_service.deliver(&t, mode);
                    AppStateUpdate::FinalText
                }

                lune_core::session::SessionEvent::Error(e) => {
                    tracing::error!(error = %e, "session error");
                    AppStateUpdate::Error
                }
                lune_core::session::SessionEvent::Stopped => AppStateUpdate::Idle,
                _ => continue,
            };
            update_state(&state_ref, &conn_clone, update).await;
        }
    });

    match manager.start(event_tx).await {
        Ok(id) => {
            tracing::info!(%id, "session started");
        }
        Err(e) => {
            tracing::error!(error = %e, "failed to start session");
            update_state(shared_state, dbus_conn, AppStateUpdate::Error).await;
        }
    }
}

#[cold]
async fn stop_dictation<A: lune_core::asr::AsrBackend + 'static>(
    manager: &SessionManager<A>,
    _shared_state: &Arc<RwLock<String>>,
    _dbus_conn: &zbus::Connection,
) {
    tracing::info!("stopping dictation");
    manager.stop().await;
    // Idle update emitted by the event forwarder via SessionEvent::Stopped
}

/// Update shared state string and emit the D-Bus signal.
///
/// Uses `replace_range` to reuse the existing heap allocation instead of
/// re-allocating on every state transition (hot path for partial-text events).
/// State strings are ASCII ≤9 bytes — always fits the pre-allocated 16-byte buffer.
#[inline]
async fn update_state(
    shared: &Arc<RwLock<String>>,
    conn: &zbus::Connection,
    update: AppStateUpdate,
) {
    // Static string slices — zero allocation, branch predicted well by CPU.
    // PartialText maps to "listening" — avoids write-lock churn on the hot path
    // when the state hasn't actually changed (already "listening").
    let state_str: &'static str = match update {
        AppStateUpdate::Idle | AppStateUpdate::FinalText => "idle",
        AppStateUpdate::Listening | AppStateUpdate::PartialText => "listening",
        AppStateUpdate::Error => "error",
    };

    {
        // Scope the write-lock as tightly as possible; release before the async
        // D-Bus emit so we don't hold the lock across an await point.
        let mut guard = shared.write().await;
        // Skip the write entirely if the state is already correct — avoids a
        // needless cache-line dirty on every PartialText event.
        if guard.as_str() != state_str {
            guard.replace_range(.., state_str);
        }
    }

    if let Err(e) = dbus::emit_state_changed(conn, state_str).await {
        tracing::warn!(error = %e, "failed to emit state_changed signal");
    }
}
