// mimalloc replaces the system allocator for better multi-threaded allocation
// throughput. The inference pipeline spawns worker threads via ORT, and the
// ASR streaming loop allocates/frees many small Vecs per chunk. Benchmarks on
// allocation-heavy workloads show 5-15% throughput gain with no behaviour change.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod cli;
mod dbus;
mod delivery;
mod events;
mod factory;
#[cfg(unix)]
mod hotkey_listener;
#[cfg(unix)]
mod uinput_typer;

use std::sync::Arc;

use tokio::sync::{RwLock, mpsc, watch};
use tracing_subscriber::{EnvFilter, fmt};

use lune_core::command::VoiceCommand;
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

    settings.migrate_pipeline_fields();

    let models = lune_persist::ensure_default_models(
        &paths.models_dir(),
        &settings.asr_backend,
        &settings.llm_backend,
    )
    .expect("ensure models downloaded");

    factory::patch_model_paths(&mut settings, &models);

    // ── Global ORT initialization ────────────────────────────────────────
    // Must happen before any Session is created. When CUDA is available and
    // the user wants GPU, register the CUDA EP globally so all ORT sessions
    // (ASR encoder, ASR decoder, LLM decoder) share the same CUDA memory arena.
    factory::init_ort_runtime(&settings);

    let (asr_backend_kind, selected_asr_dir) = match &settings.asr_backend {
        lune_core::config::AsrBackendConfig::ParakeetTdt { model_dir, .. } => {
            ("ParakeetTdt", model_dir.display().to_string())
        }
        lune_core::config::AsrBackendConfig::OrtQwenAsr { model_dir, .. } => {
            ("OrtQwenAsr", model_dir.display().to_string())
        }
        lune_core::config::AsrBackendConfig::OpenAiCompatible { .. } => {
            ("OpenAiCompatible", "<remote>".into())
        }
        _ => ("Unknown", "<unknown>".into()),
    };

    tracing::info!(
        asr_backend = asr_backend_kind,
        asr_dir = %selected_asr_dir,
        llm_dir = %models.llm_model_path.display(),
        "models ready"
    );

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
    let pipeline_cmd_tx = cmd_tx.clone();
    let hotkey_cmd_tx = cmd_tx.clone();
    // Pre-allocate capacity for the longest state string ("listening" = 9 chars)
    let shared_state = Arc::new(RwLock::new(String::with_capacity(16)));

    // Initialise to "idle" without a second allocation
    {
        let mut guard = rt.block_on(shared_state.write());
        guard.push_str("idle");
    }

    let (dictation_mode_tx, dictation_mode_rx) = watch::channel(settings.dictation_mode);
    let (activation_mode_tx, activation_mode_rx) = watch::channel(settings.activation_mode);

    let hotkey_config = settings.hotkey.clone();
    let hotkey_activation_rx = activation_mode_rx.clone();

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

    // ── Hotkey listener (evdev) ──────────────────────────────────────────
    // Spawns OS threads that read from physical keyboards and dispatch
    // AppCommands to the same channel as D-Bus. D-Bus CLI commands still
    // work as a fallback (e.g. from scripts or remote control).

    #[cfg(unix)]
    let _hotkey_handle = hotkey_listener::spawn(hotkey_config, hotkey_activation_rx, hotkey_cmd_tx);

    // ── Phase 2: pipeline inside LocalSet ────────────────────────────────

    local.block_on(&rt, async move {
        run_pipeline(
            settings,
            paths.database_file(),
            paths.adaptive_vad_file(),
            pipeline_cmd_tx,
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

#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    settings: Arc<RwLock<lune_core::config::UserSettings>>,
    db_path: std::path::PathBuf,
    vad_state_path: std::path::PathBuf,
    cmd_tx: mpsc::Sender<AppCommand>,
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
    let post_processor = match factory::build_post_processor(&snap).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "failed to build LLM post-processor");
            return;
        }
    };

    tracing::info!("LLM post-processor ready");
    let router = Arc::new(lune_commands::VoiceCommandRouter::new());

    let aligner = match (&snap.aligner, asr.as_ort_qwen()) {
        (Some(cfg), Some(backend)) if cfg.enabled => {
            match backend.make_forced_aligner(&cfg.model_dir) {
                Ok(aligner) => {
                    let fa = std::sync::Arc::new(aligner);
                    // Attach to ASR backend for timestamp-based chunk merging.
                    backend.attach_aligner(std::sync::Arc::clone(&fa));
                    Some(fa)
                }
                Err(e) => {
                    tracing::error!(error = %e, model_dir = %cfg.model_dir.display(), "failed to build forced aligner");
                    return;
                }
            }
        }
        (Some(cfg), None) if cfg.enabled => {
            tracing::warn!("forced aligner requires OrtQwen ASR backend; skipping");
            None
        }
        _ => None,
    };

    tracing::info!("opening database...");
    factory::ensure_db_path(&db_path).expect("ensure DB path");
    let db = Arc::new(tokio::sync::Mutex::new(
        lune_persist::Database::open(&db_path).expect("open database"),
    ));

    let dict_db = lune_persist::Database::open_in_memory().expect("open dict database");
    let dict = Arc::new(tokio::sync::Mutex::new(
        lune_dict::DictionaryManager::new(dict_db).expect("init dictionary"),
    ));

    let pipeline = asr.pipeline_config().with_overrides(&snap.pipeline);
    let config = SessionManagerConfig {
        asr_backend_kind: asr.kind(),
        asr_config: lune_core::asr::AsrSessionConfig {
            language: Some(snap.language.clone()),
            ..Default::default()
        },
        pipeline,
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
    .with_vad_state_path(vad_state_path);
    if let Some(aligner) = aligner {
        manager = manager.with_aligner(aligner);
    }

    let delivery_service = delivery::DeliveryService::new();

    tracing::info!("pipeline ready, waiting for commands");

    // ── Signal handler for graceful shutdown ──────────────────────────────

    // SIGTERM: resolves on Unix when the process receives SIGTERM; on other
    // platforms this future never resolves (pending forever), which is correct
    // since those platforms don't have SIGTERM.
    let sigterm = wait_for_sigterm();
    tokio::pin!(sigterm);

    // ── Command loop ─────────────────────────────────────────────────────

    loop {
        let cmd = tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("SIGINT received, shutting down");
                manager.stop().await;
                update_state(&shared_state, dbus_conn, AppStateUpdate::Idle).await;
                break;
            }
            _ = &mut sigterm => {
                tracing::info!("SIGTERM received, shutting down");
                manager.stop().await;
                update_state(&shared_state, dbus_conn, AppStateUpdate::Idle).await;
                break;
            }
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(c) => c,
                    None => {
                        tracing::info!("command channel closed, shutting down");
                        manager.stop().await;
                        break;
                    }
                }
            }
        };

        match cmd {
            AppCommand::ToggleDictation => {
                if manager.active_session_id().await.is_some() {
                    stop_dictation(&manager, &shared_state, dbus_conn).await;
                } else {
                    start_dictation(
                        &manager,
                        cmd_tx.clone(),
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
                        cmd_tx.clone(),
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
    cmd_tx: mpsc::Sender<AppCommand>,
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
                lune_core::session::SessionEvent::Processing => AppStateUpdate::Processing,
                lune_core::session::SessionEvent::PartialText(_) => AppStateUpdate::PartialText,
                lune_core::session::SessionEvent::FinalizedText(t) => {
                    // `borrow()` is lock-free on watch channels — cheap hot path
                    let mode = *dictation_mode_rx.borrow();
                    delivery_service.deliver(&t, mode);
                    AppStateUpdate::FinalText
                }
                lune_core::session::SessionEvent::CommandDetected(payload) => {
                    let mode = *dictation_mode_rx.borrow();
                    handle_voice_command_payload(&payload, mode, &delivery_service, &cmd_tx).await;
                    continue;
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

#[inline]
async fn handle_voice_command_payload(
    payload: &str,
    mode: DictationMode,
    delivery_service: &delivery::DeliveryService,
    cmd_tx: &mpsc::Sender<AppCommand>,
) {
    let Some(command) = parse_voice_command_payload(payload) else {
        tracing::warn!(payload = payload, "failed to parse voice command payload");
        return;
    };

    match command {
        VoiceCommand::StopDictation => {
            if cmd_tx.send(AppCommand::StopDictation).await.is_err() {
                tracing::warn!("failed to dispatch stop_dictation voice command");
            }
        }
        VoiceCommand::NewLine => delivery_service.deliver("\n", mode),
        VoiceCommand::NewParagraph => delivery_service.deliver("\n\n", mode),
        VoiceCommand::ListItem => delivery_service.deliver("\n- ", mode),
        // Handled inside the session loop so new utterances use the new preset immediately.
        VoiceCommand::SwitchStyle(_) => {}
        VoiceCommand::InsertSnippet(snippet) => {
            tracing::info!(snippet, "voice command insert_snippet not yet implemented");
        }
        VoiceCommand::ScratchThat => {
            tracing::info!("voice command scratch_that not yet implemented");
        }
        _ => {
            tracing::info!("voice command variant not yet implemented");
        }
    }
}

#[inline]
fn parse_voice_command_payload(payload: &str) -> Option<VoiceCommand> {
    if let Ok(cmd) = serde_json::from_str::<VoiceCommand>(payload) {
        return Some(cmd);
    }

    match payload.trim() {
        "StopDictation" => Some(VoiceCommand::StopDictation),
        "NewLine" => Some(VoiceCommand::NewLine),
        "NewParagraph" => Some(VoiceCommand::NewParagraph),
        "ListItem" => Some(VoiceCommand::ListItem),
        "ScratchThat" => Some(VoiceCommand::ScratchThat),
        _ => None,
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
        AppStateUpdate::Processing => "processing",
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

/// Resolves when SIGTERM is received on Unix; never resolves on other platforms.
#[cfg(unix)]
async fn wait_for_sigterm() {
    let mut sig = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("register SIGTERM handler");
    sig.recv().await;
}

#[cfg(not(unix))]
async fn wait_for_sigterm() {
    std::future::pending::<()>().await
}

#[cfg(test)]
mod tests {
    use super::parse_voice_command_payload;
    use lune_core::command::VoiceCommand;

    #[test]
    fn parses_voice_command_json_payload() {
        let payload = serde_json::to_string(&VoiceCommand::SwitchStyle("formal".into())).unwrap();
        let parsed = parse_voice_command_payload(&payload);
        match parsed {
            Some(VoiceCommand::SwitchStyle(style)) => assert_eq!(style, "formal"),
            other => panic!("unexpected parse result: {other:?}"),
        }
    }

    #[test]
    fn parses_legacy_debug_payload() {
        assert!(matches!(
            parse_voice_command_payload("StopDictation"),
            Some(VoiceCommand::StopDictation)
        ));
        assert!(matches!(
            parse_voice_command_payload("NewParagraph"),
            Some(VoiceCommand::NewParagraph)
        ));
    }
}
