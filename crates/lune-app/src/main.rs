mod cli;
mod dbus;
mod events;
mod factory;
mod notes;

use std::sync::Arc;

use tokio::sync::{RwLock, mpsc};
use tracing_subscriber::{EnvFilter, fmt};

use lune_app::{SessionManager, SessionManagerConfig};

use crate::events::{AppCommand, AppStateUpdate};

fn main() {
    let subcommand = std::env::args().nth(1);

    match subcommand.as_deref() {
        Some("toggle" | "cancel" | "status" | "quit") => {
            cli::run_cli(subcommand.as_deref().unwrap());
        }
        Some("daemon") | None => run_daemon(),
        Some(other) => {
            eprintln!("unknown subcommand: {other}");
            eprintln!("usage: lune-voice [toggle|cancel|status|quit|daemon]");
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

    let settings_store = lune_persist::SettingsStore::new(paths.settings_file());
    let mut settings = settings_store.load().expect("load settings");

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
    eprintln!("    Ctrl+Space  →  lune-voice toggle");
    eprintln!("    (optional)  Escape  →  lune-voice cancel");
    eprintln!();

    // ── Build tokio current_thread runtime ────────────────────────────────

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let local = tokio::task::LocalSet::new();

    // ── Phase 1: D-Bus connection OUTSIDE LocalSet ───────────────────────
    // zbus internally calls tokio::spawn, which panics inside LocalSet.

    let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(64);
    let shared_state = Arc::new(RwLock::new("idle".to_string()));

    let dbus_iface = dbus::DaemonInterface::new(cmd_tx, Arc::clone(&shared_state));

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
        )
        .await;
    });

    tracing::info!("lune-voice stopped");
}

async fn run_pipeline(
    settings: lune_core::config::UserSettings,
    db_path: std::path::PathBuf,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
    shared_state: Arc<RwLock<String>>,
    dbus_conn: &zbus::Connection,
) {
    // ── Build pipeline components ────────────────────────────────────────

    tracing::info!("building ASR backend...");
    let asr = match factory::build_asr_backend(&settings) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(error = %e, "failed to build ASR backend");
            return;
        }
    };

    tracing::info!("ASR backend ready, building LLM post-processor...");
    let post_processor = match factory::build_post_processor(&settings) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "failed to build LLM post-processor");
            return;
        }
    };

    tracing::info!("LLM post-processor ready");
    let audio =
        lune_audio::AudioCaptureService::new(lune_core::audio::AudioCaptureConfig::default());
    let router = lune_commands::VoiceCommandRouter::new();

    tracing::info!("opening database...");
    factory::ensure_db_path(&db_path).expect("ensure DB path");
    let db = lune_persist::Database::open(&db_path).expect("open database");

    let dict_db = lune_persist::Database::open_in_memory().expect("open dict database");
    let dict = lune_dict::DictionaryManager::new(dict_db).expect("init dictionary");

    let style_preset = lune_core::config::StylePreset::all_builtin()
        .into_iter()
        .find(|p| p.id == settings.default_style)
        .unwrap_or_else(lune_core::config::StylePreset::general);

    let config = SessionManagerConfig {
        style_preset,
        output_mode: settings.output_mode,
        asr_backend_kind: match settings.asr_backend {
            lune_core::config::AsrBackendConfig::ParakeetTdt { .. } => {
                lune_core::asr::AsrBackendKind::ParakeetTdt
            }
            lune_core::config::AsrBackendConfig::OpenAiCompatible { .. } => {
                lune_core::asr::AsrBackendKind::OpenAiCompatible
            }
            _ => lune_core::asr::AsrBackendKind::ParakeetTdt,
        },
        ..Default::default()
    };

    let manager = SessionManager::new(audio, asr, router, post_processor, dict, db, config);

    // ── Terminal notes view ──────────────────────────────────────────────

    let mut notes = notes::NotesView::new();
    notes.render(&AppStateUpdate::Idle);

    tracing::info!("pipeline ready, waiting for commands");

    // ── Command loop ─────────────────────────────────────────────────────

    loop {
        match cmd_rx.recv().await {
            Some(AppCommand::ToggleDictation) => {
                if manager.active_session_id().await.is_some() {
                    tracing::info!("stopping dictation");
                    manager.stop().await;
                    // Idle update emitted by forwarder via SessionEvent::Stopped
                } else {
                    tracing::info!("starting dictation");
                    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);

                    let state_ref = Arc::clone(&shared_state);
                    let conn_clone = dbus_conn.clone();
                    let mut notes_fwd = notes::NotesView::new();
                    tokio::task::spawn_local(async move {
                        while let Some(event) = event_rx.recv().await {
                            let update = match event {
                                lune_core::session::SessionEvent::Started => {
                                    AppStateUpdate::Listening
                                }
                                lune_core::session::SessionEvent::PartialText(t) => {
                                    AppStateUpdate::PartialText(t)
                                }
                                lune_core::session::SessionEvent::FinalizedText(t) => {
                                    AppStateUpdate::FinalText(t)
                                }
                                lune_core::session::SessionEvent::Error(e) => {
                                    AppStateUpdate::Error(e)
                                }
                                lune_core::session::SessionEvent::Stopped => AppStateUpdate::Idle,
                                _ => continue,
                            };
                            update_state(&state_ref, &mut notes_fwd, &conn_clone, update).await;
                        }
                    });

                    match manager.start(event_tx).await {
                        Ok(id) => {
                            // Listening update emitted by forwarder via SessionEvent::Started
                            tracing::info!(%id, "session started");
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "failed to start session");
                            update_state(
                                &shared_state,
                                &mut notes,
                                dbus_conn,
                                AppStateUpdate::Error(e.to_string()),
                            )
                            .await;
                        }
                    }
                }
            }
            Some(AppCommand::CancelDictation) => {
                tracing::info!("cancelling dictation");
                manager.cancel().await;
                // Idle update emitted by forwarder via SessionEvent::Stopped
            }
            Some(AppCommand::Quit) => {
                tracing::info!("quit command received");
                manager.stop().await;
                update_state(&shared_state, &mut notes, dbus_conn, AppStateUpdate::Idle).await;
                break;
            }
            None => {
                tracing::info!("command channel closed, shutting down");
                manager.stop().await;
                break;
            }
        }
    }
}

async fn update_state(
    shared: &Arc<RwLock<String>>,
    notes: &mut notes::NotesView,
    conn: &zbus::Connection,
    update: AppStateUpdate,
) {
    let state_str = match &update {
        AppStateUpdate::Idle => "idle",
        AppStateUpdate::Listening => "listening",
        AppStateUpdate::PartialText(_) => "listening",
        AppStateUpdate::FinalText(_) => "idle",
        AppStateUpdate::Error(_) => "error",
    };

    *shared.write().await = state_str.to_string();
    notes.render(&update);

    if let Err(e) = dbus::emit_state_changed(conn, state_str).await {
        tracing::warn!(error = %e, "failed to emit state_changed signal");
    }
}
