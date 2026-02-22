use std::sync::Arc;

use tokio::sync::{RwLock, mpsc, watch};
use zbus::{Connection, connection::Builder, interface, object_server::SignalEmitter};

use lune_core::config::{ActivationMode, DictationMode};

use crate::events::AppCommand;

const BUS_NAME: &str = "com.lunevoice.Daemon1";
const OBJECT_PATH: &str = "/com/lunevoice/Daemon1";

pub struct DaemonInterface {
    cmd_tx: mpsc::Sender<AppCommand>,
    state: Arc<RwLock<String>>,
    dictation_mode_tx: watch::Sender<DictationMode>,
    activation_mode_tx: watch::Sender<ActivationMode>,
    settings_store: Arc<lune_persist::SettingsStore>,
}

impl DaemonInterface {
    pub fn new(
        cmd_tx: mpsc::Sender<AppCommand>,
        state: Arc<RwLock<String>>,
        dictation_mode_tx: watch::Sender<DictationMode>,
        activation_mode_tx: watch::Sender<ActivationMode>,
        settings_store: Arc<lune_persist::SettingsStore>,
    ) -> Self {
        Self {
            cmd_tx,
            state,
            dictation_mode_tx,
            activation_mode_tx,
            settings_store,
        }
    }
}

#[interface(name = "com.lunevoice.Daemon1")]
impl DaemonInterface {
    async fn toggle(&self) {
        let _ = self.cmd_tx.send(AppCommand::ToggleDictation).await;
    }

    async fn start(&self) {
        let _ = self.cmd_tx.send(AppCommand::StartDictation).await;
    }

    async fn stop(&self) {
        let _ = self.cmd_tx.send(AppCommand::StopDictation).await;
    }

    async fn cancel(&self) {
        let _ = self.cmd_tx.send(AppCommand::CancelDictation).await;
    }

    async fn quit(&self) {
        let _ = self.cmd_tx.send(AppCommand::Quit).await;
    }

    async fn status(&self) -> String {
        self.state.read().await.clone()
    }

    async fn get_dictation_mode(&self) -> String {
        self.dictation_mode_tx.borrow().to_string()
    }

    async fn set_dictation_mode(&self, mode_str: &str) -> zbus::fdo::Result<()> {
        let mode: DictationMode = mode_str
            .parse()
            .map_err(|e: String| zbus::fdo::Error::InvalidArgs(e))?;
        self.dictation_mode_tx.send_replace(mode);
        self.persist_modes();
        Ok(())
    }

    async fn get_activation_mode(&self) -> String {
        self.activation_mode_tx.borrow().to_string()
    }

    async fn set_activation_mode(&self, mode_str: &str) -> zbus::fdo::Result<()> {
        let mode: ActivationMode = mode_str
            .parse()
            .map_err(|e: String| zbus::fdo::Error::InvalidArgs(e))?;
        self.activation_mode_tx.send_replace(mode);
        self.persist_modes();
        Ok(())
    }

    #[zbus(signal)]
    async fn state_changed(emitter: &SignalEmitter<'_>, state: &str) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn dictation_mode_changed(emitter: &SignalEmitter<'_>, mode: &str) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn activation_mode_changed(emitter: &SignalEmitter<'_>, mode: &str) -> zbus::Result<()>;
}

impl DaemonInterface {
    fn persist_modes(&self) {
        let store = Arc::clone(&self.settings_store);
        let dictation = *self.dictation_mode_tx.borrow();
        let activation = *self.activation_mode_tx.borrow();
        // Fire-and-forget persist — failure is logged, not fatal.
        tokio::task::spawn_local(async move {
            match store.load() {
                Ok(mut settings) => {
                    settings.dictation_mode = dictation;
                    settings.activation_mode = activation;
                    if let Err(e) = store.save(&settings) {
                        tracing::warn!(error = %e, "failed to persist mode change");
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load settings for mode persist");
                }
            }
        });
    }
}

/// Build the D-Bus connection, register the interface, and claim the bus name.
///
/// Must be called outside `LocalSet` — zbus internally uses `tokio::spawn`.
pub async fn build_connection(iface: DaemonInterface) -> zbus::Result<Connection> {
    Builder::session()?
        .name(BUS_NAME)?
        .serve_at(OBJECT_PATH, iface)?
        .build()
        .await
}

/// Emit a `state_changed` signal on the existing connection.
pub async fn emit_state_changed(conn: &Connection, state: &str) -> zbus::Result<()> {
    let iface_ref = conn
        .object_server()
        .interface::<_, DaemonInterface>(OBJECT_PATH)
        .await?;
    let emitter = iface_ref.signal_emitter();
    DaemonInterface::state_changed(emitter, state).await
}

/// Emit a `dictation_mode_changed` signal.
#[allow(dead_code)]
pub async fn emit_dictation_mode_changed(conn: &Connection, mode: &str) -> zbus::Result<()> {
    let iface_ref = conn
        .object_server()
        .interface::<_, DaemonInterface>(OBJECT_PATH)
        .await?;
    let emitter = iface_ref.signal_emitter();
    DaemonInterface::dictation_mode_changed(emitter, mode).await
}

/// Emit an `activation_mode_changed` signal.
#[allow(dead_code)]
pub async fn emit_activation_mode_changed(conn: &Connection, mode: &str) -> zbus::Result<()> {
    let iface_ref = conn
        .object_server()
        .interface::<_, DaemonInterface>(OBJECT_PATH)
        .await?;
    let emitter = iface_ref.signal_emitter();
    DaemonInterface::activation_mode_changed(emitter, mode).await
}
