use std::sync::Arc;

use tokio::sync::{RwLock, mpsc};
use zbus::{Connection, connection::Builder, interface, object_server::SignalEmitter};

use crate::events::AppCommand;

const BUS_NAME: &str = "com.lunevoice.Daemon1";
const OBJECT_PATH: &str = "/com/lunevoice/Daemon1";

pub struct DaemonInterface {
    cmd_tx: mpsc::Sender<AppCommand>,
    state: Arc<RwLock<String>>,
}

impl DaemonInterface {
    pub fn new(cmd_tx: mpsc::Sender<AppCommand>, state: Arc<RwLock<String>>) -> Self {
        Self { cmd_tx, state }
    }
}

#[interface(name = "com.lunevoice.Daemon1")]
impl DaemonInterface {
    async fn toggle(&self) {
        let _ = self.cmd_tx.send(AppCommand::ToggleDictation).await;
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

    #[zbus(signal)]
    async fn state_changed(emitter: &SignalEmitter<'_>, state: &str) -> zbus::Result<()>;
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
    DaemonInterface::state_changed(&emitter, state).await
}
