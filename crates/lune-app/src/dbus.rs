use std::sync::Arc;

use tokio::sync::{RwLock, mpsc, watch};
use zbus::{Connection, connection::Builder, interface, object_server::SignalEmitter};

use lune_core::config::{
    ActivationMode, AsrBackendConfig, DictationMode, HotkeyConfig, LlmBackendConfig, PrivacyMode,
    UserSettings, builtin_registry,
};

use crate::events::AppCommand;

/// Lightweight BCP-47 well-formedness check (RFC 5646 §2.1).
/// Validates structural rules only — does not consult the IANA subtag registry.
/// Primary subtag: 2-8 ASCII alpha. Subsequent subtags: 1-8 ASCII alphanumeric.
fn is_wellformed_bcp47(tag: &str) -> bool {
    let mut parts = tag.split('-');
    match parts.next() {
        Some(p) if (2..=8).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_alphabetic()) => {}
        _ => return false,
    }
    parts.all(|p| {
        let len = p.len();
        (1..=8).contains(&len) && p.bytes().all(|b| b.is_ascii_alphanumeric())
    })
}

const BUS_NAME: &str = "com.lunevoice.Daemon1";
const OBJECT_PATH: &str = "/com/lunevoice/Daemon1";

pub struct DaemonInterface {
    cmd_tx: mpsc::Sender<AppCommand>,
    state: Arc<RwLock<String>>,
    settings: Arc<RwLock<UserSettings>>,
    dictation_mode_tx: watch::Sender<DictationMode>,
    activation_mode_tx: watch::Sender<ActivationMode>,
    settings_store: Arc<lune_persist::SettingsStore>,
    rt_handle: tokio::runtime::Handle,
    persist_gate: Arc<tokio::sync::Mutex<()>>,
}

impl DaemonInterface {
    #[inline]
    pub fn new(
        cmd_tx: mpsc::Sender<AppCommand>,
        state: Arc<RwLock<String>>,
        settings: Arc<RwLock<UserSettings>>,
        dictation_mode_tx: watch::Sender<DictationMode>,
        activation_mode_tx: watch::Sender<ActivationMode>,
        settings_store: Arc<lune_persist::SettingsStore>,
        rt_handle: tokio::runtime::Handle,
    ) -> Self {
        Self {
            cmd_tx,
            state,
            settings,
            dictation_mode_tx,
            activation_mode_tx,
            settings_store,
            rt_handle,
            persist_gate: Arc::new(tokio::sync::Mutex::new(())),
        }
    }
}

#[interface(name = "com.lunevoice.Daemon1")]
impl DaemonInterface {
    async fn toggle(&self) {
        self.send_cmd(AppCommand::ToggleDictation);
    }

    async fn start(&self) {
        self.send_cmd(AppCommand::StartDictation);
    }

    async fn stop(&self) {
        self.send_cmd(AppCommand::StopDictation);
    }

    async fn cancel(&self) {
        self.send_cmd(AppCommand::CancelDictation);
    }

    async fn quit(&self) {
        self.send_cmd(AppCommand::Quit);
    }

    async fn status(&self) -> String {
        if let Ok(guard) = self.state.try_read() {
            return guard.clone();
        }
        self.state.read().await.clone()
    }

    // ── Hot-path settings (watch channels) ───────────────────────────────

    async fn get_dictation_mode(&self) -> String {
        self.dictation_mode_tx.borrow().to_string()
    }

    async fn set_dictation_mode(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        mode_str: &str,
    ) -> zbus::fdo::Result<()> {
        let mode: DictationMode = mode_str.parse().map_err(zbus::fdo::Error::InvalidArgs)?;
        if *self.dictation_mode_tx.borrow() != mode {
            self.dictation_mode_tx.send_replace(mode);
            self.settings.write().await.dictation_mode = mode;
            self.persist_setting();
            let mode_s = mode.to_string();
            let _ = emitter.dictation_mode_changed(&mode_s).await;
            let _ = emitter.settings_changed("dictation_mode").await;
        }
        Ok(())
    }

    async fn get_activation_mode(&self) -> String {
        self.activation_mode_tx.borrow().to_string()
    }

    async fn set_activation_mode(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        mode_str: &str,
    ) -> zbus::fdo::Result<()> {
        let mode: ActivationMode = mode_str.parse().map_err(zbus::fdo::Error::InvalidArgs)?;
        if *self.activation_mode_tx.borrow() != mode {
            self.activation_mode_tx.send_replace(mode);
            self.settings.write().await.activation_mode = mode;
            self.persist_setting();
            let mode_s = mode.to_string();
            let _ = emitter.activation_mode_changed(&mode_s).await;
            let _ = emitter.settings_changed("activation_mode").await;
        }
        Ok(())
    }

    // ── Session settings (applied at next session start) ─────────────────────────────

    async fn get_default_style(&self) -> String {
        self.settings.read().await.default_style.clone()
    }

    async fn set_default_style(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        id: &str,
    ) -> zbus::fdo::Result<()> {
        if builtin_registry().get(id).is_none() {
            return Err(zbus::fdo::Error::InvalidArgs(format!(
                "unknown style preset: {id}"
            )));
        }
        {
            let mut s = self.settings.write().await;
            if s.default_style == id {
                return Ok(());
            }
            s.default_style = id.into();
        }
        self.persist_setting();
        let _ = emitter.settings_changed("default_style").await;
        Ok(())
    }

    async fn get_language(&self) -> String {
        self.settings.read().await.language.clone()
    }

    async fn set_language(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        lang: &str,
    ) -> zbus::fdo::Result<()> {
        if !is_wellformed_bcp47(lang) {
            return Err(zbus::fdo::Error::InvalidArgs(
                "language must be a well-formed BCP-47 tag (e.g. \"en\", \"en-US\", \"pt-BR\")"
                    .into(),
            ));
        }
        {
            let mut s = self.settings.write().await;
            if s.language == lang {
                return Ok(());
            }
            s.language = lang.into();
        }
        self.persist_setting();
        let _ = emitter.settings_changed("language").await;
        Ok(())
    }

    // ── Cold settings (persist only) ─────────────────────────────────────

    async fn get_privacy_mode(&self) -> String {
        self.settings.read().await.privacy_mode.to_string()
    }

    async fn set_privacy_mode(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        mode_str: &str,
    ) -> zbus::fdo::Result<()> {
        let mode: PrivacyMode = mode_str.parse().map_err(zbus::fdo::Error::InvalidArgs)?;
        {
            let mut s = self.settings.write().await;
            if s.privacy_mode == mode {
                return Ok(());
            }
            s.privacy_mode = mode;
        }
        self.persist_setting();
        let _ = emitter.settings_changed("privacy_mode").await;
        Ok(())
    }

    async fn get_session_history_days(&self) -> i32 {
        match self.settings.read().await.session_history_days {
            None => -1,
            Some(n) => n as i32,
        }
    }

    async fn set_session_history_days(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        days: i32,
    ) -> zbus::fdo::Result<()> {
        let value = if days < 0 { None } else { Some(days as u32) };
        {
            let mut s = self.settings.write().await;
            if s.session_history_days == value {
                return Ok(());
            }
            s.session_history_days = value;
        }
        self.persist_setting();
        let _ = emitter.settings_changed("session_history_days").await;
        Ok(())
    }

    async fn get_hotkey(&self) -> zbus::fdo::Result<String> {
        let h = self.settings.read().await.hotkey.clone();
        serde_json::to_string(&h).map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }

    async fn set_hotkey(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        json: &str,
    ) -> zbus::fdo::Result<()> {
        let hotkey: HotkeyConfig =
            serde_json::from_str(json).map_err(|e| zbus::fdo::Error::InvalidArgs(e.to_string()))?;
        self.settings.write().await.hotkey = hotkey;
        self.persist_setting();
        let _ = emitter.settings_changed("hotkey").await;
        Ok(())
    }

    // ── Backend settings (applied on daemon restart) ────────────────────────────────

    async fn get_asr_backend(&self) -> zbus::fdo::Result<String> {
        let cfg = self.settings.read().await.asr_backend.clone();
        serde_json::to_string(&cfg).map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }

    async fn set_asr_backend(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        json: &str,
    ) -> zbus::fdo::Result<()> {
        let cfg: AsrBackendConfig =
            serde_json::from_str(json).map_err(|e| zbus::fdo::Error::InvalidArgs(e.to_string()))?;
        self.settings.write().await.asr_backend = cfg;
        self.persist_setting();
        tracing::info!("ASR backend config updated — restart daemon to load new model");
        let _ = emitter.settings_changed("asr_backend").await;
        Ok(())
    }

    async fn get_llm_backend(&self) -> zbus::fdo::Result<String> {
        let cfg = self.settings.read().await.llm_backend.clone();
        serde_json::to_string(&cfg).map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }

    async fn set_llm_backend(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        json: &str,
    ) -> zbus::fdo::Result<()> {
        let cfg: LlmBackendConfig =
            serde_json::from_str(json).map_err(|e| zbus::fdo::Error::InvalidArgs(e.to_string()))?;
        self.settings.write().await.llm_backend = cfg;
        self.persist_setting();
        tracing::info!("LLM backend config updated — restart daemon to load new model");
        let _ = emitter.settings_changed("llm_backend").await;
        Ok(())
    }

    // ── Read-only dump ───────────────────────────────────────────────────

    async fn get_all_settings(&self) -> zbus::fdo::Result<String> {
        let snapshot = self.settings.read().await.clone();
        toml::to_string_pretty(&snapshot).map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }

    // ── Signals ──────────────────────────────────────────────────────────

    #[zbus(signal)]
    async fn state_changed(emitter: &SignalEmitter<'_>, state: &str) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn dictation_mode_changed(emitter: &SignalEmitter<'_>, mode: &str) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn activation_mode_changed(emitter: &SignalEmitter<'_>, mode: &str) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn settings_changed(emitter: &SignalEmitter<'_>, key: &str) -> zbus::Result<()>;
}

impl DaemonInterface {
    /// Non-async command dispatch. `try_send` avoids async suspension + future allocation
    /// on the hot path; drops silently on backpressure (channel full) with a debug log.
    #[inline(always)]
    fn send_cmd(&self, cmd: AppCommand) {
        if self.cmd_tx.try_send(cmd).is_err() {
            tracing::debug!("command channel full, dropping d-bus command");
        }
    }

    /// Fire-and-forget persist of the full settings snapshot.
    #[inline]
    fn persist_setting(&self) {
        let settings = Arc::clone(&self.settings);
        let store = Arc::clone(&self.settings_store);
        let persist_gate = Arc::clone(&self.persist_gate);
        self.rt_handle.spawn(async move {
            let snapshot = settings.read().await.clone();
            let _guard = persist_gate.lock().await;
            match tokio::task::spawn_blocking(move || store.save(&snapshot)).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, "failed to persist settings"),
                Err(e) => tracing::warn!(error = %e, "settings persist task join failure"),
            }
        });
    }
}

/// Shared interface-ref lookup used by all signal emitters — avoids repeating the
/// object-server lookup and reduces binary size.
#[inline]
async fn get_iface_ref(
    conn: &Connection,
) -> zbus::Result<zbus::object_server::InterfaceRef<DaemonInterface>> {
    conn.object_server()
        .interface::<_, DaemonInterface>(OBJECT_PATH)
        .await
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
#[inline]
pub async fn emit_state_changed(conn: &Connection, state: &str) -> zbus::Result<()> {
    let iface_ref = get_iface_ref(conn).await?;
    DaemonInterface::state_changed(iface_ref.signal_emitter(), state).await
}

/// Emit a `dictation_mode_changed` signal.
#[allow(dead_code)]
#[inline]
pub async fn emit_dictation_mode_changed(conn: &Connection, mode: &str) -> zbus::Result<()> {
    let iface_ref = get_iface_ref(conn).await?;
    DaemonInterface::dictation_mode_changed(iface_ref.signal_emitter(), mode).await
}

/// Emit an `activation_mode_changed` signal.
#[allow(dead_code)]
#[inline]
pub async fn emit_activation_mode_changed(conn: &Connection, mode: &str) -> zbus::Result<()> {
    let iface_ref = get_iface_ref(conn).await?;
    DaemonInterface::activation_mode_changed(iface_ref.signal_emitter(), mode).await
}

/// Emit a `settings_changed` signal.
#[allow(dead_code)]
#[inline]
pub async fn emit_settings_changed(conn: &Connection, key: &str) -> zbus::Result<()> {
    let iface_ref = get_iface_ref(conn).await?;
    DaemonInterface::settings_changed(iface_ref.signal_emitter(), key).await
}

#[cfg(test)]
mod tests {
    use super::is_wellformed_bcp47;

    #[test]
    fn valid_bcp47_tags() {
        for tag in [
            "en",
            "en-US",
            "en-GB",
            "pt-BR",
            "zh-Hans",
            "zh-Hans-CN",
            "sr-Latn-RS",
        ] {
            assert!(is_wellformed_bcp47(tag), "should accept {tag}");
        }
    }

    #[test]
    fn rejects_malformed_tags() {
        for tag in [
            "",
            "e",
            "123",
            "en-",
            "-en",
            "en--US",
            "en-abcdefghi",
            "en US",
        ] {
            assert!(!is_wellformed_bcp47(tag), "should reject {tag:?}");
        }
    }
}
