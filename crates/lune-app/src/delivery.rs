use std::sync::mpsc;

use lune_core::config::{DictationMode, TerminalMode};
use tracing::{debug, info, warn};

/// Message sent to the persistent delivery thread.
struct DeliveryMsg {
    text: Box<str>,
    mode: DictationMode,
}

/// Handle to the background delivery thread.
#[derive(Clone)]
pub struct DeliveryService {
    tx: mpsc::Sender<DeliveryMsg>,
}

impl DeliveryService {
    pub fn new(terminal_mode: TerminalMode) -> Self {
        let (tx, rx) = mpsc::channel::<DeliveryMsg>();
        std::thread::Builder::new()
            .name("delivery".into())
            .spawn(move || delivery_thread(rx, terminal_mode))
            .expect("spawn delivery thread");
        Self { tx }
    }

    #[inline]
    pub fn deliver(&self, text: &str, mode: DictationMode) {
        if text.is_empty() {
            return;
        }
        debug!(?mode, text_len = text.len(), "delivering text");
        if let Err(e) = self.tx.send(DeliveryMsg {
            text: text.into(),
            mode,
        }) {
            warn!(error = %e, "delivery channel disconnected");
        }
    }

    #[inline]
    pub fn deliver_owned(&self, text: String, mode: DictationMode) {
        if text.is_empty() {
            return;
        }
        debug!(?mode, text_len = text.len(), "delivering text");
        if let Err(e) = self.tx.send(DeliveryMsg {
            text: text.into_boxed_str(),
            mode,
        }) {
            warn!(error = %e, "delivery channel disconnected");
        }
    }
}

fn delivery_thread(rx: mpsc::Receiver<DeliveryMsg>, _terminal_mode: TerminalMode) {
    let mut driver = match PlatformDriver::new() {
        Ok(d) => d,
        Err(e) => {
            warn!(error = %e, "text delivery unavailable on this platform");
            return;
        }
    };

    info!(platform = driver.platform_name(), "delivery thread ready");

    while let Ok(msg) = rx.recv() {
        let text: &str = &msg.text;
        match msg.mode {
            DictationMode::Clipboard | DictationMode::Both => driver.type_text(text),
            DictationMode::TypeInPlace => driver.type_keys(text),
            _ => driver.type_text(text),
        }
    }

    debug!("delivery thread exiting");
}

// ─── Cross-platform: arboard + enigo ────────────────────────────────────────

struct PlatformDriver {
    clipboard: arboard::Clipboard,
    enigo: enigo::Enigo,
}

impl PlatformDriver {
    fn new() -> Result<Self, String> {
        let clipboard =
            arboard::Clipboard::new().map_err(|e| format!("clipboard init failed: {e}"))?;
        let enigo = enigo::Enigo::new(&enigo::Settings::default())
            .map_err(|e| format!("enigo init failed: {e}"))?;
        Ok(Self { clipboard, enigo })
    }

    fn platform_name(&self) -> &'static str {
        if cfg!(target_os = "linux") {
            "linux-arboard-enigo"
        } else if cfg!(target_os = "macos") {
            "macos-arboard-enigo"
        } else {
            "windows-arboard-enigo"
        }
    }

    fn type_text(&mut self, text: &str) {
        if let Err(e) = self.clipboard.set_text(text) {
            warn!(error = %e, "clipboard set failed, falling back to per-key typing");
            self.type_keys(text);
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(30));
        self.emit_paste();
    }

    fn emit_paste(&mut self) {
        use enigo::{Direction, Key, Keyboard};
        let modifier = if cfg!(target_os = "macos") {
            Key::Meta
        } else {
            Key::Control
        };
        let _ = self.enigo.key(modifier, Direction::Press);
        let _ = self.enigo.key(Key::Unicode('v'), Direction::Click);
        let _ = self.enigo.key(modifier, Direction::Release);
    }

    fn type_keys(&mut self, text: &str) {
        use enigo::Keyboard;
        if let Err(e) = self.enigo.text(text) {
            warn!(error = %e, "per-key typing failed");
        }
    }
}
