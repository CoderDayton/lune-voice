use std::sync::mpsc;

use lune_core::config::DictationMode;
use tracing::{debug, warn};

/// Message sent to the persistent delivery thread.
struct DeliveryMsg {
    text: String,
    mode: DictationMode,
}

/// Handle to the background delivery thread.
///
/// Enigo and arboard are initialized once on a dedicated OS thread and reused
/// for every delivery. This avoids per-call Wayland connection churn and memory
/// leaks from accumulating clipboard/input contexts.
#[derive(Clone)]
pub struct DeliveryService {
    tx: mpsc::SyncSender<DeliveryMsg>,
}

impl DeliveryService {
    /// Spawn the delivery thread. Call once at startup.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::sync_channel::<DeliveryMsg>(16);

        std::thread::Builder::new()
            .name("delivery".into())
            .spawn(move || delivery_thread(rx))
            .expect("spawn delivery thread");

        Self { tx }
    }

    /// Queue text for delivery. Non-blocking from async context.
    #[inline]
    pub fn deliver(&self, text: &str, mode: DictationMode) {
        debug!(?mode, text_len = text.len(), "delivering text");

        if let Err(e) = self.tx.try_send(DeliveryMsg {
            text: text.to_owned(),
            mode,
        }) {
            warn!(error = %e, "delivery channel full or disconnected");
        }
    }
}

/// Long-lived thread that owns Enigo + Clipboard and processes deliveries.
fn delivery_thread(rx: mpsc::Receiver<DeliveryMsg>) {
    use arboard::Clipboard;
    use enigo::{Enigo, Settings};

    let mut clipboard = match Clipboard::new() {
        Ok(cb) => cb,
        Err(e) => {
            warn!(error = %e, "failed to init clipboard — delivery disabled");
            return;
        }
    };

    let mut enigo = match Enigo::new(&Settings::default()) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "failed to init enigo — delivery disabled");
            return;
        }
    };

    debug!("delivery thread ready");

    while let Ok(msg) = rx.recv() {
        match msg.mode {
            DictationMode::TypeInPlace => {
                paste_instant(&mut clipboard, &mut enigo, &msg.text);
            }
            DictationMode::Clipboard => {
                clipboard_only(&mut clipboard, &msg.text);
            }
            DictationMode::Both => {
                paste_instant(&mut clipboard, &mut enigo, &msg.text);
            }
            _ => {
                warn!(mode = ?msg.mode, "unknown dictation mode, falling back to TypeInPlace");
                paste_instant(&mut clipboard, &mut enigo, &msg.text);
            }
        }
    }

    debug!("delivery thread exiting");
}

#[inline]
fn paste_instant(clipboard: &mut arboard::Clipboard, enigo: &mut enigo::Enigo, text: &str) {
    use enigo::{Direction, Key, Keyboard};

    if let Err(e) = clipboard.set_text(text) {
        warn!(error = %e, "failed to set clipboard");
        return;
    }

    if let Err(e) = enigo.key(Key::Control, Direction::Press) {
        warn!(error = %e, "failed to press Ctrl");
        return;
    }
    if let Err(e) = enigo.key(Key::Unicode('v'), Direction::Click) {
        warn!(error = %e, "failed to click V");
    }
    if let Err(e) = enigo.key(Key::Control, Direction::Release) {
        warn!(error = %e, "failed to release Ctrl");
    }

    debug!("paste complete");
}

#[inline]
fn clipboard_only(clipboard: &mut arboard::Clipboard, text: &str) {
    if let Err(e) = clipboard.set_text(text) {
        warn!(error = %e, "failed to set clipboard");
    }
}
