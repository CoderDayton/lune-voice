use std::sync::mpsc;

use lune_core::config::DictationMode;
use tracing::{debug, warn};

/// Message sent to the persistent delivery thread.
struct DeliveryMsg {
    text: Box<str>,
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
        // Capacity 64: delivery is fast (clipboard+keypress ~1ms), bursts up to
        // ~60 wps are realistic; 64 slots prevents backpressure under load.
        let (tx, rx) = mpsc::sync_channel::<DeliveryMsg>(64);

        std::thread::Builder::new()
            .name("delivery".into())
            .spawn(move || delivery_thread(rx))
            .expect("spawn delivery thread");

        Self { tx }
    }

    /// Queue text for delivery. Non-blocking from async context.
    ///
    /// Hot path: single branch + one channel send. `Box<str>` avoids the
    /// Vec capacity overhead of `String` for immutable payloads.
    #[inline]
    pub fn deliver(&self, text: &str, mode: DictationMode) {
        if text.is_empty() {
            return;
        }

        debug!(?mode, text_len = text.len(), "delivering text");

        if let Err(e) = self.tx.try_send(DeliveryMsg {
            text: text.into(),
            mode,
        }) {
            warn!(error = %e, "delivery channel full or disconnected");
        }
    }
}

/// Long-lived thread that owns Enigo + Clipboard and processes deliveries.
///
/// Both resources are reused across calls — Wayland/X11 connection setup is
/// amortised to startup. The loop drains the channel without allocation.
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

    // Pre-resolve the two hot variants at loop entry to eliminate repeated
    // enum dispatch overhead. `recv()` blocks until a message arrives or the
    // sender is dropped, so no busy-spin.
    while let Ok(msg) = rx.recv() {
        let text: &str = &msg.text;

        match msg.mode {
            DictationMode::Clipboard => {
                clipboard_only(&mut clipboard, text);
            }
            DictationMode::TypeInPlace | DictationMode::Both => {
                paste_instant(&mut clipboard, &mut enigo, text);
            }
            _ => {
                warn!(mode = ?msg.mode, "unknown dictation mode, falling back to TypeInPlace");
                paste_instant(&mut clipboard, &mut enigo, text);
            }
        }
    }

    debug!("delivery thread exiting");
}

/// Paste text at the cursor via clipboard + Ctrl+V.
///
/// Critical hot path: clipboard write dominates (~0.5 ms). The key sequence
/// is three syscalls; bail on first error to avoid stuck modifier keys.
#[inline(always)]
fn paste_instant(clipboard: &mut arboard::Clipboard, enigo: &mut enigo::Enigo, text: &str) {
    use enigo::{Direction, Key, Keyboard};

    if let Err(e) = clipboard.set_text(text) {
        warn!(error = %e, "failed to set clipboard");
        return;
    }

    // Atomic Ctrl+V; on any error release Ctrl to prevent stuck modifiers.
    let r = enigo
        .key(Key::Control, Direction::Press)
        .and_then(|_| enigo.key(Key::Unicode('v'), Direction::Click))
        .and_then(|_| enigo.key(Key::Control, Direction::Release));

    if let Err(e) = r {
        warn!(error = %e, "input injection failed");
        let _ = enigo.key(Key::Control, Direction::Release);
        return;
    }

    debug!("paste complete");
}

/// Write text to the clipboard without triggering any keyboard events.
#[inline(always)]
fn clipboard_only(clipboard: &mut arboard::Clipboard, text: &str) {
    if let Err(e) = clipboard.set_text(text) {
        warn!(error = %e, "failed to set clipboard");
    }
}
