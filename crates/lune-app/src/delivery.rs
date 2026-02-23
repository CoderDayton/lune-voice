use std::sync::mpsc;

use lune_core::config::DictationMode;
use tracing::{debug, warn};

use crate::uinput_typer::UinputTyper;

/// Message sent to the persistent delivery thread.
struct DeliveryMsg {
    text: Box<str>,
    mode: DictationMode,
}

/// Handle to the background delivery thread.
#[derive(Clone)]
pub struct DeliveryService {
    tx: mpsc::SyncSender<DeliveryMsg>,
}

impl DeliveryService {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::sync_channel::<DeliveryMsg>(64);
        std::thread::Builder::new()
            .name("delivery".into())
            .spawn(move || delivery_thread(rx))
            .expect("spawn delivery thread");
        Self { tx }
    }

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

fn delivery_thread(rx: mpsc::Receiver<DeliveryMsg>) {
    use arboard::Clipboard;
    use enigo::{Enigo, Settings};

    // Primary: uinput ghost keyboard — kernel-level, works on all compositors.
    let mut uinput = match UinputTyper::new() {
        Ok(t) => {
            debug!("delivery: uinput ghost keyboard active");
            Some(t)
        }
        Err(e) => {
            warn!(error = %e, "uinput unavailable, falling back to clipboard paste");
            None
        }
    };

    // Fallback: clipboard + Ctrl+V, used when uinput is unavailable or text
    // contains chars unmappable on US QWERTY (accented letters, smart quotes…).
    let mut clipboard = match Clipboard::new() {
        Ok(cb) => Some(cb),
        Err(e) => {
            warn!(error = %e, "failed to init clipboard");
            None
        }
    };

    let mut enigo = match Enigo::new(&Settings::default()) {
        Ok(e) => Some(e),
        Err(e) => {
            warn!(error = %e, "failed to init enigo — Ctrl+V paste unavailable");
            None
        }
    };

    if uinput.is_none() && clipboard.is_none() {
        warn!("no delivery backend available — delivery disabled");
        return;
    }

    debug!("delivery thread ready");

    while let Ok(msg) = rx.recv() {
        let text: &str = &msg.text;

        match msg.mode {
            DictationMode::Clipboard => {
                if let Some(cb) = &mut clipboard {
                    clipboard_only(cb, text);
                }
            }
            DictationMode::TypeInPlace => {
                type_in_place(text, &mut uinput, &mut clipboard, &mut enigo)
            }
            DictationMode::Both => {
                type_in_place(text, &mut uinput, &mut clipboard, &mut enigo);
                // Also land on clipboard regardless of which path typed it.
                if let Some(cb) = &mut clipboard {
                    clipboard_only(cb, text);
                }
            }
            _ => {
                warn!(mode = ?msg.mode, "unknown dictation mode, falling back to TypeInPlace");
                type_in_place(text, &mut uinput, &mut clipboard, &mut enigo);
            }
        }
    }

    debug!("delivery thread exiting");
}

/// Type text at the cursor.
///
/// Uses uinput when available and all chars are ASCII-mappable.
/// Falls back to clipboard + Ctrl+V for non-ASCII or when uinput is absent.
fn type_in_place(
    text: &str,
    uinput: &mut Option<UinputTyper>,
    clipboard: &mut Option<arboard::Clipboard>,
    enigo: &mut Option<enigo::Enigo>,
) {
    if let Some(t) = uinput {
        if !UinputTyper::has_unmapped_chars(text) {
            t.type_str(text);
            return;
        }
    }
    // Non-ASCII or uinput unavailable — clipboard paste.
    match (clipboard.as_mut(), enigo.as_mut()) {
        (Some(cb), Some(en)) => paste_instant(cb, en, text),
        (Some(cb), None) => {
            clipboard_only(cb, text);
            debug!("text on clipboard (enigo unavailable — paste manually)");
        }
        _ => {}
    }
}

#[inline(always)]
fn paste_instant(clipboard: &mut arboard::Clipboard, enigo: &mut enigo::Enigo, text: &str) {
    use enigo::{Direction, Key, Keyboard};

    if let Err(e) = clipboard.set_text(text) {
        warn!(error = %e, "failed to set clipboard");
        return;
    }

    let r = enigo
        .key(Key::Control, Direction::Press)
        .and_then(|_| enigo.key(Key::Unicode('v'), Direction::Click))
        .and_then(|_| enigo.key(Key::Control, Direction::Release));

    if let Err(e) = r {
        warn!(error = %e, "input injection failed");
        let _ = enigo.key(Key::Control, Direction::Release);
    }
}

#[inline(always)]
fn clipboard_only(clipboard: &mut arboard::Clipboard, text: &str) {
    if let Err(e) = clipboard.set_text(text) {
        warn!(error = %e, "failed to set clipboard");
    }
}
