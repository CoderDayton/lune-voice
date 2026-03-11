use std::sync::mpsc;

use lune_core::config::{DictationMode, TerminalMode};
use tracing::{debug, warn};

#[cfg(unix)]
use crate::uinput_typer::{self, UinputTyper};

/// Stub for non-Linux platforms where uinput is unavailable.
#[cfg(not(unix))]
mod uinput_typer {
    pub fn set_clipboard(_: &str) -> Result<(), String> {
        Err("not supported on this platform".to_owned())
    }
}
#[cfg(not(unix))]
struct UinputTyper;
#[cfg(not(unix))]
impl UinputTyper {
    fn with_terminal_mode(_: TerminalMode) -> Result<Self, String> {
        Err("uinput not supported on this platform".to_owned())
    }
    fn warm_up(&self) {}
    fn type_str(&mut self, _: &str) {}
}

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

fn delivery_thread(rx: mpsc::Receiver<DeliveryMsg>, terminal_mode: TerminalMode) {
    let mut typer = match UinputTyper::with_terminal_mode(terminal_mode) {
        Ok(t) => {
            debug!("delivery: uinput clipboard-paste active");
            t.warm_up();
            t
        }
        Err(e) => {
            warn!(error = %e, "uinput unavailable — delivery disabled");
            return;
        }
    };

    debug!("delivery thread ready");

    while let Ok(msg) = rx.recv() {
        let text: &str = &msg.text;

        match msg.mode {
            DictationMode::TypeInPlace => {
                typer.type_str(text);
            }
            DictationMode::Clipboard => {
                if let Err(e) = uinput_typer::set_clipboard(text) {
                    warn!(error = %e, "clipboard-only delivery failed");
                }
            }
            DictationMode::Both => {
                // type_str already sets clipboard as a side effect
                typer.type_str(text);
            }
            _ => {
                warn!(mode = ?msg.mode, "unknown dictation mode, falling back to TypeInPlace");
                typer.type_str(text);
            }
        }
    }

    debug!("delivery thread exiting");
}
