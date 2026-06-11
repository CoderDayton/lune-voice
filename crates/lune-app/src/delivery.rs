use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use lune_core::config::{DictationMode, TerminalMode};
use tracing::{debug, info, warn};

#[cfg(unix)]
use crate::uinput_typer::WindowTarget;

#[cfg(unix)]
use crate::uinput_typer::UinputTyper;

/// Message sent to the persistent delivery thread.
struct DeliveryMsg {
    text: Box<str>,
    mode: DictationMode,
    #[cfg(unix)]
    target: Option<WindowTarget>,
}

/// Tail after a delivery finishes during which it still reports busy — lets
/// in-flight synthetic key events drain out of the compositor and app queues.
const DELIVERY_SETTLE: Duration = Duration::from_millis(300);

/// Handle to the background delivery thread.
#[derive(Clone)]
pub struct DeliveryService {
    tx: mpsc::Sender<DeliveryMsg>,
    busy_until: Arc<Mutex<Option<Instant>>>,
}

impl DeliveryService {
    pub fn new(terminal_mode: TerminalMode) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<DeliveryMsg>();
        let busy_until = Arc::new(Mutex::new(None));
        let busy = Arc::clone(&busy_until);
        std::thread::Builder::new()
            .name("delivery".into())
            .spawn(move || delivery_thread(rx, terminal_mode, busy))
            .map_err(|e| format!("spawn delivery thread: {e}"))?;
        Ok(Self { tx, busy_until })
    }

    pub fn deliver(&self, text: &str, mode: DictationMode) -> bool {
        if text.is_empty() {
            return true;
        }
        debug!(?mode, text_len = text.len(), "delivering text");
        if let Err(e) = self.tx.send(DeliveryMsg {
            text: text.into(),
            mode,
            #[cfg(unix)]
            target: None,
        }) {
            warn!(error = %e, "delivery thread dead — text will not be typed");
            return false;
        }
        true
    }

    #[cfg(unix)]
    pub fn deliver_to(&self, text: &str, mode: DictationMode, target: &WindowTarget) -> bool {
        if text.is_empty() {
            return true;
        }
        debug!(?mode, text_len = text.len(), "delivering text to target");
        if let Err(e) = self.tx.send(DeliveryMsg {
            text: text.into(),
            mode,
            target: Some(target.clone()),
        }) {
            warn!(error = %e, "delivery thread dead — text will not be typed");
            return false;
        }
        true
    }

    pub fn deliver_owned(&self, text: String, mode: DictationMode) -> bool {
        self.deliver(&text, mode)
    }

    /// True while queued text is being typed/pasted, plus a short settle
    /// tail. Dictation starts are suppressed during this window: our own
    /// synthetic keystrokes landing on a focused UI control (e.g. the toggle
    /// button, re-activated by every typed space) must never be able to
    /// drive sessions in a feedback loop.
    pub fn is_busy(&self) -> bool {
        self.busy_until
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some_and(|t| Instant::now() < t)
    }

    pub fn is_alive(&self) -> bool {
        // A zero-capacity test: if the receiver is dropped, send returns Err.
        // We send empty text which `deliver()` guards against, but the thread
        // harmlessly ignores if it somehow arrives.
        self.tx
            .send(DeliveryMsg {
                text: "".into(),
                mode: DictationMode::TypeInPlace,
                #[cfg(unix)]
                target: None,
            })
            .is_ok()
    }
}

// ─── Platform-specific delivery ──────────────────────────────────────────────

fn delivery_thread(
    rx: mpsc::Receiver<DeliveryMsg>,
    terminal_mode: TerminalMode,
    busy_until: Arc<Mutex<Option<Instant>>>,
) {
    let mut driver = match PlatformDriver::new(terminal_mode) {
        Ok(d) => d,
        Err(e) => {
            warn!(error = %e, "text delivery unavailable on this platform");
            return;
        }
    };

    info!(platform = driver.platform_name(), "delivery thread ready");

    while let Ok(msg) = rx.recv() {
        let text: &str = &msg.text;
        // Empty `is_alive` probes must not register as busy.
        if text.is_empty() {
            continue;
        }
        // Busy from dequeue until the driver returns; typing can take
        // seconds (~2 ms/char), and senders only see the queue, not this.
        *busy_until.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Instant::now() + Duration::from_secs(3600));
        #[cfg(unix)]
        let has_target = msg.target.is_some();
        #[cfg(not(unix))]
        let has_target = false;

        match msg.mode {
            DictationMode::TypeInPlace => driver.type_keys(text),
            DictationMode::Both => {
                if let Err(e) = driver.copy_to_clipboard(text) {
                    warn!(error = %e, "clipboard copy failed in Both mode");
                }
                driver.type_keys(text);
            }
            _ => {
                #[cfg(unix)]
                if let Some(ref target) = msg.target {
                    driver.type_text_to(text, target);
                } else {
                    driver.type_text(text);
                }
                #[cfg(not(unix))]
                driver.type_text(text);
            }
        }
        let _ = has_target;
        *busy_until.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Instant::now() + DELIVERY_SETTLE);
    }

    debug!("delivery thread exiting");
}

// ─── Linux: uinput (existing battle-tested path) ─────────────────────────────

#[cfg(unix)]
struct PlatformDriver {
    typer: UinputTyper,
}

#[cfg(unix)]
impl PlatformDriver {
    fn new(terminal_mode: TerminalMode) -> Result<Self, String> {
        let typer = UinputTyper::with_terminal_mode(terminal_mode)?;
        typer.warm_up();
        debug!("delivery: uinput clipboard-paste active");
        Ok(Self { typer })
    }

    fn platform_name(&self) -> &'static str {
        "linux-uinput"
    }

    fn type_text(&mut self, text: &str) {
        self.typer.type_str(text);
    }

    fn type_text_to(&mut self, text: &str, target: &crate::uinput_typer::WindowTarget) {
        self.typer.type_str_to(text, target);
    }

    fn type_keys(&mut self, text: &str) {
        self.typer.type_str_keys(text);
    }

    fn copy_to_clipboard(&mut self, text: &str) -> Result<(), String> {
        crate::uinput_typer::set_clipboard(text)
    }
}

// ─── macOS / Windows: arboard + enigo ────────────────────────────────────────

#[cfg(not(unix))]
struct PlatformDriver {
    clipboard: arboard::Clipboard,
    enigo: enigo::Enigo,
}

#[cfg(not(unix))]
impl PlatformDriver {
    fn new(_terminal_mode: TerminalMode) -> Result<Self, String> {
        let clipboard =
            arboard::Clipboard::new().map_err(|e| format!("clipboard init failed: {e}"))?;
        let enigo = enigo::Enigo::new(&enigo::Settings::default())
            .map_err(|e| format!("enigo init failed: {e}"))?;
        Ok(Self { clipboard, enigo })
    }

    fn platform_name(&self) -> &'static str {
        if cfg!(target_os = "macos") {
            "macos-arboard-enigo"
        } else {
            "windows-arboard-enigo"
        }
    }

    fn type_text(&mut self, text: &str) {
        // Set clipboard, then emit paste keystroke
        if let Err(e) = self.clipboard.set_text(text) {
            warn!(error = %e, "clipboard set failed, falling back to per-key typing");
            self.type_keys(text);
            return;
        }

        // Small delay for clipboard to settle
        std::thread::sleep(std::time::Duration::from_millis(30));

        self.emit_paste();
    }

    fn emit_paste(&mut self) {
        use enigo::{Direction, Key, Keyboard};

        // macOS: Cmd+V, Windows: Ctrl+V
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

        // Fallback: type each character directly
        if let Err(e) = self.enigo.text(text) {
            warn!(error = %e, "per-key typing failed");
        }
    }

    fn copy_to_clipboard(&mut self, text: &str) -> Result<(), String> {
        self.clipboard
            .set_text(text)
            .map_err(|e| format!("clipboard set failed: {e}"))
    }
}
