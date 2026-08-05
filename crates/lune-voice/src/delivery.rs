use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use lune_core::config::{DictationMode, TerminalMode};
use tracing::{debug, info, warn};
use uuid::Uuid;

#[cfg(unix)]
use crate::uinput_typer::WindowTarget;

#[cfg(unix)]
use crate::uinput_typer::UinputTyper;

/// What a driver's type call actually accomplished.
///
/// Two independent facts, previously conflated into one `bool`: whether the
/// clipboard now holds the dictated text (so the explicit copy can be skipped)
/// and whether the text reached the target application at all. The second one
/// exists because `UinputTyper` refuses to type text per-key when the
/// clipboard is unavailable and the text contains a newline — dropping the
/// delivery is safer than pressing Enter, but the user has to be told.
#[derive(Clone, Copy, Debug)]
pub struct TypeOutcome {
    /// The dictated text is left on the clipboard afterwards.
    pub clipboard_has_text: bool,
    /// The text actually reached the focused application.
    pub delivered: bool,
}

impl TypeOutcome {
    /// Delivered; `clipboard_has_text` says whether the text stayed behind.
    pub(crate) const fn delivered(clipboard_has_text: bool) -> Self {
        Self {
            clipboard_has_text,
            delivered: true,
        }
    }

    /// Nothing reached the application and nothing was left on the clipboard.
    pub(crate) const DROPPED: Self = Self {
        clipboard_has_text: false,
        delivered: false,
    };
}

/// Message sent to the persistent delivery thread.
struct DeliveryMsg {
    /// Dictation request this delivery belongs to, for end-to-end tracing.
    /// `None` for ad-hoc injections (newline/bullet commands) and liveness probes.
    session_id: Option<Uuid>,
    text: Box<str>,
    mode: DictationMode,
    /// Leave the dictated text on the clipboard after delivery (orthogonal to
    /// `mode`). Structural injections (newline/bullet) pass `false`.
    copy_to_clipboard: bool,
    #[cfg(unix)]
    target: Option<WindowTarget>,
    /// Reports what the driver managed to do, once it has run. `None` for
    /// callers that don't care (structural injections).
    outcome: Option<mpsc::Sender<TypeOutcome>>,
}

/// A unit of work for the delivery thread. `Prime` and `Probe` carry no
/// payload; only `Deliver` types text.
enum DeliveryJob {
    /// Snapshot the clipboard ahead of an imminent paste — sent at dictation
    /// start so the up-to-2s `wl-paste` round-trip overlaps recording + ASR
    /// instead of freezing the paste. See `DeliveryService::prime_clipboard`.
    Prime,
    /// Liveness probe — the thread drops it without registering as busy.
    Probe,
    /// Deliver dictated (or structural) text.
    Deliver(DeliveryMsg),
}

/// Tail after a delivery finishes during which it still reports busy — lets
/// in-flight synthetic key events drain out of the compositor and app queues.
const DELIVERY_SETTLE: Duration = Duration::from_millis(300);

/// Handle to the background delivery thread.
#[derive(Clone)]
pub struct DeliveryService {
    tx: mpsc::Sender<DeliveryJob>,
    busy_until: Arc<Mutex<Option<Instant>>>,
}

impl DeliveryService {
    pub fn new(terminal_mode: TerminalMode) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<DeliveryJob>();
        let busy_until = Arc::new(Mutex::new(None));
        let busy = Arc::clone(&busy_until);
        std::thread::Builder::new()
            .name("delivery".into())
            .spawn(move || delivery_thread(rx, terminal_mode, busy))
            .map_err(|e| format!("spawn delivery thread: {e}"))?;
        Ok(Self { tx, busy_until })
    }

    pub fn deliver(
        &self,
        session_id: Option<Uuid>,
        text: &str,
        mode: DictationMode,
        copy_to_clipboard: bool,
    ) -> bool {
        if text.is_empty() {
            return true;
        }
        debug!(?session_id, ?mode, text_len = text.len(), "delivering text");
        self.enqueue(DeliveryMsg {
            session_id,
            text: text.into(),
            mode,
            copy_to_clipboard,
            #[cfg(unix)]
            target: None,
            outcome: None,
        })
    }

    /// Like [`deliver`](Self::deliver), but hands back a receiver that resolves
    /// once the driver has actually run. Returns `None` when the delivery
    /// thread is dead (nothing will ever be typed). Callers use it to tell the
    /// user when text was dropped instead of typed — enqueueing successfully is
    /// not the same as delivering.
    pub fn deliver_tracked(
        &self,
        session_id: Option<Uuid>,
        text: &str,
        mode: DictationMode,
        copy_to_clipboard: bool,
    ) -> Option<mpsc::Receiver<TypeOutcome>> {
        let (tx, rx) = mpsc::channel();
        debug!(?session_id, ?mode, text_len = text.len(), "delivering text");
        self.enqueue(DeliveryMsg {
            session_id,
            text: text.into(),
            mode,
            copy_to_clipboard,
            #[cfg(unix)]
            target: None,
            outcome: Some(tx),
        })
        .then_some(rx)
    }

    /// Queue a delivery message, returning `false` if the delivery thread has
    /// died (its receiver dropped, so `send` errors).
    fn enqueue(&self, msg: DeliveryMsg) -> bool {
        if let Err(e) = self.tx.send(DeliveryJob::Deliver(msg)) {
            warn!(error = %e, "delivery thread dead — text will not be typed");
            return false;
        }
        true
    }

    /// Snapshot the clipboard ahead of delivery. Called at dictation start so
    /// the restore-after-paste read (up to 2s of `wl-paste` on a slow clipboard
    /// owner) overlaps recording + ASR instead of freezing the paste. The
    /// delivery thread processes this before the paste it precedes (single
    /// ordered queue), so the snapshot is ready by delivery time. Best-effort:
    /// a dead thread just means the paste falls back to a synchronous snapshot.
    pub fn prime_clipboard(&self) {
        if let Err(e) = self.tx.send(DeliveryJob::Prime) {
            debug!(error = %e, "clipboard prime skipped — delivery thread unavailable");
        }
    }

    #[cfg(unix)]
    pub fn deliver_to(
        &self,
        session_id: Option<Uuid>,
        text: &str,
        mode: DictationMode,
        copy_to_clipboard: bool,
        target: &WindowTarget,
    ) -> bool {
        if text.is_empty() {
            return true;
        }
        debug!(
            ?session_id,
            ?mode,
            text_len = text.len(),
            "delivering text to target"
        );
        self.enqueue(DeliveryMsg {
            session_id,
            text: text.into(),
            mode,
            copy_to_clipboard,
            target: Some(target.clone()),
            outcome: None,
        })
    }

    /// [`deliver_to`](Self::deliver_to) with the outcome receiver of
    /// [`deliver_tracked`](Self::deliver_tracked).
    #[cfg(unix)]
    pub fn deliver_to_tracked(
        &self,
        session_id: Option<Uuid>,
        text: &str,
        mode: DictationMode,
        copy_to_clipboard: bool,
        target: &WindowTarget,
    ) -> Option<mpsc::Receiver<TypeOutcome>> {
        let (tx, rx) = mpsc::channel();
        debug!(
            ?session_id,
            ?mode,
            text_len = text.len(),
            "delivering text to target"
        );
        self.enqueue(DeliveryMsg {
            session_id,
            text: text.into(),
            mode,
            copy_to_clipboard,
            target: Some(target.clone()),
            outcome: Some(tx),
        })
        .then_some(rx)
    }

    pub fn deliver_owned(
        &self,
        session_id: Option<Uuid>,
        text: String,
        mode: DictationMode,
        copy_to_clipboard: bool,
    ) -> bool {
        self.deliver(session_id, &text, mode, copy_to_clipboard)
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
        // Liveness probe: if the delivery thread (the receiver) has exited,
        // `send` returns Err. The empty text means the thread harmlessly
        // skips it (the loop `continue`s on empty text) if it does arrive.
        self.tx.send(DeliveryJob::Probe).is_ok()
    }
}

// ─── Platform-specific delivery ──────────────────────────────────────────────

fn delivery_thread(
    rx: mpsc::Receiver<DeliveryJob>,
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

    while let Ok(job) = rx.recv() {
        let msg = match job {
            // Liveness probe — nothing to do, and must not register as busy.
            DeliveryJob::Probe => continue,
            // Pre-snapshot the clipboard (dictation start). Runs on this
            // background thread during the ASR window, so its subprocess cost
            // never reaches the paste hot path.
            DeliveryJob::Prime => {
                driver.prime_clipboard();
                continue;
            }
            DeliveryJob::Deliver(msg) => msg,
        };
        let text: &str = &msg.text;
        // Defensive: empty text carries no output (probes no longer arrive here).
        if text.is_empty() {
            continue;
        }
        // Correlated with the upstream pipeline via session_id (see DeliveryMsg).
        debug!(session_id = ?msg.session_id, ?msg.mode, text_len = text.len(), "delivery: typing");
        // Busy from dequeue until the driver returns; per-key typing is paced
        // (~8-12 ms/char) and can take seconds, and senders only see the queue.
        *busy_until.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Instant::now() + Duration::from_secs(3600));

        // Paste delivery borrows the clipboard as a transport for the Ctrl+V
        // chord. Tell the driver whether to keep the dictated text afterwards
        // (copy_to_clipboard) or restore the user's previous clipboard — without
        // this the uinput path silently overwrites the clipboard on every
        // dictation (the IBus path commits directly and never touches it).
        let keep_clipboard = msg.copy_to_clipboard;
        // Delivery returns whether the dictated text now sits on the clipboard,
        // so the explicit copy below can skip re-setting it (the uinput paste
        // path with copy_to_clipboard on already leaves it there).
        let outcome = match msg.mode {
            DictationMode::TypeInPlace => driver.type_keys(text, keep_clipboard),
            _ => {
                #[cfg(unix)]
                {
                    if let Some(ref target) = msg.target {
                        driver.type_text_to(text, target, keep_clipboard)
                    } else {
                        driver.type_text(text, keep_clipboard)
                    }
                }
                #[cfg(not(unix))]
                {
                    driver.type_text(text, keep_clipboard)
                }
            }
        };
        if !outcome.delivered {
            warn!(
                session_id = ?msg.session_id,
                "delivery dropped — text reached neither the application nor the clipboard"
            );
        }
        // Best-effort: a caller that stopped listening just doesn't hear back.
        if let Some(tx) = msg.outcome {
            let _ = tx.send(outcome);
        }
        let clipboard_has_text = outcome.clipboard_has_text;
        // Orthogonal copy toggle: ensure the dictated text lands on the clipboard
        // when requested, for paths that didn't already place it there (IBus
        // commit, per-key typing). Skipped when delivery already left it, so we
        // don't redundantly re-set the clipboard (a wasted subprocess).
        if msg.copy_to_clipboard && !clipboard_has_text {
            if let Err(e) = driver.copy_to_clipboard(text) {
                warn!(error = %e, "clipboard copy failed");
            }
        }
        *busy_until.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Instant::now() + DELIVERY_SETTLE);
    }

    debug!("delivery thread exiting");
}

// ─── Linux: uinput (existing battle-tested path) ─────────────────────────────

#[cfg(unix)]
enum LinuxBackend {
    /// In-process IBus engine — preferred on Linux when IBus is the active input
    /// method. Commits Unicode straight into the focused input context, so it
    /// needs no terminal detection or paste chord (fixes GNOME Wayland). See
    /// `ibus.rs`. Linux-only; absent on macOS-as-unix.
    #[cfg(target_os = "linux")]
    Ibus(crate::ibus::IbusTyper),
    /// uinput keystroke / clipboard-paste path — the universal fallback.
    Uinput(UinputTyper),
}

#[cfg(unix)]
struct PlatformDriver {
    backend: LinuxBackend,
}

#[cfg(unix)]
impl PlatformDriver {
    fn new(terminal_mode: TerminalMode) -> Result<Self, String> {
        // Prefer IBus when it's the active input method. On GNOME Wayland the
        // uinput path can't detect terminals (so it can't pick Ctrl+Shift+V),
        // but IBus commits directly to the focused input context. Fall back to
        // the battle-tested uinput path when IBus is unavailable.
        #[cfg(target_os = "linux")]
        if crate::ibus::is_active_im() {
            match crate::ibus::IbusTyper::try_new() {
                Ok(typer) => {
                    typer.warm_up();
                    info!("delivery: IBus engine active");
                    return Ok(Self {
                        backend: LinuxBackend::Ibus(typer),
                    });
                }
                Err(e) => warn!(error = %e, "ibus init failed; falling back to uinput"),
            }
        }

        let typer = UinputTyper::with_terminal_mode(terminal_mode)?;
        typer.warm_up();
        debug!("delivery: uinput clipboard-paste active");
        Ok(Self {
            backend: LinuxBackend::Uinput(typer),
        })
    }

    fn platform_name(&self) -> &'static str {
        match &self.backend {
            #[cfg(target_os = "linux")]
            LinuxBackend::Ibus(_) => "linux-ibus",
            LinuxBackend::Uinput(_) => "linux-uinput",
        }
    }

    /// Snapshot the clipboard now so the post-paste restore doesn't read it on
    /// the delivery hot path. Only the uinput paste path borrows the clipboard;
    /// IBus commits directly and never touches it, so priming is a no-op there.
    fn prime_clipboard(&mut self) {
        match &mut self.backend {
            #[cfg(target_os = "linux")]
            LinuxBackend::Ibus(_) => {}
            LinuxBackend::Uinput(t) => t.prime_clipboard(),
        }
    }

    fn type_text(&mut self, text: &str, keep_clipboard: bool) -> TypeOutcome {
        match &mut self.backend {
            // IBus commits directly to the input context — no clipboard, so the
            // keep/restore flag is irrelevant and the text is never staged there.
            #[cfg(target_os = "linux")]
            LinuxBackend::Ibus(t) => {
                t.commit_text(text);
                TypeOutcome::delivered(false)
            }
            LinuxBackend::Uinput(t) => t.type_str(text, keep_clipboard),
        }
    }

    fn type_text_to(
        &mut self,
        text: &str,
        target: &crate::uinput_typer::WindowTarget,
        keep_clipboard: bool,
    ) -> TypeOutcome {
        match &mut self.backend {
            // IBus commits to the focused input context directly; the captured
            // WindowTarget (for focus restoration) is only needed by uinput.
            #[cfg(target_os = "linux")]
            LinuxBackend::Ibus(t) => {
                t.commit_text(text);
                TypeOutcome::delivered(false)
            }
            LinuxBackend::Uinput(t) => t.type_str_to(text, target, keep_clipboard),
        }
    }

    fn type_keys(&mut self, text: &str, keep_clipboard: bool) -> TypeOutcome {
        match &mut self.backend {
            // IBus commits Unicode directly — no US-QWERTY restriction, so
            // type-in-place of accented/CJK text "just works".
            #[cfg(target_os = "linux")]
            LinuxBackend::Ibus(t) => {
                t.commit_text(text);
                TypeOutcome::delivered(false)
            }
            LinuxBackend::Uinput(t) => t.type_str_keys(text, keep_clipboard),
        }
    }

    fn copy_to_clipboard(&mut self, text: &str) -> Result<(), String> {
        // Clipboard is backend-independent; reuse the existing helper.
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

    /// No-op: macOS/Windows leave the dictated text on the clipboard and never
    /// restore the prior contents, so there is nothing to snapshot ahead.
    fn prime_clipboard(&mut self) {}

    fn type_text(&mut self, text: &str, keep_clipboard: bool) -> TypeOutcome {
        // Set clipboard, then emit paste keystroke. (macOS/Windows keep the
        // dictated text on the clipboard regardless; restore is Linux-only.)
        if let Err(e) = self.clipboard.set_text(text) {
            warn!(error = %e, "clipboard set failed, falling back to per-key typing");
            return self.type_keys(text, keep_clipboard);
        }

        // Small delay for clipboard to settle
        std::thread::sleep(std::time::Duration::from_millis(30));

        self.emit_paste();
        // The dictated text remains on the clipboard as the paste source.
        TypeOutcome::delivered(true)
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

    /// Per-key typing never touches the clipboard. enigo types Unicode text
    /// directly (no US-QWERTY keycode mapping), so there is no newline hazard
    /// to guard against here — a failure is reported, not silently dropped.
    fn type_keys(&mut self, text: &str, _keep_clipboard: bool) -> TypeOutcome {
        use enigo::Keyboard;

        // Fallback: type each character directly
        if let Err(e) = self.enigo.text(text) {
            warn!(error = %e, "per-key typing failed");
            return TypeOutcome::DROPPED;
        }
        TypeOutcome::delivered(false)
    }

    fn copy_to_clipboard(&mut self, text: &str) -> Result<(), String> {
        self.clipboard
            .set_text(text)
            .map_err(|e| format!("clipboard set failed: {e}"))
    }
}
