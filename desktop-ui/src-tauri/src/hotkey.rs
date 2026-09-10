//! Platform-neutral global-hotkey core.
//!
//! Everything here is true of a hotkey listener on any OS: what a press
//! *means* (`HotkeyAction`), how a listener publishes intent without ever
//! blocking (`HotkeySink`/`HotkeyIntents`), and what the app may ask a
//! listener to do afterwards (`HotkeyBackend`). The OS-specific halves live in
//! `hotkey_linux` (evdev grab) and `hotkey_windows` (low-level keyboard hook),
//! and are reachable only through `start`.
//!
//! The split exists because the two platforms disagree about almost
//! everything *except* this contract: Linux must grab devices exclusively and
//! re-inject what it does not consume, Windows may observe the stream and
//! swallow only the combo it matched. Neither detail belongs in the caller,
//! and neither can be expressed in the other's vocabulary.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use tokio::sync::mpsc;
use tracing::{info, warn};

use lune_core::config::{ActivationMode, HotkeyConfig};

// ─── What a hotkey means ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyAction {
    Toggle,
    Start,
    Stop,
    Cancel,
}

// ─── Intent channel ──────────────────────────────────────────────────────────
//
// The listener thread carries two obligations with very different weights.
//
// Forwarding is absolute: on Linux the kernel routes the grabbed keyboards to
// this process alone, so every non-hotkey keystroke reaches the user's desktop
// only because this thread re-injects it; on Windows the low-level hook sits
// in the delivery path of every key the machine sees. Miss that deadline and
// the whole machine appears frozen, not just this app.
//
// Publishing intent is best-effort: the runtime that consumes it applies
// actions serially against one state machine, and a single start can take
// seconds while a model loads. Miss that deadline and one dictation is lost.
//
// So the channel between them is built to make the second incapable of
// delaying the first — not merely unlikely to.

/// Priority-lane codes for the actions that *end* a session. Ordered so
/// `fetch_max` merges two overflows into the stronger of the pair: someone who
/// reached for cancel gets cancel, not stop.
const SAFETY_NONE: u8 = 0;
const SAFETY_STOP: u8 = 1;
const SAFETY_CANCEL: u8 = 2;

/// How many intents may be in flight before the sink starts shedding.
///
/// Depth buys latency here, not reliability: the consumer is serial and one
/// action can take seconds, so anything queued behind it is stale before it
/// runs. Sized for a burst of double-taps and nothing more.
const INTENT_CAPACITY: usize = 16;

impl HotkeyAction {
    /// Which lane this action travels in when the channel is full.
    ///
    /// The two directions fail differently, so they cannot share a shedding
    /// policy. `Start`/`Toggle` begin work: losing one means nothing happens
    /// and the next press still works. `Stop`/`Cancel` end it: losing one
    /// leaves a live microphone the user believes they already closed.
    const fn safety_code(self) -> u8 {
        match self {
            Self::Stop => SAFETY_STOP,
            Self::Cancel => SAFETY_CANCEL,
            Self::Start | Self::Toggle => SAFETY_NONE,
        }
    }
}

/// The listener's end of the intent channel. Cannot block, by construction.
///
/// Handing the listener thread a bare `Sender` left `blocking_send` one edit
/// away, and that call is what turns a slow runtime into a system-wide
/// keyboard freeze. This type exposes no blocking operation to reach for.
#[derive(Clone)]
pub struct HotkeySink {
    tx: mpsc::Sender<HotkeyAction>,
    deferred_stop: Arc<AtomicU8>,
}

impl HotkeySink {
    /// Publish an intent. Returns immediately, always.
    pub fn emit(&self, action: HotkeyAction) {
        info!(?action, "hotkey dispatched");
        match self.tx.try_send(action) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(action)) => self.shed(action),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                warn!("hotkey consumer is gone; intent discarded");
            }
        }
    }

    /// A full queue means the consumer has not run in a long time. What that
    /// costs depends entirely on which direction the action was going.
    fn shed(&self, action: HotkeyAction) {
        let code = action.safety_code();
        if code == SAFETY_NONE {
            warn!(?action, "hotkey queue full; intent dropped");
            return;
        }
        self.deferred_stop.fetch_max(code, Ordering::Release);
        warn!(
            ?action,
            "hotkey queue full; stop moved to the priority lane"
        );
    }
}

/// The consumer's end of the intent channel.
pub struct HotkeyIntents {
    rx: mpsc::Receiver<HotkeyAction>,
    deferred_stop: Arc<AtomicU8>,
}

impl HotkeyIntents {
    /// The next intent to act on, or `None` once the listener is gone.
    ///
    /// A deferred stop preempts the queue. It was pressed *after* everything
    /// waiting there, and asking to stop makes those older intents moot —
    /// replaying them would restart exactly what the user just ended.
    ///
    /// No wakeup can be lost, and the channel needs no side-band notification
    /// to prove it: `deferred_stop` is only ever set when `try_send` reported
    /// the channel full, a full channel holds at least one item, and a
    /// non-empty channel makes the `recv` below ready immediately. A consumer
    /// parked on an empty channel is therefore a consumer no sender is
    /// shedding into, so checking before awaiting is sufficient.
    pub async fn recv(&mut self) -> Option<HotkeyAction> {
        if let Some(action) = self.take_deferred_stop() {
            let discarded = self.drain();
            if discarded > 0 {
                warn!(discarded, "deferred stop preempted stale hotkey intents");
            }
            return Some(action);
        }
        self.rx.recv().await
    }

    fn take_deferred_stop(&self) -> Option<HotkeyAction> {
        match self.deferred_stop.swap(SAFETY_NONE, Ordering::Acquire) {
            SAFETY_STOP => Some(HotkeyAction::Stop),
            SAFETY_CANCEL => Some(HotkeyAction::Cancel),
            _ => None,
        }
    }

    fn drain(&mut self) -> usize {
        let mut discarded = 0;
        while self.rx.try_recv().is_ok() {
            discarded += 1;
        }
        discarded
    }
}

/// Build the listener/consumer pair. Keeping both ends and the shedding policy
/// in one place is deliberate: neither side can misuse the channel on its own.
pub fn intent_channel() -> (HotkeySink, HotkeyIntents) {
    let (tx, rx) = mpsc::channel(INTENT_CAPACITY);
    let deferred_stop = Arc::new(AtomicU8::new(SAFETY_NONE));
    (
        HotkeySink {
            tx,
            deferred_stop: Arc::clone(&deferred_stop),
        },
        HotkeyIntents { rx, deferred_stop },
    )
}

// ─── The platform contract ───────────────────────────────────────────────────

/// A running OS hotkey listener, seen from the app side.
///
/// Deliberately tiny. A listener is started once and then never stopped — both
/// backends run for the process lifetime — so the only thing the app ever asks
/// of one afterwards is to re-target: apply a new binding or activation mode
/// without dropping the keyboard grab (Linux) or re-installing the hook
/// (Windows). Anything larger would be a contract only one platform could
/// honour.
pub trait HotkeyBackend: Send + Sync + 'static {
    /// Apply a new binding and activation mode in place.
    ///
    /// On an unrecognised key name the previous configuration must be kept and
    /// an error returned: a half-applied hotkey is worse than a stale one.
    fn update(&self, config: &HotkeyConfig, mode: ActivationMode) -> Result<(), String>;

    /// Short identifier for logs, e.g. `"evdev"` or `"win-hook"`.
    fn name(&self) -> &'static str;
}

/// Start the listener this platform uses, if it has one.
///
/// `None` means no native listener exists here and the caller should fall back
/// to `tauri-plugin-global-shortcut` — which only ever reports a press, so it
/// cannot serve push-to-talk. Both real backends can.
pub fn start(
    config: &HotkeyConfig,
    mode: ActivationMode,
    sink: HotkeySink,
) -> Option<Arc<dyn HotkeyBackend>> {
    #[cfg(target_os = "linux")]
    {
        crate::hotkey_linux::start(config, mode, sink)
    }
    #[cfg(target_os = "windows")]
    {
        crate::hotkey_windows::start(config, mode, sink)
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = (config, mode, sink);
        None
    }
}

// ─── Controller ──────────────────────────────────────────────────────────────

/// The toggle-only `tauri-plugin-global-shortcut` binding, seen from here.
///
/// Implemented over the app handle in `lib.rs`. It exists as a trait so the
/// controller can retire the plugin the moment a native listener takes over —
/// leaving both bound means one keypress fires two toggles.
pub trait PluginHotkeys: Send + Sync + 'static {
    /// Bind toggle and cancel, replacing whatever pair was bound before.
    fn rebind(&self, config: &HotkeyConfig) -> Result<(), String>;

    /// Unbind everything this app registered with the plugin.
    fn clear(&self);
}

type Starter = Box<
    dyn Fn(&HotkeyConfig, ActivationMode, HotkeySink) -> Option<Arc<dyn HotkeyBackend>>
        + Send
        + Sync,
>;

/// Whichever hotkey listener is live, and the way back to a working one.
///
/// A native backend is not guaranteed to exist at startup: `start` also returns
/// `None` when the configured key name does not resolve, which leaves the
/// plugin holding a binding the user is about to repair. So `update` is not a
/// simple delegation — it starts the native listener late when one becomes
/// startable, and otherwise re-registers the plugin. Without that, a binding
/// fixed in the UI would not take effect until the next launch.
pub struct HotkeyController {
    state: std::sync::Mutex<ControllerState>,
    plugin: Box<dyn PluginHotkeys>,
    starter: Starter,
}

struct ControllerState {
    native: Option<Arc<dyn HotkeyBackend>>,
    /// Kept for a late start. Holding a clone also keeps the intent channel
    /// open, so the consumer task survives a startup with no native listener.
    sink: HotkeySink,
}

impl HotkeyController {
    pub fn new(
        native: Option<Arc<dyn HotkeyBackend>>,
        sink: HotkeySink,
        plugin: Box<dyn PluginHotkeys>,
    ) -> Self {
        Self::with_starter(native, sink, plugin, Box::new(start))
    }

    fn with_starter(
        native: Option<Arc<dyn HotkeyBackend>>,
        sink: HotkeySink,
        plugin: Box<dyn PluginHotkeys>,
        starter: Starter,
    ) -> Self {
        Self {
            state: std::sync::Mutex::new(ControllerState { native, sink }),
            plugin,
            starter,
        }
    }

    /// Apply a new binding and activation mode to whatever is listening.
    pub fn update(&self, config: &HotkeyConfig, mode: ActivationMode) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(native) = state.native.as_ref() {
            return native.update(config, mode);
        }

        if let Some(backend) = (self.starter)(config, mode, state.sink.clone()) {
            info!(
                backend = backend.name(),
                "native hotkey listener started from a settings save"
            );
            self.plugin.clear();
            state.native = Some(backend);
            return Ok(());
        }

        self.plugin.rebind(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fill the channel without consuming anything, leaving the sink in the
    /// state that used to freeze the keyboard.
    fn saturated() -> (HotkeySink, HotkeyIntents) {
        let (sink, intents) = intent_channel();
        for _ in 0..INTENT_CAPACITY {
            sink.emit(HotkeyAction::Toggle);
        }
        (sink, intents)
    }

    /// The defect this channel exists to prevent: the listener thread also
    /// re-injects every non-hotkey keystroke, so a send that waits on the
    /// runtime freezes the user's keyboard system-wide. If `emit` ever waits
    /// again, this test stops finishing — hence the timeout, which turns a
    /// hang into a failure instead of a stuck CI job.
    #[tokio::test]
    async fn emit_returns_while_the_queue_is_full() {
        let (sink, _intents) = saturated();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::task::spawn_blocking(move || {
                for _ in 0..1000 {
                    sink.emit(HotkeyAction::Toggle);
                    sink.emit(HotkeyAction::Start);
                }
            })
            .await
            .expect("emit must never block the listener thread");
        })
        .await
        .expect("emit blocked on a full queue");
    }

    #[tokio::test]
    async fn a_dropped_start_costs_only_that_press() {
        let (sink, mut intents) = saturated();
        sink.emit(HotkeyAction::Start);
        // Nothing was promoted, so the queue still holds only what fit.
        for _ in 0..INTENT_CAPACITY {
            assert!(matches!(intents.recv().await, Some(HotkeyAction::Toggle)));
        }
        assert!(
            intents.rx.try_recv().is_err(),
            "the Start was shed, not kept"
        );
    }

    /// Losing a stop leaves the microphone open on a session the user believes
    /// they closed, so it must survive a full queue.
    #[tokio::test]
    async fn a_stop_survives_a_full_queue_and_preempts_it() {
        let (sink, mut intents) = saturated();
        sink.emit(HotkeyAction::Stop);

        assert!(
            matches!(intents.recv().await, Some(HotkeyAction::Stop)),
            "the stop must arrive before the backlog it was pressed after"
        );
        assert!(
            intents.rx.try_recv().is_err(),
            "intents older than the stop must not replay and restart it"
        );
    }

    #[tokio::test]
    async fn cancel_wins_over_stop_when_both_overflow() {
        let (sink, mut intents) = saturated();
        sink.emit(HotkeyAction::Stop);
        sink.emit(HotkeyAction::Cancel);
        assert!(matches!(intents.recv().await, Some(HotkeyAction::Cancel)));

        // Order of arrival must not change the answer.
        let (sink, mut intents) = saturated();
        sink.emit(HotkeyAction::Cancel);
        sink.emit(HotkeyAction::Stop);
        assert!(matches!(intents.recv().await, Some(HotkeyAction::Cancel)));
    }

    #[tokio::test]
    async fn an_uncontended_channel_is_plain_fifo() {
        let (sink, mut intents) = intent_channel();
        sink.emit(HotkeyAction::Start);
        sink.emit(HotkeyAction::Toggle);
        sink.emit(HotkeyAction::Stop);
        assert!(matches!(intents.recv().await, Some(HotkeyAction::Start)));
        assert!(matches!(intents.recv().await, Some(HotkeyAction::Toggle)));
        assert!(matches!(intents.recv().await, Some(HotkeyAction::Stop)));
    }

    #[tokio::test]
    async fn recv_ends_when_the_listener_is_gone() {
        let (sink, mut intents) = intent_channel();
        sink.emit(HotkeyAction::Start);
        drop(sink);
        assert!(matches!(intents.recv().await, Some(HotkeyAction::Start)));
        assert!(intents.recv().await.is_none());
    }

    // ── Controller ──────────────────────────────────────────────────────────

    use std::sync::atomic::AtomicUsize;

    fn combo(key: &str) -> lune_core::config::KeyCombo {
        lune_core::config::KeyCombo {
            key: key.to_string(),
            modifiers: Vec::new(),
        }
    }

    fn config(key: &str) -> HotkeyConfig {
        HotkeyConfig {
            toggle_dictation: combo(key),
            cancel_dictation: combo("escape"),
        }
    }

    #[derive(Default)]
    struct FakeBackend {
        updates: AtomicUsize,
        reject: bool,
    }

    impl HotkeyBackend for FakeBackend {
        fn update(&self, _config: &HotkeyConfig, _mode: ActivationMode) -> Result<(), String> {
            self.updates.fetch_add(1, Ordering::SeqCst);
            if self.reject {
                return Err("unknown key".into());
            }
            Ok(())
        }
        fn name(&self) -> &'static str {
            "fake"
        }
    }

    #[derive(Default)]
    struct FakePlugin {
        rebinds: AtomicUsize,
        clears: AtomicUsize,
    }

    impl PluginHotkeys for Arc<FakePlugin> {
        fn rebind(&self, _config: &HotkeyConfig) -> Result<(), String> {
            self.rebinds.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn clear(&self) {
            self.clears.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Number of times the injected starter was asked to start a listener.
    type Starts = Arc<AtomicUsize>;

    fn controller(
        native: Option<Arc<dyn HotkeyBackend>>,
        started_by_retry: Option<Arc<dyn HotkeyBackend>>,
    ) -> (HotkeyController, Arc<FakePlugin>, Starts) {
        let (sink, _intents) = intent_channel();
        // The receiver is dropped here; nothing in these tests reads intents,
        // and the controller never sends.
        let plugin = Arc::new(FakePlugin::default());
        let starts: Starts = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&starts);
        let controller = HotkeyController::with_starter(
            native,
            sink,
            Box::new(Arc::clone(&plugin)),
            Box::new(move |_c, _m, _s| {
                seen.fetch_add(1, Ordering::SeqCst);
                started_by_retry.clone()
            }),
        );
        (controller, plugin, starts)
    }

    #[test]
    fn a_live_native_listener_is_retargeted_in_place() {
        let backend = Arc::new(FakeBackend::default());
        let (controller, plugin, starts) = controller(Some(backend.clone()), None);

        assert!(
            controller
                .update(&config("f9"), ActivationMode::Toggle)
                .is_ok()
        );
        assert_eq!(backend.updates.load(Ordering::SeqCst), 1);
        assert_eq!(starts.load(Ordering::SeqCst), 0, "no restart was needed");
        assert_eq!(plugin.rebinds.load(Ordering::SeqCst), 0);
    }

    /// The repair path: a key name that did not resolve at startup left the
    /// plugin in charge, and the user has just fixed it in the UI.
    #[test]
    fn a_repaired_binding_starts_the_native_listener_and_retires_the_plugin() {
        let started = Arc::new(FakeBackend::default());
        let (controller, plugin, starts) = controller(None, Some(started.clone()));

        assert!(
            controller
                .update(&config("f9"), ActivationMode::Hold)
                .is_ok()
        );
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(
            plugin.clears.load(Ordering::SeqCst),
            1,
            "the plugin binding must go, or both fire on one press"
        );
        assert_eq!(plugin.rebinds.load(Ordering::SeqCst), 0);

        // Now adopted: a second save re-targets it rather than starting again.
        assert!(
            controller
                .update(&config("f10"), ActivationMode::Hold)
                .is_ok()
        );
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(started.updates.load(Ordering::SeqCst), 1);
    }

    /// No native backend on this platform, or the new binding is broken too:
    /// the plugin must still pick up the change instead of staying stale.
    #[test]
    fn without_a_native_listener_the_plugin_binding_is_re_registered() {
        let (controller, plugin, starts) = controller(None, None);

        assert!(
            controller
                .update(&config("f9"), ActivationMode::Toggle)
                .is_ok()
        );
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(plugin.rebinds.load(Ordering::SeqCst), 1);
        assert_eq!(plugin.clears.load(Ordering::SeqCst), 0);
    }

    /// A backend that refuses the new binding keeps the old one, and the
    /// rejection has to reach the caller rather than being papered over with a
    /// plugin registration on top of a live native listener.
    #[test]
    fn a_rejected_binding_surfaces_and_does_not_fall_back() {
        let backend = Arc::new(FakeBackend {
            updates: AtomicUsize::new(0),
            reject: true,
        });
        let (controller, plugin, _) = controller(Some(backend.clone()), None);

        assert!(
            controller
                .update(&config("nonsense"), ActivationMode::Toggle)
                .is_err()
        );
        assert_eq!(plugin.rebinds.load(Ordering::SeqCst), 0);
    }
}
