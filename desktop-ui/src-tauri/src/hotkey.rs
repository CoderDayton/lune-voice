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
}
