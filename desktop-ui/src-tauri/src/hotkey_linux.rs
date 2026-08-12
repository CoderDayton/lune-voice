//! Linux-only global hotkey detection via evdev with device grab.
//!
//! Grabs physical keyboards exclusively and re-injects non-hotkey events
//! through a uinput passthrough device. This works on both X11 and Wayland.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, mpsc as std_mpsc};

use evdev::{
    AttributeSet, Device, EventType, InputEvent, KeyCode, RelativeAxisCode, uinput::VirtualDevice,
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use lune_core::config::{ActivationMode, HotkeyConfig, Modifier};

// ─── Public types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
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
// Forwarding is absolute: while the keyboards are grabbed the kernel routes
// their events to this process alone, so every non-hotkey keystroke reaches
// the user's desktop only because this thread re-injects it. Miss that
// deadline and the whole machine appears frozen, not just this app.
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

/// Live-updatable hotkey configuration shared between the settings-save
/// path and the evdev listener thread. The listener polls `generation` per
/// key event (a single atomic load) and re-reads the snapshot only when it
/// changed, so every hotkey setting — trigger key, modifiers, cancel key,
/// activation mode — applies immediately without restarting the listener.
pub struct LiveHotkeyConfig {
    inner: RwLock<HotkeySnapshot>,
    generation: AtomicU64,
}

#[derive(Clone)]
struct HotkeySnapshot {
    main_trigger: KeyCode,
    main_mods: Vec<Modifier>,
    cancel_trigger: KeyCode,
    cancel_mods: Vec<Modifier>,
    mode: ActivationMode,
}

impl HotkeySnapshot {
    fn from_config(config: &HotkeyConfig, mode: ActivationMode) -> Option<Self> {
        let main_trigger = key_name_to_code(&config.toggle_dictation.key)?;
        let cancel_trigger = key_name_to_code(&config.cancel_dictation.key).unwrap_or_else(|| {
            warn!(
                key = %config.cancel_dictation.key,
                "unknown cancel hotkey key; cancel hotkey disabled"
            );
            KeyCode::KEY_RESERVED
        });
        Some(Self {
            main_trigger,
            main_mods: config.toggle_dictation.modifiers.clone(),
            cancel_trigger,
            cancel_mods: config.cancel_dictation.modifiers.clone(),
            mode,
        })
    }
}

impl LiveHotkeyConfig {
    /// Returns `None` when the trigger key name is unknown.
    pub fn new(config: &HotkeyConfig, mode: ActivationMode) -> Option<Self> {
        Some(Self {
            inner: RwLock::new(HotkeySnapshot::from_config(config, mode)?),
            generation: AtomicU64::new(0),
        })
    }

    /// Swap in a new configuration. On an unknown trigger key the previous
    /// config is kept and an error returned — a half-applied hotkey is worse
    /// than a stale one.
    pub fn update(&self, config: &HotkeyConfig, mode: ActivationMode) -> Result<(), String> {
        let snap = HotkeySnapshot::from_config(config, mode)
            .ok_or_else(|| format!("unknown hotkey key name: {}", config.toggle_dictation.key))?;
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = snap;
        self.generation.fetch_add(1, Ordering::Release);
        info!(
            key = %config.toggle_dictation.key,
            ?mode,
            "hotkey configuration hot-reloaded"
        );
        Ok(())
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn snapshot(&self) -> HotkeySnapshot {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Spawn the evdev hotkey listener on a background OS thread.
///
/// Returns `None` if the hotkey config references unknown key names or no
/// keyboards are accessible.
pub fn spawn(live: Arc<LiveHotkeyConfig>, sink: HotkeySink) -> Option<std::thread::JoinHandle<()>> {
    let handle = std::thread::Builder::new()
        .name("hotkey-evdev".into())
        .spawn(move || {
            run_listener(live, sink);
        })
        .map_err(|e| warn!(error = %e, "failed to spawn hotkey listener thread"))
        .ok()?;

    Some(handle)
}

// ─── Listener core ───────────────────────────────────────────────────────────

fn build_combo_mods(mods: &[Modifier]) -> PressedCodes {
    let mut combo = PressedCodes::default();
    for m in mods {
        let (l, r) = modifier_keycodes(*m);
        combo.insert(l);
        combo.insert(r);
    }
    combo
}

fn run_listener(live: Arc<LiveHotkeyConfig>, sink: HotkeySink) {
    let keyboards = enumerate_keyboards();
    if keyboards.is_empty() {
        warn!("no accessible keyboard devices found; hotkey detection disabled");
        return;
    }
    info!(
        count = keyboards.len(),
        "keyboard devices opened for hotkey detection"
    );

    let passthrough = match build_passthrough_device() {
        Ok(dev) => dev,
        Err(e) => {
            warn!(error = %e, "failed to create passthrough device; hotkey grab disabled");
            return;
        }
    };
    let mut fwd = Forwarder::new(passthrough);

    let (event_tx, event_rx) = std_mpsc::channel::<(InputEvent, bool)>();

    for dev in keyboards {
        let tx = event_tx.clone();
        let name = dev.name().unwrap_or("unknown").to_string();
        if let Err(e) = std::thread::Builder::new()
            .name(format!("evdev-{name}"))
            .spawn(move || read_device(dev, tx))
        {
            warn!(error = %e, device = %name, "failed to spawn evdev reader thread");
        }
    }
    drop(event_tx);

    let mut config_gen = live.generation();
    let mut snap = live.snapshot();
    let mut combo_mod_codes = build_combo_mods(&snap.main_mods);

    let mut state = HotkeyState::new(
        snap.main_trigger,
        snap.main_mods.clone(),
        snap.cancel_trigger,
        snap.cancel_mods.clone(),
    );
    let mut consumed = PressedCodes::default();
    let mut deferred: Vec<InputEvent> = Vec::new();

    for (raw, grabbed) in event_rx {
        if raw.event_type() != EventType::KEY {
            if grabbed {
                fwd.forward(raw);
            }
            continue;
        }

        let code = KeyCode(raw.code());
        let value = raw.value();

        // Auto-repeat
        if value == 2 {
            if grabbed && !consumed.contains(code) && !combo_mod_codes.contains(code) {
                fwd.forward(raw);
            }
            continue;
        }

        let pressed = value == 1;
        // Hot-reload: one atomic load per event; rebuild targeting only when
        // a settings save bumped the generation. Pressed-key tracking is
        // preserved so a mid-combo reload can't wedge the state machine.
        let current_gen = live.generation();
        if current_gen != config_gen {
            config_gen = current_gen;
            snap = live.snapshot();
            combo_mod_codes = build_combo_mods(&snap.main_mods);
            state.retarget(
                snap.main_trigger,
                snap.main_mods.clone(),
                snap.cancel_trigger,
                snap.cancel_mods.clone(),
            );
        }
        let activation_mode = snap.mode;

        if !grabbed {
            let action = state.handle(code, pressed, activation_mode);
            if let Some(a) = action {
                sink.emit(a);
            }
            continue;
        }

        if let Some(action) = process_grabbed_key(
            raw,
            code,
            pressed,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
            &combo_mod_codes,
            activation_mode,
        ) {
            sink.emit(action);
        }
    }
    fwd.release_all();
    info!("all keyboard devices disconnected; hotkey listener exiting");
}

/// A destination for re-injected key events: the real uinput `VirtualDevice`
/// in production, a recorder in tests.
trait KeySink {
    fn send_event(&mut self, event: InputEvent);
}

impl KeySink for VirtualDevice {
    fn send_event(&mut self, event: InputEvent) {
        let _ = self.emit(&[event, InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0)]);
    }
}

/// Wraps the passthrough sink and tracks which key codes were forwarded as
/// currently-down. This guards two invariants:
///   1. A release is never swallowed for a key whose press we forwarded — that
///      would strand the key down in apps (a stuck modifier).
///   2. On listener exit, every still-down forwarded key gets a synthetic
///      release, so a teardown mid-hold can't leave a key wedged.
struct Forwarder<S: KeySink> {
    sink: S,
    down: PressedCodes,
}

impl<S: KeySink> Forwarder<S> {
    fn new(sink: S) -> Self {
        Self {
            sink,
            down: PressedCodes::default(),
        }
    }

    /// Forward an event, recording key press/release in the forwarded-down set.
    fn forward(&mut self, event: InputEvent) {
        if event.event_type() == EventType::KEY {
            let code = KeyCode(event.code());
            match event.value() {
                1 => self.down.insert(code),
                0 => self.down.remove(code),
                _ => {}
            }
        }
        self.sink.send_event(event);
    }

    fn flush(&mut self, deferred: &mut Vec<InputEvent>) {
        for ev in deferred.drain(..) {
            self.forward(ev);
        }
    }

    /// Whether a press for `code` was forwarded without a matching release.
    fn is_down(&self, code: KeyCode) -> bool {
        self.down.contains(code)
    }

    /// Emit a synthetic release for every still-down forwarded key. Called on
    /// listener exit so a teardown while keys are held can't strand them.
    fn release_all(&mut self) {
        for code in self.down.drain_codes() {
            self.sink
                .send_event(InputEvent::new(EventType::KEY.0, code.code(), 0));
        }
    }
}

/// Handle one non-repeat key event from a grabbed device: drive the state
/// machine and forward or swallow the raw event through `fwd`. Returns the
/// action to dispatch, if any.
#[allow(clippy::too_many_arguments)]
fn process_grabbed_key<S: KeySink>(
    raw: InputEvent,
    code: KeyCode,
    pressed: bool,
    state: &mut HotkeyState,
    fwd: &mut Forwarder<S>,
    consumed: &mut PressedCodes,
    deferred: &mut Vec<InputEvent>,
    combo_mod_codes: &PressedCodes,
    mode: ActivationMode,
) -> Option<HotkeyAction> {
    // Defer combo modifier presses. If the modifier press itself completes a
    // hotkey (trigger key was already down), drop the deferred queue and
    // dispatch the action instead of forwarding the modifier through to apps.
    // Without this, the Start action is silently dropped and the next release
    // emits a stray Stop.
    if pressed && combo_mod_codes.contains(code) {
        deferred.push(raw);
        if let Some(action) = state.handle(code, pressed, mode) {
            deferred.clear();
            consumed.insert(code);
            return Some(action);
        }
        return None;
    }

    match state.handle(code, pressed, mode) {
        Some(HotkeyAction::Cancel) => {
            fwd.flush(deferred);
            fwd.forward(raw);
            Some(HotkeyAction::Cancel)
        }
        Some(a) => {
            deferred.clear();
            if pressed {
                consumed.insert(code);
            } else {
                consumed.remove(code);
                // Keep apps balanced: if this key's press reached them, its
                // release must too, or the action just stranded it down.
                if fwd.is_down(code) {
                    fwd.forward(raw);
                }
            }
            Some(a)
        }
        None if !pressed && consumed.contains(code) => {
            consumed.remove(code);
            if fwd.is_down(code) {
                fwd.forward(raw);
            }
            None
        }
        None if !pressed && combo_mod_codes.contains(code) => {
            fwd.flush(deferred);
            fwd.forward(raw);
            None
        }
        None => {
            fwd.flush(deferred);
            fwd.forward(raw);
            None
        }
    }
}

// ─── Device enumeration ──────────────────────────────────────────────────────

const SELF_PREFIX: &str = "lune-voice";

fn enumerate_keyboards() -> Vec<Device> {
    evdev::enumerate()
        .filter_map(|(path, dev)| {
            if dev.name().is_some_and(|n| n.starts_with(SELF_PREFIX)) {
                debug!(path = %path.display(), "skipping self device");
                return None;
            }
            let keys = dev.supported_keys()?;
            // A pointer (mouse) is never a typing keyboard, even when it
            // advertises a macro keyboard interface. Gate on pointer *movement*
            // (REL_X) or mouse buttons — NOT any relative axis, or a keyboard's
            // rotary knob / volume wheel (REL_WHEEL / REL_DIAL) would wrongly
            // mark the keyboard a pointer and we'd drop it.
            let is_pointer = keys.contains(KeyCode::BTN_LEFT)
                || dev
                    .supported_relative_axes()
                    .is_some_and(|a| a.contains(RelativeAxisCode::REL_X));
            if !is_typing_keyboard_caps(
                keys.contains(KeyCode::KEY_A),
                keys.contains(KeyCode::KEY_SPACE),
                keys.contains(KeyCode::KEY_LEFTSHIFT) || keys.contains(KeyCode::KEY_RIGHTSHIFT),
                keys.contains(KeyCode::KEY_ENTER),
                is_pointer,
            ) {
                debug!(name = ?dev.name(), path = %path.display(), "skipping non-typing-keyboard node");
                return None;
            }
            info!(name = ?dev.name(), path = %path.display(), "found keyboard");
            Some(dev)
        })
        .collect()
}

/// Whether a device's capabilities mark it a real typing keyboard: letters,
/// space, a Shift, and Enter, and not a pointer. Pure so the rule is
/// unit-tested. The old filter accepted any device exposing `A`+`Space`, which
/// also matched a QMK board's secondary consumer-control node and a gaming
/// mouse's macro interface — so a single keyboard got grabbed twice and a mouse
/// got grabbed at all. Requiring Shift+Enter (and rejecting pointers) leaves
/// exactly the real keyboard nodes, each grabbed once.
fn is_typing_keyboard_caps(
    has_a: bool,
    has_space: bool,
    has_shift: bool,
    has_enter: bool,
    is_pointer: bool,
) -> bool {
    has_a && has_space && has_shift && has_enter && !is_pointer
}

fn read_device(mut dev: Device, tx: std_mpsc::Sender<(InputEvent, bool)>) {
    let name = dev.name().unwrap_or("unknown").to_string();
    let grabbed = match dev.grab() {
        Ok(()) => {
            info!(name, "grabbed keyboard");
            true
        }
        Err(e) => {
            debug!(error = %e, name, "grab failed; observe-only");
            false
        }
    };
    loop {
        match dev.fetch_events() {
            Ok(events) => {
                for ev in events {
                    if tx.send((ev, grabbed)).is_err() {
                        return;
                    }
                }
            }
            Err(e) => {
                debug!(error = %e, name, "device disconnected");
                return;
            }
        }
    }
}

fn build_passthrough_device() -> Result<VirtualDevice, String> {
    let mut keys = AttributeSet::<KeyCode>::new();
    for i in 1u16..768 {
        keys.insert(KeyCode(i));
    }
    let dev = VirtualDevice::builder()
        .map_err(|e| format!("passthrough builder: {e}"))?
        .name("lune-voice-passthrough")
        .with_keys(&keys)
        .map_err(|e| format!("passthrough with_keys: {e}"))?
        .build()
        .map_err(|e| format!("passthrough build: {e}"))?;
    std::thread::sleep(std::time::Duration::from_millis(200));
    debug!("passthrough virtual keyboard ready");
    Ok(dev)
}

// ─── Hotkey state machine ────────────────────────────────────────────────────

struct HotkeyState {
    pressed: PressedCodes,
    main_trigger: KeyCode,
    main_mods: Vec<Modifier>,
    cancel_trigger: KeyCode,
    cancel_mods: Vec<Modifier>,
    hold_active: bool,
}

impl HotkeyState {
    fn new(
        main_trigger: KeyCode,
        main_mods: Vec<Modifier>,
        cancel_trigger: KeyCode,
        cancel_mods: Vec<Modifier>,
    ) -> Self {
        Self {
            pressed: PressedCodes::default(),
            main_trigger,
            main_mods,
            cancel_trigger,
            cancel_mods,
            hold_active: false,
        }
    }

    /// Swap key/modifier targets in place, preserving the physical
    /// pressed-key set. `hold_active` resets — a hold in progress under the
    /// old combo must not emit a Stop for the new one.
    fn retarget(
        &mut self,
        main_trigger: KeyCode,
        main_mods: Vec<Modifier>,
        cancel_trigger: KeyCode,
        cancel_mods: Vec<Modifier>,
    ) {
        self.main_trigger = main_trigger;
        self.main_mods = main_mods;
        self.cancel_trigger = cancel_trigger;
        self.cancel_mods = cancel_mods;
        self.hold_active = false;
    }

    fn handle(
        &mut self,
        code: KeyCode,
        pressed: bool,
        mode: ActivationMode,
    ) -> Option<HotkeyAction> {
        let was_trigger = self.pressed.contains(self.main_trigger);
        let was_hold = self.hold_active;

        if pressed {
            self.pressed.insert(code);
        } else {
            self.pressed.remove(code);
        }

        // Cancel (Escape) in any activation mode. A session can be started by
        // clicking the pill — which has no key to release — so Escape must be
        // able to stop it even in Hold mode. Emitting Cancel here is harmless
        // when idle: the dispatcher no-ops unless a session is active, and the
        // raw Escape is still forwarded on to apps either way.
        if pressed
            && code == self.cancel_trigger
            && self.cancel_trigger != KeyCode::KEY_RESERVED
            && self.mods_down(&self.cancel_mods)
            && !self.extra_mods(&self.cancel_mods)
        {
            return Some(HotkeyAction::Cancel);
        }

        // Main
        let trigger = self.pressed.contains(self.main_trigger);
        let exact = self.mods_down(&self.main_mods) && !self.extra_mods(&self.main_mods);

        match mode {
            ActivationMode::Toggle => {
                if !was_trigger && trigger && exact {
                    return Some(HotkeyAction::Toggle);
                }
            }
            ActivationMode::Hold => {
                self.hold_active = trigger && exact;
                if !was_hold && self.hold_active {
                    return Some(HotkeyAction::Start);
                }
                if was_hold && !self.hold_active {
                    return Some(HotkeyAction::Stop);
                }
            }
            _ => {
                if !was_trigger && trigger && exact {
                    return Some(HotkeyAction::Toggle);
                }
            }
        }
        None
    }

    fn mods_down(&self, mods: &[Modifier]) -> bool {
        mods.iter().all(|m| {
            let (l, r) = modifier_keycodes(*m);
            self.pressed.contains(l) || self.pressed.contains(r)
        })
    }

    fn extra_mods(&self, required: &[Modifier]) -> bool {
        for m in &[
            Modifier::Ctrl,
            Modifier::Shift,
            Modifier::Alt,
            Modifier::Meta,
        ] {
            if required.contains(m) {
                continue;
            }
            let (l, r) = modifier_keycodes(*m);
            if self.pressed.contains(l) || self.pressed.contains(r) {
                return true;
            }
        }
        false
    }
}

// ─── Key mapping ─────────────────────────────────────────────────────────────

fn modifier_keycodes(m: Modifier) -> (KeyCode, KeyCode) {
    match m {
        Modifier::Ctrl => (KeyCode::KEY_LEFTCTRL, KeyCode::KEY_RIGHTCTRL),
        Modifier::Shift => (KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_RIGHTSHIFT),
        Modifier::Alt => (KeyCode::KEY_LEFTALT, KeyCode::KEY_RIGHTALT),
        Modifier::Meta => (KeyCode::KEY_LEFTMETA, KeyCode::KEY_RIGHTMETA),
        _ => (KeyCode::KEY_RESERVED, KeyCode::KEY_RESERVED),
    }
}

fn key_name_to_code(name: &str) -> Option<KeyCode> {
    Some(match name.to_ascii_lowercase().as_str() {
        "space" => KeyCode::KEY_SPACE,
        "escape" | "esc" => KeyCode::KEY_ESC,
        "return" | "enter" => KeyCode::KEY_ENTER,
        "tab" => KeyCode::KEY_TAB,
        "backspace" => KeyCode::KEY_BACKSPACE,
        "delete" => KeyCode::KEY_DELETE,
        "insert" => KeyCode::KEY_INSERT,
        "home" => KeyCode::KEY_HOME,
        "end" => KeyCode::KEY_END,
        "pageup" => KeyCode::KEY_PAGEUP,
        "pagedown" => KeyCode::KEY_PAGEDOWN,
        "up" => KeyCode::KEY_UP,
        "down" => KeyCode::KEY_DOWN,
        "left" => KeyCode::KEY_LEFT,
        "right" => KeyCode::KEY_RIGHT,
        "capslock" => KeyCode::KEY_CAPSLOCK,
        "pause" => KeyCode::KEY_PAUSE,
        "printscreen" | "sysrq" => KeyCode::KEY_SYSRQ,
        "f1" => KeyCode::KEY_F1,
        "f2" => KeyCode::KEY_F2,
        "f3" => KeyCode::KEY_F3,
        "f4" => KeyCode::KEY_F4,
        "f5" => KeyCode::KEY_F5,
        "f6" => KeyCode::KEY_F6,
        "f7" => KeyCode::KEY_F7,
        "f8" => KeyCode::KEY_F8,
        "f9" => KeyCode::KEY_F9,
        "f10" => KeyCode::KEY_F10,
        "f11" => KeyCode::KEY_F11,
        "f12" => KeyCode::KEY_F12,
        "a" => KeyCode::KEY_A,
        "b" => KeyCode::KEY_B,
        "c" => KeyCode::KEY_C,
        "d" => KeyCode::KEY_D,
        "e" => KeyCode::KEY_E,
        "f" => KeyCode::KEY_F,
        "g" => KeyCode::KEY_G,
        "h" => KeyCode::KEY_H,
        "i" => KeyCode::KEY_I,
        "j" => KeyCode::KEY_J,
        "k" => KeyCode::KEY_K,
        "l" => KeyCode::KEY_L,
        "m" => KeyCode::KEY_M,
        "n" => KeyCode::KEY_N,
        "o" => KeyCode::KEY_O,
        "p" => KeyCode::KEY_P,
        "q" => KeyCode::KEY_Q,
        "r" => KeyCode::KEY_R,
        "s" => KeyCode::KEY_S,
        "t" => KeyCode::KEY_T,
        "u" => KeyCode::KEY_U,
        "v" => KeyCode::KEY_V,
        "w" => KeyCode::KEY_W,
        "x" => KeyCode::KEY_X,
        "y" => KeyCode::KEY_Y,
        "z" => KeyCode::KEY_Z,
        "0" => KeyCode::KEY_0,
        "1" => KeyCode::KEY_1,
        "2" => KeyCode::KEY_2,
        "3" => KeyCode::KEY_3,
        "4" => KeyCode::KEY_4,
        "5" => KeyCode::KEY_5,
        "6" => KeyCode::KEY_6,
        "7" => KeyCode::KEY_7,
        "8" => KeyCode::KEY_8,
        "9" => KeyCode::KEY_9,
        "minus" => KeyCode::KEY_MINUS,
        "equal" | "equals" => KeyCode::KEY_EQUAL,
        "leftbrace" => KeyCode::KEY_LEFTBRACE,
        "rightbrace" => KeyCode::KEY_RIGHTBRACE,
        "backslash" => KeyCode::KEY_BACKSLASH,
        "semicolon" => KeyCode::KEY_SEMICOLON,
        "apostrophe" | "quote" => KeyCode::KEY_APOSTROPHE,
        "grave" | "backtick" => KeyCode::KEY_GRAVE,
        "comma" => KeyCode::KEY_COMMA,
        "dot" | "period" => KeyCode::KEY_DOT,
        "slash" => KeyCode::KEY_SLASH,
        _ => return None,
    })
}

// ─── Pressed-key bitset ──────────────────────────────────────────────────────

const LIMIT: usize = 1024;
const WORDS: usize = LIMIT / 64;

struct PressedCodes {
    w: [u64; WORDS],
}

impl Default for PressedCodes {
    fn default() -> Self {
        Self { w: [0; WORDS] }
    }
}

impl PressedCodes {
    fn insert(&mut self, c: KeyCode) {
        let i = c.code() as usize;
        if i < LIMIT {
            self.w[i / 64] |= 1u64 << (i % 64);
        }
    }
    fn remove(&mut self, c: KeyCode) {
        let i = c.code() as usize;
        if i < LIMIT {
            self.w[i / 64] &= !(1u64 << (i % 64));
        }
    }
    fn contains(&self, c: KeyCode) -> bool {
        let i = c.code() as usize;
        i < LIMIT && (self.w[i / 64] >> (i % 64)) & 1 != 0
    }
    /// Drain all set codes, clearing the set. Used to release every key still
    /// marked down on listener exit.
    fn drain_codes(&mut self) -> Vec<KeyCode> {
        let mut codes = Vec::new();
        for (wi, word) in self.w.iter_mut().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                codes.push(KeyCode((wi * 64 + b) as u16));
            }
            *word = 0;
        }
        codes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(code: KeyCode, pressed: bool) -> InputEvent {
        InputEvent::new(EventType::KEY.0, code.code(), if pressed { 1 } else { 0 })
    }

    /// A `KeySink` that records the (code, value) of every event it receives.
    #[derive(Default)]
    struct Recorder {
        events: Vec<(u16, i32)>,
    }

    impl KeySink for Recorder {
        fn send_event(&mut self, event: InputEvent) {
            self.events.push((event.code(), event.value()));
        }
    }

    fn ctrl_space_state() -> HotkeyState {
        HotkeyState::new(
            KeyCode::KEY_SPACE,
            vec![Modifier::Ctrl],
            KeyCode::KEY_ESC,
            vec![],
        )
    }

    #[test]
    fn live_config_update_bumps_generation_and_rejects_unknown_keys() {
        let config = HotkeyConfig::default();
        let live = LiveHotkeyConfig::new(&config, ActivationMode::Toggle).unwrap();
        let g0 = live.generation();
        live.update(&config, ActivationMode::Hold).unwrap();
        assert!(live.generation() > g0);
        assert_eq!(live.snapshot().mode, ActivationMode::Hold);

        let mut bad = config.clone();
        bad.toggle_dictation.key = "definitely-not-a-key".into();
        assert!(live.update(&bad, ActivationMode::Toggle).is_err());
        // Failed update keeps the previous config.
        assert_eq!(live.snapshot().mode, ActivationMode::Hold);
    }

    #[test]
    fn retarget_preserves_pressed_keys_and_clears_hold() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Hold;
        s.handle(KeyCode::KEY_LEFTCTRL, true, m);
        s.handle(KeyCode::KEY_SPACE, true, m); // Start; hold_active = true
        s.retarget(KeyCode::KEY_F9, vec![], KeyCode::KEY_ESC, vec![]);
        // Old combo's release must not emit a Stop for the new combo.
        assert!(s.handle(KeyCode::KEY_SPACE, false, m).is_none());
        // Still-held Ctrl counts as an extra modifier for the bare-F9 combo —
        // pressed-key tracking survived the retarget.
        assert!(s.handle(KeyCode::KEY_F9, true, m).is_none());
        s.handle(KeyCode::KEY_F9, false, m);
        s.handle(KeyCode::KEY_LEFTCTRL, false, m);
        // With Ctrl released, the new trigger works.
        assert!(matches!(
            s.handle(KeyCode::KEY_F9, true, m),
            Some(HotkeyAction::Start)
        ));
    }

    #[test]
    fn hold_mode_emits_start_on_press_and_stop_on_release() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Hold;
        assert!(s.handle(KeyCode::KEY_LEFTCTRL, true, m).is_none());
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, m),
            Some(HotkeyAction::Start)
        ));
        // Releasing the trigger stops, regardless of release order.
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, false, m),
            Some(HotkeyAction::Stop)
        ));
        assert!(s.handle(KeyCode::KEY_LEFTCTRL, false, m).is_none());
    }

    #[test]
    fn hold_mode_stops_when_modifier_released_first() {
        let mut s = ctrl_space_state();
        let m = ActivationMode::Hold;
        s.handle(KeyCode::KEY_LEFTCTRL, true, m);
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, m),
            Some(HotkeyAction::Start)
        ));
        assert!(matches!(
            s.handle(KeyCode::KEY_LEFTCTRL, false, m),
            Some(HotkeyAction::Stop)
        ));
    }

    #[test]
    fn mode_switch_mid_session_does_not_panic_or_double_fire() {
        // Listener was in Toggle when the user pressed; settings flip to
        // Hold before release — the release must not emit a spurious action.
        let mut s = ctrl_space_state();
        s.handle(KeyCode::KEY_LEFTCTRL, true, ActivationMode::Toggle);
        assert!(matches!(
            s.handle(KeyCode::KEY_SPACE, true, ActivationMode::Toggle),
            Some(HotkeyAction::Toggle)
        ));
        // Hold-mode evaluation on release: hold_active was never set, so the
        // release is silent rather than a stray Stop.
        assert!(
            s.handle(KeyCode::KEY_SPACE, false, ActivationMode::Hold)
                .is_none()
        );
    }

    #[test]
    fn escape_cancels_in_both_activation_modes() {
        // A pill-started session has no held key to release, so the cancel key
        // (Escape) must emit Cancel regardless of activation mode — the
        // dispatcher gates on an active session, not on the mode.
        for mode in [ActivationMode::Toggle, ActivationMode::Hold] {
            let mut s = ctrl_space_state(); // cancel = bare Escape
            assert!(
                matches!(
                    s.handle(KeyCode::KEY_ESC, true, mode),
                    Some(HotkeyAction::Cancel)
                ),
                "Escape should cancel in {mode:?} mode"
            );
        }
    }

    #[test]
    fn typing_keyboard_filter_rejects_secondary_nodes_and_pointers() {
        // A real keyboard: letters, space, shift, enter, not a pointer.
        assert!(is_typing_keyboard_caps(true, true, true, true, false));
        // The Keychron's secondary node (event7): A+Space but no Shift → reject,
        // so the keyboard is no longer grabbed twice.
        assert!(!is_typing_keyboard_caps(true, true, false, true, false));
        // A gaming mouse's macro interface that advertises a full keyboard but
        // carries pointer axes → reject.
        assert!(!is_typing_keyboard_caps(true, true, true, true, true));
        // A media/macro pad without Enter → reject.
        assert!(!is_typing_keyboard_caps(true, true, true, false, false));
    }

    #[test]
    fn forwarded_modifier_release_is_not_swallowed_by_action() {
        // Hold mode, Ctrl+Space. The user holds Ctrl for an unrelated chord
        // (Ctrl+X) — leaking Ctrl-down through to apps — then triggers
        // dictation and releases Ctrl first. The Stop action must not eat the
        // Ctrl release, or Ctrl stays stuck down in every app.
        let m = ActivationMode::Hold;
        let combo = build_combo_mods(&[Modifier::Ctrl]);
        let mut state = ctrl_space_state();
        let mut fwd = Forwarder::new(Recorder::default());
        let mut consumed = PressedCodes::default();
        let mut deferred: Vec<InputEvent> = Vec::new();

        let ctrl = KeyCode::KEY_LEFTCTRL;
        let x = KeyCode::KEY_X;
        let space = KeyCode::KEY_SPACE;

        let step = |code: KeyCode,
                    pressed: bool,
                    state: &mut HotkeyState,
                    fwd: &mut Forwarder<Recorder>,
                    consumed: &mut PressedCodes,
                    deferred: &mut Vec<InputEvent>| {
            process_grabbed_key(
                ev(code, pressed),
                code,
                pressed,
                state,
                fwd,
                consumed,
                deferred,
                &combo,
                m,
            )
        };

        // Ctrl down — deferred, not yet forwarded.
        step(
            ctrl,
            true,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
        );
        // X down — flushes the deferred Ctrl down, then X, to apps.
        step(x, true, &mut state, &mut fwd, &mut consumed, &mut deferred);
        assert!(fwd.is_down(ctrl), "Ctrl press should have reached apps");

        // Space down — completes the combo: Start.
        let start = step(
            space,
            true,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
        );
        assert!(matches!(start, Some(HotkeyAction::Start)));

        // Release Ctrl first — Hold mode emits Stop.
        let stop = step(
            ctrl,
            false,
            &mut state,
            &mut fwd,
            &mut consumed,
            &mut deferred,
        );
        assert!(matches!(stop, Some(HotkeyAction::Stop)));

        // The release must have reached apps, balancing the earlier press.
        assert!(
            !fwd.is_down(ctrl),
            "Ctrl release was swallowed by Stop — Ctrl is stuck down"
        );
    }

    #[test]
    fn release_all_releases_every_forwarded_down_key() {
        let mut fwd = Forwarder::new(Recorder::default());
        fwd.forward(ev(KeyCode::KEY_LEFTCTRL, true));
        fwd.forward(ev(KeyCode::KEY_A, true));
        fwd.forward(ev(KeyCode::KEY_A, false));
        assert!(fwd.is_down(KeyCode::KEY_LEFTCTRL));
        assert!(!fwd.is_down(KeyCode::KEY_A));

        fwd.release_all();

        assert!(!fwd.is_down(KeyCode::KEY_LEFTCTRL));
        assert!(
            fwd.sink
                .events
                .iter()
                .any(|&(c, v)| c == KeyCode::KEY_LEFTCTRL.code() && v == 0),
            "expected a synthetic Ctrl release"
        );
    }

    // ─── Intent channel ──────────────────────────────────────────────────────

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
