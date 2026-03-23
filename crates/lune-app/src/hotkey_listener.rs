//! Global hotkey detection via evdev with device grab.
//!
//! Scans physical keyboards (`/dev/input/event*`), grabs them exclusively,
//! tracks modifier state with a boolean bitset, and dispatches [`AppCommand`]s
//! when configured key combinations are detected. Non-hotkey events are
//! re-injected through a passthrough virtual device so normal typing is
//! unaffected. Filters out `lune-voice*` virtual devices to prevent
//! self-triggering from uinput output.

use std::sync::mpsc as std_mpsc;

use evdev::{AttributeSet, Device, EventType, InputEvent, KeyCode, uinput::VirtualDevice};
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};

use lune_core::config::{ActivationMode, HotkeyConfig, Modifier};

use crate::events::AppCommand;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Spawn the hotkey listener on background OS threads.
///
/// Returns `None` if the hotkey config references unknown key names.
/// The listener runs until all keyboard reader threads exit (device disconnect
/// or process shutdown).
pub fn spawn(
    hotkey_config: HotkeyConfig,
    activation_mode_rx: watch::Receiver<ActivationMode>,
    cmd_tx: mpsc::Sender<AppCommand>,
) -> Option<std::thread::JoinHandle<()>> {
    let main_trigger = match key_name_to_code(&hotkey_config.toggle_dictation.key) {
        Some(k) => k,
        None => {
            warn!(
                key = %hotkey_config.toggle_dictation.key,
                "unknown toggle key name; hotkey listener disabled"
            );
            return None;
        }
    };

    let cancel_trigger = match key_name_to_code(&hotkey_config.cancel_dictation.key) {
        Some(k) => k,
        None => {
            warn!(
                key = %hotkey_config.cancel_dictation.key,
                "unknown cancel key name; cancel hotkey disabled"
            );
            // Still spawn — main hotkey works, cancel just won't fire.
            // Use KEY_RESERVED as a sentinel that never matches real input.
            KeyCode::KEY_RESERVED
        }
    };

    let main_modifiers = hotkey_config.toggle_dictation.modifiers;
    let cancel_modifiers = hotkey_config.cancel_dictation.modifiers;

    let handle = std::thread::Builder::new()
        .name("hotkey-listener".into())
        .spawn(move || {
            run_listener(
                main_trigger,
                main_modifiers,
                cancel_trigger,
                cancel_modifiers,
                activation_mode_rx,
                cmd_tx,
            );
        })
        .expect("spawn hotkey listener thread");

    Some(handle)
}

// ─── Listener core ───────────────────────────────────────────────────────────

fn run_listener(
    main_trigger: KeyCode,
    main_modifiers: Vec<Modifier>,
    cancel_trigger: KeyCode,
    cancel_modifiers: Vec<Modifier>,
    activation_mode_rx: watch::Receiver<ActivationMode>,
    cmd_tx: mpsc::Sender<AppCommand>,
) {
    let keyboards = enumerate_keyboards();
    if keyboards.is_empty() {
        warn!(
            "no accessible keyboard devices found; hotkey detection disabled. \
             Ensure the current user has read access to /dev/input/event* \
             (add user to 'input' group or check logind seat assignment)"
        );
        return;
    }

    info!(
        count = keyboards.len(),
        "keyboard devices opened for hotkey detection"
    );

    // Build the passthrough device before any reader thread can grab a real
    // keyboard. If this fails, we bail out without intercepting input.
    let mut passthrough = match build_passthrough_device() {
        Ok(dev) => dev,
        Err(e) => {
            warn!(error = %e, "failed to create passthrough device; hotkey grab disabled");
            return;
        }
    };

    // Fan-in channel: each reader thread sends raw events here.
    let (event_tx, event_rx) = std_mpsc::channel::<(InputEvent, bool)>();

    for dev in keyboards {
        let tx = event_tx.clone();
        let name = dev.name().unwrap_or("unknown").to_string();
        std::thread::Builder::new()
            .name(format!("evdev-{name}"))
            .spawn(move || read_device(dev, tx))
            .expect("spawn evdev reader thread");
    }
    // Drop our sender so the channel closes when all readers exit.
    drop(event_tx);

    // Collect all modifier keycodes used by the main combo so we can defer
    // their press events until we know the trigger key won't follow.
    let mut combo_modifier_codes = PressedCodes::default();
    for m in &main_modifiers {
        let (left, right) = modifier_keycodes(*m);
        combo_modifier_codes.insert(left);
        combo_modifier_codes.insert(right);
    }

    let mut state = HotkeyState::new(
        main_trigger,
        main_modifiers,
        cancel_trigger,
        cancel_modifiers,
    );

    // Tracks which trigger keys had their press event consumed, so we also
    // consume the corresponding release (prevent orphan key-up reaching apps).
    let mut consumed_triggers = PressedCodes::default();

    // Deferred modifier press events waiting to see if the trigger follows.
    let mut deferred_mods: Vec<InputEvent> = Vec::new();

    for (raw_event, grabbed) in event_rx {
        if raw_event.event_type() != EventType::KEY {
            // Non-KEY events (SYN, MSC, LED): forward only for grabbed
            // devices. Ungrabbed devices already reach the focused app.
            if grabbed {
                forward(&mut passthrough, raw_event);
            }
            continue;
        }

        let code = KeyCode(raw_event.code());
        let value = raw_event.value();

        // Auto-repeat (value=2): forward only if the key wasn't consumed and
        // the device is grabbed.
        if value == 2 {
            if grabbed && !consumed_triggers.contains(code) && !combo_modifier_codes.contains(code)
            {
                forward(&mut passthrough, raw_event);
            }
            continue;
        }

        let pressed = value == 1;

        if !grabbed {
            // Ungrabbed devices already deliver their native events to the app.
            // Keep the state machine hotkey detection, but do not reinject.
            let signal = InputSignal { code, pressed };
            let mode = *activation_mode_rx.borrow();
            let action = state.handle_signal(signal, mode);
            if action != Action::None {
                dispatch(action, &cmd_tx);
            }
            continue;
        }

        // Defer combo modifier press events — don't forward yet.
        if pressed && combo_modifier_codes.contains(code) {
            deferred_mods.push(raw_event);
            // Still update the hotkey state so handle_signal sees the modifier.
            let signal = InputSignal { code, pressed };
            let mode = *activation_mode_rx.borrow();
            state.handle_signal(signal, mode);
            continue;
        }

        let signal = InputSignal { code, pressed };
        let mode = *activation_mode_rx.borrow();
        let action = state.handle_signal(signal, mode);

        if action == Action::Cancel {
            // Cancel hotkey: dispatch command but ALSO forward the key so
            // Escape (default cancel) still works normally in apps.
            // Flush any deferred modifiers first.
            flush_deferred(&mut passthrough, &mut deferred_mods);
            dispatch(action, &cmd_tx);
            forward(&mut passthrough, raw_event);
        } else if action != Action::None {
            // Main hotkey combo triggered — consume the trigger key AND
            // discard deferred modifier events (they were part of the combo).
            deferred_mods.clear();
            if pressed {
                consumed_triggers.insert(code);
            } else {
                consumed_triggers.remove(code);
            }
            dispatch(action, &cmd_tx);
        } else if !pressed && consumed_triggers.contains(code) {
            // Release of a previously consumed key (e.g. Toggle mode:
            // Space release after Ctrl+Space press) — consume silently.
            consumed_triggers.remove(code);
        } else if !pressed && combo_modifier_codes.contains(code) {
            // Release of a combo modifier — flush its deferred press (if any)
            // followed by the release so apps see a matched press+release.
            flush_deferred(&mut passthrough, &mut deferred_mods);
            forward(&mut passthrough, raw_event);
        } else {
            // Non-combo key event — flush any deferred modifiers first
            // (the combo won't fire) then forward this key.
            flush_deferred(&mut passthrough, &mut deferred_mods);
            forward(&mut passthrough, raw_event);
        }
    }

    info!("all keyboard devices disconnected; hotkey listener exiting");
}

fn dispatch(action: Action, cmd_tx: &mpsc::Sender<AppCommand>) {
    let cmd = match action {
        Action::Toggle => AppCommand::ToggleDictation,
        Action::Start => AppCommand::StartDictation,
        Action::Stop => AppCommand::StopDictation,
        Action::Cancel => AppCommand::CancelDictation,
        Action::None => return,
    };
    debug!(?cmd, "hotkey dispatched");
    // try_send: non-blocking, bounded(64). At human input rates this
    // never fills up. If it does, dropping one command is acceptable.
    let _ = cmd_tx.try_send(cmd);
}

/// Flush any deferred modifier press events to the passthrough device.
#[inline]
fn flush_deferred(passthrough: &mut VirtualDevice, deferred: &mut Vec<InputEvent>) {
    for event in deferred.drain(..) {
        forward(passthrough, event);
    }
}

/// Forward an event through the passthrough virtual device.
#[inline]
fn forward(passthrough: &mut VirtualDevice, event: InputEvent) {
    // Emit the event + SYN_REPORT for proper event grouping.
    let _ = passthrough.emit(&[event, InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0)]);
}

// ─── Device enumeration ──────────────────────────────────────────────────────

/// Prefix for our virtual devices — must be filtered out during enumeration.
const SELF_DEVICE_PREFIX: &str = "lune-voice";

fn enumerate_keyboards() -> Vec<Device> {
    let mut keyboards = Vec::new();

    for (path, dev) in evdev::enumerate() {
        // Skip our own uinput virtual devices (typing + passthrough).
        if dev
            .name()
            .is_some_and(|n| n.starts_with(SELF_DEVICE_PREFIX))
        {
            debug!(path = %path.display(), name = ?dev.name(), "skipping self device");
            continue;
        }

        // A real keyboard has alphabetic keys + space. This filters out mice,
        // gamepads, power buttons, and other non-keyboard input devices.
        let is_keyboard = match dev.supported_keys() {
            Some(keys) => keys.contains(KeyCode::KEY_A) && keys.contains(KeyCode::KEY_SPACE),
            None => false,
        };

        if !is_keyboard {
            continue;
        }

        info!(
            name = ?dev.name(),
            path = %path.display(),
            "found keyboard device"
        );
        keyboards.push(dev);
    }

    keyboards
}

/// Blocking event reader for a single device. Grabs the device exclusively
/// so hotkey events don't leak to the focused application.
fn read_device(mut dev: Device, tx: std_mpsc::Sender<(InputEvent, bool)>) {
    let name = dev.name().unwrap_or("unknown").to_string();

    // Grab the device so key events don't reach the compositor directly.
    // If grab fails, we still read events for hotkey detection, but the
    // physical device continues to deliver native input to the focused app.
    // We do not reinject those events through the passthrough device, or they
    // would be duplicated.
    let grabbed = match dev.grab() {
        Ok(()) => {
            info!(name, "grabbed keyboard device");
            true
        }
        Err(e) => {
            warn!(
                error = %e,
                name,
                "failed to grab keyboard; hotkey detection will be observe-only"
            );
            false
        }
    };

    loop {
        match dev.fetch_events() {
            Ok(events) => {
                for ev in events {
                    if tx.send((ev, grabbed)).is_err() {
                        return; // Consumer dropped — shutting down.
                    }
                }
            }
            Err(e) => {
                debug!(error = %e, name, "keyboard device disconnected");
                return;
            }
        }
    }
}

// ─── Passthrough virtual device ──────────────────────────────────────────────

/// Build a uinput virtual keyboard for re-injecting non-hotkey events.
fn build_passthrough_device() -> Result<VirtualDevice, String> {
    let mut keys = AttributeSet::<KeyCode>::new();
    // Register all standard keycodes (0..768 covers KEY_MAX).
    for i in 1u16..768 {
        keys.insert(KeyCode(i));
    }

    let device = VirtualDevice::builder()
        .map_err(|e| format!("passthrough builder: {e}"))?
        .name("lune-voice-passthrough")
        .with_keys(&keys)
        .map_err(|e| format!("passthrough with_keys: {e}"))?
        .build()
        .map_err(|e| format!("passthrough build: {e}"))?;

    // Give the compositor time to register the new virtual device.
    std::thread::sleep(std::time::Duration::from_millis(200));

    debug!("passthrough virtual keyboard ready");
    Ok(device)
}

// ─── Hotkey state machine ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputSignal {
    code: KeyCode,
    pressed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Toggle mode: rising edge detected → ToggleDictation
    Toggle,
    /// Hold mode: combo pressed → StartDictation
    Start,
    /// Hold mode: combo released → StopDictation
    Stop,
    /// Cancel combo pressed → CancelDictation
    Cancel,
    /// No action
    None,
}

struct HotkeyState {
    pressed: PressedCodes,
    main_trigger: KeyCode,
    main_modifiers: Vec<Modifier>,
    cancel_trigger: KeyCode,
    cancel_modifiers: Vec<Modifier>,
    /// Tracks whether the main combo is currently active (for Hold mode transitions).
    hold_active: bool,
}

impl HotkeyState {
    fn new(
        main_trigger: KeyCode,
        main_modifiers: Vec<Modifier>,
        cancel_trigger: KeyCode,
        cancel_modifiers: Vec<Modifier>,
    ) -> Self {
        Self {
            pressed: PressedCodes::default(),
            main_trigger,
            main_modifiers,
            cancel_trigger,
            cancel_modifiers,
            hold_active: false,
        }
    }

    fn handle_signal(&mut self, signal: InputSignal, mode: ActivationMode) -> Action {
        // Snapshot state BEFORE updating the bitset (for rising-edge detection).
        let was_main_trigger_down = self.pressed.contains(self.main_trigger);
        let was_hold_active = self.hold_active;

        // Update pressed state.
        if signal.pressed {
            self.pressed.insert(signal.code);
        } else {
            self.pressed.remove(signal.code);
        }

        // ── Cancel hotkey (always fires on rising edge) ──────────────
        if signal.pressed
            && signal.code == self.cancel_trigger
            && self.cancel_trigger != KeyCode::KEY_RESERVED
            && self.modifiers_are_down(&self.cancel_modifiers)
            && !self.has_extra_modifiers(&self.cancel_modifiers.clone())
        {
            return Action::Cancel;
        }

        // ── Main hotkey ──────────────────────────────────────────────
        let main_trigger_down = self.pressed.contains(self.main_trigger);
        let mods_down = self.modifiers_are_down(&self.main_modifiers);
        // Don't fire if extra modifiers are held — that's a different combo
        // (e.g. Ctrl+Shift+Space when our hotkey is Ctrl+Space).
        let exact_mods = mods_down && !self.has_extra_modifiers(&self.main_modifiers);

        match mode {
            ActivationMode::Toggle => {
                // Fire on rising edge of trigger while exact modifiers are held.
                if !was_main_trigger_down && main_trigger_down && exact_mods {
                    return Action::Toggle;
                }
            }
            ActivationMode::Hold => {
                self.hold_active = main_trigger_down && exact_mods;
                if !was_hold_active && self.hold_active {
                    return Action::Start;
                }
                if was_hold_active && !self.hold_active {
                    return Action::Stop;
                }
            }
            _ => {
                // Future activation modes: treat as Toggle.
                if !was_main_trigger_down && main_trigger_down && exact_mods {
                    return Action::Toggle;
                }
            }
        }

        Action::None
    }

    /// Returns true if ALL required modifiers have at least one variant (left/right) pressed.
    /// Empty modifier list → always true (no modifiers required).
    fn modifiers_are_down(&self, modifiers: &[Modifier]) -> bool {
        modifiers.iter().all(|m| {
            let (left, right) = modifier_keycodes(*m);
            self.pressed.contains(left) || self.pressed.contains(right)
        })
    }

    /// Returns true if any modifier key is pressed that is NOT in the required set.
    /// This prevents Ctrl+Space from firing when Ctrl+Shift+Space is pressed.
    fn has_extra_modifiers(&self, required: &[Modifier]) -> bool {
        const ALL_MODIFIERS: &[Modifier] = &[
            Modifier::Ctrl,
            Modifier::Shift,
            Modifier::Alt,
            Modifier::Meta,
        ];
        for m in ALL_MODIFIERS {
            if required.contains(m) {
                continue;
            }
            let (left, right) = modifier_keycodes(*m);
            if self.pressed.contains(left) || self.pressed.contains(right) {
                return true;
            }
        }
        false
    }
}

// ─── Key name mapping ────────────────────────────────────────────────────────

fn modifier_keycodes(m: Modifier) -> (KeyCode, KeyCode) {
    match m {
        Modifier::Ctrl => (KeyCode::KEY_LEFTCTRL, KeyCode::KEY_RIGHTCTRL),
        Modifier::Shift => (KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_RIGHTSHIFT),
        Modifier::Alt => (KeyCode::KEY_LEFTALT, KeyCode::KEY_RIGHTALT),
        Modifier::Meta => (KeyCode::KEY_LEFTMETA, KeyCode::KEY_RIGHTMETA),
        // Non-exhaustive: future variants fall back to unreachable codes.
        _ => (KeyCode::KEY_RESERVED, KeyCode::KEY_RESERVED),
    }
}

fn key_name_to_code(name: &str) -> Option<KeyCode> {
    // Case-insensitive matching for user convenience.
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
        "numlock" => KeyCode::KEY_NUMLOCK,
        "scrolllock" => KeyCode::KEY_SCROLLLOCK,
        "pause" => KeyCode::KEY_PAUSE,
        "printscreen" | "sysrq" => KeyCode::KEY_SYSRQ,
        "menu" | "compose" => KeyCode::KEY_COMPOSE,

        // Function keys
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

        // Letters
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

        // Digits
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

        // Punctuation
        "minus" => KeyCode::KEY_MINUS,
        "equal" | "equals" => KeyCode::KEY_EQUAL,
        "leftbrace" | "bracketleft" => KeyCode::KEY_LEFTBRACE,
        "rightbrace" | "bracketright" => KeyCode::KEY_RIGHTBRACE,
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

pub(crate) fn is_supported_hotkey_key(name: &str) -> bool {
    key_name_to_code(name).is_some()
}

// ─── Pressed-key bitset ──────────────────────────────────────────────────────
//
// Boolean bitset (not reference-counted). Each bit = whether a key is currently
// pressed. This ensures synthetic events from the uinput typer cannot
// accumulate extra "presses" that would desync the tracker.

const TRACKED_KEYCODE_LIMIT: usize = 1024;
const TRACKED_KEYCODE_WORDS: usize = TRACKED_KEYCODE_LIMIT / u64::BITS as usize;

struct PressedCodes {
    words: [u64; TRACKED_KEYCODE_WORDS],
}

impl Default for PressedCodes {
    fn default() -> Self {
        Self {
            words: [0; TRACKED_KEYCODE_WORDS],
        }
    }
}

impl PressedCodes {
    fn insert(&mut self, code: KeyCode) {
        if let Some(idx) = tracked_index(code) {
            self.words[idx / 64] |= 1u64 << (idx % 64);
        }
    }

    fn remove(&mut self, code: KeyCode) {
        if let Some(idx) = tracked_index(code) {
            self.words[idx / 64] &= !(1u64 << (idx % 64));
        }
    }

    fn contains(&self, code: KeyCode) -> bool {
        tracked_index(code)
            .map(|idx| (self.words[idx / 64] >> (idx % 64)) & 1 != 0)
            .unwrap_or(false)
    }
}

fn tracked_index(code: KeyCode) -> Option<usize> {
    let idx = code.code() as usize;
    if idx >= TRACKED_KEYCODE_LIMIT {
        return None;
    }
    Some(idx)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn press(code: KeyCode) -> InputSignal {
        InputSignal {
            code,
            pressed: true,
        }
    }

    fn release(code: KeyCode) -> InputSignal {
        InputSignal {
            code,
            pressed: false,
        }
    }

    fn default_state() -> HotkeyState {
        HotkeyState::new(
            KeyCode::KEY_SPACE,
            vec![Modifier::Ctrl],
            KeyCode::KEY_ESC,
            vec![],
        )
    }

    #[test]
    fn toggle_mode_activates_on_ctrl_space() {
        let mut state = default_state();

        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_LEFTCTRL), ActivationMode::Toggle),
            Action::None
        );
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::Toggle
        );
    }

    #[test]
    fn toggle_mode_ignores_space_without_modifier() {
        let mut state = default_state();

        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::None
        );
    }

    #[test]
    fn toggle_mode_ignores_auto_repeat() {
        let mut state = default_state();

        state.handle_signal(press(KeyCode::KEY_LEFTCTRL), ActivationMode::Toggle);
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::Toggle
        );
        // Second press without release = auto-repeat → no action
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::None
        );
    }

    #[test]
    fn toggle_mode_fires_again_after_release_cycle() {
        let mut state = default_state();

        state.handle_signal(press(KeyCode::KEY_LEFTCTRL), ActivationMode::Toggle);
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::Toggle
        );
        state.handle_signal(release(KeyCode::KEY_SPACE), ActivationMode::Toggle);
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::Toggle
        );
    }

    #[test]
    fn hold_mode_start_and_stop() {
        let mut state = default_state();

        state.handle_signal(press(KeyCode::KEY_LEFTCTRL), ActivationMode::Hold);
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Hold),
            Action::Start
        );
        assert_eq!(
            state.handle_signal(release(KeyCode::KEY_SPACE), ActivationMode::Hold),
            Action::Stop
        );
    }

    #[test]
    fn hold_mode_deactivates_on_modifier_release() {
        let mut state = default_state();

        state.handle_signal(press(KeyCode::KEY_LEFTCTRL), ActivationMode::Hold);
        state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Hold);
        // Release modifier while trigger still held → deactivate
        assert_eq!(
            state.handle_signal(release(KeyCode::KEY_LEFTCTRL), ActivationMode::Hold),
            Action::Stop
        );
    }

    #[test]
    fn cancel_fires_on_escape() {
        let mut state = default_state();

        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_ESC), ActivationMode::Toggle),
            Action::Cancel
        );
    }

    #[test]
    fn right_modifier_works() {
        let mut state = default_state();

        state.handle_signal(press(KeyCode::KEY_RIGHTCTRL), ActivationMode::Toggle);
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::Toggle
        );
    }

    #[test]
    fn extra_modifier_blocks_combo() {
        let mut state = default_state(); // Ctrl+Space

        // Press Ctrl+Shift+Space — should NOT trigger our Ctrl+Space combo
        state.handle_signal(press(KeyCode::KEY_LEFTCTRL), ActivationMode::Toggle);
        state.handle_signal(press(KeyCode::KEY_LEFTSHIFT), ActivationMode::Toggle);
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::None
        );
    }

    #[test]
    fn extra_modifier_released_then_combo_fires() {
        let mut state = default_state(); // Ctrl+Space

        // Press Ctrl+Shift, release Shift, then press Space → should fire
        state.handle_signal(press(KeyCode::KEY_LEFTCTRL), ActivationMode::Toggle);
        state.handle_signal(press(KeyCode::KEY_LEFTSHIFT), ActivationMode::Toggle);
        state.handle_signal(release(KeyCode::KEY_LEFTSHIFT), ActivationMode::Toggle);
        assert_eq!(
            state.handle_signal(press(KeyCode::KEY_SPACE), ActivationMode::Toggle),
            Action::Toggle
        );
    }

    #[test]
    fn key_name_mapping_case_insensitive() {
        assert_eq!(key_name_to_code("Space"), Some(KeyCode::KEY_SPACE));
        assert_eq!(key_name_to_code("space"), Some(KeyCode::KEY_SPACE));
        assert_eq!(key_name_to_code("SPACE"), Some(KeyCode::KEY_SPACE));
        assert_eq!(key_name_to_code("Escape"), Some(KeyCode::KEY_ESC));
        assert_eq!(key_name_to_code("F1"), Some(KeyCode::KEY_F1));
        assert_eq!(key_name_to_code("unknown_key"), None);
    }
}
