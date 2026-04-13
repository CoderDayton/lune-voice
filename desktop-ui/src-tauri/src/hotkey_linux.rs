//! Linux-only global hotkey detection via evdev with device grab.
//!
//! Grabs physical keyboards exclusively and re-injects non-hotkey events
//! through a uinput passthrough device. This works on both X11 and Wayland.

use std::sync::mpsc as std_mpsc;

use evdev::{AttributeSet, Device, EventType, InputEvent, KeyCode, uinput::VirtualDevice};
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

/// Spawn the evdev hotkey listener on a background OS thread.
///
/// Returns `None` if the hotkey config references unknown key names or no
/// keyboards are accessible.
pub fn spawn(
    config: HotkeyConfig,
    activation_mode: ActivationMode,
    tx: mpsc::Sender<HotkeyAction>,
) -> Option<std::thread::JoinHandle<()>> {
    let main_trigger = key_name_to_code(&config.toggle_dictation.key)?;
    let cancel_trigger =
        key_name_to_code(&config.cancel_dictation.key).unwrap_or(KeyCode::KEY_RESERVED);

    let main_mods = config.toggle_dictation.modifiers;
    let cancel_mods = config.cancel_dictation.modifiers;

    let handle = std::thread::Builder::new()
        .name("hotkey-evdev".into())
        .spawn(move || {
            run_listener(
                main_trigger,
                main_mods,
                cancel_trigger,
                cancel_mods,
                activation_mode,
                tx,
            );
        })
        .expect("spawn hotkey listener thread");

    Some(handle)
}

// ─── Listener core ───────────────────────────────────────────────────────────

fn run_listener(
    main_trigger: KeyCode,
    main_mods: Vec<Modifier>,
    cancel_trigger: KeyCode,
    cancel_mods: Vec<Modifier>,
    activation_mode: ActivationMode,
    cmd_tx: mpsc::Sender<HotkeyAction>,
) {
    let keyboards = enumerate_keyboards();
    if keyboards.is_empty() {
        warn!("no accessible keyboard devices found; hotkey detection disabled");
        return;
    }
    info!(
        count = keyboards.len(),
        "keyboard devices opened for hotkey detection"
    );

    let mut passthrough = match build_passthrough_device() {
        Ok(dev) => dev,
        Err(e) => {
            warn!(error = %e, "failed to create passthrough device; hotkey grab disabled");
            return;
        }
    };

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

    let mut combo_mod_codes = PressedCodes::default();
    for m in &main_mods {
        let (l, r) = modifier_keycodes(*m);
        combo_mod_codes.insert(l);
        combo_mod_codes.insert(r);
    }

    let mut state = HotkeyState::new(main_trigger, main_mods, cancel_trigger, cancel_mods);
    let mut consumed = PressedCodes::default();
    let mut deferred: Vec<InputEvent> = Vec::new();

    for (raw, grabbed) in event_rx {
        if raw.event_type() != EventType::KEY {
            if grabbed {
                forward(&mut passthrough, raw);
            }
            continue;
        }

        let code = KeyCode(raw.code());
        let value = raw.value();

        // Auto-repeat
        if value == 2 {
            if grabbed && !consumed.contains(code) && !combo_mod_codes.contains(code) {
                forward(&mut passthrough, raw);
            }
            continue;
        }

        let pressed = value == 1;

        if !grabbed {
            let action = state.handle(code, pressed, activation_mode);
            if let Some(a) = action {
                dispatch(a, &cmd_tx);
            }
            continue;
        }

        // Defer combo modifier presses
        if pressed && combo_mod_codes.contains(code) {
            deferred.push(raw);
            state.handle(code, pressed, activation_mode);
            continue;
        }

        let action = state.handle(code, pressed, activation_mode);

        match action {
            Some(HotkeyAction::Cancel) => {
                flush(&mut passthrough, &mut deferred);
                dispatch(HotkeyAction::Cancel, &cmd_tx);
                forward(&mut passthrough, raw);
            }
            Some(a) => {
                deferred.clear();
                if pressed {
                    consumed.insert(code);
                } else {
                    consumed.remove(code);
                }
                dispatch(a, &cmd_tx);
            }
            None if !pressed && consumed.contains(code) => {
                consumed.remove(code);
            }
            None if !pressed && combo_mod_codes.contains(code) => {
                flush(&mut passthrough, &mut deferred);
                forward(&mut passthrough, raw);
            }
            None => {
                flush(&mut passthrough, &mut deferred);
                forward(&mut passthrough, raw);
            }
        }
    }
    info!("all keyboard devices disconnected; hotkey listener exiting");
}

fn dispatch(action: HotkeyAction, tx: &mpsc::Sender<HotkeyAction>) {
    info!(?action, "hotkey dispatched");
    if let Err(e) = tx.try_send(action) {
        warn!(error = %e, "hotkey channel send failed");
    }
}

#[inline]
fn flush(passthrough: &mut VirtualDevice, deferred: &mut Vec<InputEvent>) {
    for ev in deferred.drain(..) {
        forward(passthrough, ev);
    }
}

#[inline]
fn forward(passthrough: &mut VirtualDevice, event: InputEvent) {
    let _ = passthrough.emit(&[event, InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0)]);
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
            let is_kb = dev
                .supported_keys()
                .is_some_and(|k| k.contains(KeyCode::KEY_A) && k.contains(KeyCode::KEY_SPACE));
            if !is_kb {
                return None;
            }
            info!(name = ?dev.name(), path = %path.display(), "found keyboard");
            Some(dev)
        })
        .collect()
}

fn read_device(mut dev: Device, tx: std_mpsc::Sender<(InputEvent, bool)>) {
    let name = dev.name().unwrap_or("unknown").to_string();
    let grabbed = match dev.grab() {
        Ok(()) => {
            info!(name, "grabbed keyboard");
            true
        }
        Err(e) => {
            warn!(error = %e, name, "grab failed; observe-only");
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

        // Cancel
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
}
