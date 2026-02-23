//! Universal kernel-level keyboard via uinput.
//!
//! Bypasses all display server protocols (Wayland virtual keyboard, XTest)
//! by spawning a ghost virtual hardware device. The kernel treats it as a
//! physical USB keyboard — it types into X11, GNOME, COSMIC, KDE, TTY, and
//! full-screen games unconditionally.
//!
//! First run: if /dev/uinput is not writable, uses pkexec to install a
//! systemd-logind uaccess udev rule via a native GUI password dialog.
//! After that, access is permanent — no further prompts.

use std::{fs::OpenOptions, process::Command, time::Duration};

use evdev::{AttributeSet, EventType, InputEvent, KeyCode, uinput::VirtualDevice};
use tracing::{debug, info};

// ─── Bootstrap ───────────────────────────────────────────────────────────────

const UDEV_RULE: &str =
    r#"KERNEL=="uinput", SUBSYSTEM=="misc", OPTIONS+="static_node=uinput", TAG+="uaccess""#;

const UDEV_RULE_PATH: &str = "/etc/udev/rules.d/99-lune-voice-uinput.rules";

fn ensure_kernel_access() -> Result<(), String> {
    if OpenOptions::new().write(true).open("/dev/uinput").is_ok() {
        return Ok(());
    }

    info!("first-time setup: requesting uinput access via pkexec");

    let script = format!(
        "echo '{}' > {} && udevadm control --reload-rules && udevadm trigger",
        UDEV_RULE, UDEV_RULE_PATH
    );

    let status = Command::new("pkexec")
        .arg("sh")
        .arg("-c")
        .arg(&script)
        .status()
        .map_err(|e| format!("pkexec spawn failed: {e}"))?;

    if !status.success() {
        return Err("pkexec authentication cancelled or failed".into());
    }

    // Give udevd 500ms to apply the new ACL.
    std::thread::sleep(Duration::from_millis(500));

    if OpenOptions::new().write(true).open("/dev/uinput").is_err() {
        return Err(
            "udev rule installed but /dev/uinput still inaccessible — reboot may be required"
                .into(),
        );
    }

    info!("uinput access granted permanently via uaccess rule");
    Ok(())
}

// ─── Key mapping (US QWERTY) ─────────────────────────────────────────────────

fn char_to_key(ch: char) -> Option<(KeyCode, bool)> {
    let pair = match ch {
        'a' | 'A' => (KeyCode::KEY_A, ch.is_uppercase()),
        'b' | 'B' => (KeyCode::KEY_B, ch.is_uppercase()),
        'c' | 'C' => (KeyCode::KEY_C, ch.is_uppercase()),
        'd' | 'D' => (KeyCode::KEY_D, ch.is_uppercase()),
        'e' | 'E' => (KeyCode::KEY_E, ch.is_uppercase()),
        'f' | 'F' => (KeyCode::KEY_F, ch.is_uppercase()),
        'g' | 'G' => (KeyCode::KEY_G, ch.is_uppercase()),
        'h' | 'H' => (KeyCode::KEY_H, ch.is_uppercase()),
        'i' | 'I' => (KeyCode::KEY_I, ch.is_uppercase()),
        'j' | 'J' => (KeyCode::KEY_J, ch.is_uppercase()),
        'k' | 'K' => (KeyCode::KEY_K, ch.is_uppercase()),
        'l' | 'L' => (KeyCode::KEY_L, ch.is_uppercase()),
        'm' | 'M' => (KeyCode::KEY_M, ch.is_uppercase()),
        'n' | 'N' => (KeyCode::KEY_N, ch.is_uppercase()),
        'o' | 'O' => (KeyCode::KEY_O, ch.is_uppercase()),
        'p' | 'P' => (KeyCode::KEY_P, ch.is_uppercase()),
        'q' | 'Q' => (KeyCode::KEY_Q, ch.is_uppercase()),
        'r' | 'R' => (KeyCode::KEY_R, ch.is_uppercase()),
        's' | 'S' => (KeyCode::KEY_S, ch.is_uppercase()),
        't' | 'T' => (KeyCode::KEY_T, ch.is_uppercase()),
        'u' | 'U' => (KeyCode::KEY_U, ch.is_uppercase()),
        'v' | 'V' => (KeyCode::KEY_V, ch.is_uppercase()),
        'w' | 'W' => (KeyCode::KEY_W, ch.is_uppercase()),
        'x' | 'X' => (KeyCode::KEY_X, ch.is_uppercase()),
        'y' | 'Y' => (KeyCode::KEY_Y, ch.is_uppercase()),
        'z' | 'Z' => (KeyCode::KEY_Z, ch.is_uppercase()),

        '1' | '!' => (KeyCode::KEY_1, ch == '!'),
        '2' | '@' => (KeyCode::KEY_2, ch == '@'),
        '3' | '#' => (KeyCode::KEY_3, ch == '#'),
        '4' | '$' => (KeyCode::KEY_4, ch == '$'),
        '5' | '%' => (KeyCode::KEY_5, ch == '%'),
        '6' | '^' => (KeyCode::KEY_6, ch == '^'),
        '7' | '&' => (KeyCode::KEY_7, ch == '&'),
        '8' | '*' => (KeyCode::KEY_8, ch == '*'),
        '9' | '(' => (KeyCode::KEY_9, ch == '('),
        '0' | ')' => (KeyCode::KEY_0, ch == ')'),

        '-' | '_' => (KeyCode::KEY_MINUS, ch == '_'),
        '=' | '+' => (KeyCode::KEY_EQUAL, ch == '+'),
        '[' | '{' => (KeyCode::KEY_LEFTBRACE, ch == '{'),
        ']' | '}' => (KeyCode::KEY_RIGHTBRACE, ch == '}'),
        '\\' | '|' => (KeyCode::KEY_BACKSLASH, ch == '|'),
        ';' | ':' => (KeyCode::KEY_SEMICOLON, ch == ':'),
        '\'' | '"' => (KeyCode::KEY_APOSTROPHE, ch == '"'),
        ',' | '<' => (KeyCode::KEY_COMMA, ch == '<'),
        '.' | '>' => (KeyCode::KEY_DOT, ch == '>'),
        '/' | '?' => (KeyCode::KEY_SLASH, ch == '?'),
        '`' | '~' => (KeyCode::KEY_GRAVE, ch == '~'),

        ' ' => (KeyCode::KEY_SPACE, false),
        '\n' => (KeyCode::KEY_ENTER, false),
        '\t' => (KeyCode::KEY_TAB, false),

        _ => return None,
    };
    Some(pair)
}

// ─── Typer ───────────────────────────────────────────────────────────────────

pub struct UinputTyper {
    device: VirtualDevice,
}

impl UinputTyper {
    pub fn new() -> Result<Self, String> {
        ensure_kernel_access()?;

        let mut keys = AttributeSet::<KeyCode>::new();
        for i in 1u16..255 {
            keys.insert(KeyCode(i));
        }

        let device = VirtualDevice::builder()
            .map_err(|e| format!("uinput builder: {e}"))?
            .name("lune-voice")
            .with_keys(&keys)
            .map_err(|e| format!("uinput with_keys: {e}"))?
            .build()
            .map_err(|e| format!("uinput build: {e}"))?;

        // Give the compositor 200ms to register the new virtual device.
        std::thread::sleep(Duration::from_millis(200));

        debug!("uinput ghost keyboard ready");
        Ok(Self { device })
    }

    /// Returns true if `s` contains chars with no US-QWERTY evdev mapping.
    pub fn has_unmapped_chars(s: &str) -> bool {
        s.chars().any(|c| char_to_key(c).is_none())
    }

    #[inline]
    fn emit_raw(&mut self, ev_type: EventType, code: u16, value: i32) {
        let _ = self.device.emit(&[
            InputEvent::new(ev_type.0, code, value),
            InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
        ]);
    }

    fn type_char(&mut self, ch: char) {
        let Some((key, needs_shift)) = char_to_key(ch) else {
            return;
        };
        if needs_shift {
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 1);
        }
        self.emit_raw(EventType::KEY, key.0, 1);
        self.emit_raw(EventType::KEY, key.0, 0);
        if needs_shift {
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 0);
        }
    }

    pub fn type_str(&mut self, text: &str) {
        for ch in text.chars() {
            self.type_char(ch);
            // 500µs between chars prevents kernel input buffer overflow on bursts.
            std::thread::sleep(Duration::from_micros(500));
        }
    }
}
