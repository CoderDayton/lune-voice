//! Clipboard-paste delivery via uinput virtual keyboard.
//!
//! Primary path: set clipboard via `wl-copy`/`xclip` → detect if focused window
//! is a terminal → emit Ctrl+V or Ctrl+Shift+V via uinput evdev events.
//!
//! Fallback: per-key typing for paste-hostile apps or when clipboard tools are
//! missing. ASCII-only, ~2ms/char.
//!
//! First run: if /dev/uinput is not writable, uses pkexec to install a
//! systemd-logind uaccess udev rule via a native GUI password dialog.
//! After that, access is permanent — no further prompts.

use std::{
    fs::OpenOptions,
    process::Command,
    time::{Duration, Instant},
};

use evdev::{AttributeSet, EventType, InputEvent, KeyCode, uinput::VirtualDevice};
use lune_core::config::TerminalMode;
use tracing::{debug, info, warn};

// ─── Window target (focus lock) ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WindowTarget {
    pub class: Option<String>,
    pub is_terminal: bool,
    pub focus_token: Option<FocusToken>,
}

#[derive(Debug, Clone)]
pub enum FocusToken {
    HyprlandAddress(String),
    SwayConId(i64),
    X11WindowId(String),
}

// ─── Bootstrap ───────────────────────────────────────────────────────────────

const UDEV_RULE: &str =
    r#"KERNEL=="uinput", SUBSYSTEM=="misc", OPTIONS+="static_node=uinput", TAG+="uaccess""#;

const UDEV_RULE_PATH: &str = "/etc/udev/rules.d/99-lune-voice-uinput.rules";

fn ensure_kernel_access() -> Result<(), String> {
    if OpenOptions::new().write(true).open("/dev/uinput").is_ok() {
        return Ok(());
    }

    info!("first-time setup: requesting uinput access via pkexec");

    use std::io::Write;
    let mut child = Command::new("pkexec")
        .args(["tee", UDEV_RULE_PATH])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("pkexec spawn failed: {e}"))?;
    if let Some(ref mut stdin) = child.stdin {
        let _ = stdin.write_all(UDEV_RULE.as_bytes());
    }
    let status = child
        .wait()
        .map_err(|e| format!("pkexec wait failed: {e}"))?;
    if !status.success() {
        return Err("pkexec authentication cancelled or failed".into());
    }

    let reload = Command::new("pkexec")
        .args(["udevadm", "control", "--reload-rules"])
        .status();
    let trigger = Command::new("pkexec").args(["udevadm", "trigger"]).status();
    if reload.map(|s| !s.success()).unwrap_or(true) || trigger.map(|s| !s.success()).unwrap_or(true)
    {
        warn!("udevadm reload/trigger may have failed — rule may require reboot");
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

// ─── Terminal detection ──────────────────────────────────────────────────────

/// Known terminal emulator WM_CLASS / app_id values (lowercase).
const KNOWN_TERMINALS: &[&str] = &[
    "alacritty",
    "com.raggesilver.blackbox",
    "contour",
    "cool-retro-term",
    "cosmic-term",
    "deepin-terminal",
    "eterm",
    "foot",
    "footclient",
    "ghostty",
    "gnome-terminal",
    "gnome-terminal-server",
    "guake",
    "havoc",
    "hyper",
    "io.elementary.terminal",
    "kgx",
    "kitty",
    "konsole",
    "lxterminal",
    "mate-terminal",
    "mlterm",
    "org.codeberg.dnkl.foot",
    "org.gnome.console",
    "org.gnome.terminal",
    "org.kde.konsole",
    "org.wezfurlong.wezterm",
    "qterminal",
    "rio",
    "roxterm",
    "sakura",
    "st",
    "tabby",
    "terminator",
    "terminology",
    "termite",
    "tilix",
    "urxvt",
    "warp",
    "wezterm",
    "wezterm-gui",
    "xfce4-terminal",
    "xterm",
    "yakuake",
    "zellij",
    "zutty",
];

/// Query the focused window's app_id / WM_CLASS from the compositor.
///
/// Detection chain: Hyprland → Sway → COSMIC → KDE (D-Bus) → GNOME (D-Bus) → xdotool → None
fn focused_window_class() -> Option<String> {
    // Hyprland: hyprctl activewindow -j → .class
    if std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok()
        && let Some(class) = run_json_query("hyprctl", &["activewindow", "-j"], &["class"])
    {
        return Some(class);
    }

    // Sway: swaymsg -t get_tree → find focused node's app_id
    if std::env::var("SWAYSOCK").is_ok()
        && let Some(class) = sway_focused_app_id()
    {
        return Some(class);
    }

    // COSMIC: uses wlr-foreign-toplevel-management but no CLI tool exposes it.
    // cosmic-comp is a standard Wayland compositor — wl-copy/wl-paste work natively.
    // For focused window detection, try cosmic-query (if installed) or fall through
    // to xdotool (XWayland) / None (defaults to Ctrl+V, safe for all GUI apps).
    if std::env::var("XDG_CURRENT_DESKTOP")
        .map(|d| d.to_lowercase().contains("cosmic"))
        .unwrap_or(false)
        && let Some(class) = cosmic_focused_app_id()
    {
        return Some(class);
        // Fall through to KDE/GNOME/xdotool chains (XWayland apps visible there)
    }

    // KDE Plasma: D-Bus org.kde.KWin /Scripting
    if std::env::var("KDE_FULL_SESSION").is_ok()
        && let Some(class) = kde_focused_class()
    {
        return Some(class);
    }

    // GNOME: D-Bus org.gnome.Shell eval
    if std::env::var("XDG_CURRENT_DESKTOP")
        .map(|d| d.to_lowercase().contains("gnome"))
        .unwrap_or(false)
        && let Some(class) = gnome_focused_class()
    {
        return Some(class);
    }

    // X11 fallback: xdotool
    if std::env::var("DISPLAY").is_ok()
        && let Some(class) = xdotool_focused_class()
    {
        return Some(class);
    }

    None
}

fn focused_is_terminal() -> bool {
    let Some(class) = focused_window_class() else {
        info!("focused window: unknown (defaulting to Ctrl+V)");
        return false;
    };
    let lower = class.to_lowercase();
    let is_term = KNOWN_TERMINALS.iter().any(|t| lower == *t);
    info!(class = %class, is_term, "focused window detected");
    is_term
}

fn is_terminal_class(class: &str) -> bool {
    let lower = class.to_lowercase();
    KNOWN_TERMINALS.iter().any(|t| lower == *t)
}

pub fn capture_focus_target() -> WindowTarget {
    if std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok()
        && let Some((class, addr)) = hyprland_active_info()
    {
        let is_terminal = is_terminal_class(&class);
        return WindowTarget {
            class: Some(class),
            is_terminal,
            focus_token: Some(FocusToken::HyprlandAddress(addr)),
        };
    }

    if std::env::var("SWAYSOCK").is_ok()
        && let Some((class, con_id)) = sway_focused_with_id()
    {
        let is_terminal = is_terminal_class(&class);
        return WindowTarget {
            class: Some(class),
            is_terminal,
            focus_token: Some(FocusToken::SwayConId(con_id)),
        };
    }

    if std::env::var("DISPLAY").is_ok()
        && let Some(wid) = xdotool_active_id()
    {
        let class = xdotool_focused_class();
        let is_terminal = class.as_ref().is_some_and(|c| is_terminal_class(c));
        return WindowTarget {
            class,
            is_terminal,
            focus_token: Some(FocusToken::X11WindowId(wid)),
        };
    }

    let class = focused_window_class();
    let is_terminal = class.as_ref().is_some_and(|c| is_terminal_class(c));
    WindowTarget {
        class,
        is_terminal,
        focus_token: None,
    }
}

fn hyprland_active_info() -> Option<(String, String)> {
    let output = Command::new("hyprctl")
        .args(["activewindow", "-j"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let val: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let class = val.get("class")?.as_str()?.to_owned();
    let addr = val.get("address")?.as_str()?;
    if !addr.starts_with("0x") || !addr[2..].chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some((class, addr.to_owned()))
}

fn sway_focused_with_id() -> Option<(String, i64)> {
    let output = Command::new("swaymsg")
        .args(["-t", "get_tree"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tree: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    find_focused_with_id(&tree)
}

fn find_focused_with_id(node: &serde_json::Value) -> Option<(String, i64)> {
    if node.get("focused").and_then(|v| v.as_bool()) == Some(true) {
        let con_id = node.get("id").and_then(|v| v.as_i64())?;
        let class = node
            .get("app_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty() && *s != "null")
            .or_else(|| {
                node.get("window_properties")
                    .and_then(|wp| wp.get("class"))
                    .and_then(|v| v.as_str())
            })?
            .to_owned();
        return Some((class, con_id));
    }
    for key in &["nodes", "floating_nodes"] {
        if let Some(children) = node.get(key).and_then(|v| v.as_array()) {
            for child in children {
                if let Some(found) = find_focused_with_id(child) {
                    return Some(found);
                }
            }
        }
    }
    None
}

fn xdotool_active_id() -> Option<String> {
    let output = Command::new("xdotool")
        .arg("getactivewindow")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let id = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_digit()) {
        None
    } else {
        Some(id)
    }
}

fn restore_focus(token: &FocusToken) -> bool {
    match token {
        FocusToken::HyprlandAddress(addr) => Command::new("hyprctl")
            .args(["dispatch", "focuswindow", &format!("address:{addr}")])
            .status()
            .ok()
            .is_some_and(|s| s.success()),
        FocusToken::SwayConId(id) => Command::new("swaymsg")
            .arg(format!("[con_id={id}] focus"))
            .status()
            .ok()
            .is_some_and(|s| s.success()),
        FocusToken::X11WindowId(wid) => Command::new("xdotool")
            .args(["windowactivate", "--sync", wid])
            .status()
            .ok()
            .is_some_and(|s| s.success()),
    }
}

/// Run a command, parse stdout as JSON, extract a string field.
fn run_json_query(cmd: &str, args: &[&str], path: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let val: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let mut current = &val;
    for key in path {
        current = current.get(key)?;
    }
    current.as_str().map(|s| s.to_owned())
}

/// Sway: walk the tree to find the focused node's app_id.
fn sway_focused_app_id() -> Option<String> {
    let output = Command::new("swaymsg")
        .args(["-t", "get_tree"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tree: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    find_focused_node(&tree)
}

fn find_focused_node(node: &serde_json::Value) -> Option<String> {
    if node.get("focused").and_then(|v| v.as_bool()) == Some(true) {
        // Prefer app_id (Wayland native), fall back to window_properties.class (XWayland)
        if let Some(app_id) = node.get("app_id").and_then(|v| v.as_str())
            && !app_id.is_empty()
            && app_id != "null"
        {
            return Some(app_id.to_owned());
        }
        if let Some(class) = node
            .get("window_properties")
            .and_then(|wp| wp.get("class"))
            .and_then(|v| v.as_str())
        {
            return Some(class.to_owned());
        }
    }
    // Recurse into child nodes
    for key in &["nodes", "floating_nodes"] {
        if let Some(children) = node.get(key).and_then(|v| v.as_array()) {
            for child in children {
                if let Some(found) = find_focused_node(child) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// COSMIC: attempt to read focused toplevel from cosmic-comp.
///
/// cosmic-comp implements wlr-foreign-toplevel-management but doesn't expose
/// a CLI query tool yet. We try `cosmic-toplevel-info` (community tool) and
/// fall back to checking /proc for the focused PID via loginctl.
fn cosmic_focused_app_id() -> Option<String> {
    // Try community tool first (may be installed on some setups)
    if let Some(class) = run_json_query(
        "cosmic-toplevel-info",
        &["--focused", "--json"],
        &["app_id"],
    ) {
        return Some(class);
    }
    // Fallback: use loginctl to find the foreground session's leader,
    // then check /proc/<pid>/comm. This is a rough heuristic.
    None
}

/// KDE: use gdbus to call KWin scripting for the active window caption/resourceClass.
fn kde_focused_class() -> Option<String> {
    // KDE exposes active window info via qdbus/gdbus
    let output = Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.kde.KWin",
            "--object-path",
            "/KWin",
            "--method",
            "org.kde.KWin.getActiveWindow",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    // Response format varies; try parsing resource class from the variant
    let stdout = String::from_utf8_lossy(&output.stdout);
    // KWin returns a dict with 'resourceClass' key — attempt basic extraction
    // This is best-effort; if KDE changes the API, we fall back gracefully.
    if let Some(start) = stdout.find("resourceClass") {
        let rest = &stdout[start..];
        // Extract the string value after the key
        if let Some(q1) = rest.find('\'') {
            let after = &rest[q1 + 1..];
            if let Some(q2) = after.find('\'') {
                return Some(after[..q2].to_owned());
            }
        }
    }
    None
}

/// GNOME: use gdbus to eval JS in the shell for the focused window's WM_CLASS.
fn gnome_focused_class() -> Option<String> {
    let output = Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.gnome.Shell",
            "--object-path",
            "/org/gnome/Shell",
            "--method",
            "org.gnome.Shell.Eval",
            "global.display.focus_window?.get_wm_class() ?? ''",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Response: (true, 'ClassName') — extract between quotes
    extract_quoted_value(&stdout)
}

/// X11: xdotool getactivewindow + xprop for WM_CLASS.
fn xdotool_focused_class() -> Option<String> {
    let wid = Command::new("xdotool")
        .arg("getactivewindow")
        .output()
        .ok()?;
    if !wid.status.success() {
        return None;
    }
    let window_id = String::from_utf8_lossy(&wid.stdout).trim().to_owned();
    if window_id.is_empty() {
        return None;
    }
    let xprop = Command::new("xprop")
        .args(["-id", &window_id, "WM_CLASS"])
        .output()
        .ok()?;
    if !xprop.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&xprop.stdout);
    // WM_CLASS(STRING) = "instance", "Class"  — we want the second value
    if let Some(eq) = stdout.find('=') {
        let after = &stdout[eq + 1..];
        let parts: Vec<&str> = after.split(',').collect();
        if parts.len() >= 2 {
            return extract_quoted_value(parts[1]);
        }
        // Single class
        return extract_quoted_value(after);
    }
    None
}

/// Extract a string between single or double quotes.
fn extract_quoted_value(s: &str) -> Option<String> {
    for quote in ['"', '\''] {
        if let Some(start) = s.find(quote) {
            let rest = &s[start + 1..];
            if let Some(end) = rest.find(quote) {
                let val = rest[..end].trim();
                if !val.is_empty() {
                    return Some(val.to_owned());
                }
            }
        }
    }
    None
}

// ─── Clipboard ───────────────────────────────────────────────────────────────

fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
}

/// Set clipboard contents. Uses `wl-copy` on Wayland, `xclip` on X11.
pub fn set_clipboard(text: &str) -> Result<(), String> {
    let (cmd, args): (&str, &[&str]) = if is_wayland() {
        ("wl-copy", &["--"])
    } else {
        ("xclip", &["-selection", "clipboard"])
    };

    let mut child = Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("{cmd} spawn failed: {e}"))?;

    use std::io::Write;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| format!("{cmd} write failed: {e}"))?;
    }

    let status = child
        .wait()
        .map_err(|e| format!("{cmd} wait failed: {e}"))?;
    if !status.success() {
        return Err(format!("{cmd} exited with {status}"));
    }
    Ok(())
}

// ─── Key mapping (US QWERTY) ─────────────────────────────────────────────────

/// Maps each lowercase ASCII letter to its Linux key code (US QWERTY layout).
/// The order follows the alphabet (a–z); codes are non-sequential because they
/// mirror physical QWERTY rows, so we cannot compute them arithmetically.
const LETTER_KEYS: [(u8, KeyCode); 26] = [
    (b'a', KeyCode::KEY_A),
    (b'b', KeyCode::KEY_B),
    (b'c', KeyCode::KEY_C),
    (b'd', KeyCode::KEY_D),
    (b'e', KeyCode::KEY_E),
    (b'f', KeyCode::KEY_F),
    (b'g', KeyCode::KEY_G),
    (b'h', KeyCode::KEY_H),
    (b'i', KeyCode::KEY_I),
    (b'j', KeyCode::KEY_J),
    (b'k', KeyCode::KEY_K),
    (b'l', KeyCode::KEY_L),
    (b'm', KeyCode::KEY_M),
    (b'n', KeyCode::KEY_N),
    (b'o', KeyCode::KEY_O),
    (b'p', KeyCode::KEY_P),
    (b'q', KeyCode::KEY_Q),
    (b'r', KeyCode::KEY_R),
    (b's', KeyCode::KEY_S),
    (b't', KeyCode::KEY_T),
    (b'u', KeyCode::KEY_U),
    (b'v', KeyCode::KEY_V),
    (b'w', KeyCode::KEY_W),
    (b'x', KeyCode::KEY_X),
    (b'y', KeyCode::KEY_Y),
    (b'z', KeyCode::KEY_Z),
];

fn char_to_key(ch: char) -> Option<(KeyCode, bool)> {
    // Handle all 26 letters with a single table lookup.
    if ch.is_ascii_alphabetic() {
        let lower = ch.to_ascii_lowercase() as u8;
        let key = LETTER_KEYS
            .iter()
            .find(|(b, _)| *b == lower)
            .map(|&(_, k)| k)?;
        return Some((key, ch.is_uppercase()));
    }

    let pair = match ch {
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

// ─── Adaptive paste timing ───────────────────────────────────────────────────

/// Minimum post-paste delay — floor even on the fastest systems.
const MIN_POST_PASTE_MS: u64 = 40;
/// Maximum post-paste delay — ceiling for very slow systems / Electron apps.
const MAX_POST_PASTE_MS: u64 = 300;
/// Initial post-paste delay before any adaptation.
const INITIAL_POST_PASTE_MS: u64 = 100;
/// EMA smoothing factor (0..1). Higher = faster adaptation, more jitter.
const EMA_ALPHA: f64 = 0.3;
/// Multiplier applied to clipboard round-trip EMA to derive post-paste delay.
/// Paste read is typically faster than a full clipboard round-trip, so 2× is
/// conservative while still adapting to system speed.
const RTT_MULTIPLIER: f64 = 2.0;

/// Settle delay: let modifier keys from hotkey release settle before emitting paste.
const MODIFIER_SETTLE_MS: u64 = 30;

/// Tracks clipboard operation latency and adapts the post-paste delay.
struct AdaptivePasteDelay {
    /// EMA of clipboard set+get round-trip in ms.
    rtt_ema_ms: f64,
    /// Current computed delay.
    current_ms: u64,
}

impl AdaptivePasteDelay {
    fn new() -> Self {
        Self {
            rtt_ema_ms: INITIAL_POST_PASTE_MS as f64 / RTT_MULTIPLIER,
            current_ms: INITIAL_POST_PASTE_MS,
        }
    }

    /// Record a clipboard round-trip measurement and update the delay.
    fn record_rtt(&mut self, rtt: Duration) {
        let sample_ms = rtt.as_secs_f64() * 1000.0;
        self.rtt_ema_ms = EMA_ALPHA * sample_ms + (1.0 - EMA_ALPHA) * self.rtt_ema_ms;
        let derived = (self.rtt_ema_ms * RTT_MULTIPLIER) as u64;
        self.current_ms = derived.clamp(MIN_POST_PASTE_MS, MAX_POST_PASTE_MS);
    }

    fn delay(&self) -> Duration {
        Duration::from_millis(self.current_ms)
    }
}

pub struct UinputTyper {
    device: VirtualDevice,
    terminal_mode: TerminalMode,
    paste_delay: AdaptivePasteDelay,
}

impl UinputTyper {
    #[allow(dead_code)]
    pub fn new() -> Result<Self, String> {
        Self::with_terminal_mode(TerminalMode::Auto)
    }

    pub fn with_terminal_mode(terminal_mode: TerminalMode) -> Result<Self, String> {
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

        debug!(?terminal_mode, "uinput ghost keyboard ready");
        Ok(Self {
            device,
            terminal_mode,
            paste_delay: AdaptivePasteDelay::new(),
        })
    }

    /// Pre-warm compositor detection so the first `type_str()` call has no
    /// cold-start latency from spawning detection subprocesses.
    pub fn warm_up(&self) {
        // Fire-and-forget: prime the detection chain (subprocess caches, D-Bus
        // connections) so subsequent calls are near-instant.
        let _ = focused_window_class();
        debug!("warm_up: compositor detection primed");
    }

    #[inline]
    fn emit_raw(&mut self, ev_type: EventType, code: u16, value: i32) {
        if let Err(e) = self.device.emit(&[
            InputEvent::new(ev_type.0, code, value),
            InputEvent::new(EventType::SYNCHRONIZATION.0, 0, 0),
        ]) {
            tracing::warn!(error = %e, code, value, "uinput emit failed");
        }
    }

    /// Emit Ctrl+V (normal) or Ctrl+Shift+V (terminal) paste chord.
    fn emit_paste(&mut self, terminal: bool) {
        // Settle after hotkey modifier release
        std::thread::sleep(Duration::from_millis(MODIFIER_SETTLE_MS));

        // Press Ctrl
        self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTCTRL.0, 1);

        if terminal {
            // Press Shift for terminal paste
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 1);
        }

        // Press+release V
        self.emit_raw(EventType::KEY, KeyCode::KEY_V.0, 1);
        self.emit_raw(EventType::KEY, KeyCode::KEY_V.0, 0);

        if terminal {
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 0);
        }

        // Release Ctrl
        self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTCTRL.0, 0);
    }

    /// Primary delivery: clipboard paste with terminal detection.
    ///
    /// Sets clipboard → emits paste chord (Ctrl+V or Ctrl+Shift+V) → leaves
    /// text on clipboard so the user can re-paste manually if needed.
    ///
    /// The post-paste delay adapts to system speed by tracking an EMA of
    /// clipboard operation latency. Fast systems converge toward 40ms;
    /// slow systems (Electron, heavy toolkits) grow up to 300ms.
    pub fn type_str(&mut self, text: &str) {
        let is_terminal = match self.terminal_mode {
            TerminalMode::Auto => focused_is_terminal(),
            TerminalMode::Terminal => true,
            TerminalMode::Normal => false,
            _ => focused_is_terminal(),
        };
        self.do_clipboard_paste(text, is_terminal, "emitting paste");
    }

    pub fn type_str_to(&mut self, text: &str, target: &WindowTarget) {
        if let Some(ref token) = target.focus_token {
            if !restore_focus(token) {
                warn!("could not restore focus to original window");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        self.do_clipboard_paste(text, target.is_terminal, "emitting paste to target");
    }

    /// Set clipboard, emit the appropriate paste chord, and wait for the
    /// target app to consume it.  Falls back to per-key typing on clipboard
    /// failure.  `label` is passed through to the info log.
    fn do_clipboard_paste(&mut self, text: &str, is_terminal: bool, label: &str) {
        let rtt_start = Instant::now();
        if let Err(e) = set_clipboard(text) {
            warn!(error = %e, "clipboard set failed, falling back to per-key typing");
            self.type_str_keys(text);
            return;
        }
        self.paste_delay.record_rtt(rtt_start.elapsed());

        let chord = if is_terminal {
            "Ctrl+Shift+V"
        } else {
            "Ctrl+V"
        };
        info!(
            chord,
            post_paste_ms = self.paste_delay.current_ms,
            "{label}"
        );
        self.emit_paste(is_terminal);
        std::thread::sleep(self.paste_delay.delay());
    }

    /// Fallback per-key typing. ASCII-only, ~2ms/char.
    pub fn type_str_keys(&mut self, text: &str) {
        std::thread::sleep(Duration::from_millis(MODIFIER_SETTLE_MS));
        for ch in text.chars() {
            self.type_char(ch);
            std::thread::sleep(Duration::from_millis(2));
        }
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
}
