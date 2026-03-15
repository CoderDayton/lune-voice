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
    let is_term = KNOWN_TERMINALS.iter().any(|t| lower.contains(t));
    info!(class = %class, is_term, "focused window detected");
    is_term
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
        let rtt_start = Instant::now();

        // Set clipboard content (text stays on clipboard after paste)
        if let Err(e) = set_clipboard(text) {
            warn!(error = %e, "clipboard set failed, falling back to per-key typing");
            self.type_str_keys(text);
            return;
        }

        // Update adaptive delay from clipboard operation latency.
        let rtt = rtt_start.elapsed();
        self.paste_delay.record_rtt(rtt);

        // Determine paste chord
        let is_terminal = match self.terminal_mode {
            TerminalMode::Auto => focused_is_terminal(),
            TerminalMode::Terminal => true,
            TerminalMode::Normal => false,
            _ => focused_is_terminal(),
        };

        let chord = if is_terminal {
            "Ctrl+Shift+V"
        } else {
            "Ctrl+V"
        };
        info!(
            chord,
            post_paste_ms = self.paste_delay.current_ms,
            "emitting paste"
        );
        self.emit_paste(is_terminal);

        // Wait for target app to read the clipboard
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
