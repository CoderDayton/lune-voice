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
    borrow::Cow,
    fs::OpenOptions,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

use evdev::{AttributeSet, EventType, InputEvent, KeyCode, uinput::VirtualDevice};
use lune_core::config::TerminalMode;
use tracing::{debug, error, info, warn};

use crate::delivery::TypeOutcome;

// Hard ceiling on every compositor / IPC subprocess we spawn during delivery.
// These tools normally return in single-digit ms; anything beyond means the
// daemon is wedged and we'd rather fall through than stall paste.
//
// `capture_focus_target` and `active_class_for_terminal_detection` may try
// several backends in sequence (Hyprland → Sway → KDE → GNOME → X11), so the
// worst-case latency on a system where every gating env var is set AND every
// daemon is wedged is bounded by N × SUBPROCESS_TIMEOUT. In practice only one
// backend matches per session, so the typical worst case is one timeout.
const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(1);

// Clipboard tooling gets a slightly looser budget because wl-copy/xclip
// have to round-trip through the compositor's clipboard manager.
const CLIPBOARD_TIMEOUT: Duration = Duration::from_secs(2);

/// Spawn `cmd` with `args`, collect stdout/stderr, kill if it doesn't finish
/// within `timeout`. Returns `None` on spawn failure or timeout.
///
/// `try_wait` polling at 10ms; the upper bound on overshoot is ~10ms. We
/// must close stdin (Stdio::null) so children that read stdin don't block.
fn output_with_timeout(cmd: &str, args: &[&str], timeout: Duration) -> Option<Output> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    debug!(
                        cmd,
                        timeout_ms = timeout.as_millis() as u64,
                        "subprocess timeout"
                    );
                    return None;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Status-only variant of `output_with_timeout` for fire-and-forget commands.
fn status_with_timeout(
    cmd: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<std::process::ExitStatus> {
    output_with_timeout(cmd, args, timeout).map(|o| o.status)
}

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

/// Where the runtime installs the rule when nothing else has.
///
/// `/etc/udev/rules.d` is the local-administrator directory, which is what a
/// self-installing rule belongs in — and, unlike the vendor directory, it is
/// ours to remove again ([`remove_uinput_rule`]).
///
/// The .deb and .rpm ship the same rule to `/usr/lib/udev/rules.d` instead, so
/// a packaged install never reaches this path and the package manager owns the
/// file's whole life. Only AppImage and `cargo run` fall back to installing it
/// here at runtime.
const UDEV_RULE_PATH: &str = "/etc/udev/rules.d/99-lune-voice-uinput.rules";

/// Whether the runtime-installed rule is present.
///
/// Only the copy this app wrote counts. A rule shipped by the package manager
/// is not ours to report on or remove — `dpkg`/`rpm` owns it, and deleting it
/// from under them would leave the package believing it is still installed.
pub fn uinput_rule_is_self_installed() -> bool {
    std::path::Path::new(UDEV_RULE_PATH).is_file()
}

/// Remove the rule this app installed, restoring the system to how it was.
///
/// Worth offering because the rule is not a private setting: `TAG+="uaccess"`
/// grants **every** process running as the logged-in user permanent read/write
/// on `/dev/uinput`, which is a keystroke-injection primitive into any
/// application. Leaving that behind after someone stops using the app — or
/// uninstalls it — is not a decision to make silently on their behalf.
///
/// Blocking: raises a polkit prompt. Returns `Ok(false)` when there was nothing
/// to remove.
pub fn remove_uinput_rule() -> Result<bool, String> {
    if !uinput_rule_is_self_installed() {
        return Ok(false);
    }

    let status = Command::new("pkexec")
        .args(["rm", "-f", UDEV_RULE_PATH])
        .status()
        .map_err(|e| format!("pkexec spawn failed: {e}"))?;
    if !status.success() {
        return Err("pkexec authentication cancelled or failed".into());
    }

    reload_udev_rules();

    if uinput_rule_is_self_installed() {
        return Err("the udev rule is still present after removal".into());
    }
    info!("removed the uinput udev rule");
    Ok(true)
}

/// Ask udevd to re-read its rules and re-apply them to existing devices.
///
/// Best-effort on purpose: both halves failing means the change lands on the
/// next boot rather than now, which is worth a warning and nothing more.
fn reload_udev_rules() {
    let reload = Command::new("pkexec")
        .args(["udevadm", "control", "--reload-rules"])
        .status();
    let trigger = Command::new("pkexec").args(["udevadm", "trigger"]).status();
    if reload.map(|s| !s.success()).unwrap_or(true) || trigger.map(|s| !s.success()).unwrap_or(true)
    {
        warn!("udevadm reload/trigger may have failed — the change may require a reboot");
    }
}

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
    let mut wrote = false;
    if let Some(ref mut stdin) = child.stdin {
        wrote = stdin.write_all(UDEV_RULE.as_bytes()).is_ok();
    }
    let status = child
        .wait()
        .map_err(|e| format!("pkexec wait failed: {e}"))?;
    if !status.success() {
        return Err("pkexec authentication cancelled or failed".into());
    }
    // `tee` exits 0 having written an empty file if the pipe broke, which would
    // otherwise leave a silently useless root-owned rule in /etc.
    if !wrote {
        let _ = Command::new("pkexec")
            .args(["rm", "-f", UDEV_RULE_PATH])
            .status();
        return Err("could not write the udev rule".into());
    }

    reload_udev_rules();

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
/// Detection chain: Hyprland → Sway → wlr-foreign-toplevel (COSMIC, KDE-Wayland,
/// Wayfire, river, …) → KDE (D-Bus) → GNOME (D-Bus) → xdotool (X11) → None
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

    // Generic Wayland: try wlr-foreign-toplevel-management first (covers
    // KDE Plasma 5.27+, Wayfire, river, niri, Mir-based compositors —
    // also Hyprland/Sway as a fallback though those have faster IPC paths
    // above). Then try COSMIC's zcosmic_toplevel_info_v1 (cosmic-comp does
    // NOT advertise wlr-foreign-toplevel — verified by live globals probe).
    if let Some(class) = crate::wayland_toplevel::wlr_active_app_id() {
        return Some(class);
    }
    if let Some(class) = crate::cosmic_toplevel::cosmic_active_app_id() {
        return Some(class);
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

pub(crate) fn focused_is_terminal() -> bool {
    let Some(class) = focused_window_class() else {
        info!("focused window: unknown (defaulting to Ctrl+V)");
        return false;
    };
    let is_term = is_terminal_class(&class);
    info!(class = %class, is_term, "focused window detected");
    is_term
}

/// Match a window class / app_id against `KNOWN_TERMINALS`.
///
/// Compositors report classes in three shapes that all need to map to the
/// same logical terminal name:
///
/// - bare token: `kitty`, `konsole`
/// - reverse-DNS app_id: `dev.warp.Warp`, `org.gnome.Console`
/// - hyphenated bare: `cosmic-term`, `gnome-terminal`
///
/// We normalize by taking the last `.`-separated segment, lowercasing, and
/// stripping non-alphanumeric chars on both sides of the comparison. So
/// `com.system76.CosmicTerm` and `cosmic-term` both reduce to `cosmicterm`,
/// `dev.warp.Warp` and `warp` both reduce to `warp`.
fn is_terminal_class(class: &str) -> bool {
    let key = terminal_key(class);
    if key.is_empty() {
        return false;
    }
    KNOWN_TERMINALS.iter().any(|t| terminal_key(t) == key)
}

fn terminal_key(s: &str) -> String {
    s.rsplit('.')
        .next()
        .unwrap_or(s)
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
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

    // Generic Wayland: try both wlr-foreign-toplevel-management and
    // cosmic-toplevel-info. No focus_token — neither protocol exposes a
    // portable re-focus request, but delivery only needs the class to pick
    // the paste chord. The window already has focus because the user just
    // released the hotkey there.
    if std::env::var_os("WAYLAND_DISPLAY").is_some()
        && let Some(class) = crate::wayland_toplevel::wlr_active_app_id()
            .or_else(crate::cosmic_toplevel::cosmic_active_app_id)
    {
        let is_terminal = is_terminal_class(&class);
        return WindowTarget {
            class: Some(class),
            is_terminal,
            focus_token: None,
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
    let output = output_with_timeout("hyprctl", &["activewindow", "-j"], SUBPROCESS_TIMEOUT)?;
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
    let output = output_with_timeout("swaymsg", &["-t", "get_tree"], SUBPROCESS_TIMEOUT)?;
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
    let output = output_with_timeout("xdotool", &["getactivewindow"], SUBPROCESS_TIMEOUT)?;
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
        FocusToken::HyprlandAddress(addr) => status_with_timeout(
            "hyprctl",
            &["dispatch", "focuswindow", &format!("address:{addr}")],
            SUBPROCESS_TIMEOUT,
        )
        .is_some_and(|s| s.success()),
        FocusToken::SwayConId(id) => status_with_timeout(
            "swaymsg",
            &[&format!("[con_id={id}] focus")],
            SUBPROCESS_TIMEOUT,
        )
        .is_some_and(|s| s.success()),
        FocusToken::X11WindowId(wid) => status_with_timeout(
            "xdotool",
            &["windowactivate", "--sync", wid],
            SUBPROCESS_TIMEOUT,
        )
        .is_some_and(|s| s.success()),
    }
}

/// Run a command, parse stdout as JSON, extract a string field.
fn run_json_query(cmd: &str, args: &[&str], path: &[&str]) -> Option<String> {
    let output = output_with_timeout(cmd, args, SUBPROCESS_TIMEOUT)?;
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
    let output = output_with_timeout("swaymsg", &["-t", "get_tree"], SUBPROCESS_TIMEOUT)?;
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

/// KDE: use gdbus to call KWin scripting for the active window caption/resourceClass.
fn kde_focused_class() -> Option<String> {
    // KDE exposes active window info via qdbus/gdbus
    let output = output_with_timeout(
        "gdbus",
        &[
            "call",
            "--session",
            "--dest",
            "org.kde.KWin",
            "--object-path",
            "/KWin",
            "--method",
            "org.kde.KWin.getActiveWindow",
        ],
        SUBPROCESS_TIMEOUT,
    )?;
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
    let output = output_with_timeout(
        "gdbus",
        &[
            "call",
            "--session",
            "--dest",
            "org.gnome.Shell",
            "--object-path",
            "/org/gnome/Shell",
            "--method",
            "org.gnome.Shell.Eval",
            "global.display.focus_window?.get_wm_class() ?? ''",
        ],
        SUBPROCESS_TIMEOUT,
    )?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Response: (true, 'ClassName') — extract between quotes
    extract_quoted_value(&stdout)
}

/// X11: xdotool getactivewindow + xprop for WM_CLASS.
fn xdotool_focused_class() -> Option<String> {
    let wid = output_with_timeout("xdotool", &["getactivewindow"], SUBPROCESS_TIMEOUT)?;
    if !wid.status.success() {
        return None;
    }
    let window_id = String::from_utf8_lossy(&wid.stdout).trim().to_owned();
    if window_id.is_empty() {
        return None;
    }
    let xprop = output_with_timeout(
        "xprop",
        &["-id", &window_id, "WM_CLASS"],
        SUBPROCESS_TIMEOUT,
    )?;
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
///
/// Capped at `CLIPBOARD_TIMEOUT` — a wedged clipboard daemon used to block
/// delivery indefinitely. On timeout the child is killed and the caller
/// falls back to per-key typing.
pub fn set_clipboard(text: &str) -> Result<(), String> {
    let (cmd, args): (&str, &[&str]) = if is_wayland() {
        ("wl-copy", &["--"])
    } else {
        ("xclip", &["-selection", "clipboard"])
    };
    pipe_text_to_command(cmd, args, text, CLIPBOARD_TIMEOUT)
}

/// Set the clipboard to `text` for a *transient* paste, tagged with the
/// `x-kde-passwordManagerHint` exclusion MIME so clipboard-history managers
/// (Klipper, cliphist, clipman, wl-clip-persist, …) don't durably record the
/// dictated text during the brief set → paste → restore window.
///
/// Uses `arboard` (native `wlr-data-control` on Wayland, X11 selections
/// otherwise) — the only Linux clipboard path here that can co-advertise the
/// exclusion hint *alongside* the text. `wl-copy`/`xclip` serve a single target
/// and can't (co-advertising via `wl-copy --type … --type …` drops the text
/// target entirely), which is why the transient path can't reuse `set_clipboard`.
///
/// The returned owner MUST be kept alive until the paste has been consumed: on
/// X11 its background thread serves the selection for exactly as long as the
/// value lives; on Wayland a detached daemon is forked to serve it and the
/// value is incidental. Callers fall back to `set_clipboard` on `Err` so
/// delivery still works when arboard can't initialise (e.g. no display).
fn set_clipboard_excluded(text: &str) -> Result<arboard::Clipboard, String> {
    use arboard::SetExtLinux;
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("arboard init failed: {e}"))?;
    clipboard
        .set()
        .exclude_from_history()
        .text(text.to_owned())
        .map_err(|e| format!("arboard set failed: {e}"))?;
    Ok(clipboard)
}

/// A byte-for-byte snapshot of the clipboard: the source's preferred MIME type
/// and its exact bytes.
///
/// Wayland/X11 clipboards aren't buffers you can hold a handle to — the data is
/// served live by the owning app, and the instant we set our own text we seize
/// ownership with no protocol way to hand it back. So preserving the user's
/// clipboard means copying its bytes out and re-serving them ourselves.
/// Capturing the raw bytes of the source's preferred format (rather than
/// forcing text/plain) round-trips images and other binary content intact. The
/// one case CLI tooling can't fully reproduce is a single clipboard offering
/// several independent formats at once — we re-serve only the primary one.
struct ClipboardSnapshot {
    /// Empty on X11 — xclip restores the default selection without a target.
    mime: String,
    bytes: Vec<u8>,
}

/// A clipboard MIME type is remote-controlled data (the owning app chooses it),
/// and it is passed as a `--type` argument to `wl-paste`/`wl-copy`. It is never
/// shell-expanded, so it cannot inject a command, but an unexpected value could
/// still be a flag-like string or a malformed type. Accept only well-formed
/// RFC 6838 MIME types (optionally with `;key=value` parameters): no leading
/// `-`, no whitespace, no control or shell-interesting characters.
fn is_safe_mime(mime: &str) -> bool {
    if mime.is_empty() || mime.starts_with('-') {
        return false;
    }
    // The base type must be `type/subtype` with exactly one `/` and non-empty
    // sides (RFC 6838). `charset=utf-8` etc. live after the first `;`.
    let base = mime.split(';').next().unwrap_or("");
    let mut parts = base.split('/');
    let (Some(t), Some(s)) = (parts.next(), parts.next()) else {
        return false;
    };
    if t.is_empty() || s.is_empty() || parts.next().is_some() {
        return false;
    }
    // No empty parameter separators/values: `;;`, trailing `;`, `=`, or `;=`.
    if mime.contains(";;") || mime.ends_with(';') || mime.ends_with('=') || mime.contains(";=") {
        return false;
    }
    mime.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '+' | '-' | ';' | '='))
}

/// Capture the current clipboard so it can be restored after a paste.
///
/// `Ok(None)` means the clipboard is genuinely empty — safe to clear on
/// restore. `Err` means we couldn't read it reliably: a timeout (e.g. a
/// multi-MB image exceeding `CLIPBOARD_TIMEOUT`) or a missing tool. The caller
/// must NOT clear in that case, or it would destroy content it failed to
/// snapshot. Uses `wl-paste` on Wayland, `xclip -o` on X11.
fn get_clipboard() -> Result<Option<ClipboardSnapshot>, String> {
    if is_wayland() {
        // Pick the source's preferred (first-listed) offered type. A killed or
        // timed-out child yields None from output_with_timeout → propagate as
        // Err (don't clear); a clean non-success exit means nothing is copied.
        let types = output_with_timeout("wl-paste", &["--list-types"], CLIPBOARD_TIMEOUT)
            .ok_or("wl-paste --list-types failed or timed out")?;
        if !types.status.success() {
            return Ok(None);
        }
        let mime = match String::from_utf8_lossy(&types.stdout)
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
        {
            Some(m) if is_safe_mime(m) => m.to_owned(),
            // A type we refuse to forward means there IS content here that we
            // cannot snapshot. That is the `Err` case, not `Ok(None)`: the
            // caller clears the clipboard on `Ok(None)`, which would destroy
            // the very content we just failed to read.
            Some(_) => return Err("clipboard advertises an unsupported MIME type".into()),
            // Nothing advertised at all: the clipboard is genuinely empty.
            None => return Ok(None),
        };
        // Strip wl-paste's appended trailing newline for text only — doing so to
        // binary (an image whose last byte happens to be 0x0a) would corrupt it.
        let mut read_args: Vec<&str> = Vec::new();
        if mime.starts_with("text/") {
            read_args.push("--no-newline");
        }
        read_args.push("--type");
        read_args.push(&mime);
        let out = output_with_timeout("wl-paste", &read_args, CLIPBOARD_TIMEOUT)
            .ok_or("wl-paste --type read failed or timed out")?;
        if !out.status.success() || out.stdout.is_empty() {
            return Ok(None);
        }
        Ok(Some(ClipboardSnapshot {
            mime,
            bytes: out.stdout,
        }))
    } else {
        let out = output_with_timeout(
            "xclip",
            &["-selection", "clipboard", "-o"],
            CLIPBOARD_TIMEOUT,
        )
        .ok_or("xclip -o failed or timed out")?;
        if !out.status.success() || out.stdout.is_empty() {
            return Ok(None);
        }
        Ok(Some(ClipboardSnapshot {
            mime: String::new(),
            bytes: out.stdout,
        }))
    }
}

/// Current clipboard contents as text, if the clipboard holds text — used by
/// snippet `{clipboard}` placeholders. Best-effort: `None` when the clipboard
/// is empty, unreadable, or holds non-text content (e.g. an image).
pub fn clipboard_text() -> Option<String> {
    let snap = get_clipboard().ok().flatten()?;
    // Empty mime = X11 default selection (text); Wayland reports a text/* type.
    if !(snap.mime.is_empty() || snap.mime.starts_with("text/")) {
        return None;
    }
    let text = String::from_utf8_lossy(&snap.bytes).into_owned();
    (!text.is_empty()).then_some(text)
}

/// Restore a captured clipboard snapshot verbatim.
fn set_clipboard_snapshot(snap: &ClipboardSnapshot) -> Result<(), String> {
    if is_wayland() {
        pipe_bytes_to_command(
            "wl-copy",
            &["--type", &snap.mime],
            &snap.bytes,
            CLIPBOARD_TIMEOUT,
        )
    } else {
        pipe_bytes_to_command(
            "xclip",
            &["-selection", "clipboard"],
            &snap.bytes,
            CLIPBOARD_TIMEOUT,
        )
    }
}

/// Empty the clipboard — the restore-to-nothing path when there was no prior
/// text to put back. `wl-copy --clear` on Wayland; an empty selection on X11.
fn clear_clipboard() -> Result<(), String> {
    if is_wayland() {
        match status_with_timeout("wl-copy", &["--clear"], CLIPBOARD_TIMEOUT) {
            Some(s) if s.success() => Ok(()),
            Some(s) => Err(format!("wl-copy --clear exited with {s}")),
            None => Err("wl-copy --clear failed or timed out".into()),
        }
    } else {
        pipe_text_to_command("xclip", &["-selection", "clipboard"], "", CLIPBOARD_TIMEOUT)
    }
}

/// Spawn `cmd` with a piped stdin, write `text`, close the pipe so the
/// child sees EOF, then poll for exit up to `timeout`. On timeout the
/// child is killed and an error is returned.
fn pipe_text_to_command(
    cmd: &str,
    args: &[&str],
    text: &str,
    timeout: Duration,
) -> Result<(), String> {
    pipe_bytes_to_command(cmd, args, text.as_bytes(), timeout)
}

/// Byte-exact variant of `pipe_text_to_command` — used to restore a captured
/// clipboard snapshot verbatim, including binary payloads like images.
fn pipe_bytes_to_command(
    cmd: &str,
    args: &[&str],
    bytes: &[u8],
    timeout: Duration,
) -> Result<(), String> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{cmd} spawn failed: {e}"))?;

    // Take ownership of stdin so dropping it closes the pipe and signals
    // EOF — without this, wl-copy/xclip read forever and we always time out.
    use std::io::Write;
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(e) = stdin.write_all(bytes) {
            // Reap the child explicitly: `Child` does not kill on drop,
            // so without this a misbehaving wl-copy/xclip keeps running.
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("{cmd} write failed: {e}"));
        }
    }

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Err(format!("{cmd} exited with {status}"));
                }
                return Ok(());
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("{cmd} timed out after {timeout:?}"));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(format!("{cmd} wait failed: {e}")),
        }
    }
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
        // No `'\t' => KEY_TAB`. Pressing Tab is an action, not a character:
        // in a GUI it moves focus to the next widget, so the rest of the
        // transcript lands in a different control, and at a shell prompt it
        // requests completion. A tab therefore has no keystroke and routes
        // the whole text to a clipboard paste, where it stays literal.
        _ => return None,
    };
    Some(pair)
}

/// True when every character in `text` can be emitted as a US-QWERTY
/// keystroke. Anything else (accents, CJK, emoji, smart quotes) must be
/// delivered via the clipboard rather than silently dropped.
fn is_keystroke_typeable(text: &str) -> bool {
    text.chars().all(|c| char_to_key(c).is_some())
}

/// True when per-key typing is unsafe or impossible for `text`, so delivery
/// must go through a clipboard paste instead.
///
/// Two reasons: a character with no US-QWERTY keystroke — which now includes
/// tab, see `char_to_key` — so it would be dropped, or a newline, which
/// `char_to_key` turns into KEY_ENTER and which submits the form or runs the
/// shell line being typed into instead of inserting a line break.
fn needs_paste_delivery(text: &str) -> bool {
    !is_keystroke_typeable(text) || text.contains('\n')
}

/// Whether `do_clipboard_paste`'s clipboard-failure fallback may type `text`
/// per-key. Deliberately the inverse of `needs_paste_delivery`: text routed to
/// the clipboard *because* it cannot be typed safely must not be typed anyway
/// when the clipboard turns out to be unavailable (missing `wl-copy`/`xclip`,
/// wedged clipboard daemon, no display) — a dropped delivery beats a literal
/// Enter submitting the form or running the shell line.
fn keystroke_fallback_allowed(text: &str) -> bool {
    !needs_paste_delivery(text)
}

/// Substitute spaces for tabs so the per-key fallback can still run.
///
/// A tab has no keystroke by design (see `char_to_key`), which routes text
/// containing one to a clipboard paste. When the clipboard is what failed,
/// that same rule would refuse to type the text at all and drop the whole
/// utterance over one whitespace character.
fn tabs_to_spaces(text: &str) -> Cow<'_, str> {
    if text.contains('\t') {
        Cow::Owned(text.replace('\t', " "))
    } else {
        Cow::Borrowed(text)
    }
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

/// Gap between the key-down/up/modifier events *within* one synthetic
/// character. Without it a fast shift release races the next key (stray
/// mid-word capitals) and a too-brief key press is coalesced away (dropped
/// spaces) on busy Wayland/X11 input pipelines.
const KEY_EVENT_GAP_MS: u64 = 2;
/// Gap *between* characters in per-key typing. The old 2 ms was fast enough
/// that the compositor dropped keys (notably SPACE); this gives it headroom.
const PER_CHAR_GAP_MS: u64 = 6;

/// Above this many characters, per-key typing is routed to a clipboard paste
/// instead.
///
/// At `PER_CHAR_GAP_MS` + `KEY_EVENT_GAP_MS` a character costs ~8 ms, so 80
/// characters is roughly 0.6 s of visible typing — about where a paste (flat,
/// ~100–360 ms including the clipboard restore) becomes the faster of the two.
/// Below it, per-key wins and leaves the clipboard untouched.
const PASTE_THRESHOLD_CHARS: usize = 80;

/// Whether `text` is long enough that pasting beats typing it out.
fn is_long_enough_to_paste(text: &str) -> bool {
    text.chars().count() > PASTE_THRESHOLD_CHARS
}

/// Gap between the key events of the one-shot paste chord (`emit_paste`). Uses
/// the same drop-safe headroom as `PER_CHAR_GAP_MS` rather than the tighter
/// `KEY_EVENT_GAP_MS`: the chord fires once per dictation, so the few extra
/// milliseconds are free, and a dropped modifier *release* here strands
/// Shift/Ctrl logically down across the desktop (a stuck modifier).
const CHORD_GAP_MS: u64 = 6;

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
    /// Clipboard captured at dictation start (see `prime_clipboard`) so the
    /// snapshot's up-to-2s subprocess round-trip overlaps ASR instead of
    /// freezing the paste. Consumed once by the next `do_clipboard_paste`.
    primed_clipboard: Option<Result<Option<ClipboardSnapshot>, String>>,
}

impl UinputTyper {
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
            primed_clipboard: None,
        })
    }

    /// Snapshot the clipboard ahead of the paste so the restore-after-paste
    /// read doesn't sit on the delivery hot path. Called at dictation start
    /// (via the delivery thread's `Prime` job), it overlaps the seconds of
    /// recording + ASR that follow — by paste time the bytes are already
    /// captured. Overwrites any prior prime; only the latest dictation matters.
    ///
    /// Because the capture happens at start, a clipboard change the user makes
    /// *during* the dictation is not preserved on restore — an accepted trade
    /// for keeping the paste responsive (the snapshot's up-to-2s read no longer
    /// blocks it).
    pub fn prime_clipboard(&mut self) {
        let start = Instant::now();
        let snap = get_clipboard();
        debug!(
            prime_ms = start.elapsed().as_millis() as u64,
            captured = matches!(snap, Ok(Some(_))),
            "clipboard primed at dictation start"
        );
        self.primed_clipboard = Some(snap);
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
        let gap = Duration::from_millis(CHORD_GAP_MS);

        // Press the modifiers, each given time to register before V — the same
        // chord race `type_char` guards against (without the gaps a composited
        // system can drop V or apply it without the modifier).
        self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTCTRL.0, 1);
        std::thread::sleep(gap);
        if terminal {
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 1);
            std::thread::sleep(gap); // shift must register before the key
        }

        // Press+release V, holding it down long enough to register.
        self.emit_raw(EventType::KEY, KeyCode::KEY_V.0, 1);
        std::thread::sleep(gap);
        self.emit_raw(EventType::KEY, KeyCode::KEY_V.0, 0);
        std::thread::sleep(gap); // V must release before the modifiers, or they leak

        // Release the modifiers in reverse order, each spaced by `gap`. The gap
        // BETWEEN the two releases matters as much as the rest: without it a
        // busy compositor can coalesce or drop the Shift release and strand
        // Shift logically down — a stuck modifier that outlives the dictation.
        if terminal {
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 0);
            std::thread::sleep(gap);
        }
        self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTCTRL.0, 0);
    }

    /// Resolve whether the focused window should receive a terminal-style
    /// paste chord, honoring the configured `terminal_mode`.
    fn resolve_terminal(&self) -> bool {
        match self.terminal_mode {
            TerminalMode::Auto => focused_is_terminal(),
            TerminalMode::Terminal => true,
            TerminalMode::Normal => false,
            _ => focused_is_terminal(),
        }
    }

    pub fn type_str(&mut self, text: &str, keep_clipboard: bool) -> TypeOutcome {
        let is_terminal = self.resolve_terminal();
        self.do_clipboard_paste(text, is_terminal, "emitting paste", keep_clipboard)
    }

    pub fn type_str_to(
        &mut self,
        text: &str,
        target: &WindowTarget,
        keep_clipboard: bool,
    ) -> TypeOutcome {
        if let Some(ref token) = target.focus_token {
            if !restore_focus(token) {
                warn!("could not restore focus to original window");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        self.do_clipboard_paste(
            text,
            target.is_terminal,
            "emitting paste to target",
            keep_clipboard,
        )
    }

    /// Set clipboard, emit the appropriate paste chord, and wait for the
    /// target app to consume it.  Falls back to per-key typing on clipboard
    /// failure.  `label` is passed through to the info log.
    ///
    /// Reports both whether the text reached the app and whether it is left on
    /// the clipboard afterwards (the latter true only when `keep_clipboard`
    /// and the paste succeeded).
    fn do_clipboard_paste(
        &mut self,
        text: &str,
        is_terminal: bool,
        label: &str,
        keep_clipboard: bool,
    ) -> TypeOutcome {
        // A terminal runs what it receives: every newline in a paste is a line
        // the shell executes and every tab is a completion request. Collapse
        // both to spaces so the target gets one inert line and nothing happens
        // until the user presses Enter themselves. Only the terminal path is
        // affected; ordinary text fields keep their line breaks.
        //
        // This assumes everything arriving here is transcript content. A future
        // caller that wants to press a real Enter (a "new line" voice command,
        // say) must not route through the paste path — it would get a space.
        //
        // The clipboard below carries the flattened text too, so what the user
        // keeps (`copy_to_clipboard`) is what was actually pasted.
        let flattened = if is_terminal {
            crate::sanitize::flatten_for_terminal(text)
        } else {
            Cow::Borrowed(text)
        };
        if flattened != text {
            warn!("collapsed line breaks and tabs in text pasted into a terminal");
        }
        let text: &str = &flattened;
        // Flattening a transcript that was nothing but line breaks leaves
        // nothing to paste. `delivery.rs` screens empty text, but it does so
        // before this step.
        if text.is_empty() {
            return TypeOutcome::DROPPED;
        }

        // The clipboard is only a transport for the paste chord. Unless the user
        // opted to keep the dictated text (copy_to_clipboard), preserve whatever
        // they had on the clipboard so we can put it back after the paste —
        // otherwise every dictation silently overwrites it. The snapshot is
        // normally captured ahead of time at dictation start (see
        // `prime_clipboard`) to keep its wl-paste cost off this hot path.
        //
        // Prefer that primed snapshot; always `take()` so a primed-but-unused
        // one (when keeping our text) can't leak into a later paste's restore.
        // Fall back to a synchronous read only when this paste wasn't primed.
        let primed = self.primed_clipboard.take();
        let prior = if keep_clipboard {
            Ok(None)
        } else if let Some(snap) = primed {
            snap
        } else {
            let snap_start = Instant::now();
            let snap = get_clipboard();
            debug!(
                snapshot_ms = snap_start.elapsed().as_millis() as u64,
                captured = matches!(snap, Ok(Some(_))),
                "clipboard snapshot taken (unprimed fallback)"
            );
            snap
        };

        let rtt_start = Instant::now();
        // On the transient path (dictated text not kept) set the clipboard via
        // arboard with the history-exclusion hint so clipboard-history managers
        // skip recording it, and hold the returned owner alive until after the
        // paste+restore below — on X11 the selection is served only while it
        // lives. On the keep path the user wants the text on their clipboard, so
        // set it plainly (and let it be recorded). Either set failing falls back
        // to per-key typing; an arboard-specific failure first retries the plain
        // CLI set so a missing display never drops the paste.
        let _excluded_owner = if keep_clipboard {
            if let Err(e) = set_clipboard(text) {
                warn!(error = %e, "clipboard set failed, falling back to per-key typing");
                // Prior contents are still intact and our text never reached the
                // clipboard — nothing to restore.
                return self.fallback_to_keys(text);
            }
            None
        } else {
            match set_clipboard_excluded(text) {
                Ok(owner) => Some(owner),
                Err(e) => {
                    warn!(error = %e, "history-excluded clipboard set failed, falling back to plain set");
                    if let Err(e) = set_clipboard(text) {
                        warn!(error = %e, "clipboard set failed, falling back to per-key typing");
                        return self.fallback_to_keys(text);
                    }
                    None
                }
            }
        };
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

        // Restore the user's clipboard now that the target app has had the
        // post-paste window to read our text.
        if !keep_clipboard {
            match prior {
                // Had content — put it back byte-for-byte.
                Ok(Some(prev)) => {
                    if let Err(e) = set_clipboard_snapshot(&prev) {
                        warn!(error = %e, "clipboard restore failed; dictated text left on clipboard");
                    }
                }
                // Genuinely empty before — clear our staged text back to empty.
                Ok(None) => {
                    if let Err(e) = clear_clipboard() {
                        warn!(error = %e, "clipboard clear failed; dictated text left on clipboard");
                    }
                }
                // Couldn't read it (timeout / missing tool) — must NOT clear, or
                // we'd destroy content we failed to snapshot; leave our text.
                Err(e) => {
                    warn!(error = %e, "clipboard snapshot failed; dictated text left on clipboard");
                }
            }
        }

        TypeOutcome::delivered(keep_clipboard)
    }

    /// Per-key typing entry point. Keystrokes are US-QWERTY/ASCII only, so if
    /// `text` contains any character that can't be represented as a keystroke
    /// (accented, CJK, emoji, smart quotes) the whole string is delivered via
    /// the clipboard instead — non-ASCII transcripts are never silently
    /// dropped.
    pub fn type_str_keys(&mut self, text: &str, keep_clipboard: bool) -> TypeOutcome {
        // Multi-line snippet bodies are the normal source of newlines here;
        // see `needs_paste_delivery` for why they must not be typed per-key.
        if needs_paste_delivery(text) {
            let is_terminal = self.resolve_terminal();
            let reason = if text.contains('\n') {
                "multi-line text — emitting paste (Enter would submit)"
            } else {
                "non-ASCII text — emitting paste"
            };
            return self.do_clipboard_paste(text, is_terminal, reason, keep_clipboard);
        }
        // Per-key typing costs ~8 ms per character, all of it after the
        // transcript is ready: a long paragraph types for seconds while the
        // user watches. A paste is flat in length. Short text still goes
        // per-key — it is faster than a clipboard round trip and leaves the
        // clipboard alone, which is what people expect for a quick phrase.
        if is_long_enough_to_paste(text) {
            let is_terminal = self.resolve_terminal();
            return self.do_clipboard_paste(
                text,
                is_terminal,
                "long text — emitting paste (per-key typing would take seconds)",
                keep_clipboard,
            );
        }
        // Per-key typing never touches the clipboard.
        self.type_keys_raw(text);
        TypeOutcome::delivered(false)
    }

    /// Last-resort per-key typing when the clipboard itself is unavailable.
    ///
    /// Applies the same rule as `needs_paste_delivery`: text the clipboard was
    /// chosen for because it *cannot* be typed per-key must not be typed
    /// anyway. A `'\n'` maps to `KEY_ENTER`, which submits the form or runs the
    /// shell line instead of inserting a line break — dropping the delivery is
    /// strictly safer than that. The refusal is reported through the returned
    /// [`TypeOutcome`] so `delivery.rs` can warn the user rather than let a
    /// dropped dictation look like a successful one.
    ///
    /// Tabs are the one exception. A tab has no keystroke by design (see
    /// `char_to_key`) but is only whitespace, so it becomes a space here
    /// instead of costing the user the whole utterance over one character.
    fn fallback_to_keys(&mut self, text: &str) -> TypeOutcome {
        let text = tabs_to_spaces(text);
        if !keystroke_fallback_allowed(&text) {
            error!(
                multi_line = text.contains('\n'),
                untypeable = !is_keystroke_typeable(&text),
                "clipboard unavailable and text cannot be typed per-key safely; \
                 dictation not delivered"
            );
            return TypeOutcome::DROPPED;
        }
        self.type_keys_raw(&text);
        TypeOutcome::delivered(false)
    }

    /// Raw per-key emitter. Paced (see `KEY_EVENT_GAP_MS` / `PER_CHAR_GAP_MS`)
    /// so the compositor reliably registers each key in order — at speed the
    /// pipeline drops spaces and leaks the shift modifier into later letters.
    /// Used once `text` is known to be keystroke-typeable, and as the last
    /// resort when the clipboard itself is unavailable — in that case any
    /// non-typeable char is unavoidably skipped.
    fn type_keys_raw(&mut self, text: &str) {
        std::thread::sleep(Duration::from_millis(MODIFIER_SETTLE_MS));
        for ch in text.chars() {
            self.type_char(ch);
            std::thread::sleep(Duration::from_millis(PER_CHAR_GAP_MS));
        }
    }

    fn type_char(&mut self, ch: char) {
        let Some((key, needs_shift)) = char_to_key(ch) else {
            warn!(%ch, "uinput: no US-QWERTY keystroke for char, skipping (clipboard unavailable)");
            return;
        };
        let gap = Duration::from_millis(KEY_EVENT_GAP_MS);
        if needs_shift {
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 1);
            std::thread::sleep(gap); // shift must register before the key
        }
        self.emit_raw(EventType::KEY, key.0, 1);
        std::thread::sleep(gap); // hold the key down long enough to register
        self.emit_raw(EventType::KEY, key.0, 0);
        if needs_shift {
            std::thread::sleep(gap); // key must release before shift, or shift leaks
            self.emit_raw(EventType::KEY, KeyCode::KEY_LEFTSHIFT.0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PASTE_THRESHOLD_CHARS, is_long_enough_to_paste, is_safe_mime, is_terminal_class,
        keystroke_fallback_allowed, needs_paste_delivery, pipe_text_to_command, tabs_to_spaces,
    };

    /// A short phrase types faster than the clipboard round trip and leaves the
    /// clipboard alone; a paragraph does not, and at ~8 ms per character it
    /// spends seconds visibly typing after the transcript is already known.
    #[test]
    fn only_long_text_is_routed_to_a_paste() {
        assert!(!is_long_enough_to_paste("send this"));
        assert!(!is_long_enough_to_paste(&"a".repeat(PASTE_THRESHOLD_CHARS)));
        assert!(is_long_enough_to_paste(
            &"a".repeat(PASTE_THRESHOLD_CHARS + 1)
        ));
    }

    /// The threshold counts characters, not bytes: a multi-byte transcript must
    /// not be pushed over it by its encoding.
    #[test]
    fn the_paste_threshold_counts_characters_not_bytes() {
        let text = "é".repeat(PASTE_THRESHOLD_CHARS);
        assert!(
            text.len() > PASTE_THRESHOLD_CHARS,
            "multi-byte by construction"
        );
        assert!(!is_long_enough_to_paste(&text));
    }
    use std::borrow::Cow;
    use std::time::{Duration, Instant};

    /// A clipboard failure must not reopen the Enter hazard: the fallback
    /// `do_clipboard_paste` takes when `set_clipboard` fails refuses exactly
    /// the text that was routed to the clipboard in the first place.
    #[test]
    fn clipboard_failure_fallback_refuses_paste_only_text() {
        for text in ["Best regards,\nDayton", "line one\n", "café", "日本語"] {
            assert!(needs_paste_delivery(text));
            assert!(
                !keystroke_fallback_allowed(text),
                "{text:?} must never be typed per-key"
            );
        }
        assert!(keystroke_fallback_allowed("plain single line"));
    }

    /// A newline typed per-key is KEY_ENTER, which submits whatever the text
    /// is going into — the multi-line snippet body "Best regards,\nDayton"
    /// would send the message after the first line and type the rest into
    /// whatever gained focus. Such text must go through the clipboard.
    #[test]
    fn multi_line_text_is_routed_to_the_clipboard() {
        assert!(needs_paste_delivery("Best regards,\nDayton"));
        assert!(needs_paste_delivery("line one\n"));
        assert!(needs_paste_delivery("a\n\nb"));
    }

    #[test]
    fn single_line_ascii_is_typed_per_key() {
        assert!(!needs_paste_delivery("Best regards, Dayton"));
        // A tab has no keystroke: pressing Tab would move focus mid-transcript.
        assert!(needs_paste_delivery("tabs\tgo through the clipboard"));
    }

    /// Routing a tab to the clipboard must not mean losing the dictation when
    /// the clipboard is the thing that failed. `fallback_to_keys` substitutes
    /// spaces first, so only the genuinely untypeable cases are refused.
    #[test]
    fn a_tab_does_not_cost_the_whole_utterance_when_the_clipboard_is_down() {
        assert!(
            keystroke_fallback_allowed(&tabs_to_spaces("review\tthis")),
            "a tab is whitespace; it must degrade to a space, not drop the text"
        );
        assert_eq!(tabs_to_spaces("review\tthis"), "review this");
        assert!(matches!(tabs_to_spaces("no tabs here"), Cow::Borrowed(_)));

        // The two refusals that must survive: Enter is an action, and a
        // character with no keystroke would be silently swallowed.
        assert!(!keystroke_fallback_allowed(&tabs_to_spaces("a\tb\nc")));
        assert!(!keystroke_fallback_allowed(&tabs_to_spaces("café\tx")));
        assert!(!needs_paste_delivery(""));
    }

    /// Unchanged by the newline rule: text with no US-QWERTY keystroke still
    /// has to paste, or those characters would be dropped.
    #[test]
    fn non_ascii_still_requires_paste() {
        assert!(needs_paste_delivery("café"));
        assert!(needs_paste_delivery("日本語"));
    }

    #[test]
    fn safe_mimes_are_well_formed_rfc6838_types() {
        for mime in [
            "text/plain",
            "text/plain;charset=utf-8",
            "image/png",
            "application/x-www-form-urlencoded",
            "text/plain;charset=utf-8;q=0.9",
        ] {
            assert!(is_safe_mime(mime), "{mime} should be accepted");
        }
    }

    #[test]
    fn unsafe_mimes_are_rejected() {
        for mime in [
            "",
            "-o",
            "--type",
            "text/plain --flag",
            "text/plain;charset=utf-8\nimage/png",
            "text/plain\nother",
            "text/<script>",
            "text/plain;charset=ü",
            "image/png;",
            "text/plain;;charset=utf-8",
            "text/plain;charset=",
            "text/plain;=utf-8",
            "text/plain;charset=utf-8=",
        ] {
            assert!(!is_safe_mime(mime), "{mime:?} should be rejected");
        }
    }

    #[test]
    fn matches_bare_terminal_names() {
        assert!(is_terminal_class("kitty"));
        assert!(is_terminal_class("konsole"));
        assert!(is_terminal_class("alacritty"));
        assert!(is_terminal_class("Konsole")); // case-insensitive
    }

    #[test]
    fn matches_reverse_dns_app_ids() {
        assert!(is_terminal_class("dev.warp.Warp"));
        assert!(is_terminal_class("org.gnome.Console"));
        assert!(is_terminal_class("org.kde.konsole"));
        assert!(is_terminal_class("com.raggesilver.BlackBox"));
    }

    #[test]
    fn matches_camelcase_against_hyphenated_list_entries() {
        // `cosmic-term` (list) → "cosmicterm"
        // `com.system76.CosmicTerm` (compositor) → "CosmicTerm" → "cosmicterm"
        assert!(is_terminal_class("com.system76.CosmicTerm"));
        // `gnome-terminal` ↔ `org.gnome.Terminal`
        assert!(is_terminal_class("org.gnome.Terminal"));
    }

    #[test]
    fn rejects_non_terminals() {
        assert!(!is_terminal_class("firefox"));
        assert!(!is_terminal_class("org.mozilla.firefox"));
        assert!(!is_terminal_class("code-oss"));
        assert!(!is_terminal_class("discord"));
        assert!(!is_terminal_class("google-chrome"));
        assert!(!is_terminal_class(""));
    }

    /// Regression: `pipe_text_to_command` must `take()` the child's stdin
    /// so dropping the writer closes the pipe and the child sees EOF.
    /// Earlier code left stdin owned by `Child`, so wl-copy/xclip kept
    /// reading forever and every `set_clipboard` call timed out. `cat`
    /// is a POSIX stand-in that reads stdin to EOF then exits 0.
    #[test]
    fn pipe_text_to_command_closes_stdin_so_child_exits() {
        let result = pipe_text_to_command("cat", &[], "hello", Duration::from_secs(2));
        assert!(result.is_ok(), "expected Ok; got {result:?}");
    }

    /// Regression: the polling loop must kill the child once `timeout`
    /// elapses. `sleep 5` ignores stdin entirely and is a wedged-child
    /// proxy — we assert the helper terminates well before the sleep
    /// would have finished.
    #[test]
    fn pipe_text_to_command_kills_after_timeout() {
        let started = Instant::now();
        let result = pipe_text_to_command("sleep", &["5"], "x", Duration::from_millis(200));
        let elapsed = started.elapsed();
        assert!(result.is_err(), "expected timeout Err; got {result:?}");
        assert!(
            elapsed < Duration::from_secs(2),
            "expected ~200ms termination via kill; elapsed: {elapsed:?}"
        );
    }

    /// Regression: when `write_all` fails (e.g. broken pipe because the
    /// child closed its stdin), the helper must kill and reap the child
    /// instead of dropping it. `std::process::Child` does NOT kill on
    /// drop, so without explicit cleanup the spawned process keeps
    /// running for the lifetime of the parent.
    ///
    /// The script closes its own stdin (forcing EPIPE on our large
    /// write), then sleeps and touches a sentinel file. If the child
    /// was killed on the write-failure path, the sentinel never
    /// appears; without the fix, it shows up within ~400ms.
    #[test]
    fn pipe_text_to_command_kills_child_on_write_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sentinel = dir.path().join("touched");
        let sentinel_str = sentinel.to_str().expect("utf-8 path");
        let script = format!("exec </dev/null; sleep 0.4; touch {sentinel_str}");

        let big = "x".repeat(200_000);
        let result = pipe_text_to_command("sh", &["-c", &script], &big, Duration::from_secs(10));
        assert!(
            result.is_err(),
            "expected write-failure Err; got {result:?}"
        );

        std::thread::sleep(Duration::from_millis(900));
        assert!(
            !sentinel.exists(),
            "child was not killed on write failure — sentinel file exists",
        );
    }
}
