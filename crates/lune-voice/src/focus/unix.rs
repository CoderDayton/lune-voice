//! Asking a Unix desktop which window has focus, and putting focus back.
//!
//! There is no one answer: each compositor exposes the focused window through
//! its own IPC, and several expose nothing at all. The probes are tried in
//! order of how cheap and how specific they are — a compositor's own socket
//! first, then the generic Wayland toplevel protocols, then D-Bus, then X11.
//!
//! Every probe is bounded ([`SUBPROCESS_TIMEOUT`]): a wedged daemon must cost
//! delivery one timeout, not the utterance.

use std::time::Duration;

use crate::focus::{FocusToken, WindowTarget, is_terminal_class};
use crate::probe::{output_with_timeout, status_with_timeout};

/// Hard ceiling on every compositor / IPC subprocess spawned while resolving
/// focus. These tools normally return in single-digit ms; anything beyond
/// means the daemon is wedged and we'd rather fall through than stall paste.
///
/// [`capture_focus_target`] and [`focused_window_class`] may try several
/// backends in sequence (Hyprland → Sway → KDE → GNOME → X11), so the
/// worst-case latency on a system where every gating env var is set AND every
/// daemon is wedged is bounded by N × `SUBPROCESS_TIMEOUT`. In practice only
/// one backend matches per session, so the typical worst case is one timeout.
const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(1);

/// Query the focused window's app_id / WM_CLASS from the compositor.
///
/// Detection chain: Hyprland → Sway → wlr-foreign-toplevel (COSMIC, KDE-Wayland,
/// Wayfire, river, …) → KDE (D-Bus) → GNOME (D-Bus) → xdotool (X11) → None
pub(crate) fn focused_window_class() -> Option<String> {
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

pub(crate) fn capture() -> WindowTarget {
    if std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok()
        && let Some((class, addr)) = hyprland_active_info()
    {
        return WindowTarget::from_class(class, Some(FocusToken::HyprlandAddress(addr)));
    }

    if std::env::var("SWAYSOCK").is_ok()
        && let Some((class, con_id)) = sway_focused_with_id()
    {
        return WindowTarget::from_class(class, Some(FocusToken::SwayConId(con_id)));
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
        return WindowTarget::from_class(class, None);
    }

    if std::env::var("DISPLAY").is_ok()
        && let Some(wid) = xdotool_active_id()
    {
        // The class comes from the id we just resolved: asking xdotool a second
        // time costs another spawn and can name a window that took focus in
        // between, pairing a class with the wrong token.
        let class = xprop_class(&wid);
        return with_token(class, Some(FocusToken::X11WindowId(wid)));
    }

    with_token(focused_window_class(), None)
}

/// A target from a class the compositor may or may not have named.
fn with_token(class: Option<String>, focus_token: Option<FocusToken>) -> WindowTarget {
    WindowTarget {
        is_terminal: class.as_deref().is_some_and(is_terminal_class),
        class,
        focus_token,
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
        return Some((focused_node_class(node)?, con_id));
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

/// Put focus back on the window a dictation started in.
///
/// The pill and the app's own windows can take focus between the capture and
/// the paste, so delivery re-focuses before emitting the chord.
pub(crate) fn restore_focus(token: &FocusToken) -> bool {
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
        // A Win32 handle cannot reach this backend: nothing on Unix mints one.
        FocusToken::Win32Window(_) => false,
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
    if node.get("focused").and_then(|v| v.as_bool()) == Some(true)
        && let Some(class) = focused_node_class(node)
    {
        return Some(class);
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

/// A sway tree node's class: `app_id` (Wayland native), falling back to
/// `window_properties.class` (XWayland).
///
/// Sway writes the JSON string `"null"` for a missing `app_id` on some
/// versions, so an empty or literal-`null` id counts as absent.
fn focused_node_class(node: &serde_json::Value) -> Option<String> {
    node.get("app_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty() && *s != "null")
        .or_else(|| {
            node.get("window_properties")
                .and_then(|wp| wp.get("class"))
                .and_then(|v| v.as_str())
        })
        .map(str::to_owned)
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
    let rest = &stdout[stdout.find("resourceClass")?..];
    let q1 = rest.find('\'')?;
    let after = &rest[q1 + 1..];
    let q2 = after.find('\'')?;
    Some(after[..q2].to_owned())
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
    xprop_class(&xdotool_active_id()?)
}

/// The WM_CLASS of a window we already have the id of.
fn xprop_class(window_id: &str) -> Option<String> {
    let xprop = output_with_timeout("xprop", &["-id", window_id, "WM_CLASS"], SUBPROCESS_TIMEOUT)?;
    if !xprop.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&xprop.stdout);
    // WM_CLASS(STRING) = "instance", "Class"  — we want the second value
    let after = &stdout[stdout.find('=')? + 1..];
    let parts: Vec<&str> = after.split(',').collect();
    if parts.len() >= 2 {
        return extract_quoted_value(parts[1]);
    }
    // Single class
    extract_quoted_value(after)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_sway_node_prefers_its_app_id() {
        let node = json!({ "app_id": "kitty", "window_properties": { "class": "Kitty" } });
        assert_eq!(focused_node_class(&node).as_deref(), Some("kitty"));
    }

    /// XWayland clients carry no `app_id`; sway reports the string `"null"`
    /// rather than a JSON null on some versions.
    #[test]
    fn a_sway_node_without_an_app_id_falls_back_to_wm_class() {
        for absent in [json!("null"), json!("")] {
            let node = json!({ "app_id": absent, "window_properties": { "class": "Firefox" } });
            assert_eq!(focused_node_class(&node).as_deref(), Some("Firefox"));
        }
        let node = json!({ "window_properties": { "class": "Firefox" } });
        assert_eq!(focused_node_class(&node).as_deref(), Some("Firefox"));
    }

    #[test]
    fn a_node_naming_nothing_stays_unnamed() {
        assert_eq!(focused_node_class(&json!({})), None);
    }

    #[test]
    fn the_focused_node_is_found_below_floating_containers() {
        let tree = json!({
            "focused": false,
            "nodes": [{ "focused": false, "floating_nodes": [
                { "focused": true, "id": 42, "app_id": "konsole" }
            ]}]
        });
        assert_eq!(find_focused_node(&tree).as_deref(), Some("konsole"));
        assert_eq!(
            find_focused_with_id(&tree),
            Some(("konsole".to_owned(), 42))
        );
    }

    #[test]
    fn quoted_values_are_read_out_of_either_quote_style() {
        assert_eq!(
            extract_quoted_value("(true, 'Alacritty')").as_deref(),
            Some("Alacritty")
        );
        assert_eq!(
            extract_quoted_value(r#" "Konsole" "#).as_deref(),
            Some("Konsole")
        );
        assert_eq!(extract_quoted_value("''"), None);
        assert_eq!(extract_quoted_value("no quotes here"), None);
    }
}
