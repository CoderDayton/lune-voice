// SPDX-License-Identifier: GPL-3.0-or-later

//! What window the dictation is going into, and how to name it.
//!
//! Delivery has to answer two questions about the focused window: *is it a
//! terminal* (which changes the paste chord and whether newlines are safe to
//! send) and *how do I get back to it* (the pill and the app's own windows can
//! take focus between the capture and the paste). Both answers are captured
//! once, when a dictation starts, and consumed seconds later when it ends.
//!
//! The capture is the only platform-specific part. Everything about what a
//! `WindowTarget` *means* — including terminal classification, which is just
//! string matching — is shared, so a new platform implements one function and
//! inherits the rest.

use lune_core::config::TerminalMode;

/// The focused window at the moment a dictation started.
#[derive(Debug, Clone, Default)]
pub struct WindowTarget {
    /// The window's app_id / WM_CLASS (Unix) or executable stem (Windows).
    /// `None` when the desktop will not say — GNOME Wayland, notably.
    pub class: Option<String>,
    pub is_terminal: bool,
    /// A handle for restoring focus, where the platform offers one that
    /// survives being handed to another thread.
    pub focus_token: Option<FocusToken>,
}

impl WindowTarget {
    /// The answer when the desktop will not name the focused window.
    ///
    /// Not an error: delivery degrades to the non-terminal chord and pastes
    /// wherever focus happens to be, which is the pre-existing behaviour on
    /// every compositor that withholds this.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Build from a class name alone, classifying it on the way in.
    pub fn from_class(class: String, focus_token: Option<FocusToken>) -> Self {
        Self {
            is_terminal: is_terminal_class(&class),
            class: Some(class),
            focus_token,
        }
    }
}

/// A platform's handle to the window that was focused.
///
/// Every variant is compiled on every platform. Gating them per-OS would put a
/// `cfg` on each arm of every match that consumes one, to save a few bytes in
/// an enum that exists once per dictation.
#[derive(Debug, Clone)]
pub enum FocusToken {
    HyprlandAddress(String),
    SwayConId(i64),
    X11WindowId(String),
    /// Win32 `HWND`, as an integer so the token stays `Send`.
    Win32Window(isize),
}

// ─── Terminal classification ─────────────────────────────────────────────────

/// Window classes that mean "a shell is reading this as a command line".
///
/// Kept as one list across platforms: several of these (alacritty, kitty,
/// wezterm, tabby, ghostty) run on Windows under the same name, and a
/// Windows-only split would have to repeat them.
const KNOWN_TERMINALS: &[&str] = &[
    "alacritty",
    "cmd",
    "com.raggesilver.blackbox",
    "conhost",
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
    "mintty",
    "mlterm",
    "org.codeberg.dnkl.foot",
    "org.gnome.console",
    "org.gnome.terminal",
    "org.kde.konsole",
    "org.wezfurlong.wezterm",
    "powershell",
    "pwsh",
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
    "windowsterminal",
    "wt",
    "xfce4-terminal",
    "xterm",
    "yakuake",
    "zellij",
    "zutty",
];

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
///
/// The `.`-segment rule also does the right thing on Windows, where the class
/// arrives as an executable name: `WindowsTerminal.exe` reduces to `exe` —
/// which is why the Windows capture strips the extension before calling this.
pub fn is_terminal_class(class: &str) -> bool {
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

/// The file stem of a Windows image path: last `\` segment, extension removed.
///
/// Lives here rather than beside the Win32 calls that feed it so it is
/// compiled — and tested — on every platform. The extension has to go before
/// `is_terminal_class` sees the name, or its `.`-segment rule reduces every
/// Windows class to `exe`.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn windows_image_stem(path: &str) -> Option<String> {
    let file = path.rsplit(['\\', '/']).next()?;
    let stem = file.rsplit_once('.').map_or(file, |(before, _)| before);
    (!stem.is_empty()).then(|| stem.to_string())
}

// ─── Capture ─────────────────────────────────────────────────────────────────

#[cfg(unix)]
mod unix;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(unix)]
pub(crate) use unix::{focused_window_class, restore_focus};
#[cfg(target_os = "windows")]
pub(crate) use windows::restore_focus;

/// Snapshot the focused window.
///
/// Blocking, and slow enough to matter: the Unix probes shell out or take a
/// Wayland round-trip and can cost ~500 ms on COSMIC. Callers run it off the
/// hot path.
pub fn capture_focus_target() -> WindowTarget {
    #[cfg(unix)]
    let target = unix::capture();
    #[cfg(target_os = "windows")]
    let target = windows::capture();
    #[cfg(not(any(unix, target_os = "windows")))]
    let target = WindowTarget::unknown();

    if target.class.is_some() {
        note_window_named();
    }
    target
}

/// Prime the compositor-detection chain (subprocess caches, D-Bus
/// connections) so the first real probe pays no cold-start cost.
///
/// Fire-and-forget; the result is discarded. Called once when the delivery
/// driver is built.
pub(crate) fn warm_probes() {
    #[cfg(unix)]
    {
        let _ = focused_window_class();
        tracing::debug!("focus: compositor detection primed");
    }
}

/// Whether any focus probe has named a window in this process.
static FOCUS_EVER_NAMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn note_window_named() {
    FOCUS_EVER_NAMED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether this desktop has ever identified a focused window.
///
/// Separates a desktop that cannot name windows at all — GNOME Wayland, whose
/// `Shell.Eval` endpoint is disabled and whose native windows `xdotool` cannot
/// see — from one that normally names them and simply failed on this window.
/// Only the second case makes an unnamed window unusual enough to be worth
/// treating as a possible shell.
fn focus_is_identifiable() -> bool {
    FOCUS_EVER_NAMED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether delivery should treat the focused window as a terminal (the paste
/// chord), and whether it must collapse newlines and tabs first (safety).
///
/// Both answers come from one focus probe. Resolving them separately would
/// put two compositor round-trips on the delivery hot path and let them
/// disagree when focus moves in between — Ctrl+Shift+V into what is no longer
/// a terminal.
///
/// Under `Auto` the captured target answers when there is one, and the
/// compositor is asked when there is not. Best effort: where the compositor
/// will not name the focused window — notably GNOME Wayland — the chord stays
/// Ctrl+V (Ctrl+Shift+V in an ordinary text field is wrong) while flattening
/// still applies, because an unnameable window might be a shell and every
/// newline pasted into a shell is a line it runs.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn paste_decisions(mode: TerminalMode, target: Option<&WindowTarget>) -> (bool, bool) {
    match mode {
        TerminalMode::Terminal => (true, true),
        TerminalMode::Normal => (false, false),
        _ => match target {
            Some(t) => decide_from_target(t, focus_is_identifiable()),
            None => decide_from_kind(focused_kind(), focus_is_identifiable()),
        },
    }
}

/// A captured target with no class is the unknown case; one that was named has
/// already been classified. An unnamed window only counts as a possible shell
/// where this desktop names windows at all.
fn decide_from_target(target: &WindowTarget, identifiable: bool) -> (bool, bool) {
    (
        target.is_terminal,
        target.is_terminal || (target.class.is_none() && identifiable),
    )
}

fn decide_from_kind(kind: FocusKind, identifiable: bool) -> (bool, bool) {
    match kind {
        FocusKind::Terminal => (true, true),
        FocusKind::NotTerminal => (false, false),
        FocusKind::Unknown => (false, identifiable),
    }
}

/// Whether delivery must collapse newlines and tabs before it emits the text.
///
/// Deliberately **not** the same question as the paste chord. The chord
/// is a convenience and errs towards Ctrl+V, because Ctrl+Shift+V in an
/// ordinary text field is wrong. Flattening is a safety measure and errs the
/// other way: on a desktop that names windows, one it would not name might be
/// a shell, and every newline pasted into a shell is a line it runs.
///
/// Where nothing can be named at all the caution buys nothing and costs every
/// line break, so it does not apply — see [`focus_is_identifiable`]. Users who
/// dictate into terminals there can force it with `TerminalMode::Terminal`.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn must_flatten(mode: TerminalMode, target: Option<&WindowTarget>) -> bool {
    match mode {
        TerminalMode::Terminal => true,
        TerminalMode::Normal => false,
        _ => paste_decisions(mode, target).1,
    }
}

/// What delivery could learn about the window focused right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) enum FocusKind {
    Terminal,
    NotTerminal,
    /// The desktop would not name the focused window — GNOME Wayland with no
    /// xdotool, a compositor with no focus API.
    Unknown,
}

#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn focused_kind() -> FocusKind {
    #[cfg(unix)]
    {
        match unix::focused_window_class() {
            Some(class) => {
                note_window_named();
                let is_term = is_terminal_class(&class);
                tracing::info!(class = %class, is_term, "focused window detected");
                if is_term {
                    FocusKind::Terminal
                } else {
                    FocusKind::NotTerminal
                }
            }
            None => {
                tracing::info!(
                    identifiable = focus_is_identifiable(),
                    "focused window: unknown (Ctrl+V; flattened only where windows are nameable)"
                );
                FocusKind::Unknown
            }
        }
    }
    #[cfg(not(unix))]
    {
        let target = capture_focus_target();
        match (target.is_terminal, target.class.is_some()) {
            (true, _) => FocusKind::Terminal,
            (false, true) => FocusKind::NotTerminal,
            (false, false) => FocusKind::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(class: Option<&str>) -> WindowTarget {
        match class {
            Some(c) => WindowTarget::from_class(c.to_owned(), None),
            None => WindowTarget::unknown(),
        }
    }

    /// The paste chord is a convenience: Ctrl+Shift+V in an ordinary text
    /// field is wrong, so only a window known to be a terminal gets it.
    #[test]
    fn only_a_known_terminal_gets_the_terminal_chord() {
        let chord = |mode, t: &WindowTarget| paste_decisions(mode, Some(t)).0;
        assert!(chord(TerminalMode::Auto, &target(Some("kitty"))));
        assert!(!chord(TerminalMode::Auto, &target(Some("firefox"))));
        assert!(!chord(TerminalMode::Auto, &target(None)));
        assert!(chord(TerminalMode::Terminal, &target(None)));
        assert!(!chord(TerminalMode::Normal, &target(Some("kitty"))));
    }

    /// Flattening is a safety measure, so it goes the other way: on a desktop
    /// that names windows, one it would not name might be a shell, and every
    /// newline in a paste into a shell is a line it runs.
    #[test]
    fn an_unnameable_window_is_flattened_even_though_it_gets_ctrl_v() {
        let flatten = |t: &WindowTarget| decide_from_target(t, true).1;
        assert!(flatten(&target(None)));
        assert!(flatten(&target(Some("kitty"))));
        assert!(!flatten(&target(Some("firefox"))));
        // `Normal` is the explicit opt-out, and it still means what it says.
        assert!(!must_flatten(TerminalMode::Normal, Some(&target(None))));
        assert!(must_flatten(TerminalMode::Terminal, Some(&target(None))));
    }

    /// Where nothing can ever be named — GNOME Wayland — the same caution
    /// would strip the line breaks out of every dictation, so it does not
    /// apply. A named terminal is still flattened there.
    #[test]
    fn an_unnameable_window_keeps_its_line_breaks_where_nothing_is_nameable() {
        assert!(!decide_from_target(&target(None), false).1);
        assert!(!decide_from_kind(FocusKind::Unknown, false).1);
        assert!(decide_from_kind(FocusKind::Unknown, true).1);
        // Positive identification still decides on its own either way.
        assert!(decide_from_target(&target(Some("kitty")), false).1);
        assert!(!decide_from_target(&target(Some("firefox")), true).1);
    }

    #[test]
    fn the_three_class_shapes_all_reduce_to_the_same_terminal() {
        assert!(is_terminal_class("kitty"));
        assert!(is_terminal_class("alacritty"));
        assert!(is_terminal_class("Konsole")); // case-insensitive
        assert!(is_terminal_class("org.kde.konsole"));
        assert!(is_terminal_class("org.gnome.Console"));
        assert!(is_terminal_class("com.raggesilver.BlackBox"));
        assert!(is_terminal_class("cosmic-term"));
        // `cosmic-term` (list) and `com.system76.CosmicTerm` (compositor)
        // both reduce to `cosmicterm`; likewise `gnome-terminal`.
        assert!(is_terminal_class("com.system76.CosmicTerm"));
        assert!(is_terminal_class("org.gnome.Terminal"));
        assert!(is_terminal_class("dev.warp.Warp"));
    }

    #[test]
    fn windows_shells_are_recognised_once_the_extension_is_stripped() {
        assert!(is_terminal_class("WindowsTerminal"));
        assert!(is_terminal_class("powershell"));
        assert!(is_terminal_class("cmd"));
        assert!(
            !is_terminal_class("WindowsTerminal.exe"),
            "the `.`-segment rule reduces this to `exe`; the caller must strip it"
        );
    }

    #[test]
    fn ordinary_apps_are_not_terminals() {
        assert!(!is_terminal_class("firefox"));
        assert!(!is_terminal_class("org.mozilla.firefox"));
        assert!(!is_terminal_class("Code"));
        assert!(!is_terminal_class("code-oss"));
        assert!(!is_terminal_class("discord"));
        assert!(!is_terminal_class("google-chrome"));
        assert!(!is_terminal_class(""));
        assert!(!is_terminal_class("..."));
    }

    #[test]
    fn an_unknown_target_claims_nothing() {
        let t = WindowTarget::unknown();
        assert!(t.class.is_none());
        assert!(!t.is_terminal);
        assert!(t.focus_token.is_none());
    }

    #[test]
    fn a_full_image_path_reduces_to_its_stem() {
        assert_eq!(
            windows_image_stem(r"C:\Program Files\WindowsApps\WindowsTerminal.exe").as_deref(),
            Some("WindowsTerminal")
        );
        assert_eq!(
            windows_image_stem(r"C:\Windows\System32\cmd.exe").as_deref(),
            Some("cmd")
        );
    }

    /// A dotted directory must not eat the file name, and a name with no
    /// extension is already a stem.
    #[test]
    fn odd_image_paths_still_produce_a_usable_class() {
        assert_eq!(
            windows_image_stem(r"C:\my.tools\bin\foo").as_deref(),
            Some("foo")
        );
        assert_eq!(
            windows_image_stem("firefox.exe").as_deref(),
            Some("firefox")
        );
        assert_eq!(windows_image_stem(""), None);
        assert_eq!(
            windows_image_stem(r"C:\dir\.exe"),
            None,
            "no stem left to name"
        );
    }

    /// The whole point of stripping the extension: the classifier has to see
    /// `WindowsTerminal`, not `exe`.
    #[test]
    fn a_windows_terminal_path_classifies_as_a_terminal() {
        let stem = windows_image_stem(r"C:\Windows\System32\WindowsTerminal.exe").unwrap();
        assert!(is_terminal_class(&stem));
    }

    #[test]
    fn from_class_classifies_on_the_way_in() {
        let t = WindowTarget::from_class("konsole".into(), None);
        assert!(t.is_terminal);
        let t = WindowTarget::from_class("firefox".into(), None);
        assert!(!t.is_terminal);
    }
}
