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

/// Snapshot the focused window.
///
/// Blocking, and slow enough to matter: the Unix probes shell out or take a
/// Wayland round-trip and can cost ~500 ms on COSMIC. Callers run it off the
/// hot path.
pub fn capture_focus_target() -> WindowTarget {
    #[cfg(unix)]
    {
        unix::capture()
    }
    #[cfg(target_os = "windows")]
    {
        windows::capture()
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        WindowTarget::unknown()
    }
}

/// Whether the window focused *right now* is a terminal.
///
/// The live counterpart to a captured `WindowTarget`, for delivery paths that
/// never captured one.
/// Only the Linux paste-chord decision needs the live answer; every other
/// driver reads `is_terminal` off a captured target.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn focused_is_terminal() -> bool {
    #[cfg(unix)]
    {
        unix::focused_is_terminal()
    }
    #[cfg(not(unix))]
    {
        capture_focus_target().is_terminal
    }
}

/// Whether delivery should treat the focused window as a terminal.
///
/// Under `Auto` the captured target answers when there is one, and the
/// compositor is asked when there is not. Both delivery backends decide this
/// the same way, so the rule lives here rather than once per backend.
///
/// Best effort under `Auto`: where the compositor will not name the focused
/// window — notably GNOME Wayland — this answers `false` and the text is
/// delivered unchanged.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn targets_terminal(mode: TerminalMode, target: Option<&WindowTarget>) -> bool {
    match mode {
        TerminalMode::Terminal => true,
        TerminalMode::Normal => false,
        _ => target.map_or_else(focused_is_terminal, |t| t.is_terminal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
