use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PrivacyMode {
    /// Reject any remote backend usage.
    LocalOnly,
    /// Use remote only if explicitly configured + consented.
    LocalPreferred,
    /// Use whatever is configured.
    AllowRemote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum DictationMode {
    /// Type finalized text into focused input via wtype.
    #[default]
    TypeInPlace,
    /// Copy finalized text to clipboard via wl-copy.
    Clipboard,
    /// Type keystrokes into the focused app AND copy text to the clipboard.
    /// Lets the user re-paste the dictation later without overwriting it.
    Both,
}

impl std::fmt::Display for DictationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeInPlace => f.write_str("TypeInPlace"),
            Self::Clipboard => f.write_str("Clipboard"),
            Self::Both => f.write_str("Both"),
        }
    }
}

impl std::str::FromStr for DictationMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "TypeInPlace" => Ok(Self::TypeInPlace),
            "Clipboard" => Ok(Self::Clipboard),
            "Both" => Ok(Self::Both),
            _ => Err(dictation_mode_parse_err(s)),
        }
    }
}

#[cold]
#[inline(never)]
fn dictation_mode_parse_err(s: &str) -> String {
    format!("unknown dictation mode: {s} (expected TypeInPlace|Clipboard|Both)")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ActivationMode {
    /// Press once to start, press again to stop.
    #[default]
    Toggle,
    /// Hold key to dictate, release to stop.
    Hold,
}

impl std::fmt::Display for ActivationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Toggle => f.write_str("Toggle"),
            Self::Hold => f.write_str("Hold"),
        }
    }
}

impl std::str::FromStr for ActivationMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Toggle" => Ok(Self::Toggle),
            "Hold" => Ok(Self::Hold),
            _ => Err(activation_mode_parse_err(s)),
        }
    }
}

#[cold]
#[inline(never)]
fn activation_mode_parse_err(s: &str) -> String {
    format!("unknown activation mode: {s} (expected Toggle|Hold)")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum PillVisibility {
    /// Show the pill whenever the service is ready.
    #[default]
    Always,
    /// Show the pill only while actively dictating.
    WhenDictating,
    /// Never show the pill overlay.
    Never,
}

impl std::fmt::Display for PillVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => f.write_str("Always"),
            Self::WhenDictating => f.write_str("WhenDictating"),
            Self::Never => f.write_str("Never"),
        }
    }
}

impl std::str::FromStr for PillVisibility {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Always" => Ok(Self::Always),
            "WhenDictating" => Ok(Self::WhenDictating),
            "Never" => Ok(Self::Never),
            _ => Err(pill_visibility_parse_err(s)),
        }
    }
}

#[cold]
#[inline(never)]
fn pill_visibility_parse_err(s: &str) -> String {
    format!("unknown pill visibility: {s} (expected Always|WhenDictating|Never)")
}

impl std::fmt::Display for PrivacyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalOnly => f.write_str("LocalOnly"),
            Self::LocalPreferred => f.write_str("LocalPreferred"),
            Self::AllowRemote => f.write_str("AllowRemote"),
        }
    }
}

impl std::str::FromStr for PrivacyMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "LocalOnly" => Ok(Self::LocalOnly),
            "LocalPreferred" => Ok(Self::LocalPreferred),
            "AllowRemote" => Ok(Self::AllowRemote),
            _ => Err(privacy_mode_parse_err(s)),
        }
    }
}

#[cold]
#[inline(never)]
fn privacy_mode_parse_err(s: &str) -> String {
    format!("unknown privacy mode: {s} (expected LocalOnly|LocalPreferred|AllowRemote)")
}

/// Controls how the delivery system detects terminal windows for paste chords.
///
/// Terminals use Ctrl+Shift+V to paste, while other apps use Ctrl+V.
/// On compositors that expose focused window info (Hyprland, Sway, KDE, GNOME),
/// `Auto` detects this automatically. On COSMIC and others without such APIs,
/// set `Terminal` if you primarily dictate into terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum TerminalMode {
    /// Probe compositor for focused window class. Falls back to Ctrl+V if
    /// detection is unavailable (e.g. COSMIC without xdotool).
    #[default]
    Auto,
    /// Always use Ctrl+Shift+V (force terminal paste).
    Terminal,
    /// Always use Ctrl+V (force normal paste).
    Normal,
}

impl std::fmt::Display for TerminalMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("Auto"),
            Self::Terminal => f.write_str("Terminal"),
            Self::Normal => f.write_str("Normal"),
        }
    }
}

impl std::str::FromStr for TerminalMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Auto" => Ok(Self::Auto),
            "Terminal" => Ok(Self::Terminal),
            "Normal" => Ok(Self::Normal),
            _ => Err(terminal_mode_parse_err(s)),
        }
    }
}

#[cold]
#[inline(never)]
fn terminal_mode_parse_err(s: &str) -> String {
    format!("unknown terminal mode: {s} (expected Auto|Terminal|Normal)")
}
