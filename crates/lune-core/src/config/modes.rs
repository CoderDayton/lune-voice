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
    /// Both type and copy to clipboard.
    Both,
}

impl std::fmt::Display for DictationMode {
    #[inline]
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
    #[inline]
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
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Toggle => f.write_str("Toggle"),
            Self::Hold => f.write_str("Hold"),
        }
    }
}

impl std::str::FromStr for ActivationMode {
    type Err = String;
    #[inline]
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

impl std::fmt::Display for PrivacyMode {
    #[inline]
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
    #[inline]
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
