use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub toggle_dictation: KeyCombo,
    pub cancel_dictation: KeyCombo,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            toggle_dictation: KeyCombo {
                modifiers: vec![Modifier::Ctrl],
                key: "Space".into(),
            },
            cancel_dictation: KeyCombo {
                modifiers: vec![],
                key: "Escape".into(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyCombo {
    pub modifiers: Vec<Modifier>,
    pub key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Modifier {
    Ctrl,
    Shift,
    Alt,
    Meta,
}
