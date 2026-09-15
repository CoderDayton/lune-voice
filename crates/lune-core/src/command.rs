// SPDX-License-Identifier: GPL-3.0-or-later

use serde::{Deserialize, Serialize};

/// Voice commands the router can emit.
///
/// Both are parametric and both are acted on in the dictation pipeline itself.
/// There are no fixed built-in phrases: every one that used to exist either did
/// nothing or was a formatting shortcut, and a recognized command swallows the
/// whole utterance, so a false positive typed nothing at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VoiceCommand {
    SwitchStyle(String),
    InsertSnippet(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_command_round_trip() {
        let cmds = vec![
            VoiceCommand::SwitchStyle("formal".into()),
            VoiceCommand::InsertSnippet("greeting".into()),
        ];
        for cmd in &cmds {
            let json = serde_json::to_string(cmd).unwrap();
            let parsed: VoiceCommand = serde_json::from_str(&json).unwrap();
            assert_eq!(&parsed, cmd);
        }
    }
}
