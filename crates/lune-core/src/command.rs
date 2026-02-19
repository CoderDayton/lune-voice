use serde::{Deserialize, Serialize};

/// Built-in voice commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VoiceCommand {
    StopDictation,
    NewLine,
    NewParagraph,
    ListItem,
    SwitchStyle(String),
    InsertSnippet(String),
    ScratchThat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_command_round_trip() {
        let cmds = vec![
            VoiceCommand::StopDictation,
            VoiceCommand::NewLine,
            VoiceCommand::SwitchStyle("formal".into()),
            VoiceCommand::InsertSnippet("greeting".into()),
            VoiceCommand::ScratchThat,
        ];
        for cmd in &cmds {
            let json = serde_json::to_string(cmd).unwrap();
            let parsed: VoiceCommand = serde_json::from_str(&json).unwrap();
            assert_eq!(&parsed, cmd);
        }
    }
}
