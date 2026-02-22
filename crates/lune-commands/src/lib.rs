use lune_core::command::VoiceCommand;

/// Result of routing a text through the command recognizer.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Commands detected in the text, in order of appearance.
    pub commands: Vec<VoiceCommand>,
    /// Remaining text after stripping command phrases.
    pub remaining: String,
}

/// Routes raw ASR text to built-in voice commands before LLM post-processing.
///
/// Performs case-insensitive exact phrase matching. Matched phrases are stripped
/// from the text; the cleaned remainder is returned for LLM post-processing.
pub struct VoiceCommandRouter {
    /// Additional user-defined phrase → command mappings loaded from settings.
    custom: Vec<(String, VoiceCommand)>,
}

// Static builtin patterns — avoids re-allocating on every call to all_patterns().
static BUILTINS: &[(&str, VoiceCommand)] = &[
    ("stop dictation", VoiceCommand::StopDictation),
    ("stop", VoiceCommand::StopDictation),
    ("new paragraph", VoiceCommand::NewParagraph),
    ("new line", VoiceCommand::NewLine),
    ("list item", VoiceCommand::ListItem),
    ("scratch that", VoiceCommand::ScratchThat),
];

impl VoiceCommandRouter {
    #[inline]
    pub fn new() -> Self {
        Self { custom: Vec::new() }
    }

    /// Add a custom phrase mapping (case-insensitive).
    #[inline]
    pub fn add_custom(&mut self, phrase: impl Into<String>, command: VoiceCommand) {
        self.custom.push((phrase.into().to_lowercase(), command));
    }

    /// Route `text`, returning detected commands and the remaining text.
    ///
    /// Commands are detected left-to-right; each matched span is removed once.
    pub fn route(&self, text: &str) -> RouteResult {
        let lower = text.to_lowercase();
        let mut commands = Vec::new();
        let mut removals: Vec<(usize, usize)> = Vec::new();

        for (phrase, cmd) in self.all_patterns() {
            let mut search_from = 0;
            while let Some(pos) = lower[search_from..].find(phrase) {
                let start = search_from + pos;
                let end = start + phrase.len();
                if removals.iter().any(|&(rs, re)| rs < end && re > start) {
                    search_from = end;
                    continue;
                }
                commands.push(cmd.clone());
                removals.push((start, end));
                search_from = end;
            }
        }

        removals.sort_unstable_by_key(|&(s, _)| s);
        let mut remaining = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for (start, end) in removals {
            remaining.push_str(&text[cursor..start]);
            cursor = end;
        }
        remaining.push_str(&text[cursor..]);

        let remaining = collapse_whitespace(&remaining);

        RouteResult {
            commands,
            remaining,
        }
    }

    /// Iterator over all (lowercase_phrase, VoiceCommand) pairs, built-ins first.
    #[inline]
    fn all_patterns(&self) -> impl Iterator<Item = (&str, VoiceCommand)> + '_ {
        BUILTINS
            .iter()
            .map(|(p, c)| (*p, c.clone()))
            .chain(self.custom.iter().map(|(p, c)| (p.as_str(), c.clone())))
    }

    /// Check for parametric patterns that capture an argument.
    fn parametric_matches(lower: &str) -> Vec<(usize, usize, VoiceCommand)> {
        let mut out = Vec::new();

        for prefix in &["switch to ", "switch preset to "] {
            let mut pos = 0;
            while let Some(idx) = lower[pos..].find(prefix) {
                let start = pos + idx;
                let arg_start = start + prefix.len();
                let arg_end = lower[arg_start..]
                    .find([',', '.', ';'])
                    .map(|i| arg_start + i)
                    .unwrap_or(lower.len());
                let arg = lower[arg_start..arg_end].trim().to_string();
                if !arg.is_empty() {
                    out.push((start, arg_end, VoiceCommand::SwitchStyle(arg)));
                }
                pos = arg_end;
            }
        }

        for prefix in &["insert ", "expand snippet "] {
            let mut pos = 0;
            while let Some(idx) = lower[pos..].find(prefix) {
                let start = pos + idx;
                let arg_start = start + prefix.len();
                let arg_end = lower[arg_start..]
                    .find([',', '.', ';'])
                    .map(|i| arg_start + i)
                    .unwrap_or(lower.len());
                let arg = lower[arg_start..arg_end].trim().to_string();
                if !arg.is_empty() {
                    out.push((start, arg_end, VoiceCommand::InsertSnippet(arg)));
                }
                pos = arg_end;
            }
        }

        out
    }

    /// Full route including parametric patterns.
    pub fn route_full(&self, text: &str) -> RouteResult {
        let lower = text.to_lowercase();
        let mut commands: Vec<VoiceCommand> = Vec::new();
        let mut removals: Vec<(usize, usize)> = Vec::new();

        for (start, end, cmd) in Self::parametric_matches(&lower) {
            commands.push(cmd);
            removals.push((start, end));
        }

        for (phrase, cmd) in self.all_patterns() {
            let mut search_from = 0;
            while let Some(pos) = lower[search_from..].find(phrase) {
                let start = search_from + pos;
                let end = start + phrase.len();
                if removals.iter().any(|&(rs, re)| rs < end && re > start) {
                    search_from = end;
                    continue;
                }
                commands.push(cmd.clone());
                removals.push((start, end));
                search_from = end;
            }
        }

        removals.sort_unstable_by_key(|&(s, _)| s);
        let mut remaining = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for (start, end) in removals {
            remaining.push_str(&text[cursor..start]);
            cursor = end;
        }
        remaining.push_str(&text[cursor..]);

        RouteResult {
            commands,
            remaining: collapse_whitespace(&remaining),
        }
    }
}

impl Default for VoiceCommandRouter {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = true;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> VoiceCommandRouter {
        VoiceCommandRouter::new()
    }

    #[test]
    fn stop_dictation_full() {
        let r = router().route_full("please stop dictation now");
        assert!(r.commands.contains(&VoiceCommand::StopDictation));
        assert!(!r.remaining.to_lowercase().contains("stop dictation"));
    }

    #[test]
    fn stop_alone() {
        let r = router().route_full("stop");
        assert!(r.commands.contains(&VoiceCommand::StopDictation));
        assert!(r.remaining.is_empty());
    }

    #[test]
    fn new_line_command() {
        let r = router().route_full("hello new line world");
        assert!(r.commands.contains(&VoiceCommand::NewLine));
        assert_eq!(r.remaining, "hello world");
    }

    #[test]
    fn new_paragraph_command() {
        let r = router().route_full("end of section new paragraph continue here");
        assert!(r.commands.contains(&VoiceCommand::NewParagraph));
        assert!(!r.remaining.to_lowercase().contains("new paragraph"));
    }

    #[test]
    fn list_item_command() {
        let r = router().route_full("list item fix the bug");
        assert!(r.commands.contains(&VoiceCommand::ListItem));
        assert_eq!(r.remaining, "fix the bug");
    }

    #[test]
    fn scratch_that_command() {
        let r = router().route_full("I made a mistake scratch that");
        assert!(r.commands.contains(&VoiceCommand::ScratchThat));
        assert_eq!(r.remaining, "I made a mistake");
    }

    #[test]
    fn switch_style_command() {
        let r = router().route_full("switch to email, please clean this up");
        assert!(
            r.commands
                .iter()
                .any(|c| matches!(c, VoiceCommand::SwitchStyle(s) if s == "email"))
        );
    }

    #[test]
    fn insert_snippet_command() {
        let r = router().route_full("insert greeting here");
        assert!(
            r.commands
                .iter()
                .any(|c| matches!(c, VoiceCommand::InsertSnippet(s) if s == "greeting here"))
        );
    }

    #[test]
    fn multiple_commands_in_one_utterance() {
        let r = router().route_full("new line list item fix the bug new line");
        let new_line_count = r
            .commands
            .iter()
            .filter(|c| **c == VoiceCommand::NewLine)
            .count();
        assert!(new_line_count >= 1);
        assert!(r.commands.contains(&VoiceCommand::ListItem));
    }

    #[test]
    fn no_commands_passthrough() {
        let r = router().route_full("hello world how are you");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "hello world how are you");
    }

    #[test]
    fn case_insensitive_match() {
        let r = router().route_full("STOP DICTATION right now");
        assert!(r.commands.contains(&VoiceCommand::StopDictation));
    }

    #[test]
    fn custom_command() {
        let mut r = VoiceCommandRouter::new();
        r.add_custom("send message", VoiceCommand::StopDictation);
        let result = r.route_full("please send message now");
        assert!(result.commands.contains(&VoiceCommand::StopDictation));
    }

    #[test]
    fn collapse_whitespace_fn() {
        assert_eq!(collapse_whitespace("  hello   world  "), "hello world");
        assert_eq!(collapse_whitespace(""), "");
        assert_eq!(collapse_whitespace("no extra"), "no extra");
    }
}
