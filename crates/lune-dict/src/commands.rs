use lune_core::command::VoiceCommand;

/// Result of routing a text through the command recognizer.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Detected command, if the whole utterance was one (at most one).
    pub commands: Vec<VoiceCommand>,
    /// The dictation text. Empty when the utterance was a command; otherwise
    /// the original text, unchanged.
    pub remaining: String,
}

/// Routes raw ASR text to built-in voice commands before LLM post-processing.
///
/// A command is recognized **only when the entire utterance is that command**,
/// compared word-for-word and ignoring case and surrounding punctuation
/// (`"Stop, dictation!"` == `"stop dictation"`). Anything else passes through
/// untouched, so ordinary words like `"stop"` in `"tell him to stop"` are never
/// stripped from dictation.
///
/// Parametric commands (`"switch to <preset>"`, `"insert <snippet>"`) match when
/// the utterance *begins* with the exact command words and has at least one more
/// word; the remainder is the argument.
pub struct VoiceCommandRouter {
    /// User-defined phrase → command mappings, stored as normalized word lists.
    custom: Vec<(Vec<String>, VoiceCommand)>,
}

// Fixed commands. Each matches only when it is the entire utterance. Bare
// "stop" is intentionally absent — it collided with the ordinary word "stop"
// in dictation; stop the session with "stop dictation" / "stop listening".
static BUILTINS: &[(&str, VoiceCommand)] = &[
    ("stop dictation", VoiceCommand::StopDictation),
    ("stop listening", VoiceCommand::StopDictation),
    ("new paragraph", VoiceCommand::NewParagraph),
    ("scratch that", VoiceCommand::ScratchThat),
    ("list item", VoiceCommand::ListItem),
    ("new line", VoiceCommand::NewLine),
];

#[derive(Copy, Clone)]
enum ParametricKind {
    SwitchStyle,
    InsertSnippet,
}

// Parametric command prefixes, matched only at the very start of an utterance.
// Longer prefixes first so "switch preset to" wins over "switch to".
static PARAMETRIC_PREFIXES: &[(&str, ParametricKind)] = &[
    ("switch preset to", ParametricKind::SwitchStyle),
    ("switch to", ParametricKind::SwitchStyle),
    ("expand snippet", ParametricKind::InsertSnippet),
    ("insert", ParametricKind::InsertSnippet),
];

impl VoiceCommandRouter {
    pub fn new() -> Self {
        Self { custom: Vec::new() }
    }

    pub fn add_custom(&mut self, phrase: impl Into<String>, command: VoiceCommand) {
        let words = normalize_words(&phrase.into());
        if !words.is_empty() {
            self.custom.push((words, command));
        }
    }

    /// Route `text`, returning the detected command (if the whole utterance is
    /// one) and the remaining dictation text.
    pub fn route(&self, text: &str) -> RouteResult {
        self.route_inner(text, false)
    }

    /// Full route, including parametric commands.
    pub fn route_full(&self, text: &str) -> RouteResult {
        self.route_inner(text, true)
    }

    fn route_inner(&self, text: &str, parametric: bool) -> RouteResult {
        let words = normalize_words(text);
        let passthrough = || RouteResult {
            commands: Vec::new(),
            remaining: text.trim().to_string(),
        };
        if words.is_empty() {
            return passthrough();
        }

        // Parametric: the utterance must begin with the exact command words and
        // carry at least one argument word after them.
        if parametric {
            for &(prefix, kind) in PARAMETRIC_PREFIXES {
                let pre: Vec<&str> = prefix.split_whitespace().collect();
                if words.len() > pre.len()
                    && words[..pre.len()].iter().zip(&pre).all(|(w, p)| w == p)
                {
                    let arg = words[pre.len()..].join(" ");
                    let cmd = match kind {
                        ParametricKind::SwitchStyle => VoiceCommand::SwitchStyle(arg),
                        ParametricKind::InsertSnippet => VoiceCommand::InsertSnippet(arg),
                    };
                    return RouteResult {
                        commands: vec![cmd],
                        remaining: String::new(),
                    };
                }
            }
        }

        // Fixed/custom commands: the whole utterance must equal the command.
        for (phrase, cmd) in self.all_patterns() {
            if words.iter().map(String::as_str).eq(phrase.iter().copied()) {
                return RouteResult {
                    commands: vec![cmd.clone()],
                    remaining: String::new(),
                };
            }
        }

        passthrough()
    }

    fn all_patterns(&self) -> impl Iterator<Item = (Vec<&str>, &VoiceCommand)> + '_ {
        BUILTINS
            .iter()
            .map(|(p, c)| (p.split_whitespace().collect::<Vec<_>>(), c))
            .chain(
                self.custom
                    .iter()
                    .map(|(words, c)| (words.iter().map(String::as_str).collect::<Vec<_>>(), c)),
            )
    }
}

impl Default for VoiceCommandRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize an utterance to lowercase, punctuation-free words for command
/// comparison. Each whitespace token is reduced to its alphanumeric characters
/// (so `"Stop,"` and `"stop"` compare equal); tokens with no alphanumerics are
/// dropped. Used only for *matching* — the passed-through dictation keeps its
/// original casing and punctuation.
fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|tok| {
            let w: String = tok
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect();
            (!w.is_empty()).then_some(w)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> VoiceCommandRouter {
        VoiceCommandRouter::new()
    }

    #[test]
    fn stop_dictation_exact() {
        let r = router().route_full("stop dictation");
        assert!(r.commands.contains(&VoiceCommand::StopDictation));
        assert!(r.remaining.is_empty());
    }

    #[test]
    fn stop_dictation_ignores_case_and_punctuation() {
        let r = router().route_full("Stop, Dictation!");
        assert!(r.commands.contains(&VoiceCommand::StopDictation));
        assert!(r.remaining.is_empty());
    }

    #[test]
    fn stop_listening_exact() {
        let r = router().route_full("stop listening");
        assert!(r.commands.contains(&VoiceCommand::StopDictation));
    }

    #[test]
    fn midsentence_stop_is_kept_as_dictation() {
        // Regression: the bare-"stop" substring command used to strip the word.
        let r = router().route_full("tell him to stop now");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "tell him to stop now");
    }

    #[test]
    fn bare_stop_is_not_a_command() {
        let r = router().route_full("stop");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "stop");
    }

    #[test]
    fn word_containing_command_is_not_matched() {
        // "nonstop" / "stopwatch" must not trip the stop command.
        let r = router().route_full("we ran nonstop");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "we ran nonstop");
    }

    #[test]
    fn command_only_when_whole_utterance() {
        let r = router().route_full("hello new line world");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "hello new line world");
    }

    #[test]
    fn new_line_exact() {
        let r = router().route_full("new line");
        assert!(r.commands.contains(&VoiceCommand::NewLine));
        assert!(r.remaining.is_empty());
    }

    #[test]
    fn new_paragraph_exact() {
        let r = router().route_full("New paragraph.");
        assert!(r.commands.contains(&VoiceCommand::NewParagraph));
    }

    #[test]
    fn list_item_exact() {
        let r = router().route_full("list item");
        assert!(r.commands.contains(&VoiceCommand::ListItem));
    }

    #[test]
    fn scratch_that_exact() {
        let r = router().route_full("scratch that");
        assert!(r.commands.contains(&VoiceCommand::ScratchThat));
    }

    #[test]
    fn switch_style_takes_rest_as_argument() {
        let r = router().route_full("switch to email");
        assert!(
            r.commands
                .iter()
                .any(|c| matches!(c, VoiceCommand::SwitchStyle(s) if s == "email"))
        );
        assert!(r.remaining.is_empty());
    }

    #[test]
    fn switch_preset_to_multiword_argument() {
        let r = router().route_full("switch preset to formal email");
        assert!(
            r.commands
                .iter()
                .any(|c| matches!(c, VoiceCommand::SwitchStyle(s) if s == "formal email"))
        );
    }

    #[test]
    fn insert_snippet_takes_rest_as_argument() {
        let r = router().route_full("insert greeting here");
        assert!(
            r.commands
                .iter()
                .any(|c| matches!(c, VoiceCommand::InsertSnippet(s) if s == "greeting here"))
        );
    }

    #[test]
    fn parametric_prefix_alone_is_not_a_command() {
        // "insert" with no argument is ordinary dictation, not a snippet command.
        let r = router().route_full("insert");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "insert");
    }

    #[test]
    fn no_commands_passthrough() {
        let r = router().route_full("hello world how are you");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "hello world how are you");
    }

    #[test]
    fn custom_command_exact_only() {
        let mut r = VoiceCommandRouter::new();
        r.add_custom("send message", VoiceCommand::StopDictation);
        assert!(
            r.route_full("send message")
                .commands
                .contains(&VoiceCommand::StopDictation)
        );
        // Not the whole utterance → not a command.
        assert!(r.route_full("please send message now").commands.is_empty());
    }
}
