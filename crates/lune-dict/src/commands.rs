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

/// Routes raw ASR text to voice commands before LLM post-processing.
///
/// There are no fixed built-in commands. Every phrase that once lived here was
/// either unimplemented or a formatting shortcut, and each one cost more in
/// swallowed dictation than it returned — a recognized command consumes the
/// whole utterance, so a false positive types nothing at all.
///
/// What remains is two parametric commands, `"switch to <preset>"` and
/// `"insert <snippet>"`, which match when the utterance *begins* with the exact
/// command words and has at least one more word; the remainder is the argument.
/// Callers may register their own exact-match phrases with [`Self::add_custom`],
/// which are recognized only when they are the entire utterance, compared
/// word-for-word and ignoring case and surrounding punctuation.
pub struct VoiceCommandRouter {
    /// User-defined phrase → command mappings, stored as normalized word lists.
    custom: Vec<(Vec<String>, VoiceCommand)>,
}

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

        // Custom commands: the whole utterance must equal the command.
        for (phrase, cmd) in &self.custom {
            if words == *phrase {
                return RouteResult {
                    commands: vec![cmd.clone()],
                    remaining: String::new(),
                };
            }
        }

        passthrough()
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
pub(crate) fn normalize_words(text: &str) -> Vec<String> {
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
    fn no_phrase_is_a_command_on_its_own() {
        // There are no fixed builtins left: an utterance is only ever a command
        // if it starts with a parametric prefix or matches a custom phrase.
        for utterance in ["scratch that", "stop dictation", "new line", "list item"] {
            let r = router().route_full(utterance);
            assert!(
                r.commands.is_empty(),
                "{utterance:?} must be dictated, not swallowed as a command"
            );
            assert_eq!(r.remaining, utterance);
        }
    }

    #[test]
    fn parametric_command_must_start_the_utterance() {
        // A prefix mid-sentence is ordinary dictation, so "please switch to
        // decaf" types instead of silently changing the style preset.
        let r = router().route_full("please switch to decaf");
        assert!(r.commands.is_empty());
        assert_eq!(r.remaining, "please switch to decaf");
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
        r.add_custom("send message", VoiceCommand::SwitchStyle("formal".into()));
        assert!(
            r.route_full("send message")
                .commands
                .contains(&VoiceCommand::SwitchStyle("formal".into()))
        );
        // Not the whole utterance → not a command.
        assert!(r.route_full("please send message now").commands.is_empty());
    }
}
