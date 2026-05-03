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

// Static builtin patterns — sorted longest-first to prevent short-match shadowing.
// "stop dictation" must precede "stop" so the longer match wins.
static BUILTINS: &[(&str, VoiceCommand)] = &[
    ("stop listening", VoiceCommand::StopDictation),
    ("stop dictation", VoiceCommand::StopDictation),
    ("new paragraph", VoiceCommand::NewParagraph),
    ("scratch that", VoiceCommand::ScratchThat),
    ("list item", VoiceCommand::ListItem),
    ("new line", VoiceCommand::NewLine),
    ("stop", VoiceCommand::StopDictation),
];

// Parametric prefix tables — stored as (prefix_str, prefix_len, variant_tag).
// Enum tag avoids dynamic dispatch / Box overhead in the hot loop.
#[derive(Copy, Clone)]
enum ParametricKind {
    SwitchStyle,
    InsertSnippet,
}

static PARAMETRIC_PREFIXES: &[(&str, ParametricKind)] = &[
    ("switch preset to ", ParametricKind::SwitchStyle),
    ("switch to ", ParametricKind::SwitchStyle),
    ("expand snippet ", ParametricKind::InsertSnippet),
    ("insert ", ParametricKind::InsertSnippet),
];

// Delimiter set used by parametric scanning — avoids repeated char array construction.
const PARAM_DELIMITERS: [char; 3] = [',', '.', ';'];

impl VoiceCommandRouter {
    pub fn new() -> Self {
        Self { custom: Vec::new() }
    }

    pub fn add_custom(&mut self, phrase: impl Into<String>, command: VoiceCommand) {
        self.custom.push((phrase.into().to_lowercase(), command));
    }

    /// Route `text`, returning detected commands and the remaining text.
    /// Commands are detected left-to-right; each matched span is removed once.
    pub fn route(&self, text: &str) -> RouteResult {
        self.route_inner(text, false)
    }

    /// Full route including parametric patterns.
    pub fn route_full(&self, text: &str) -> RouteResult {
        self.route_inner(text, true)
    }

    fn route_inner(&self, text: &str, parametric: bool) -> RouteResult {
        let lower = text.to_lowercase();
        let cap = BUILTINS.len()
            + self.custom.len()
            + if parametric {
                PARAMETRIC_PREFIXES.len()
            } else {
                0
            };
        let mut commands: Vec<VoiceCommand> = Vec::with_capacity(cap);
        let mut removals: Vec<(usize, usize)> = Vec::with_capacity(cap);

        if parametric {
            parametric_matches(&lower, &mut commands, &mut removals);
        }
        self.match_exact(&lower, &mut commands, &mut removals);

        removals.sort_unstable_by_key(|&(s, _)| s);
        let remaining = rebuild_remaining(text, &removals);

        RouteResult {
            commands,
            remaining,
        }
    }

    fn match_exact(
        &self,
        lower: &str,
        commands: &mut Vec<VoiceCommand>,
        removals: &mut Vec<(usize, usize)>,
    ) {
        for (phrase, cmd) in self.all_patterns() {
            let plen = phrase.len();
            let mut search_from = 0usize;
            while search_from + plen <= lower.len() {
                match lower[search_from..].find(phrase) {
                    None => break,
                    Some(pos) => {
                        let start = search_from + pos;
                        let end = start + plen;
                        // Overlap check — linear over removals; expected small (<16).
                        if span_overlaps(removals, start, end) {
                            search_from = end;
                            continue;
                        }
                        commands.push(cmd.clone());
                        removals.push((start, end));
                        search_from = end;
                    }
                }
            }
        }
    }

    fn all_patterns(&self) -> impl Iterator<Item = (&str, VoiceCommand)> + '_ {
        BUILTINS
            .iter()
            .map(|(p, c)| (*p, c.clone()))
            .chain(self.custom.iter().map(|(p, c)| (p.as_str(), c.clone())))
    }
}

impl Default for VoiceCommandRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free functions — kept out of impl to allow inlining across call sites.
// ---------------------------------------------------------------------------

/// Returns true iff [start, end) overlaps any existing removal span.
#[inline(always)]
fn span_overlaps(removals: &[(usize, usize)], start: usize, end: usize) -> bool {
    removals.iter().any(|&(rs, re)| rs < end && re > start)
}

/// Scan `lower` for all parametric patterns; push matched spans + commands.
fn parametric_matches(
    lower: &str,
    commands: &mut Vec<VoiceCommand>,
    removals: &mut Vec<(usize, usize)>,
) {
    for &(prefix, kind) in PARAMETRIC_PREFIXES {
        let plen = prefix.len();
        let mut pos = 0usize;
        while pos + plen <= lower.len() {
            match lower[pos..].find(prefix) {
                None => break,
                Some(idx) => {
                    let start = pos + idx;
                    let arg_start = start + plen;
                    let arg_end = lower[arg_start..]
                        .find(PARAM_DELIMITERS)
                        .map(|i| arg_start + i)
                        .unwrap_or(lower.len());
                    let arg = lower[arg_start..arg_end].trim();
                    if !arg.is_empty() && !span_overlaps(removals, start, arg_end) {
                        let cmd = match kind {
                            ParametricKind::SwitchStyle => {
                                VoiceCommand::SwitchStyle(arg.to_string())
                            }
                            ParametricKind::InsertSnippet => {
                                VoiceCommand::InsertSnippet(arg.to_string())
                            }
                        };
                        commands.push(cmd);
                        removals.push((start, arg_end));
                    }
                    pos = arg_end;
                }
            }
        }
    }
}

/// Reconstruct the non-removed portions of `text` and collapse whitespace in
/// a single pass, avoiding an extra allocation for the intermediate string.
fn rebuild_remaining(text: &str, sorted_removals: &[(usize, usize)]) -> String {
    // Estimate capacity: full text minus removed bytes is a tight upper bound.
    let removed: usize = sorted_removals.iter().map(|(s, e)| e - s).sum();
    let cap = text.len().saturating_sub(removed);
    let mut out = String::with_capacity(cap);
    let mut cursor = 0usize;

    for &(start, end) in sorted_removals {
        if start > cursor {
            collapse_push(&mut out, &text[cursor..start]);
        }
        cursor = end;
    }
    if cursor < text.len() {
        collapse_push(&mut out, &text[cursor..]);
    }

    // Trim trailing space that collapse_push may have left.
    if out.ends_with(' ') {
        out.pop();
    }
    // Trim leading space.
    if out.starts_with(' ') {
        out.remove(0);
    }
    out
}

/// Push `s` into `out` collapsing runs of whitespace to a single space,
/// tracking whether `out` already ends with a space to avoid doubling.
#[inline]
fn collapse_push(out: &mut String, s: &str) {
    let prev_space = out.ends_with(' ');
    let mut in_space = prev_space;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_space {
                out.push(' ');
                in_space = true;
            }
        } else {
            out.push(c);
            in_space = false;
        }
    }
}

/// Collapse whitespace in a standalone string (used by tests).
#[cfg(test)]
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    collapse_push(&mut out, s);
    if out.ends_with(' ') {
        out.pop();
    }
    if out.starts_with(' ') {
        out.remove(0);
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
