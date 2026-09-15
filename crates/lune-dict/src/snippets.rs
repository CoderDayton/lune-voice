// SPDX-License-Identifier: GPL-3.0-or-later

//! Voice snippet matching and placeholder rendering.
//!
//! A snippet is a user-defined template ([`lune_core::dict::Snippet`])
//! expanded by voice in one of two ways: the whole utterance equals the
//! snippet's trigger phrase, or the utterance is the parametric command
//! `insert <name>` / `expand snippet <name>` (routed as
//! `VoiceCommand::InsertSnippet` by [`crate::commands::VoiceCommandRouter`]),
//! where `<name>` names the trigger. Matching mirrors the command router:
//! lowercase, punctuation-free word comparison, so saying `"Signature!"`
//! matches the trigger `"signature"`. Name resolution additionally tolerates
//! one ASR mishear per word (edit distance ≤ 1), but only when that resolves
//! to a *unique* snippet — a garbled name must never expand the wrong
//! template.

use lune_core::dict::Snippet;

use crate::commands::normalize_words;
use crate::lev_le1_ci;

/// Find the snippet whose trigger phrase is the entire utterance
/// (case- and punctuation-insensitive, word for word).
pub fn match_utterance<'a>(snippets: &'a [Snippet], utterance: &str) -> Option<&'a Snippet> {
    let words = normalize_words(utterance);
    if words.is_empty() {
        return None;
    }
    snippets
        .iter()
        .find(|s| normalize_words(&s.trigger_phrase) == words)
}

/// Whether two trigger phrases would match the same spoken utterance.
///
/// Matching normalizes (lowercase, punctuation dropped, whitespace collapsed),
/// so a plain string comparison is not enough to decide whether a new trigger
/// shadows an existing one: `"Meeting, header."` and `"meeting header"` are
/// different strings but the same phrase, and [`match_utterance`] would
/// silently pick whichever came first. This is what the uniqueness check on
/// the save path has to use.
pub fn triggers_collide(a: &str, b: &str) -> bool {
    let a = normalize_words(a);
    !a.is_empty() && a == normalize_words(b)
}

/// Resolve the argument of an `insert <name>` command to a snippet.
///
/// Exact normalized match wins. Failing that, a fuzzy pass accepts triggers
/// with the same word count where every word is within edit distance 1 of the
/// spoken word — but only when exactly one snippet qualifies; an ambiguous
/// fuzzy hit returns `None` rather than guessing.
pub fn resolve_name<'a>(snippets: &'a [Snippet], name: &str) -> Option<&'a Snippet> {
    let words = normalize_words(name);
    if words.is_empty() {
        return None;
    }

    if let Some(exact) = snippets
        .iter()
        .find(|s| normalize_words(&s.trigger_phrase) == words)
    {
        return Some(exact);
    }

    let mut fuzzy = snippets.iter().filter(|s| {
        let trigger = normalize_words(&s.trigger_phrase);
        trigger.len() == words.len() && trigger.iter().zip(&words).all(|(t, w)| lev_le1_ci(t, w))
    });
    let first = fuzzy.next()?;
    fuzzy.next().is_none().then_some(first)
}

/// Caller-resolved values for the dynamic placeholders. Plain strings, so this
/// crate needs no clock or clipboard access and rendering is exact under test.
#[derive(Debug, Clone, Default)]
pub struct PlaceholderValues {
    pub date: String,
    pub time: String,
    pub datetime: String,
    /// `None` when the clipboard is empty or unreadable — renders as nothing.
    pub clipboard: Option<String>,
}

/// True when `body` references `{clipboard}` — lets the caller skip the
/// (potentially slow, subprocess-backed) clipboard read otherwise.
pub fn needs_clipboard(body: &str) -> bool {
    tokens(body).any(|tok| tok.eq_ignore_ascii_case("clipboard"))
}

/// Render a snippet body, substituting `{date}`, `{time}`, `{datetime}` and
/// `{clipboard}` (token names are case-insensitive). Anything else in braces
/// passes through verbatim, so ordinary `{}` text in a template survives.
pub fn render_body(body: &str, values: &PlaceholderValues) -> String {
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find('}') {
            Some(close) => {
                let token = &after[..close];
                match resolve_token(token, values) {
                    Some(value) => out.push_str(value),
                    None => {
                        // Unknown token: keep the braces and their content.
                        out.push('{');
                        out.push_str(token);
                        out.push('}');
                    }
                }
                rest = &after[close + 1..];
            }
            None => {
                // Unclosed brace: emit the remainder as-is.
                out.push_str(&rest[open..]);
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

fn resolve_token<'a>(token: &str, values: &'a PlaceholderValues) -> Option<&'a str> {
    if token.eq_ignore_ascii_case("date") {
        Some(&values.date)
    } else if token.eq_ignore_ascii_case("time") {
        Some(&values.time)
    } else if token.eq_ignore_ascii_case("datetime") {
        Some(&values.datetime)
    } else if token.eq_ignore_ascii_case("clipboard") {
        Some(values.clipboard.as_deref().unwrap_or(""))
    } else {
        None
    }
}

/// Iterate the brace-delimited token names in `body`.
///
/// Mirrors [`render_body`]'s left-to-right scan exactly — first `{`, then the
/// next `}` after it, then resume past that `}`. Splitting on every `{`
/// independently would report tokens `render_body` never substitutes (a
/// nested `{{clipboard}}` expands to nothing), making [`needs_clipboard`] read
/// the clipboard for a body that does not use it.
fn tokens(body: &str) -> impl Iterator<Item = &str> {
    let mut rest = body;
    std::iter::from_fn(move || {
        let open = rest.find('{')?;
        let after = &rest[open + 1..];
        let close = after.find('}')?;
        let token = &after[..close];
        rest = &after[close + 1..];
        Some(token)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snip(id: &str, trigger: &str, body: &str) -> Snippet {
        Snippet {
            id: id.into(),
            trigger_phrase: trigger.into(),
            body: body.into(),
        }
    }

    fn library() -> Vec<Snippet> {
        vec![
            snip("1", "signature", "Best regards,\nDayton"),
            snip("2", "meeting header", "## Meeting — {date}\nAttendees:"),
            snip("3", "meeting footer", "Adjourned at {time}."),
        ]
    }

    fn values() -> PlaceholderValues {
        PlaceholderValues {
            date: "July 8, 2026".into(),
            time: "14:30".into(),
            datetime: "July 8, 2026 at 14:30".into(),
            clipboard: Some("PASTE".into()),
        }
    }

    // ── matching ─────────────────────────────────────────────────────────

    #[test]
    fn utterance_matches_trigger_ignoring_case_and_punctuation() {
        let lib = library();
        let hit = match_utterance(&lib, "Signature!").expect("match");
        assert_eq!(hit.id, "1");
        let hit = match_utterance(&lib, "Meeting, header.").expect("match");
        assert_eq!(hit.id, "2");
    }

    #[test]
    fn utterance_with_extra_words_does_not_match() {
        let lib = library();
        assert!(match_utterance(&lib, "my signature please").is_none());
        assert!(match_utterance(&lib, "").is_none());
    }

    #[test]
    fn resolve_name_exact() {
        let lib = library();
        assert_eq!(resolve_name(&lib, "meeting header").unwrap().id, "2");
    }

    #[test]
    fn resolve_name_tolerates_one_edit_per_word() {
        let lib = library();
        // ASR mishears: "signture" (deletion), "meating header" (substitution).
        assert_eq!(resolve_name(&lib, "signture").unwrap().id, "1");
        assert_eq!(resolve_name(&lib, "meating header").unwrap().id, "2");
    }

    #[test]
    fn resolve_name_refuses_ambiguous_fuzzy() {
        let lib = vec![snip("a", "note", "A"), snip("b", "nope", "B")];
        // "nose" is one edit from both "note" and "nope" — ambiguous, so it
        // must not expand either.
        assert!(resolve_name(&lib, "nose").is_none());
    }

    #[test]
    fn triggers_collide_across_punctuation_and_case() {
        // Both normalize to ["meeting", "header"], so saving the second would
        // shadow the first at match time.
        assert!(triggers_collide("Meeting, header.", "meeting header"));
        assert!(triggers_collide("signature", "  Signature!  "));
        assert!(!triggers_collide("meeting header", "meeting footer"));
        // A trigger that normalizes to nothing can never be spoken, so it
        // collides with nothing (the save path rejects it separately).
        assert!(!triggers_collide("!!!", "!!!"));
    }

    #[test]
    fn resolve_name_prefers_exact_over_fuzzy() {
        let lib = vec![snip("a", "note", "A"), snip("b", "notes", "B")];
        assert_eq!(resolve_name(&lib, "notes").unwrap().id, "b");
    }

    // ── rendering ────────────────────────────────────────────────────────

    #[test]
    fn renders_known_placeholders() {
        let out = render_body("## Meeting — {date}\nAt {time} ({datetime})", &values());
        assert_eq!(
            out,
            "## Meeting — July 8, 2026\nAt 14:30 (July 8, 2026 at 14:30)"
        );
    }

    #[test]
    fn placeholder_names_are_case_insensitive() {
        assert_eq!(
            render_body("{DATE} {Time}", &values()),
            "July 8, 2026 14:30"
        );
    }

    #[test]
    fn clipboard_renders_content_or_empty() {
        assert_eq!(render_body("[{clipboard}]", &values()), "[PASTE]");
        let empty = PlaceholderValues {
            clipboard: None,
            ..values()
        };
        assert_eq!(render_body("[{clipboard}]", &empty), "[]");
    }

    #[test]
    fn unknown_tokens_and_stray_braces_pass_through() {
        assert_eq!(
            render_body("fn main() { {x} }", &values()),
            "fn main() { {x} }"
        );
        assert_eq!(render_body("open { brace", &values()), "open { brace");
        assert_eq!(render_body("{unclosed", &values()), "{unclosed");
    }

    #[test]
    fn needs_clipboard_detects_token_lazily() {
        assert!(needs_clipboard("paste: {clipboard}"));
        assert!(needs_clipboard("{CLIPBOARD}"));
        assert!(!needs_clipboard("just {date} here"));
        assert!(!needs_clipboard("brace { but no token"));
    }

    #[test]
    fn needs_clipboard_agrees_with_render_body_on_nested_braces() {
        // render_body leaves these verbatim, so no clipboard read is owed.
        for body in ["{{clipboard}}", "{a{clipboard}", "fn f() { {clipboard"] {
            assert!(!needs_clipboard(body), "should not need clipboard: {body}");
            assert_eq!(
                render_body(body, &values()),
                body,
                "body should render verbatim: {body}"
            );
        }
    }
}
