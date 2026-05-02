//! Residual annotation marker stripping for ASR transcript output.
//!
//! Marker formats may appear in local passthrough or remote-LLM output,
//! so keep one utility — [`strip_markers`] — that reduces them back to
//! bare words.

/// Strip residual annotation markers from text.
///
/// Handles three formats the marker pipeline may produce:
/// - `[?word?]` — legacy uncertainty marker, strips to `word`
/// - `[word]`   — partial strip by a small model, recovers `word`
/// - `{guess|alternate}` — runner-up marker, strips to `guess` (first option)
///
/// Unclosed brackets/braces are preserved verbatim so stripping is
/// lossless on partial output.
pub fn strip_markers(text: &str) -> String {
    if !text.contains('[') && !text.contains('{') {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            let mut inner = String::new();
            let mut found_close = false;
            for ic in chars.by_ref() {
                if ic == ']' {
                    found_close = true;
                    break;
                }
                inner.push(ic);
            }
            if found_close {
                let stripped = inner.strip_prefix('?').unwrap_or(&inner);
                let stripped = stripped.strip_suffix('?').unwrap_or(stripped);
                result.push_str(stripped);
            } else {
                result.push('[');
                result.push_str(&inner);
            }
        } else if c == '{' {
            let mut inner = String::new();
            let mut found_close = false;
            for ic in chars.by_ref() {
                if ic == '}' {
                    found_close = true;
                    break;
                }
                inner.push(ic);
            }
            if found_close {
                let guess = inner.split('|').next().unwrap_or(&inner);
                result.push_str(guess.trim());
            } else {
                result.push('{');
                result.push_str(&inner);
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Collapses unnecessary spaces within numeric values like thousands separators.
///
/// Removes spaces that appear between a number and the digits that follow
/// optional comma spacing, e.g. `2, 000` -> `2,000`, `2 , 100` -> `2,100`.
///
/// Spaces are preserved elsewhere in the text (e.g. `2 apples` remains unchanged),
/// and only when a digit is eventually followed by those spaces and another digit.
pub fn fix_all_number_spaces(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Detect a digit followed by a comma
        if i + 1 < chars.len() && chars[i].is_ascii_digit() && chars[i + 1] == ',' {
            // Look ahead to see if there is a digit eventually following spaces
            let mut j = i + 2;
            while j < chars.len() && chars[j] == ' ' {
                j += 1;
            }

            // If we found a digit after those spaces, collapse them
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push(chars[i]);
                result.push(',');
                i = j; // Jump past the comma and all spaces
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_markers_question_marks() {
        assert_eq!(strip_markers("hello [?world?] there"), "hello world there");
    }

    #[test]
    fn strip_markers_bare_brackets() {
        assert_eq!(strip_markers("the [most] important"), "the most important");
    }

    #[test]
    fn strip_markers_no_brackets() {
        assert_eq!(strip_markers("hello world"), "hello world");
    }

    #[test]
    fn strip_markers_unclosed_bracket() {
        assert_eq!(strip_markers("hello [world"), "hello [world");
    }

    #[test]
    fn strip_markers_mixed() {
        assert_eq!(
            strip_markers("the [?oth?] service [provided] for [?them?]"),
            "the oth service provided for them"
        );
    }

    #[test]
    fn strip_markers_runner_up_picks_first_word() {
        assert_eq!(strip_markers("the {fox|box} jumped"), "the fox jumped");
    }

    #[test]
    fn strip_markers_runner_up_mixed_with_legacy() {
        assert_eq!(
            strip_markers("the {fox|box} [?and?] the [?dog?]"),
            "the fox and the dog"
        );
    }

    #[test]
    fn strip_markers_unclosed_brace_preserved() {
        assert_eq!(strip_markers("the {fox|box"), "the {fox|box");
    }

    #[test]
    fn strip_markers_brace_without_pipe() {
        assert_eq!(strip_markers("the {fox} jumped"), "the fox jumped");
    }

    #[test]
    fn strip_markers_no_brackets_or_braces() {
        assert_eq!(strip_markers("hello world"), "hello world");
    }
}
