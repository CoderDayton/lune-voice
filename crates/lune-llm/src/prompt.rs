//! Prompt output cleanup utilities.

/// Collapses unnecessary spaces within numeric values like thousands separators.
///
/// Handles:
/// - Comma spacing: `2, 000` → `2,000`
/// - European thousand sep: `2 000` → `2000` (when 3+ digit groups)
/// - Decimal periods: `3 . 14` → `3.14`
///
/// Spaces are preserved elsewhere in the text (e.g. `2 apples` remains unchanged).
pub fn fix_all_number_spaces(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Pattern 0: digit + space + comma + [spaces] + digit
        if i > 0
            && i + 2 < chars.len()
            && chars[i - 1].is_ascii_digit()
            && chars[i] == ' '
            && chars[i + 1] == ','
        {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push(',');
                i = j;
                continue;
            }
        }

        // Pattern 0b: digit + space + period + [spaces] + digit
        if i > 0
            && i + 2 < chars.len()
            && chars[i - 1].is_ascii_digit()
            && chars[i] == ' '
            && chars[i + 1] == '.'
        {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push('.');
                i = j;
                continue;
            }
        }

        // Pattern 1: digit + comma + [spaces] + digit -> collapse spaces
        if i + 1 < chars.len() && chars[i].is_ascii_digit() && chars[i + 1] == ',' {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push(chars[i]);
                result.push(',');
                i = j;
                continue;
            }
        }

        // Pattern 2: digit + comma + space[s] + digit
        if i + 2 < chars.len()
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars[i] == ','
            && chars[i + 1] == ' '
            && chars[i + 2].is_ascii_digit()
        {
            result.push(',');
            i = skip_spaces(&chars, i + 2);
            continue;
        }

        // Pattern 3: digit + period + [spaces] + digit -> collapse spaces (decimals)
        if i + 1 < chars.len() && chars[i].is_ascii_digit() && chars[i + 1] == '.' {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push(chars[i]);
                result.push('.');
                i = j;
                continue;
            }
        }

        // Pattern 4: digit + period + space[s] + digit
        if i + 2 < chars.len()
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars[i] == '.'
            && chars[i + 1] == ' '
            && chars[i + 2].is_ascii_digit()
        {
            result.push('.');
            i = skip_spaces(&chars, i + 2);
            continue;
        }

        // Pattern 5: European thousand separator (digit + spaces + 3 digits)
        if i + 1 < chars.len() && chars[i].is_ascii_digit() {
            let j = skip_spaces(&chars, i + 1);
            if j + 2 < chars.len()
                && chars[j].is_ascii_digit()
                && chars[j + 1].is_ascii_digit()
                && chars[j + 2].is_ascii_digit()
                && (j - i - 1) > 0
            // there were spaces
            {
                result.push(chars[i]);
                i = j;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Skip whitespace characters, returning the index of the first non-space char.
fn skip_spaces(chars: &[char], start: usize) -> usize {
    let mut j = start;
    while j < chars.len() && chars[j] == ' ' {
        j += 1;
    }
    j
}

/// True when the model emitted meta-commentary instead of clean text.
pub fn contains_forbidden_output(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        "here is",
        "corrected transcript",
        "the corrected",
        "i changed",
        "i fixed",
        "input:",
        "output:",
        "possibly misheard",
    ]
    .iter()
    .any(|m| lower.contains(m))
}

/// Tag pairs a reasoning model may wrap its private thinking in. Matched
/// without ASCII case, so `<THINK>` counts too.
const REASONING_TAGS: [(&str, &str); 4] = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
];

/// Harmony (gpt-oss) splits a response into channels and marks the answer as
/// the `final` one; everything in the other channels is reasoning.
const HARMONY_FINAL: &str = "<|channel|>final<|message|>";
const HARMONY_CHANNEL: &str = "<|channel|>";

/// Byte offset of `needle` in `hay`, compared without ASCII case. Every marker
/// here is ASCII, so a match can never land inside a multi-byte character.
fn find_ascii_ci(hay: &str, needle: &str) -> Option<usize> {
    let (hay, needle) = (hay.as_bytes(), needle.as_bytes());
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| hay[i..i + needle.len()].eq_ignore_ascii_case(needle))
}

/// Byte offset just past the close tag that ends the block opened immediately
/// before `body`. Nested openers of the same spelling raise the depth, so an
/// inner close does not end the outer block — a model that re-opens `<think>`
/// inside its own reasoning would otherwise leak the tail of it. `None` means
/// the block never closed.
fn find_block_end(body: &str, open: &str, close: &str) -> Option<usize> {
    let mut depth = 1usize;
    let mut pos = 0usize;
    while depth > 0 {
        let next_close = find_ascii_ci(&body[pos..], close)?;
        match find_ascii_ci(&body[pos..], open) {
            Some(next_open) if next_open < next_close => {
                depth += 1;
                pos += next_open + open.len();
            }
            _ => {
                depth -= 1;
                pos += next_close + close.len();
            }
        }
    }
    Some(pos)
}

/// Strip a model's private reasoning, leaving only the answer.
///
/// Reasoning delivered in a separate `reasoning_content` field never reaches
/// this function — the backend reads `message.content` alone, so that shape is
/// discarded by construction. Within the content there are three shapes:
///
/// - a wrapped block: `<think>…</think>` and its `<thinking>` / `<reasoning>` /
///   `<thought>` spellings, in any case, repeated if the model emits several;
/// - an unwrapped block ending at the closing tag, which is what a chat template
///   that pre-fills the opening tag produces — the model never emits the opener;
/// - harmony channels, where only `<|channel|>final<|message|>` is the answer.
///
/// Reasoning that never closes yields `""`, which the pipeline treats as "keep
/// the text we already had" rather than delivering half a thought.
pub fn strip_reasoning(text: &str) -> &str {
    let text = text.trim();

    if let Some(i) = find_ascii_ci(text, HARMONY_FINAL) {
        let answer = &text[i + HARMONY_FINAL.len()..];
        // The final channel runs to the next control token, when the endpoint
        // leaves one on the end (`<|end|>`, `<|return|>`).
        return match answer.find("<|") {
            Some(end) => answer[..end].trim(),
            None => answer.trim(),
        };
    }
    if find_ascii_ci(text, HARMONY_CHANNEL).is_some() {
        // Channels, but no final one: nothing here is an answer.
        return "";
    }

    let mut rest = text;
    while let Some((open, close)) = REASONING_TAGS.iter().copied().find(|(open, _)| {
        rest.get(..open.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(open))
    }) {
        let Some(end) = find_block_end(&rest[open.len()..], open, close) else {
            return "";
        };
        rest = rest[open.len() + end..].trim_start();
    }

    if rest.len() == text.len()
        && let Some((i, close_len)) = REASONING_TAGS
            .iter()
            .filter_map(|(_, close)| find_ascii_ci(rest, close).map(|i| (i, close.len())))
            .min_by_key(|&(i, _)| i)
    {
        // No opening tag, but the reasoning closed: a pre-filled template. Drop
        // everything up to that close.
        rest = rest[i + close_len..].trim_start();
    }

    rest.trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_all_number_spaces_comma() {
        assert_eq!(fix_all_number_spaces("2, 000"), "2,000");
        assert_eq!(fix_all_number_spaces("2 , 100"), "2,100");
    }

    #[test]
    fn fix_all_number_spaces_decimal() {
        assert_eq!(fix_all_number_spaces("3 . 14"), "3.14");
        assert_eq!(fix_all_number_spaces("pi is 3 . 14159"), "pi is 3.14159");
    }

    #[test]
    fn fix_all_number_spaces_european() {
        // European: 2 000 = 2000
        assert_eq!(fix_all_number_spaces("2 000 000"), "2000000");
    }

    #[test]
    fn fix_all_number_spaces_preserves_text() {
        assert_eq!(fix_all_number_spaces("2 apples"), "2 apples");
        assert_eq!(fix_all_number_spaces("hello, 2 apples"), "hello, 2 apples");
        assert_eq!(
            fix_all_number_spaces("hello , 2 apples"),
            "hello , 2 apples"
        );
    }

    #[test]
    fn strip_reasoning_passthrough_when_no_tags() {
        assert_eq!(strip_reasoning("just clean text"), "just clean text");
    }

    #[test]
    fn strip_reasoning_removes_block() {
        assert_eq!(
            strip_reasoning("<think>reasoning here</think>Hello world"),
            "Hello world"
        );
    }

    #[test]
    fn strip_reasoning_accepts_every_tag_spelling() {
        for (open, close) in REASONING_TAGS {
            assert_eq!(
                strip_reasoning(&format!("{open}private{close}Answer")),
                "Answer",
                "{open}"
            );
        }
    }

    #[test]
    fn strip_reasoning_ignores_tag_case() {
        assert_eq!(strip_reasoning("<THINK>private</Think>Answer"), "Answer");
    }

    #[test]
    fn strip_reasoning_removes_repeated_blocks() {
        assert_eq!(
            strip_reasoning("<think>one</think>\n<thinking>two</thinking>\nAnswer"),
            "Answer"
        );
    }

    #[test]
    fn strip_reasoning_removes_nested_blocks() {
        // A model that re-opens the tag inside its own reasoning: the inner
        // close must not be mistaken for the end of the outer block.
        assert_eq!(
            strip_reasoning("<think>one<think>two</think>three</think>Answer"),
            "Answer"
        );
    }

    #[test]
    fn strip_reasoning_unclosed_nested_block_is_empty() {
        assert_eq!(strip_reasoning("<think>one<think>two</think>three"), "");
    }

    #[test]
    fn strip_reasoning_handles_prefilled_opening_tag() {
        // A chat template that pre-fills `<think>` means the model only ever
        // emits the close.
        assert_eq!(
            strip_reasoning("weighing the options</think>Answer"),
            "Answer"
        );
    }

    #[test]
    fn strip_reasoning_opening_only_is_empty() {
        assert_eq!(strip_reasoning("<think>still thinking"), "");
        assert_eq!(strip_reasoning("<reasoning>still thinking"), "");
    }

    #[test]
    fn strip_reasoning_keeps_only_harmony_final_channel() {
        assert_eq!(
            strip_reasoning(
                "<|channel|>analysis<|message|>weighing it<|end|>\
                 <|start|>assistant<|channel|>final<|message|>Answer<|return|>"
            ),
            "Answer"
        );
    }

    #[test]
    fn strip_reasoning_harmony_without_final_channel_is_empty() {
        assert_eq!(
            strip_reasoning("<|channel|>analysis<|message|>still weighing it"),
            ""
        );
    }

    #[test]
    fn strip_reasoning_trims_whitespace_around_answer() {
        assert_eq!(
            strip_reasoning("  <think>x</think>\n  Final answer  "),
            "Final answer"
        );
    }

    #[test]
    fn strip_reasoning_empty_string() {
        assert_eq!(strip_reasoning(""), "");
    }

    #[test]
    fn contains_forbidden_output_detects_meta() {
        assert!(contains_forbidden_output("Here is the corrected text"));
        assert!(contains_forbidden_output("Corrected transcript: hello"));
        assert!(contains_forbidden_output("Output: hello"));
    }

    #[test]
    fn contains_forbidden_output_allows_clean_text() {
        assert!(!contains_forbidden_output("The quick brown fox jumps."));
    }
}
