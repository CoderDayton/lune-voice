//! Prompt output cleanup utilities.

/// Collapses unnecessary spaces within numeric values like thousands separators.
///
/// Handles:
/// - Comma spacing: `2, 000` → `2,000`
/// - European thousand sep: `2 000 000` → `2000000` (only when the whole run
///   groups as a number: 1-3 digits with no leading zero, then groups of
///   exactly 3)
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

        // Pattern 5: European thousand separator (digit + single space +
        // 3-digit groups). Only a run whose *whole* grouping is a valid number
        // is collapsed — first group 1–3 digits with no leading zero, every
        // later group exactly 3 — so a phone number / part number / ID like
        // "555 123 4567" (3-3-4) or "0 800 123 456" (leading zero) is left
        // untouched while "1 000 000" (1-3-3) collapses to "1000000".
        if i + 1 < chars.len() && chars[i].is_ascii_digit() && is_digit_run_start(&chars, i) {
            let run_end = find_digit_group_run(&chars, i);
            if run_end > i {
                let groups: Vec<&[char]> = chars[i..run_end].split(|&c| c == ' ').collect();
                let valid_number = groups.len() >= 2
                    && (1..=3).contains(&groups[0].len())
                    // A written number never leads with a zero, but a dialling
                    // prefix (`0 800 …`, `07 700 …`) always does — the only
                    // signal separating the two shapes.
                    && groups[0][0] != '0'
                    && groups[1..].iter().all(|g| g.len() == 3);
                if valid_number {
                    for g in &groups {
                        result.extend(g.iter().copied());
                    }
                    i = run_end;
                    continue;
                }
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

/// End (exclusive) of the maximal run of digit groups separated by *single*
/// spaces starting at `start`. A double space is a word boundary, not a number
/// grouping, so it stops the run.
fn find_digit_group_run(chars: &[char], start: usize) -> usize {
    let mut k = start;
    while k < chars.len() && chars[k].is_ascii_digit() {
        k += 1;
    }
    while k + 1 < chars.len() && chars[k] == ' ' && chars[k + 1].is_ascii_digit() {
        k += 1;
        while k < chars.len() && chars[k].is_ascii_digit() {
            k += 1;
        }
    }
    k
}

/// True when `start` begins a maximal digit-group run rather than sitting
/// inside one. The grouping check above only means anything at the true start:
/// re-entering one character later would let `1234 567` — a 4-digit lead group,
/// so not a thousands separator — collapse anyway via its `234 567` suffix.
fn is_digit_run_start(chars: &[char], start: usize) -> bool {
    match start.checked_sub(1) {
        None => true,
        Some(p) => {
            !chars[p].is_ascii_digit()
                && !(chars[p] == ' ' && p > 0 && chars[p - 1].is_ascii_digit())
        }
    }
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
    // `input:`/`output:` are only treated as label echoes when they begin a
    // line — the few-shot training format puts them at line start. A
    // mid-sentence "input:" or "output:" in the user's own transcript (e.g.
    // "I said input: the password") must not be rejected.
    let label_echo = lower.lines().any(|line| {
        let t = line.trim_start();
        t.starts_with("input:") || t.starts_with("output:")
    });
    let phrase_echo = [
        "here is",
        "corrected transcript",
        "the corrected",
        "i changed",
        "i fixed",
        "possibly misheard",
    ]
    .iter()
    .any(|m| lower.contains(m));
    label_echo || phrase_echo
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
/// Wrapped blocks are removed wherever they sit: at the head, or spliced out of
/// the middle of an answer.
///
/// Reasoning that never closes yields `""`, which the pipeline treats as "keep
/// the text we already had" rather than delivering half a thought.
pub fn strip_reasoning(text: &str) -> String {
    let text = text.trim();

    if let Some(i) = find_ascii_ci(text, HARMONY_FINAL) {
        let answer = &text[i + HARMONY_FINAL.len()..];
        // The final channel runs to the next control token, when the endpoint
        // leaves one on the end (`<|end|>`, `<|return|>`).
        return match answer.find("<|") {
            Some(end) => answer[..end].trim().to_owned(),
            None => answer.trim().to_owned(),
        };
    }
    if find_ascii_ci(text, HARMONY_CHANNEL).is_some() {
        // Channels, but no final one: nothing here is an answer.
        return String::new();
    }

    // Strip leading reasoning blocks (any spelling, repeated). A leading block
    // that never closes means the model is still thinking: no answer.
    let mut rest = text;
    while let Some((open, close)) = REASONING_TAGS.iter().copied().find(|(open, _)| {
        rest.get(..open.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(open))
    }) {
        let Some(end) = find_block_end(&rest[open.len()..], open, close) else {
            return String::new();
        };
        rest = rest[open.len() + end..].trim_start();
    }

    // No opener at the head, but a close tag present (and no angle opener
    // anywhere): a chat template that pre-fills the opening tag, so the model
    // only ever emits the close. Drop everything up to that first close.
    if rest.len() == text.len()
        && !REASONING_TAGS
            .iter()
            .any(|(open, _)| find_ascii_ci(rest, open).is_some())
        && let Some((i, close_len)) = REASONING_TAGS
            .iter()
            .filter_map(|(_, close)| find_ascii_ci(rest, close).map(|i| (i, close.len())))
            .min_by_key(|&(i, _)| i)
    {
        rest = rest[i + close_len..].trim_start();
    }

    // Splice reasoning blocks that appear anywhere in the answer, not just at
    // the head. An unclosed block means the model never finished: nothing
    // after it is an answer.
    let mut out = String::with_capacity(rest.len());
    let mut pos = 0usize;
    while let Some((rel, open, close)) = REASONING_TAGS
        .iter()
        .copied()
        .filter_map(|(open, close)| find_ascii_ci(&rest[pos..], open).map(|i| (i, open, close)))
        .min_by_key(|&(i, _, _)| i)
    {
        let block_start = pos + rel;
        out.push_str(&rest[pos..block_start]);
        let Some(end) = find_block_end(&rest[block_start + open.len()..], open, close) else {
            return out.trim().to_owned();
        };
        pos = block_start + open.len() + end;
        // Cutting the block out must not leave the spaces that flanked it
        // doubled up in the text we type into the user's document.
        let kept = out.trim_end_matches(' ').len();
        if kept < out.len() && rest[pos..].starts_with(' ') {
            out.truncate(kept);
        }
    }
    out.push_str(&rest[pos..]);
    out.trim().to_owned()
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
        assert_eq!(fix_all_number_spaces("2 000 000"), "2000000");
        // Every group boundary in the run collapses, not just the first.
        assert_eq!(fix_all_number_spaces("1 234 567 890"), "1234567890");
        // The run is bounded by its surrounding text.
        assert_eq!(
            fix_all_number_spaces("we shipped 2 000 000 units"),
            "we shipped 2000000 units"
        );
        // A double space is a word boundary, not a grouping separator.
        assert_eq!(fix_all_number_spaces("2  000"), "2  000");
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

    #[test]
    fn contains_forbidden_output_allows_label_like_text_mid_sentence() {
        // A transcript that legitimately contains "input:" or "output:" in the
        // middle of a sentence is not a model echoing the few-shot labels.
        assert!(!contains_forbidden_output("I said input: the password"));
        assert!(!contains_forbidden_output("the output: was correct"));
        assert!(!contains_forbidden_output(
            "Feed the input: and check the output:"
        ));
    }

    #[test]
    fn contains_forbidden_output_still_detects_line_started_labels() {
        // The model echoing the few-shot `Input:`/`Output:` labels puts them at
        // the start of a line, which is the shape we do want to reject.
        assert!(contains_forbidden_output("Output: hello"));
        assert!(contains_forbidden_output("some text\nInput: hello"));
    }

    #[test]
    fn fix_all_number_spaces_keeps_phone_like_groups() {
        // A spaced 3-digit group that is not a thousands separator (phone
        // numbers, part numbers, IDs) must be left untouched.
        assert_eq!(fix_all_number_spaces("555 123 4567"), "555 123 4567");
        assert_eq!(
            fix_all_number_spaces("My number is 555 123 4567."),
            "My number is 555 123 4567."
        );
        assert_eq!(
            fix_all_number_spaces("Call 1 800 555 0199 now"),
            "Call 1 800 555 0199 now"
        );
        // A lead group longer than 3 digits is not a thousands separator, and
        // must not collapse via one of its own suffixes either.
        assert_eq!(fix_all_number_spaces("1234 567"), "1234 567");
        // A leading-zero lead group is a dialling prefix, not a thousands
        // group, even though 1-3-3-3 is otherwise a valid number shape.
        assert_eq!(fix_all_number_spaces("0 800 123 456"), "0 800 123 456");
        assert_eq!(
            fix_all_number_spaces("Call 07 700 900 123 today"),
            "Call 07 700 900 123 today"
        );
    }

    #[test]
    fn fix_all_number_spaces_keeps_valid_european_thousands() {
        assert_eq!(fix_all_number_spaces("2 000"), "2000");
        assert_eq!(fix_all_number_spaces("12 345 678"), "12345678");
        assert_eq!(fix_all_number_spaces("1 000 000"), "1000000");
    }

    #[test]
    fn strip_reasoning_removes_mid_answer_angle_blocks() {
        // Reasoning can appear in the middle of a response, not just at the
        // head; the angle-bracket forms are unambiguous and must be spliced out.
        assert_eq!(
            strip_reasoning("The answer is <thinking>secret</thinking> 42"),
            "The answer is 42"
        );
        assert_eq!(
            strip_reasoning("Left <reasoning>private</reasoning> right"),
            "Left right"
        );
        // `<think>` is the most common spelling of all — a mid-answer block in
        // that spelling must not leak into the user's document, and must not
        // be mistaken for a pre-filled template that swallows the text before it.
        assert_eq!(
            strip_reasoning("The answer is <think>x</think> ok"),
            "The answer is ok"
        );
        assert_eq!(
            strip_reasoning("<think>a</think>Hello <think>b</think>World"),
            "Hello World"
        );
    }

    #[test]
    fn strip_reasoning_drops_unclosed_mid_answer_block_tail() {
        // An unclosed reasoning block in the middle of the answer means the
        // model never finished; nothing after it is an answer.
        assert_eq!(
            strip_reasoning("The answer is <thinking>still thinking"),
            "The answer is"
        );
    }
}
