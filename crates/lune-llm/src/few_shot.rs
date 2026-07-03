use lune_core::llm::LlmService;
use lune_core::{Error, Result};

pub async fn generate_few_shot_examples(
    llm: &dyn LlmService,
    style_name: &str,
    style_instructions: &str,
) -> Result<Vec<(String, String)>> {
    let system = "You generate training examples for a text styling preset applied after \
        speech-to-text transcription. Each example is a PAIR: a realistic raw transcript \
        (typical ASR quirks: no punctuation, lowercase, filler words like um/like), and that \
        same sentence rewritten to follow the style rules. Never describe, explain, or restate \
        the style rules — only output concrete spoken sentences and their rewrites. Return \
        exactly 3 pairs as a JSON array of 2-element arrays: \
        [[\"input\", \"output\"], [\"input\", \"output\"], [\"input\", \"output\"]]";

    let user = format!(
        "Style name: {style_name}\n\
         Style rules: {style_instructions}\n\n\
         Required JSON shape (placeholders only — invent your own content, do not copy this):\n\
         [[\"<a raw spoken sentence with filler words and no punctuation>\", \
         \"<that same sentence rewritten per the style rules>\"]]\n\n\
         Now generate 3 realistic raw-transcript / styled-output pairs for the style above. \
         Each input is 1-2 sentences of natural, unpunctuated speech. Each output is that same \
         sentence rewritten per the style rules — write the rewritten sentence itself, not a \
         description of the rule.\n\n\
         Return ONLY the JSON array of 3 pairs, nothing else."
    );

    let response = llm.complete(system, &user).await?;

    let result = parse_pairs(&response);
    if result.is_empty() {
        // The raw response is the only evidence of *why* generation failed
        // (degenerate loop, truncation, wrong shape) — surface it.
        tracing::warn!(
            response = truncate_for_log(&response, 500),
            "few-shot generation produced no usable pairs"
        );
        return Err(Error::llm("LLM returned no valid examples"));
    }

    Ok(result)
}

/// Extract `(input, output)` pairs from the model response. Tries a strict
/// parse of the outermost JSON array first, then falls back to salvaging any
/// complete `["…", "…"]` pairs — a small model's output is routinely
/// truncated mid-array or wrapped in prose, and the complete leading pairs
/// are still perfectly good examples.
fn parse_pairs(response: &str) -> Vec<(String, String)> {
    let strict: Option<Vec<Vec<String>>> =
        extract_json_array(response).and_then(|s| serde_json::from_str(s).ok());
    let rows = match strict {
        Some(rows) => rows,
        None => salvage_pair_rows(response),
    };

    let mut result: Vec<(String, String)> = Vec::new();
    for row in rows {
        if let [input, output] = row.as_slice()
            && !input.trim().is_empty()
            && !output.trim().is_empty()
        {
            let pair = (input.clone(), output.clone());
            // Degenerate decodes repeat one pair over and over; keep each once.
            if !result.contains(&pair) {
                result.push(pair);
            }
        }
        if result.len() >= 5 {
            break;
        }
    }
    result
}

fn extract_json_array(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

/// Scan for individually complete `["…", "…"]` JSON arrays anywhere in the
/// text. Each candidate spans a `[` to the next `]`; anything serde rejects
/// is skipped. A `]` inside a string breaks only that candidate — the scan
/// moves on and later pairs are still recovered.
fn salvage_pair_rows(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut search_from = 0usize;
    while let Some(open_rel) = text[search_from..].find('[') {
        let open = search_from + open_rel;
        let Some(close_rel) = text[open + 1..].find(']') else {
            break;
        };
        let close = open + 1 + close_rel;
        let candidate = &text[open..=close];
        // A nested opener means `open` was the outer array's bracket; retry
        // from the inner one rather than skipping past the candidate.
        if let Some(inner_rel) = candidate[1..].find('[') {
            search_from = open + 1 + inner_rel;
            continue;
        }
        match serde_json::from_str::<Vec<String>>(candidate) {
            Ok(row) => {
                rows.push(row);
                search_from = close + 1;
            }
            Err(_) => {
                search_from = open + 1;
            }
        }
    }
    rows
}

fn truncate_for_log(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let cut: String = text.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_clean_json_array() {
        let pairs = parse_pairs(r#"[["hello there", "Hello there."], ["ok", "Okay."]]"#);
        assert_eq!(
            pairs,
            vec![
                ("hello there".to_string(), "Hello there.".to_string()),
                ("ok".to_string(), "Okay.".to_string()),
            ]
        );
    }

    #[test]
    fn parses_array_wrapped_in_prose_and_fences() {
        let pairs = parse_pairs(
            "Here are the examples:\n```json\n[[\"a b\", \"A b.\"]]\n```\nHope that helps!",
        );
        assert_eq!(pairs, vec![("a b".to_string(), "A b.".to_string())]);
    }

    #[test]
    fn salvages_complete_pairs_from_truncated_output() {
        // Token budget ran out mid-third-pair: the two complete pairs survive.
        let pairs = parse_pairs(
            r#"[["first in", "First out."], ["second in", "Second out."], ["third in", "Thi"#,
        );
        assert_eq!(
            pairs,
            vec![
                ("first in".to_string(), "First out.".to_string()),
                ("second in".to_string(), "Second out.".to_string()),
            ]
        );
    }

    #[test]
    fn dedups_degenerate_repetition() {
        let pairs = parse_pairs(r#"[["same", "Same."], ["same", "Same."], ["same", "Same."]]"#);
        assert_eq!(pairs, vec![("same".to_string(), "Same.".to_string())]);
    }

    #[test]
    fn rejects_flat_string_array() {
        // A degenerate flat array of strings holds no input/output pairs.
        assert!(parse_pairs(r#"["a", "b", "c"]"#).is_empty());
    }

    #[test]
    fn rejects_rows_with_wrong_arity_or_empty_sides() {
        assert!(parse_pairs(r#"[["only one"], ["in", "out", "extra"]]"#).is_empty());
        assert!(parse_pairs(r#"[["", "out"], ["in", "  "]]"#).is_empty());
    }

    #[test]
    fn empty_and_bracketless_responses_yield_nothing() {
        assert!(parse_pairs("").is_empty());
        assert!(parse_pairs("I cannot generate examples.").is_empty());
        assert!(parse_pairs("[]").is_empty());
    }
}
