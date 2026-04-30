use std::collections::HashSet;

use lune_core::llm::{CorrectionSpan, PostProcessInput};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Edit {
    start_word: usize,
    end_word: usize,
    replacement: String,
}

pub fn apply_model_edits(input: &PostProcessInput, model_text: &str) -> String {
    let Some(json) = extract_json_array(model_text) else {
        return input.raw_text.clone();
    };
    let Ok(mut edits) = serde_json::from_str::<Vec<Edit>>(json) else {
        return input.raw_text.clone();
    };
    if edits.is_empty() {
        return input.raw_text.clone();
    }

    edits.sort_by_key(|edit| edit.start_word);

    let mut tokens: Vec<String> = input
        .raw_text
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let marked = marked_indices(&input.correction_spans);

    let mut offset: isize = 0;
    let mut last_original_end = 0usize;
    for edit in &edits {
        if !valid_edit(edit, tokens.len(), &marked, last_original_end) {
            continue;
        }

        let replacement: Vec<String> = edit
            .replacement
            .split_whitespace()
            .take(8)
            .map(str::to_owned)
            .collect();
        if replacement.is_empty() {
            continue;
        }

        let start = shifted_index(edit.start_word, offset, tokens.len());
        let end = shifted_index(edit.end_word, offset, tokens.len());
        if start >= end || end > tokens.len() {
            continue;
        }

        let original = tokens[start..end].join(" ");
        let proposed = replacement.join(" ");
        if !phonetically_plausible(&original, &proposed) {
            continue;
        }

        let removed = end - start;
        let added = replacement.len();
        tokens.splice(start..end, replacement);
        offset += added as isize - removed as isize;
        last_original_end = edit.end_word;
    }

    tokens.join(" ")
}

fn extract_json_array(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    (start <= end).then_some(&text[start..=end])
}

fn marked_indices(spans: &[CorrectionSpan]) -> HashSet<usize> {
    let mut marked = HashSet::new();
    for span in spans {
        for i in span.start_word..span.end_word {
            marked.insert(i);
        }
    }
    marked
}

fn valid_edit(
    edit: &Edit,
    token_len: usize,
    marked: &HashSet<usize>,
    last_original_end: usize,
) -> bool {
    if edit.start_word >= edit.end_word
        || edit.end_word > token_len
        || edit.start_word < last_original_end
        || edit.end_word - edit.start_word > 4
        || edit.replacement.len() > 80
        || edit.replacement.contains('\n')
    {
        return false;
    }

    (edit.start_word..edit.end_word).any(|i| marked.contains(&i))
}

fn shifted_index(original: usize, offset: isize, max: usize) -> usize {
    original.saturating_add_signed(offset).min(max)
}

fn phonetically_plausible(original: &str, proposed: &str) -> bool {
    let a = original.to_lowercase();
    let b = proposed.to_lowercase();
    if a == b {
        return false;
    }
    let a = a.trim_matches(|c: char| c.is_ascii_punctuation());
    let b = b.trim_matches(|c: char| c.is_ascii_punctuation());
    if a == b {
        return false;
    }
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return false;
    }
    let dist = edit_distance(a.as_bytes(), b.as_bytes());
    let ratio = dist as f32 / max_len as f32;
    ratio <= 0.6
}

fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lune_core::config::StylePreset;
    use lune_core::llm::{CorrectionSpan, PostProcessInput};

    use super::apply_model_edits;

    fn input(raw_text: &str) -> PostProcessInput {
        PostProcessInput {
            raw_text: raw_text.into(),
            style_preset: Arc::new(StylePreset::minimal()),
            app_category: None,
            dictionary: vec![],
            correction_spans: vec![CorrectionSpan {
                start_word: 3,
                end_word: 5,
                text: "poorly numbered".into(),
                confidence: 0.2,
                fix_score: 0.9,
            }],
            context: None,
            language: "en".into(),
        }
    }

    #[test]
    fn applies_marked_edit() {
        let input = input("the meeting had poorly numbered today");
        let out = apply_model_edits(
            &input,
            r#"[{"start_word":3,"end_word":5,"replacement":"quarterly numbers"}]"#,
        );
        assert_eq!(out, "the meeting had quarterly numbers today");
    }

    #[test]
    fn ignores_unmarked_edit() {
        let input = input("the meeting had poorly numbered today");
        let out = apply_model_edits(
            &input,
            r#"[{"start_word":0,"end_word":2,"replacement":"a meeting"}]"#,
        );
        assert_eq!(out, input.raw_text);
    }

    #[test]
    fn invalid_json_returns_input() {
        let input = input("the meeting had poorly numbered today");
        assert_eq!(apply_model_edits(&input, "rewrite it"), input.raw_text);
    }

    #[test]
    fn rejects_identity_edit() {
        let input = input("the meeting had poorly numbered today");
        let out = apply_model_edits(
            &input,
            r#"[{"start_word":3,"end_word":4,"replacement":"poorly"}]"#,
        );
        assert_eq!(out, input.raw_text);
    }

    #[test]
    fn rejects_phonetically_distant_edit() {
        let input = input("the meeting had poorly numbered today");
        let out = apply_model_edits(
            &input,
            r#"[{"start_word":3,"end_word":5,"replacement":"xyz elephant"}]"#,
        );
        assert_eq!(out, input.raw_text);
    }

    #[test]
    fn accepts_phonetically_similar_edit() {
        let input = input("the meeting had poorly numbered today");
        let out = apply_model_edits(
            &input,
            r#"[{"start_word":3,"end_word":5,"replacement":"poorly number"}]"#,
        );
        assert_eq!(out, "the meeting had poorly number today");
    }
}
