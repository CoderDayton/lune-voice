use lune_core::llm::PostProcessInput;

/// Build a ChatML-formatted prompt for Qwen3 from a PostProcessInput.
///
/// Format:
/// ```text
/// <|im_start|>system
/// {system_prompt}<|im_end|>
/// <|im_start|>user
/// {few_shot_input}<|im_end|>
/// <|im_start|>assistant
/// <think>
/// </think>
/// {few_shot_output}<|im_end|>
/// ...
/// <|im_start|>user
/// {raw_text}<|im_end|>
/// <|im_start|>assistant
/// <think>
/// </think>
///
/// ```
///
/// We start with `<think>\n</think>\n` to disable thinking mode —
/// for text cleanup we want direct output, not chain-of-thought.
pub fn build_chatml(input: &PostProcessInput) -> String {
    let style = &input.style_preset;

    // Estimate capacity from all input lengths to avoid reallocations.
    let dict_len: usize = input
        .dictionary
        .iter()
        .map(|e| e.phrase.len() + e.replacement.len() + 8)
        .sum();
    let few_shot_len: usize = style
        .few_shot_pairs
        .iter()
        .map(|(u, a)| u.len() + a.len() + 80)
        .sum();
    let ctx_len = input.context.as_ref().map_or(0, |c| c.len() + 20);
    let cap = 128
        + style.system_prompt.len()
        + dict_len
        + few_shot_len
        + ctx_len
        + style.user_prompt_template.len()
        + input.raw_text.len();

    let mut prompt = String::with_capacity(cap);

    // System message with dictionary context.
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(&style.system_prompt);

    if !input.dictionary.is_empty() {
        prompt.push_str("\n\nDictionary terms (always use these exact spellings):");
        for entry in &input.dictionary {
            prompt.push_str("\n- ");
            prompt.push_str(&entry.phrase);
            prompt.push_str(" → ");
            prompt.push_str(&entry.replacement);
        }
    }

    prompt.push_str("<|im_end|>\n");

    // Few-shot examples.
    for (user_ex, assistant_ex) in &style.few_shot_pairs {
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(user_ex);
        prompt.push_str("\n/no_think<|im_end|>\n");
        prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        prompt.push_str(assistant_ex);
        prompt.push_str("<|im_end|>\n");
    }

    // Actual user input — inline template substitution to avoid an intermediate allocation.
    prompt.push_str("<|im_start|>user\n");
    let template = &style.user_prompt_template;

    // First pass: expand {context} slot.
    let ctx_text = input.context.as_deref().unwrap_or("");
    let after_ctx = if let Some(idx) = template.find("{context}") {
        let mut buf = String::with_capacity(template.len() + ctx_text.len());
        buf.push_str(&template[..idx]);
        buf.push_str(ctx_text);
        buf.push_str(&template[idx + "{context}".len()..]);
        std::borrow::Cow::Owned(buf)
    } else {
        std::borrow::Cow::Borrowed(template.as_str())
    };

    // Second pass: expand {raw_text} slot.
    if let Some(idx) = after_ctx.find("{raw_text}") {
        prompt.push_str(&after_ctx[..idx]);
        prompt.push_str(&input.raw_text);
        prompt.push_str(&after_ctx[idx + "{raw_text}".len()..]);
    } else {
        prompt.push_str(&after_ctx);
    }
    prompt.push_str("\n/no_think<|im_end|>\n");

    // Start assistant turn with thinking disabled.
    // Double newlines inside <think> block per Qwen3 official docs.
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");

    prompt
}

/// Strip any `<think>...</think>` blocks and trim the output.
#[inline]
pub fn strip_thinking(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    const IM_END: &str = "<|im_end|>";

    let has_open = text.contains(OPEN);
    let has_close = text.contains(CLOSE);
    let has_im_end = text.contains(IM_END);

    // Allocate once at input length; result after stripping will only be smaller.
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    if has_open || has_close || has_im_end {
        // Strip matched <think>...</think> pairs.
        while let Some(start) = remaining.find(OPEN) {
            result.push_str(&remaining[..start]);
            if let Some(end) = remaining[start..].find(CLOSE) {
                remaining = &remaining[start + end + CLOSE.len()..];
            } else {
                // Unclosed think tag — drop everything after it.
                remaining = "";
                break;
            }
        }
        result.push_str(remaining);

        // Handle bare </think> without opening tag — the model continued
        // reasoning after our prefilled <think></think> and only emitted the
        // closing tag. Drain in-place to avoid a new heap allocation.
        if let Some(pos) = result.find(CLOSE) {
            result.drain(0..pos + CLOSE.len());
        }

        // Also strip <|im_end|> if the model emits it.
        if result.contains(IM_END) {
            let cleaned = result.replace(IM_END, "");
            result = cleaned;
        }
    } else {
        result.push_str(text);
    }

    // Strip reasoning preambles that small models emit without think tags.
    // These start with meta-commentary instead of the cleaned transcription.
    const REASONING_PREFIXES: &[&str] = &[
        "Okay, ",
        "Okay,\n",
        "Let me ",
        "Let's ",
        "I need to ",
        "I'll ",
        "First, ",
        "First,\n",
        "Starting with ",
        "Step ",
        "Sure, ",
        "Here's ",
        "Here is ",
        "The user ",
        "Now, ",
        "Now I ",
        "Alright, ",
        "So, ",
        "So the ",
    ];
    let trimmed = result.trim_start();
    if REASONING_PREFIXES.iter().any(|p| trimmed.starts_with(p)) {
        // The model is reasoning. Look for a double-newline break after the
        // preamble — the actual output often follows after `\n\n`.
        if let Some(pos) = result.find("\n\n") {
            let after = result[pos + 2..].trim();
            // If there's substantial text after the break, use it.
            if after.len() > 3 {
                result = after.to_string();
            }
        }
    }

    // Trim in-place — avoids the `.trim().to_string()` double-allocation pattern.
    let end_len = result.trim_end().len();
    result.truncate(end_len);
    let start_offset = result.len() - result.trim_start().len();
    if start_offset > 0 {
        result.drain(0..start_offset);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use lune_core::config::StylePreset;
    use lune_core::dict::DictionaryEntry;

    #[test]
    fn chatml_basic_structure() {
        let input = PostProcessInput {
            raw_text: "hello um world".into(),
            style_preset: Arc::new(StylePreset::general()),
            app_category: None,
            dictionary: vec![],
            context: None,
        };
        let prompt = build_chatml(&input);
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("hello um world"));
        assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn chatml_with_dictionary() {
        let input = PostProcessInput {
            raw_text: "use the lunevoice api".into(),
            style_preset: Arc::new(StylePreset::general()),
            app_category: None,
            dictionary: vec![DictionaryEntry {
                phrase: "lunevoice".into(),
                replacement: "Lune Voice".into(),
                priority: 10,
            }],
            context: None,
        };
        let prompt = build_chatml(&input);
        assert!(prompt.contains("lunevoice"));
        assert!(prompt.contains("Lune Voice"));
    }

    #[test]
    fn chatml_with_context() {
        let input = PostProcessInput {
            raw_text: "hello world".into(),
            style_preset: Arc::new(StylePreset::general()),
            app_category: None,
            dictionary: vec![],
            context: Some("Context: I was just talking about Rust.\n".into()),
        };
        let prompt = build_chatml(&input);
        assert!(prompt.contains("Context: I was just talking about Rust.\nASR: hello world"));
    }

    #[test]
    fn chatml_without_context_no_prefix() {
        let input = PostProcessInput {
            raw_text: "hello world".into(),
            style_preset: Arc::new(StylePreset::general()),
            app_category: None,
            dictionary: vec![],
            context: None,
        };
        let prompt = build_chatml(&input);
        // No context → {context} expands to empty string, template starts with "ASR:"
        assert!(prompt.contains("<|im_start|>user\nASR: hello world\n"));
    }

    #[test]
    fn chatml_includes_few_shot() {
        let input = PostProcessInput {
            raw_text: "test".into(),
            style_preset: Arc::new(StylePreset::general()),
            app_category: None,
            dictionary: vec![],
            context: None,
        };
        let prompt = build_chatml(&input);
        // General preset has 4 few-shot pairs.
        assert!(prompt.contains("ASR: so um I'm gonna go to the store"));
        assert!(prompt.contains("I'm gonna go to the store and buy some more flour for the cake"));
    }

    #[test]
    fn strip_thinking_basic() {
        let text = "<think>reasoning here</think>The answer is 42.";
        assert_eq!(strip_thinking(text), "The answer is 42.");
    }

    #[test]
    fn strip_thinking_empty() {
        let text = "<think></think>Clean output.";
        assert_eq!(strip_thinking(text), "Clean output.");
    }

    #[test]
    fn strip_thinking_no_tags() {
        let text = "Just regular text.";
        assert_eq!(strip_thinking(text), "Just regular text.");
    }

    #[test]
    fn strip_im_end() {
        let text = "Hello world.<|im_end|>";
        assert_eq!(strip_thinking(text), "Hello world.");
    }

    #[test]
    fn strip_bare_close_think() {
        // Model ignored prefill and produced reasoning with only a closing tag.
        let text = "Okay, the user wrote something...\n</think>\n\nThe meeting is at 3.";
        assert_eq!(strip_thinking(text), "The meeting is at 3.");
    }

    #[test]
    fn strip_reasoning_preamble() {
        // Model reasons in plain text without think tags — common with 0.6B.
        let text = "Okay, let's process this ASR input.\n\nThe meeting is at three.";
        assert_eq!(strip_thinking(text), "The meeting is at three.");
    }

    #[test]
    fn no_strip_legit_okay() {
        // "So, " at start is a reasoning prefix but if no \n\n break, keep as-is.
        let text = "So, the meeting is at three.";
        assert_eq!(strip_thinking(text), "So, the meeting is at three.");
    }
}
