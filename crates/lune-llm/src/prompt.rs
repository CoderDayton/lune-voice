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
    let mut prompt = String::with_capacity(2048);
    let style = &input.style_preset;

    // System message with dictionary context.
    // /no_think disables Qwen3's chain-of-thought mode at the system level.
    prompt.push_str("<|im_start|>system\n/no_think\n");
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

    if let Some(ref ctx) = input.prev_context
        && !ctx.is_empty()
    {
        prompt.push_str("\n\nPreceding context: ");
        prompt.push_str(ctx);
    }

    prompt.push_str("<|im_end|>\n");

    // Few-shot examples.
    for (user_ex, assistant_ex) in &style.few_shot_pairs {
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(user_ex);
        prompt.push_str("<|im_end|>\n");
        prompt.push_str("<|im_start|>assistant\n<think>\n</think>\n");
        prompt.push_str(assistant_ex);
        prompt.push_str("<|im_end|>\n");
    }

    // Actual user input.
    let user_text = style
        .user_prompt_template
        .replace("{raw_text}", &input.raw_text);
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(&user_text);
    prompt.push_str("<|im_end|>\n");

    // Start assistant turn with thinking disabled.
    prompt.push_str("<|im_start|>assistant\n<think>\n</think>\n");

    prompt
}

/// Strip any `<think>...</think>` blocks and trim the output.
#[inline]
pub fn strip_thinking(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    // Strip matched <think>...</think> pairs.
    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</think>") {
            remaining = &remaining[start + end + "</think>".len()..];
        } else {
            // Unclosed think tag — drop everything after it.
            remaining = "";
        }
    }
    result.push_str(remaining);

    // Handle bare </think> without opening tag — the model continued
    // reasoning after our prefilled <think></think> and only emitted the
    // closing tag. Drop everything before (and including) the bare closer.
    if let Some(pos) = result.find("</think>") {
        // drain in-place avoids a new heap allocation
        result.drain(0..pos + "</think>".len());
    }

    // Also strip <|im_end|> if the model emits it.
    // Check before replacing to avoid an allocation in the common case.
    if result.contains("<|im_end|>") {
        result = result.replace("<|im_end|>", "");
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::config::StylePreset;
    use lune_core::dict::DictionaryEntry;

    #[test]
    fn chatml_basic_structure() {
        let input = PostProcessInput {
            raw_text: "hello um world".into(),
            prev_context: None,
            style_preset: StylePreset::general(),
            app_category: None,
            dictionary: vec![],
        };
        let prompt = build_chatml(&input);
        assert!(prompt.starts_with("<|im_start|>system\n/no_think\n"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("hello um world"));
        assert!(prompt.ends_with("<think>\n</think>\n"));
    }

    #[test]
    fn chatml_with_dictionary() {
        let input = PostProcessInput {
            raw_text: "use the lunevoice api".into(),
            prev_context: None,
            style_preset: StylePreset::general(),
            app_category: None,
            dictionary: vec![DictionaryEntry {
                phrase: "lunevoice".into(),
                replacement: "Lune Voice".into(),
                priority: 10,
            }],
        };
        let prompt = build_chatml(&input);
        assert!(prompt.contains("lunevoice"));
        assert!(prompt.contains("Lune Voice"));
    }

    #[test]
    fn chatml_includes_few_shot() {
        let input = PostProcessInput {
            raw_text: "test".into(),
            prev_context: None,
            style_preset: StylePreset::general(),
            app_category: None,
            dictionary: vec![],
        };
        let prompt = build_chatml(&input);
        // General preset has 2 few-shot pairs.
        assert!(prompt.contains("so um I was thinking"));
        assert!(prompt.contains("I was thinking that we should go to the store."));
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
}
