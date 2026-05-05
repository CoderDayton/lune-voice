use lune_core::llm::LlmService;

pub async fn generate_few_shot_examples(
    llm: &dyn LlmService,
    style_name: &str,
    style_instructions: &str,
) -> Result<Vec<(String, String)>, String> {
    let system = "You generate example input/output pairs for a text styling preset. \
        Each example shows raw speech-to-text input and the correctly styled output. \
        Return exactly 3 examples as a JSON array: [[\"input\", \"output\"], ...]";

    let user = format!(
        "Style name: {style_name}\n\
         Style rules: {style_instructions}\n\n\
         Generate 3 realistic examples of raw transcribed speech and the correct styled output. \
         Each input should be 1-2 sentences of natural speech (with typical ASR errors like \
         missing punctuation, lowercase, filler words). Each output should show the same text \
         after applying the style rules.\n\n\
         Return ONLY a JSON array: [[\"input1\", \"output1\"], [\"input2\", \"output2\"], [\"input3\", \"output3\"]]"
    );

    let response = llm
        .complete(system, &user)
        .await
        .map_err(|e| format!("LLM generation failed: {e}"))?;

    let json_str = response
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let pairs: Vec<Vec<String>> =
        serde_json::from_str(json_str).map_err(|e| format!("failed to parse response: {e}"))?;

    let result: Vec<(String, String)> = pairs
        .into_iter()
        .filter_map(|pair| {
            if pair.len() == 2 {
                Some((pair[0].clone(), pair[1].clone()))
            } else {
                None
            }
        })
        .take(5)
        .collect();

    if result.is_empty() {
        return Err("LLM returned no valid examples".into());
    }

    Ok(result)
}
