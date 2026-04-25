use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};

use lune_core::config::BASE_SYSTEM_PROMPT;
use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

/// Configuration for the OpenAI-compatible LLM backend.
#[derive(Debug, Clone)]
pub struct OpenAiLlmConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for OpenAiLlmConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "gpt-4o-mini".into(),
            max_tokens: 512,
            temperature: 0.2,
        }
    }
}

/// OpenAI-compatible LLM backend for remote text post-processing.
/// Gracefully degrades: API failures return the raw text with a warning.
pub struct OpenAiLlmBackend {
    client: Client<OpenAIConfig>,
    config: OpenAiLlmConfig,
}

impl OpenAiLlmBackend {
    pub fn new(config: OpenAiLlmConfig) -> Self {
        let openai_config = OpenAIConfig::new()
            .with_api_key(&config.api_key)
            .with_api_base(&config.base_url);
        let client = Client::with_config(openai_config);

        tracing::info!(
            base_url = %config.base_url,
            model = %config.model,
            "initialized OpenAI-compatible LLM backend"
        );

        Self { client, config }
    }
}

/// Build the chat message array from a `PostProcessInput`.
///
/// Amortises all string allocations up-front; avoids per-entry format! calls.
fn build_messages(input: &PostProcessInput) -> Vec<ChatCompletionRequestMessage> {
    let style = &input.style_preset;
    let capacity = 1 + style.few_shot_pairs.len() * 2 + 1;
    let mut messages = Vec::with_capacity(capacity);

    // ── System message ────────────────────────────────────────────────────────
    let dict_len = input.dictionary.len();

    let style_len = if style.style_instructions.is_empty() {
        0
    } else {
        2 + style.style_instructions.len()
    };
    let mut system_content = String::with_capacity(
        BASE_SYSTEM_PROMPT.len() + style_len + if dict_len > 0 { 55 + dict_len * 30 } else { 0 },
    );
    system_content.push_str(BASE_SYSTEM_PROMPT);

    if !style.style_instructions.is_empty() {
        system_content.push_str("\n\n");
        system_content.push_str(&style.style_instructions);
    }

    if dict_len > 0 {
        system_content.push_str("\n\nDictionary terms (always use these exact spellings):");
        for entry in &input.dictionary {
            system_content.push_str("\n- ");
            system_content.push_str(&entry.phrase);
            system_content.push_str(" → ");
            system_content.push_str(&entry.replacement);
        }
    }

    messages.push(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_content)
            .build()
            .expect("system message build")
            .into(),
    );

    // ── Few-shot examples ─────────────────────────────────────────────────────
    for (user_ex, assistant_ex) in &style.few_shot_pairs {
        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_ex.clone())
                .build()
                .expect("user few-shot build")
                .into(),
        );
        messages.push(
            ChatCompletionRequestAssistantMessageArgs::default()
                .content(assistant_ex.clone())
                .build()
                .expect("assistant few-shot build")
                .into(),
        );
    }

    // ── Final user message ────────────────────────────────────────────────────
    // Avoid allocating a new String when the template contains exactly one
    // `{raw_text}` placeholder — replace in-place via pre-sized buffer.
    let tmpl = &style.user_prompt_template;
    let user_text = if let Some(pos) = tmpl.find("{raw_text}") {
        let raw = &input.raw_text;
        let mut buf = String::with_capacity(tmpl.len() - 10 + raw.len());
        buf.push_str(&tmpl[..pos]);
        buf.push_str(raw);
        buf.push_str(&tmpl[pos + 10..]);
        buf
    } else {
        tmpl.clone()
    };

    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_text)
            .build()
            .expect("user message build")
            .into(),
    );

    messages
}

#[inline(always)]
fn fallback_output(raw_text: &str) -> PostProcessOutput {
    PostProcessOutput {
        cleaned_text: raw_text.to_owned(),
        metadata: None,
    }
}

#[async_trait::async_trait]
impl TextPostProcessor for OpenAiLlmBackend {
    async fn transform(&self, input: &PostProcessInput) -> lune_core::Result<PostProcessOutput> {
        let messages = build_messages(input);

        let request = match CreateChatCompletionRequestArgs::default()
            .model(&self.config.model)
            .max_tokens(self.config.max_tokens)
            .temperature(self.config.temperature)
            .messages(messages)
            .build()
        {
            Ok(req) => req,
            Err(e) => {
                tracing::warn!(error = %e, "failed to build chat completion request, returning raw text");
                return Ok(fallback_output(&input.raw_text));
            }
        };

        tracing::debug!(
            model = %self.config.model,
            raw_text_len = input.raw_text.len(),
            style = %input.style_preset.id,
            "OpenAI post-process request"
        );

        match self.client.chat().create(request).await {
            Ok(response) => {
                // Extract content without an intermediate owned String when possible.
                let text = response
                    .choices
                    .first()
                    .and_then(|c| c.message.content.as_deref())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .unwrap_or(input.raw_text.trim())
                    .to_owned();

                tracing::debug!(output_len = text.len(), "OpenAI post-process complete");

                Ok(PostProcessOutput {
                    cleaned_text: crate::prompt::strip_markers(&text),
                    metadata: None,
                })
            }
            Err(e) => {
                tracing::warn!(error = %e, "OpenAI API call failed, returning raw text");
                Ok(fallback_output(&input.raw_text))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use lune_core::config::StylePreset;
    use lune_core::dict::DictionaryEntry;

    fn test_input(raw_text: &str) -> PostProcessInput {
        PostProcessInput {
            raw_text: raw_text.into(),
            style_preset: Arc::new(StylePreset::casual()),
            app_category: None,
            dictionary: vec![],
            context: None,
            language: "en".into(),
            hypotheses: Vec::new(),
        }
    }

    #[test]
    fn build_messages_basic() {
        let input = test_input("hello um world");
        let msgs = build_messages(&input);

        // system + N few-shot pairs (2*N msgs) + 1 final user
        let n = lune_core::config::StylePreset::casual()
            .few_shot_pairs
            .len();
        assert_eq!(msgs.len(), 1 + 2 * n + 1);

        let json = serde_json::to_string(&msgs).unwrap();
        assert!(json.contains("hello um world"));
        assert!(json.contains("You correct dictation transcripts"));
    }

    #[test]
    fn build_messages_with_dictionary() {
        let input = PostProcessInput {
            dictionary: vec![DictionaryEntry {
                phrase: "lunevoice".into(),
                replacement: "Lune Voice".into(),
                priority: 10,
            }],
            ..test_input("use lunevoice")
        };
        let msgs = build_messages(&input);

        let json = serde_json::to_string(&msgs).unwrap();
        assert!(json.contains("Lune Voice"));
        assert!(json.contains("Dictionary terms"));
    }

    #[test]
    fn build_messages_no_raw_text_placeholder() {
        // Template without {raw_text} — should not panic, just clone template.
        let mut preset = StylePreset::casual();
        preset.user_prompt_template = "fixed prompt".into();
        let input = PostProcessInput {
            raw_text: "hello".into(),
            style_preset: Arc::new(preset),
            app_category: None,
            dictionary: vec![],
            context: None,
            language: "en".into(),
            hypotheses: Vec::new(),
        };
        let msgs = build_messages(&input);
        let json = serde_json::to_string(&msgs).unwrap();
        assert!(json.contains("fixed prompt"));
    }

    #[test]
    fn config_defaults() {
        let cfg = OpenAiLlmConfig::default();
        assert_eq!(cfg.max_tokens, 512);
        assert!((cfg.temperature - 0.2).abs() < f32::EPSILON);
        assert_eq!(cfg.model, "gpt-4o-mini");
        assert_eq!(cfg.base_url, "https://api.openai.com/v1");
    }

    #[tokio::test]
    async fn graceful_degradation() {
        let config = OpenAiLlmConfig {
            base_url: "http://127.0.0.1:1".into(),
            api_key: "test-key".into(),
            ..Default::default()
        };
        let backend = OpenAiLlmBackend::new(config);
        let input = test_input("hello world");

        let output = backend.transform(&input).await.unwrap();
        assert_eq!(output.cleaned_text, "hello world");
        assert!(output.metadata.is_none());
    }
}
