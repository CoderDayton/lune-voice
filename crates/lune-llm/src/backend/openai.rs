use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
};

use lune_core::{
    config::BASE_SYSTEM_PROMPT,
    llm::{PostProcessInput, PostProcessOutput, TextPostProcessor},
};

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
/// Mirrors the ORT backend prompt structure: system = BASE_SYSTEM_PROMPT,
/// user = structured prompt via `OrtStyleLlm::user_prompt()`.
fn build_messages(input: &PostProcessInput) -> Vec<ChatCompletionRequestMessage> {
    let mut messages = Vec::with_capacity(2);

    let mut system_content = BASE_SYSTEM_PROMPT.replace("{lang}", &input.language);
    if !input.dictionary.is_empty() {
        system_content.push_str("\nDictionary terms (use these exact spellings):");
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

    let user_text = super::ort_llm::OrtStyleLlm::user_prompt(
        &input.raw_text,
        &input.language,
        &input.uncertain_words,
    );

    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_text)
            .build()
            .expect("user message build")
            .into(),
    );

    messages
}

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
                    cleaned_text: text,
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
            uncertain_words: Vec::new(),
        }
    }

    #[test]
    fn build_messages_basic() {
        let input = test_input("hello um world");
        let msgs = build_messages(&input);

        assert_eq!(msgs.len(), 2);

        let json = serde_json::to_string(&msgs).unwrap();
        assert!(json.contains("hello um world"));
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
    fn build_messages_includes_transcript() {
        let input = test_input("hello world");
        let msgs = build_messages(&input);
        let json = serde_json::to_string(&msgs).unwrap();
        assert!(json.contains("hello world"));
    }

    #[test]
    fn build_messages_with_uncertain_words() {
        let input = PostProcessInput {
            uncertain_words: vec![("brought".into(), 6)],
            ..test_input(
                "call with Jennifer Martinez to discuss interesting their OAuth deployment on AWS",
            )
        };
        let msgs = build_messages(&input);
        let json = serde_json::to_string(&msgs).unwrap();

        assert!(json.contains("call with Jennifer Martinez"));
        assert!(json.contains("/no_think"));
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
