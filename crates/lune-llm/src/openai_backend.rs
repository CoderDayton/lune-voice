use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};

use lune_core::llm::{PostProcessInput, PostProcessOutput, TextPostProcessor};

/// Configuration for the OpenAI-compatible LLM backend.
#[derive(Debug, Clone)]
pub struct OpenAiLlmConfig {
    /// Base URL for the API (e.g. "https://api.openai.com/v1").
    pub base_url: String,
    /// API key for authentication.
    pub api_key: String,
    /// Model identifier (e.g. "gpt-4o-mini").
    pub model: String,
    /// Maximum tokens to generate per request.
    pub max_tokens: u32,
    /// Sampling temperature.
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
///
/// Calls `POST /v1/chat/completions` on any OpenAI-compatible endpoint.
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
/// Structure:
/// 1. System message: style prompt + dictionary terms + preceding context
/// 2. Few-shot pairs as user/assistant messages
/// 3. Final user message with `{raw_text}` substituted
fn build_messages(input: &PostProcessInput) -> Vec<ChatCompletionRequestMessage> {
    let style = &input.style_preset;
    let mut messages = Vec::new();

    // System message.
    let mut system_content = style.system_prompt.clone();

    if !input.dictionary.is_empty() {
        system_content.push_str("\n\nDictionary terms (always use these exact spellings):");
        for entry in &input.dictionary {
            system_content.push_str("\n- ");
            system_content.push_str(&entry.phrase);
            system_content.push_str(" → ");
            system_content.push_str(&entry.replacement);
        }
    }

    if let Some(ref ctx) = input.prev_context
        && !ctx.is_empty()
    {
        system_content.push_str("\n\nPreceding context: ");
        system_content.push_str(ctx);
    }

    messages.push(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_content)
            .build()
            .expect("system message build")
            .into(),
    );

    // Few-shot examples.
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

    // Final user message with template substitution.
    let user_text = style
        .user_prompt_template
        .replace("{raw_text}", &input.raw_text);
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
                let text = response
                    .choices
                    .first()
                    .and_then(|c| c.message.content.as_deref())
                    .unwrap_or(&input.raw_text)
                    .trim()
                    .to_string();

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
    use lune_core::config::StylePreset;
    use lune_core::dict::DictionaryEntry;

    fn test_input(raw_text: &str) -> PostProcessInput {
        PostProcessInput {
            raw_text: raw_text.into(),
            prev_context: None,
            style_preset: StylePreset::general(),
            app_category: None,
            dictionary: vec![],
        }
    }

    #[test]
    fn build_messages_basic() {
        let input = test_input("hello um world");
        let msgs = build_messages(&input);

        // system + 2 few-shot pairs (4 msgs) + 1 final user = 6
        assert_eq!(msgs.len(), 6);

        let json = serde_json::to_string(&msgs).unwrap();
        assert!(json.contains("hello um world"));
        // System prompt from General preset
        assert!(json.contains("text post-processor"));
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
    fn build_messages_with_context() {
        let input = PostProcessInput {
            prev_context: Some("Previous sentence here.".into()),
            ..test_input("and then this")
        };
        let msgs = build_messages(&input);

        let json = serde_json::to_string(&msgs).unwrap();
        assert!(json.contains("Preceding context"));
        assert!(json.contains("Previous sentence here."));
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
        // Connect to an unreachable address — should return raw text, not error.
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
