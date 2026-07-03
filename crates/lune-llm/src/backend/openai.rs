use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
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

#[async_trait::async_trait]
impl lune_core::llm::LlmService for OpenAiLlmBackend {
    async fn complete(&self, system: &str, user: &str) -> lune_core::Result<String> {
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::with_capacity(2);
        if !system.is_empty() {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system)
                    .build()
                    .expect("system message")
                    .into(),
            );
        }
        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(user)
                .build()
                .expect("user message")
                .into(),
        );
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.config.model)
            .max_tokens(self.config.max_tokens)
            .temperature(self.config.temperature)
            .messages(messages)
            .build()
            .map_err(|e| lune_core::Error::llm(format!("build request: {e}")))?;
        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| lune_core::Error::llm(format!("API call: {e}")))?;
        response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            // A Qwen-style endpoint may emit an inline <think>…</think> block
            // (reasoning must never reach the user's document); an unclosed
            // block collapses to "", which the pipeline treats as "keep the
            // input text".
            .map(|s| crate::prompt::strip_think_tags(s).trim().to_owned())
            .ok_or_else(|| lune_core::Error::llm("empty response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::llm::LlmService;

    #[test]
    fn config_defaults() {
        let cfg = OpenAiLlmConfig::default();
        assert_eq!(cfg.max_tokens, 512);
        assert!((cfg.temperature - 0.2).abs() < f32::EPSILON);
        assert_eq!(cfg.model, "gpt-4o-mini");
        assert_eq!(cfg.base_url, "https://api.openai.com/v1");
    }

    /// Network failures surface as `Err` from `LlmService::complete`. Pipeline-
    /// level graceful degradation is tested in `pipeline.rs` against a mock
    /// `LlmService` that returns the same `Err`, so this test only needs to
    /// verify the backend reports the failure rather than swallowing it.
    #[tokio::test]
    async fn complete_errors_on_unreachable_endpoint() {
        let config = OpenAiLlmConfig {
            base_url: "http://127.0.0.1:1".into(),
            api_key: "test-key".into(),
            ..Default::default()
        };
        let backend = OpenAiLlmBackend::new(config);
        let result = backend.complete("system", "user").await;
        assert!(result.is_err());
    }
}
