use std::time::Duration;

use lune_core::config::RedactedString;
use lune_core::error::Error;

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
    /// Held redacted so a `{:?}` of this config — or of anything embedding it —
    /// cannot put the key in a log line. Zeroized on drop.
    pub api_key: RedactedString,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for OpenAiLlmConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: RedactedString::new(""),
            model: "gpt-4o-mini".into(),
            max_tokens: 512,
            temperature: 0.2,
        }
    }
}

/// Connect timeout for the styling endpoint. A host that accepts nothing must
/// fail fast rather than sit on the dictation's latency budget.
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Read timeout between response chunks. The pipeline's `llm_timeout_secs`
/// bounds the call end to end; this bounds a single stalled read, so a server
/// that accepts the connection and then dribbles bytes still terminates.
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Build the HTTP client the styling backend talks through.
///
/// reqwest's default client has no timeouts and follows up to 10 redirects.
/// Both matter here: the request body is the user's transcript and the header
/// carries the API key, and the HTTPS/locality check in [`OpenAiLlmBackend::new`]
/// only ever sees `base_url`. A redirect would move both to a host that was
/// never checked, so redirects are refused outright — an OpenAI-compatible
/// chat-completions endpoint has no legitimate reason to redirect.
fn http_client() -> Result<reqwest::Client, Error> {
    reqwest::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .read_timeout(HTTP_READ_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| Error::llm(format!("build HTTP client: {e}")))
}

/// OpenAI-compatible LLM backend for remote text post-processing.
/// Gracefully degrades: API failures return the raw text with a warning.
pub struct OpenAiLlmBackend {
    client: Client<OpenAIConfig>,
    config: OpenAiLlmConfig,
}

impl OpenAiLlmBackend {
    pub fn new(config: OpenAiLlmConfig) -> Result<Self, Error> {
        // Reject plaintext (non-TLS) unless targeting a local endpoint. The
        // styling LLM sends the API key and every dictated transcript to this
        // endpoint, so it must never cross a public-network cleartext link.
        // `is_local_endpoint` (the same test the readiness check uses) allows
        // loopback, private LAN, and `.local`/`.localhost` — a LAN Ollama or
        // llama-server is "local" and plaintext-safe, mirroring the ASR guard.
        let url = url::Url::parse(&config.base_url)
            .map_err(|e| Error::llm(format!("invalid base_url: {e}")))?;
        if url.scheme() != "https" && !lune_core::config::is_local_endpoint(&url) {
            return Err(Error::llm(format!(
                "OpenAI LLM base_url must use HTTPS (got {}://); \
                 plaintext only allowed for loopback and private local addresses",
                url.scheme()
            )));
        }

        let openai_config = OpenAIConfig::new()
            .with_api_key(config.api_key.expose())
            .with_api_base(&config.base_url);
        let client = Client::with_config(openai_config).with_http_client(http_client()?);

        tracing::info!(
            base_url = %config.base_url,
            model = %config.model,
            "initialized OpenAI-compatible LLM backend"
        );

        Ok(Self { client, config })
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
            // Read `content` only: an endpoint that returns its reasoning in a
            // separate `reasoning_content` field is handled by never looking at
            // it. Reasoning inlined *into* the content — think tags, harmony
            // channels — is stripped here, because it must never reach the
            // user's document. Reasoning that never closed collapses to "",
            // which the pipeline treats as "keep the input text".
            .map(crate::prompt::strip_reasoning)
            .ok_or_else(|| lune_core::Error::llm("empty response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::llm::LlmService;

    /// The styling key must not be reachable through the config's `Debug`.
    /// Every peer config already uses `RedactedString`; this one held a plain
    /// `String` behind a derived `Debug`, which is one stray `{:?}` away from a
    /// key in the logs.
    #[test]
    fn debug_does_not_leak_the_api_key() {
        let config = OpenAiLlmConfig {
            api_key: RedactedString::new("sk-live-do-not-log-me"),
            ..Default::default()
        };
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("sk-live"), "key leaked: {rendered}");
        assert!(rendered.contains("[REDACTED]"), "{rendered}");
    }

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
            api_key: RedactedString::new("test-key"),
            ..Default::default()
        };
        let backend = OpenAiLlmBackend::new(config).unwrap();
        let result = backend.complete("system", "user").await;
        assert!(result.is_err());
    }

    /// The styling LLM sends the API key and dictated text to `base_url`, so a
    /// non-loopback plaintext endpoint must be refused — the ASR backend has the
    /// same guard (`crates/lune-asr/src/backend/openai.rs`), and the LLM path
    /// must not be the one that ships the key over cleartext HTTP.
    #[test]
    fn rejects_plaintext_remote_endpoint() {
        let config = OpenAiLlmConfig {
            base_url: "http://api.example.com/v1".into(),
            ..Default::default()
        };
        assert!(OpenAiLlmBackend::new(config).is_err());
    }

    #[test]
    fn accepts_https_remote_endpoint() {
        let config = OpenAiLlmConfig {
            base_url: "https://api.example.com/v1".into(),
            ..Default::default()
        };
        assert!(OpenAiLlmBackend::new(config).is_ok());
    }

    /// Plaintext is only acceptable for loopback-hosted local servers (Ollama,
    /// llama-server, LM Studio) — the same carve-out the ASR backend grants.
    #[test]
    fn accepts_plaintext_loopback_endpoint() {
        for base in [
            "http://127.0.0.1:11434/v1",
            "http://localhost:1234/v1",
            "http://[::1]:8080/v1",
        ] {
            let config = OpenAiLlmConfig {
                base_url: base.into(),
                ..Default::default()
            };
            assert!(
                OpenAiLlmBackend::new(config).is_ok(),
                "{base} should be allowed as loopback"
            );
        }
    }

    /// A private-LAN server (e.g. Ollama on 192.168.x.x) is "local" by the
    /// app's own `is_local_endpoint` semantics — the readiness check even
    /// allows it keyless — so plaintext there must still be accepted, or a
    /// user with a LAN model would get a startup crash.
    #[test]
    fn accepts_plaintext_private_lan_endpoint() {
        let config = OpenAiLlmConfig {
            base_url: "http://192.168.1.50:11434/v1".into(),
            ..Default::default()
        };
        assert!(OpenAiLlmBackend::new(config).is_ok());
    }
}
