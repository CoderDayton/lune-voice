// SPDX-License-Identifier: GPL-3.0-or-later

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use lune_core::config::RedactedString;
use lune_core::error::Error;
use lune_core::retry;

use async_openai::error::OpenAIError;
use async_openai::middleware::HttpRequestFactory;

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

/// Read timeout between response chunks. The pipeline's `llm_timeout_secs`
/// bounds the call end to end; this bounds a single stalled read, so a server
/// that accepts the connection and then dribbles bytes still terminates.
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Total bound on one request attempt, matching the ASR client's default.
const HTTP_TOTAL_TIMEOUT: Duration = Duration::from_secs(30);

/// Ceiling on a response body from the styling endpoint. A styled utterance is
/// a few kilobytes of JSON. async-openai buffers whatever body it is handed, so
/// without this cap a misbehaving endpoint could exhaust memory by answering.
const MAX_RESPONSE_BODY: usize = 1024 * 1024;

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
        .connect_timeout(retry::CONNECT_TIMEOUT)
        .read_timeout(HTTP_READ_TIMEOUT)
        .timeout(HTTP_TOTAL_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| Error::llm(format!("build HTTP client: {e}")))
}

/// Executes async-openai's requests with the retry policy the ASR client uses
/// ([`lune_core::retry`]) and refuses a body over [`MAX_RESPONSE_BODY`].
///
/// Installed in place of async-openai's own retry layer, which gives up on a
/// `429` whose body is not OpenAI's JSON error object — the plain-text reply a
/// local server sends — and honours `Retry-After` without a ceiling.
#[derive(Clone)]
struct RemoteService {
    client: reqwest::Client,
}

impl tower::Service<HttpRequestFactory> for RemoteService {
    type Response = reqwest::Response;
    type Error = OpenAIError;
    type Future = Pin<Box<dyn Future<Output = Result<reqwest::Response, OpenAIError>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), OpenAIError>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: HttpRequestFactory) -> Self::Future {
        let client = self.client.clone();
        Box::pin(async move {
            let mut delay = retry::INITIAL_BACKOFF;
            let mut attempt = 1;
            loop {
                let result = client.execute(request.build().await?).await;
                // `Some(server_delay)` marks a failure worth another attempt.
                let transient = match &result {
                    Ok(r) if retry::is_transient_status(r.status().as_u16()) => {
                        Some(retry_after(r.headers()))
                    }
                    Ok(_) => None,
                    Err(_) => Some(None),
                };
                match transient {
                    Some(server_delay) if attempt < retry::MAX_ATTEMPTS => {
                        tracing::warn!(attempt, "styling request failed transiently; retrying");
                        let wait = server_delay.unwrap_or_else(|| retry::jittered(delay));
                        tokio::time::sleep(wait).await;
                        delay *= 2;
                        attempt += 1;
                    }
                    _ => return read_capped(result?).await,
                }
            }
        })
    }
}

/// `Retry-After` from a response, via [`retry::parse_retry_after`].
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    retry::parse_retry_after(headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?)
}

/// Buffer the body up to [`MAX_RESPONSE_BODY`] and hand async-openai a response
/// carrying it, so its own unbounded `bytes()` read only ever sees capped bytes.
async fn read_capped(mut response: reqwest::Response) -> Result<reqwest::Response, OpenAIError> {
    let status = response.status();
    let version = response.version();
    let headers = std::mem::take(response.headers_mut());
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > MAX_RESPONSE_BODY {
            return Err(OpenAIError::Boxed(
                format!(
                    "the styling endpoint sent more than {MAX_RESPONSE_BODY} bytes; refusing to buffer it"
                )
                .into(),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    let mut rebuilt = http::Response::new(body);
    *rebuilt.status_mut() = status;
    *rebuilt.version_mut() = version;
    *rebuilt.headers_mut() = headers;
    Ok(reqwest::Response::from(rebuilt))
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
        // The gate allows loopback, private LAN, and `.local`/`.localhost` —
        // a LAN Ollama or llama-server is "local" and plaintext-safe.
        let url = url::Url::parse(&config.base_url)
            .map_err(|e| Error::llm(format!("invalid base_url: {e}")))?;
        lune_core::config::require_tls_or_local(&url, "LLM").map_err(Error::llm)?;

        let openai_config = OpenAIConfig::new()
            .with_api_key(config.api_key.expose())
            .with_api_base(&config.base_url);
        let http = http_client()?;
        // Requests are built with `http` and executed by `RemoteService`, which
        // carries the shared retry policy and the body cap.
        let client = Client::with_config(openai_config)
            .with_http_client(http.clone())
            .with_http_service(RemoteService { client: http });

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

    const COMPLETION: &str = r#"{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"styled"},"finish_reason":"stop"}]}"#;

    fn http_response(status: &str, extra_headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n{body}",
            body.len()
        )
    }

    /// Serve one canned response per connection, in order, on a loopback port.
    /// Returns the `base_url` to point the backend at.
    async fn serve(responses: Vec<String>) -> String {
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for response in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        format!("http://{addr}/v1")
    }

    /// Read one request through its `Content-Length` body, so the response is
    /// not written while the client is still sending.
    async fn read_request(socket: &mut tokio::net::TcpStream) {
        use tokio::io::AsyncReadExt;

        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = socket.read(&mut chunk).await.unwrap();
            if n == 0 {
                return;
            }
            buf.extend_from_slice(&chunk[..n]);
            let text = String::from_utf8_lossy(&buf);
            if let Some(end) = text.find("\r\n\r\n") {
                let body_len = text[..end]
                    .lines()
                    .find_map(|line| {
                        let line = line.to_ascii_lowercase();
                        line.strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                if buf.len() >= end + 4 + body_len {
                    return;
                }
            }
        }
    }

    fn backend_at(base_url: String) -> OpenAiLlmBackend {
        OpenAiLlmBackend::new(OpenAiLlmConfig {
            base_url,
            ..Default::default()
        })
        .unwrap()
    }

    /// A local llama-server or Ollama that just restarted answers 429/5xx with
    /// plain text rather than OpenAI's JSON error object. That is still a
    /// transient failure, and the ASR client retries it.
    #[tokio::test]
    async fn retries_a_plain_text_rate_limit() {
        let base_url = serve(vec![
            http_response("429 Too Many Requests", "Retry-After: 0\r\n", "slow down"),
            http_response("200 OK", "", COMPLETION),
        ])
        .await;
        let styled = backend_at(base_url).complete("system", "user").await;
        assert_eq!(styled.unwrap(), "styled");
    }

    /// The endpoint is user-configured and untrusted; an answer larger than any
    /// styled utterance must not be buffered into memory.
    #[tokio::test]
    async fn refuses_an_oversized_response_body() {
        let oversized = COMPLETION.replace("styled", &"x".repeat(MAX_RESPONSE_BODY));
        let base_url = serve(vec![http_response("200 OK", "", &oversized)]).await;
        let err = backend_at(base_url)
            .complete("system", "user")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("refusing to buffer"), "{err}");
    }

    /// The styling LLM sends the API key and dictated text to `base_url`, so
    /// the constructor must apply `require_tls_or_local`. The full
    /// accept/reject matrix is tested once in `lune-core`; these two cases
    /// prove this backend is wired to it.
    #[test]
    fn constructor_applies_the_tls_gate() {
        let remote_plaintext = OpenAiLlmConfig {
            base_url: "http://api.example.com/v1".into(),
            ..Default::default()
        };
        assert!(OpenAiLlmBackend::new(remote_plaintext).is_err());

        let lan_plaintext = OpenAiLlmConfig {
            base_url: "http://192.168.1.50:11434/v1".into(),
            ..Default::default()
        };
        assert!(OpenAiLlmBackend::new(lan_plaintext).is_ok());
    }
}
