//! Feedback relay — posts user feedback to a self-hosted proxy that holds the
//! real chat webhook.
//!
//! The proxy exists because a desktop binary cannot keep a secret: anything
//! compiled in is recoverable with `strings`. Shipping the chat webhook itself
//! would let anyone post to that channel at will, at any rate, and the only fix
//! would be cutting a new release. A proxy token can be revoked and rotated
//! server-side instead, and the proxy is the one place that can rate-limit
//! traffic which never went through this app at all.
//!
//! The cooldown here is a courtesy guard against double-clicks and impatience.
//! It is *not* a security control — an attacker with the extracted token skips
//! this process entirely, which is why the proxy must enforce its own limits.

use serde::Serialize;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::WebviewWindow;

use crate::commands::require_window_access;

/// Baked at build time. `option_env!`, not `std::env::var`, because the value
/// has to travel inside the binary — the end user's environment has neither
/// variable set. Runtime variables of the same names still take precedence —
/// which is what lets `LUNE_FEEDBACK_PROXY_URL=… LUNE_FEEDBACK_PROXY_TOKEN=…
/// cgs dev` work without a rebuild — but only as a pair; see `proxy_config`.
const BUILD_PROXY_URL: Option<&str> = option_env!("LUNE_FEEDBACK_PROXY_URL");
const BUILD_PROXY_TOKEN: Option<&str> = option_env!("LUNE_FEEDBACK_PROXY_TOKEN");

/// Minimum gap between two accepted submissions from this process.
const COOLDOWN: Duration = Duration::from_secs(120);
/// Upper bound on message length. Discord's `content` caps at 2000 characters
/// and the header line consumes some of that, so leave headroom.
const MAX_MESSAGE_CHARS: usize = 1800;
/// A blank or near-blank report is noise; make the user say something.
const MIN_MESSAGE_CHARS: usize = 10;
/// The categories the dialog offers. Allow-listed rather than trusted: the
/// value lands verbatim in the bold header line, so an arbitrary string could
/// push `content` past Discord's 2000-character cap that `MAX_MESSAGE_CHARS`
/// budgets for, or smuggle markdown into the header.
const CATEGORIES: [&str; 7] = [
    "Bug",
    "Crash",
    "Accuracy",
    "Performance",
    "Idea",
    "Question",
    "Feedback",
];
/// Total budget for one attempt. Matches the proxy's own timeout expectations.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
/// Retries apply only to 429 — every other failure is terminal.
const MAX_ATTEMPTS: u32 = 3;
/// Clamp a hostile or absurd `Retry-After` so a bad header cannot park the
/// command for hours.
const MAX_RETRY_AFTER: u64 = 60;

/// Last accepted submission. `None` until the first send in this process, so a
/// restart clears the cooldown — acceptable, since the proxy is the real limit.
static LAST_SENT: Mutex<Option<Instant>> = Mutex::new(None);

#[derive(Serialize)]
struct WebhookPayload {
    username: String,
    content: String,
    allowed_mentions: AllowedMentions,
}

/// Empty `parse` disarms `@everyone` / `@here` / role pings, so a hostile
/// report cannot mass-notify the channel. The proxy should re-apply this on its
/// side too — this process is the untrusted party from its point of view.
#[derive(Serialize)]
struct AllowedMentions {
    parse: Vec<String>,
}

/// Whether `url` is safe to send a bearer token to: TLS, or plain HTTP only
/// when it stays on this machine (the usual shape of a local test proxy).
fn is_safe_endpoint(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    match parsed.scheme() {
        "https" => true,
        "http" => matches!(
            parsed.host_str(),
            Some("localhost" | "127.0.0.1" | "[::1]" | "::1")
        ),
        _ => false,
    }
}

/// Resolved proxy endpoint, or `None` when this build was not configured with
/// one. `None` is a normal state, not an error: the UI hides the form.
///
/// The environment override is all-or-nothing on purpose. Resolving the URL
/// and the token independently would let a lone `LUNE_FEEDBACK_PROXY_URL` in
/// the user's session redirect the *build-baked* token to an endpoint of the
/// setter's choosing — handing out the proxy credential. Either both come from
/// the environment, or both come from the build.
fn proxy_config() -> Option<(String, String)> {
    let env_pair = match (
        std::env::var("LUNE_FEEDBACK_PROXY_URL"),
        std::env::var("LUNE_FEEDBACK_PROXY_TOKEN"),
    ) {
        (Ok(url), Ok(token)) => Some((url, token)),
        _ => None,
    };
    let (url, token) = env_pair
        .or_else(|| Some((BUILD_PROXY_URL?.to_string(), BUILD_PROXY_TOKEN?.to_string())))?;
    if url.trim().is_empty() || token.trim().is_empty() {
        return None;
    }
    if !is_safe_endpoint(url.trim()) {
        tracing::warn!("feedback: proxy URL is not https (or local http); disabling the form");
        return None;
    }
    Some((url.trim().to_string(), token))
}

/// Whether `value` is one of the categories the dialog can offer.
fn is_known_category(value: &str) -> bool {
    CATEGORIES.contains(&value)
}

/// Seconds left on the cooldown, or 0 when a send is allowed.
fn cooldown_remaining() -> u64 {
    let guard = LAST_SENT.lock().unwrap_or_else(|e| e.into_inner());
    match *guard {
        Some(at) => COOLDOWN.saturating_sub(at.elapsed()).as_secs(),
        None => 0,
    }
}

/// Whether this build can send feedback at all. The dialog calls this to decide
/// between showing the form and showing only the issue/discussion links.
#[tauri::command]
pub async fn feedback_available(window: WebviewWindow) -> Result<bool, String> {
    require_window_access(&window, &["main"])?;
    Ok(proxy_config().is_some())
}

/// Seconds the UI should keep the submit button disabled. Lets the dialog show
/// an accurate countdown after a reload instead of assuming the gate is open.
#[tauri::command]
pub async fn feedback_cooldown_remaining(window: WebviewWindow) -> Result<u64, String> {
    require_window_access(&window, &["main"])?;
    Ok(cooldown_remaining())
}

/// Send one feedback report. Returns a user-facing message on every failure —
/// the dialog renders the string as-is.
#[tauri::command]
pub async fn submit_feedback(
    window: WebviewWindow,
    app: tauri::AppHandle,
    category: String,
    message: String,
) -> Result<(), String> {
    require_window_access(&window, &["main"])?;

    let category = category.trim();
    if !is_known_category(category) {
        return Err("Unknown feedback category.".into());
    }

    let trimmed = message.trim();
    if trimmed.chars().count() < MIN_MESSAGE_CHARS {
        return Err(format!(
            "Please write at least {MIN_MESSAGE_CHARS} characters so the report is actionable."
        ));
    }
    if trimmed.chars().count() > MAX_MESSAGE_CHARS {
        return Err(format!(
            "Feedback is limited to {MAX_MESSAGE_CHARS} characters. Please shorten it."
        ));
    }

    let Some((proxy_url, proxy_token)) = proxy_config() else {
        return Err("Feedback sending isn't configured in this build.".into());
    };

    // Reserve the slot before the network call so two clicks racing through
    // cannot both pass the check. Restored on failure so a network blip does
    // not cost the user two minutes.
    let previous = {
        let mut guard = LAST_SENT.lock().unwrap_or_else(|e| e.into_inner());
        let remaining = match *guard {
            Some(at) => COOLDOWN.saturating_sub(at.elapsed()).as_secs(),
            None => 0,
        };
        if remaining > 0 {
            return Err(format!(
                "Thanks — please wait {remaining}s before sending more feedback."
            ));
        }
        guard.replace(Instant::now())
    };

    let version = app.package_info().version.to_string();
    let payload = WebhookPayload {
        username: "Lune Voice".to_string(),
        content: format!(
            "**{}** · v{} · {}\n{}",
            category,
            version,
            std::env::consts::OS,
            trimmed
        ),
        allowed_mentions: AllowedMentions { parse: vec![] },
    };

    match deliver(&proxy_url, &proxy_token, &payload).await {
        Ok(()) => Ok(()),
        Err(err) => {
            let mut guard = LAST_SENT.lock().unwrap_or_else(|e| e.into_inner());
            *guard = previous;
            Err(err)
        }
    }
}

/// POST with retry on 429 only. Every error string here is shown to the user.
async fn deliver(url: &str, token: &str, payload: &WebhookPayload) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| {
            tracing::warn!(error = %e, "feedback: http client build failed");
            "Couldn't start the request. Please try again.".to_string()
        })?;

    for attempt in 1..=MAX_ATTEMPTS {
        let response = client
            .post(url)
            .bearer_auth(token)
            .json(payload)
            .send()
            .await;

        let response = match response {
            Ok(r) => r,
            Err(e) => {
                // Transport-level failure: no connection, DNS, TLS, or the
                // 15s budget expired. All of these are worth a plain retry.
                tracing::warn!(error = %e, attempt, "feedback: request failed");
                if attempt == MAX_ATTEMPTS {
                    return Err(if e.is_timeout() {
                        "The request timed out. Check your connection and try again.".into()
                    } else {
                        "Couldn't reach the feedback service. Check your connection and try again."
                            .into()
                    });
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        let status = response.status();
        if status.is_success() {
            return Ok(());
        }

        match status {
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                tracing::warn!(%status, "feedback: proxy rejected the token");
                return Err("This build isn't authorized to send feedback.".into());
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                if attempt == MAX_ATTEMPTS {
                    return Err("The feedback service is busy. Please try again later.".into());
                }
                let wait = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .unwrap_or(2)
                    .clamp(1, MAX_RETRY_AFTER);
                tracing::info!(wait, attempt, "feedback: rate limited, retrying");
                tokio::time::sleep(Duration::from_secs(wait)).await;
            }
            _ => {
                tracing::warn!(%status, "feedback: proxy returned an error");
                return Err(format!(
                    "The feedback service returned an error ({}). Please try again later.",
                    status.as_u16()
                ));
            }
        }
    }

    Err("Couldn't send your feedback after several attempts. Please try again later.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_mentions_serializes_empty_parse() {
        let json = serde_json::to_string(&AllowedMentions { parse: vec![] }).unwrap();
        assert_eq!(json, r#"{"parse":[]}"#);
    }

    #[test]
    fn payload_header_carries_category_version_and_os() {
        let payload = WebhookPayload {
            username: "Lune Voice".into(),
            content: format!("**Bug** · v1.2.3 · {}\nit broke", std::env::consts::OS),
            allowed_mentions: AllowedMentions { parse: vec![] },
        };
        let json = serde_json::to_value(&payload).unwrap();
        let content = json["content"].as_str().unwrap();
        assert!(content.starts_with("**Bug** · v1.2.3 · "));
        assert!(content.ends_with("\nit broke"));
    }

    #[test]
    fn unknown_categories_are_rejected() {
        assert!(is_known_category("Bug"));
        assert!(is_known_category("Feedback"));
        assert!(!is_known_category("bug"));
        assert!(!is_known_category(""));
        assert!(!is_known_category(&"A".repeat(4000)));
    }

    #[test]
    fn proxy_config_is_none_when_url_is_blank() {
        // Guards the "configured but empty" case: a CI build that sets the
        // variable to "" must behave as unconfigured, not post to "".
        temp_env_absent(|| {
            unsafe { std::env::set_var("LUNE_FEEDBACK_PROXY_URL", "   ") };
            unsafe { std::env::set_var("LUNE_FEEDBACK_PROXY_TOKEN", "tok") };
            assert!(proxy_config().is_none());
        });
    }

    #[test]
    fn proxy_config_reads_runtime_override() {
        temp_env_absent(|| {
            unsafe { std::env::set_var("LUNE_FEEDBACK_PROXY_URL", "https://example.test/hook") };
            unsafe { std::env::set_var("LUNE_FEEDBACK_PROXY_TOKEN", "tok") };
            let (url, token) = proxy_config().expect("configured");
            assert_eq!(url, "https://example.test/hook");
            assert_eq!(token, "tok");
        });
    }

    #[test]
    fn proxy_config_ignores_a_url_only_override() {
        // A lone URL in the environment must not pair with the build-baked
        // token: that would post the credential to whoever set the variable.
        temp_env_absent(|| {
            unsafe { std::env::set_var("LUNE_FEEDBACK_PROXY_URL", "https://evil.test/hook") };
            unsafe { std::env::remove_var("LUNE_FEEDBACK_PROXY_TOKEN") };
            // With no build-time pair either, this build is simply unconfigured.
            if BUILD_PROXY_URL.is_none() || BUILD_PROXY_TOKEN.is_none() {
                assert!(proxy_config().is_none());
            } else {
                let (url, _) = proxy_config().expect("build pair");
                assert_ne!(url, "https://evil.test/hook");
            }
        });
    }

    #[test]
    fn non_tls_endpoints_are_rejected_unless_local() {
        assert!(is_safe_endpoint("https://example.test/hook"));
        assert!(is_safe_endpoint("http://localhost:8787/hook"));
        assert!(is_safe_endpoint("http://127.0.0.1:8787/hook"));
        assert!(!is_safe_endpoint("http://example.test/hook"));
        assert!(!is_safe_endpoint("ftp://example.test/hook"));
        assert!(!is_safe_endpoint("not a url"));
    }

    /// These tests mutate process-global env, so they must not overlap.
    fn temp_env_absent(body: impl FnOnce()) {
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev_url = std::env::var("LUNE_FEEDBACK_PROXY_URL").ok();
        let prev_token = std::env::var("LUNE_FEEDBACK_PROXY_TOKEN").ok();
        // Actually make them absent first, so a developer shell that already
        // exports either one cannot leak into the case under test.
        unsafe { std::env::remove_var("LUNE_FEEDBACK_PROXY_URL") };
        unsafe { std::env::remove_var("LUNE_FEEDBACK_PROXY_TOKEN") };
        body();
        match prev_url {
            Some(v) => unsafe { std::env::set_var("LUNE_FEEDBACK_PROXY_URL", v) },
            None => unsafe { std::env::remove_var("LUNE_FEEDBACK_PROXY_URL") },
        }
        match prev_token {
            Some(v) => unsafe { std::env::set_var("LUNE_FEEDBACK_PROXY_TOKEN", v) },
            None => unsafe { std::env::remove_var("LUNE_FEEDBACK_PROXY_TOKEN") },
        }
    }
}
