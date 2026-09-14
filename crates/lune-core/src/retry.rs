//! Retry policy shared by the remote OpenAI-compatible clients (ASR and the
//! styling LLM), so both back off the same way against the same class of
//! user-configured, untrusted endpoint.

use std::time::Duration;

/// Attempts made before a transient failure becomes a hard error.
pub const MAX_ATTEMPTS: u8 = 3;

/// First backoff window; doubled on each subsequent attempt.
pub const INITIAL_BACKOFF: Duration = Duration::from_secs(2);

/// Ceiling applied to a server-supplied `Retry-After`. The header is remote
/// input: an endpoint that asks us to sleep for an hour would hang the
/// utterance far past any point the user still wants it.
pub const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

/// Connect timeout for a remote endpoint. A host that accepts nothing must fail
/// fast rather than sit on the dictation's latency budget.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Whether an HTTP status is worth retrying: server errors and `429`.
///
/// Transport errors (refused connection, timeout, reset) are transient too; the
/// caller decides that before it has a status.
pub fn is_transient_status(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

/// Equal-jitter backoff: keep half the window fixed so the delay still grows,
/// and randomise the other half.
///
/// A desktop client has no herd to stampede, so this is not about protecting the
/// server. It matters when the endpoint is a *local* llama-server or Ollama that
/// just restarted: without jitter every retry in the process lands on the same
/// tick, so the requests keep colliding on the same cold-start window instead of
/// spreading across it.
pub fn jittered(base: Duration) -> Duration {
    const NANOS_PER_SEC: u128 = 1_000_000_000;

    let half = base / 2;
    // Nanosecond phase as the entropy source. Jitter needs spread, not
    // unpredictability, so a CSPRNG (and the dependency it costs) would be
    // wasted here.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    // Scale the sub-second phase across the whole half-window rather than using
    // it as raw nanoseconds: the phase only spans 1 s, so adding it directly
    // would confine a 2 s half-window's jitter to its first second.
    let spread = u128::from(nanos) * half.as_nanos() / NANOS_PER_SEC;
    half + Duration::from_nanos(u64::try_from(spread).unwrap_or(u64::MAX))
}

/// Parse a `Retry-After` header value, clamped to [`MAX_RETRY_AFTER`].
///
/// Only the delta-seconds form is honoured; the HTTP-date form is rare on the
/// APIs this talks to and [`jittered`] backoff is a fine fallback for it.
pub fn parse_retry_after(raw: &str) -> Option<Duration> {
    let secs: u64 = raw.trim().parse().ok()?;
    Some(Duration::from_secs(secs).min(MAX_RETRY_AFTER))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_stays_within_the_backoff_window() {
        let base = Duration::from_secs(4);
        // Never below half the window (the delay must still grow across
        // attempts) and never above it (backoff must stay bounded).
        for _ in 0..256 {
            let d = jittered(base);
            assert!(d >= base / 2, "{d:?} collapsed the backoff");
            assert!(d <= base, "{d:?} overshot the window");
        }
    }

    #[test]
    fn jitter_actually_varies() {
        let base = Duration::from_secs(4);
        // A constant "jitter" would defeat the point; sample until we see two
        // different values rather than asserting on one draw.
        let first = jittered(base);
        let varied = (0..1000).any(|_| jittered(base) != first);
        assert!(varied, "jitter produced a constant delay");
    }

    #[test]
    fn jitter_handles_a_degenerate_window() {
        assert_eq!(jittered(Duration::ZERO), Duration::ZERO);
    }

    #[test]
    fn retry_after_is_honoured_and_clamped() {
        assert_eq!(parse_retry_after("5"), Some(Duration::from_secs(5)));
        // A hostile or misconfigured endpoint must not park the utterance.
        assert_eq!(parse_retry_after(" 86400 "), Some(MAX_RETRY_AFTER));
    }

    #[test]
    fn unparseable_retry_after_falls_back_to_our_own_backoff() {
        assert_eq!(parse_retry_after(""), None);
        // The HTTP-date form is deliberately not parsed.
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
    }

    #[test]
    fn only_server_errors_and_rate_limits_are_transient() {
        for status in [429, 500, 502, 503, 599] {
            assert!(is_transient_status(status), "{status}");
        }
        for status in [200, 400, 401, 404, 413] {
            assert!(!is_transient_status(status), "{status}");
        }
    }
}
