//! Bounded probing of a display server.
//!
//! Wayland and X11 clients block on socket I/O with no deadline of their own.
//! A compositor or X server that accepts the connection and then stops
//! answering — a wedged session, a display forwarded over a dead SSH link —
//! blocks the caller forever, so every probe in this crate runs on a thread
//! the caller is willing to abandon.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

/// Run `probe` on a thread we are willing to abandon, answering `on_no_answer`
/// if it has not reported back within `timeout`.
///
/// `subject` names the thing being probed; it appears in the give-up log.
///
/// An abandoned thread stays parked in its blocking read until the connection
/// errors out. It holds nothing the caller needs, and its answer is discarded
/// when it finally arrives.
pub(crate) fn bounded<T: Send + 'static>(
    timeout: Duration,
    subject: &'static str,
    on_no_answer: T,
    probe: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = mpsc::channel();
    // The receiver going away is the timeout case; the send failing there is
    // expected, not an error.
    std::thread::spawn(move || {
        let _ = tx.send(probe());
    });
    match rx.recv_timeout(timeout) {
        Ok(answer) => answer,
        Err(RecvTimeoutError::Timeout) => {
            tracing::warn!(
                subject,
                timeout_ms = timeout.as_millis() as u64,
                "probe did not answer in time; treating it as unreachable"
            );
            on_no_answer
        }
        // The sender was dropped without a value, so the probe panicked.
        // Reporting it as a timeout would send debugging after a hang that
        // never happened.
        Err(RecvTimeoutError::Disconnected) => {
            tracing::warn!(subject, "probe panicked; treating it as unreachable");
            on_no_answer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::bounded;
    use std::time::Duration;

    #[test]
    fn a_probe_that_answers_is_believed() {
        assert!(bounded(Duration::from_secs(30), "test", false, || true));
        assert!(!bounded(Duration::from_secs(30), "test", false, || false));
    }

    #[test]
    fn a_probe_may_answer_with_any_type() {
        let answer = bounded(Duration::from_secs(30), "test", None, || Some(7u32));
        assert_eq!(answer, Some(7));
    }

    #[test]
    fn a_stalled_probe_gives_up_instead_of_blocking_startup() {
        let started = std::time::Instant::now();
        let answered = bounded(Duration::from_millis(50), "test", false, || {
            std::thread::sleep(Duration::from_secs(30));
            true
        });
        assert!(
            !answered,
            "a probe that never answers must not report success"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the probe blocked for {:?} instead of timing out",
            started.elapsed()
        );
    }

    /// A panicking probe drops its sender without a value. That must read as
    /// "no answer", not as a hang, and must not propagate into the caller.
    #[test]
    fn a_panicking_probe_answers_with_the_fallback() {
        let answered = bounded(Duration::from_secs(30), "test", false, || {
            panic!("probe blew up");
        });
        assert!(!answered);
    }
}
