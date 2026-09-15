// SPDX-License-Identifier: GPL-3.0-or-later

//! Bounded probing of a display server.
//!
//! Wayland and X11 clients block on socket I/O with no deadline of their own.
//! A compositor or X server that accepts the connection and then stops
//! answering — a wedged session, a display forwarded over a dead SSH link —
//! blocks the caller forever, so every probe in this crate runs on a thread
//! the caller is willing to abandon.

use std::io::Read;
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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

/// Spawn `cmd` with `args`, collect stdout/stderr, kill if it doesn't finish
/// within `timeout`. Returns `None` on spawn failure or timeout.
///
/// `try_wait` polling at 10ms; the upper bound on overshoot is ~10ms. Stdin is
/// closed (`Stdio::null`) so children that read stdin don't block.
///
/// Both pipes are drained by their own thread for the whole wait. A pipe holds
/// about 64 KiB before the writer blocks, and waiting for exit without reading
/// would deadlock every child with more to say than that — `wl-paste` on a
/// clipboard holding an image, `swaymsg -t get_tree` on a busy session — until
/// the timeout killed a process that was working fine.
pub(crate) fn output_with_timeout(cmd: &str, args: &[&str], timeout: Duration) -> Option<Output> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // The child is gone, so both pipes are at EOF and these joins
                // return as soon as the readers finish their last chunk.
                return Some(Output {
                    status,
                    stdout: stdout.join().unwrap_or_default(),
                    stderr: stderr.join().unwrap_or_default(),
                });
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    tracing::debug!(
                        cmd,
                        timeout_ms = timeout.as_millis() as u64,
                        "subprocess timeout"
                    );
                    return None;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Read one of the child's pipes to EOF on its own thread.
///
/// Every path that spawns a child with a piped stdout or stderr needs this: a
/// pipe holds about 64 KiB before the writer blocks, so an unread pipe wedges a
/// chatty child short of exit and the caller's deadline then kills a process
/// that was working.
pub(crate) fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    })
}

/// Wait for an already-spawned child, killing it if it outlives `timeout`.
///
/// For children the caller had to spawn itself — one it writes to on stdin,
/// say — where [`output_with_timeout`] cannot be used. `None` means the child
/// was killed at the deadline or could not be waited on.
///
/// The caller owns any piped stdout/stderr: an undrained pipe wedges a chatty
/// child short of exit and this deadline then kills work in progress.
pub(crate) fn wait_with_timeout(child: &mut Child, timeout: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    tracing::debug!(
                        timeout_ms = timeout.as_millis() as u64,
                        "child did not exit within its deadline; killed"
                    );
                    return None;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Status-only variant of [`output_with_timeout`] for fire-and-forget commands.
pub(crate) fn status_with_timeout(
    cmd: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<ExitStatus> {
    output_with_timeout(cmd, args, timeout).map(|o| o.status)
}

#[cfg(test)]
mod tests {
    use super::{bounded, output_with_timeout, wait_with_timeout};
    use std::time::Duration;

    #[test]
    fn a_spawned_child_that_never_exits_is_killed_at_the_deadline() {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let started = std::time::Instant::now();
        assert!(wait_with_timeout(&mut child, Duration::from_millis(50)).is_none());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_spawned_child_that_exits_reports_its_status() {
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn true");
        let status = wait_with_timeout(&mut child, Duration::from_secs(10))
            .expect("a fast child must not read as a timeout");
        assert!(status.success());
    }

    /// A pipe holds ~64 KiB before the writer blocks. Waiting for exit without
    /// reading deadlocks any child with more than that to say — a clipboard
    /// holding an image, a sway tree with many windows — and the timeout then
    /// kills a process that was working fine.
    #[test]
    fn a_child_that_outruns_the_pipe_buffer_is_still_read() {
        let bytes = 512 * 1024;
        let out = output_with_timeout(
            "sh",
            &["-c", &format!("head -c {bytes} /dev/zero")],
            Duration::from_secs(10),
        )
        .expect("a fast, chatty child must not read as a timeout");
        assert!(out.status.success());
        assert_eq!(out.stdout.len(), bytes);
    }

    #[test]
    fn a_child_that_never_exits_is_killed() {
        let started = std::time::Instant::now();
        assert!(output_with_timeout("sleep", &["30"], Duration::from_millis(50)).is_none());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

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
