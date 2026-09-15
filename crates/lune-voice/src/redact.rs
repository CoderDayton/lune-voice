// SPDX-License-Identifier: GPL-3.0-or-later

//! Keeping dictated text out of the logs.
//!
//! Every transcript this app touches is, by definition, private speech: the
//! user may be dictating a password, a diagnosis, or a message they would never
//! paste into a bug report. Logs do not stay local — they go to journald, to
//! `RUST_LOG=info` sessions the user is told to run when something breaks, and
//! from there into screenshots and issue threads.
//!
//! So transcript text is redacted by default and only ever rendered verbatim
//! when the user opts in for that run by setting [`TRANSCRIPT_LOG_ENV`]. The
//! redacted form still carries a character count, which is what the logs were
//! actually being read for (did ASR return anything? did styling change the
//! length?) without carrying the content.
//!
//! Log call sites should never interpolate transcript text directly. Wrap it:
//!
//! ```ignore
//! info!(%session_id, transcript = %Transcript::new(&raw_text), "[ASR] complete");
//! ```

use std::fmt;
use std::sync::OnceLock;

/// Environment variable that opts this process into verbatim transcript
/// logging. Any value other than empty/`0`/`false` enables it.
///
/// Deliberately available in release builds: the realistic support path is
/// "reproduce it and send me the log", and that has to work against a shipped
/// binary. What matters is that it is off unless the user turned it on for that
/// run, not that it is impossible.
pub const TRANSCRIPT_LOG_ENV: &str = "LUNE_LOG_TRANSCRIPTS";

/// Whether verbatim transcript logging was enabled for this process.
///
/// Read once and cached — this sits on the per-segment path in `capture`, and
/// an env lookup per dropped segment is pure waste.
fn transcripts_logged() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(TRANSCRIPT_LOG_ENV) {
        Ok(v) => {
            let v = v.trim();
            !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    })
}

/// Display wrapper that renders transcript text as a redaction summary unless
/// [`TRANSCRIPT_LOG_ENV`] is set.
///
/// Borrows rather than owns so wrapping costs nothing at a log site that is
/// filtered out anyway.
pub struct Transcript<'a>(&'a str);

impl<'a> Transcript<'a> {
    pub fn new(text: &'a str) -> Self {
        Self(text)
    }
}

impl fmt::Display for Transcript<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if transcripts_logged() {
            f.write_str(self.0)
        } else {
            // Character count, not byte length: the number is there for a human
            // eyeballing "did anything come back", and bytes mislead on any
            // non-ASCII transcript.
            write!(f, "<redacted {} chars>", self.0.chars().count())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The env var is process-global and `transcripts_logged` caches on first
    /// read, so these tests assert the default (redacted) path only — flipping
    /// the toggle here would race every other test in the binary.
    #[test]
    fn transcript_text_is_redacted_by_default() {
        let rendered = Transcript::new("my banking password is hunter2").to_string();
        assert!(
            !rendered.contains("hunter2"),
            "transcript leaked into its Display: {rendered}"
        );
        assert_eq!(rendered, "<redacted 30 chars>");
    }

    #[test]
    fn redaction_counts_characters_not_bytes() {
        // 5 chars, 10 bytes in UTF-8.
        assert_eq!(Transcript::new("привет").to_string(), "<redacted 6 chars>");
    }

    #[test]
    fn empty_transcript_still_renders() {
        assert_eq!(Transcript::new("").to_string(), "<redacted 0 chars>");
    }
}
