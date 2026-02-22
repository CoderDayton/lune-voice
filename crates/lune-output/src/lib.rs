use tokio::sync::mpsc;
use tracing::debug;

use lune_core::session::SessionEvent;

/// An action the output layer should perform.
///
/// For v0.1 (NotesView), only `ReplaceAll` is used.
/// `DeleteThenType` is used for keyboard injection (v0.2+).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputAction {
    /// Replace all currently-displayed text with this string.
    ReplaceAll(String),
    /// Delete `delete_count` characters, then type `insert`.
    DeleteThenType { delete_count: usize, insert: String },
}

/// Routes post-processed text to event consumers (D-Bus, delivery, logging).
///
/// Partials are always emitted. Delivery to the focused input or clipboard
/// happens externally based on `DictationMode`.
pub struct TextOutputController {
    event_tx: mpsc::Sender<SessionEvent>,
    /// Last partial text sent, for diff computation.
    last_partial: String,
}

impl TextOutputController {
    pub fn new(event_tx: mpsc::Sender<SessionEvent>) -> Self {
        Self {
            event_tx,
            last_partial: String::new(),
        }
    }

    /// Emit a partial (in-progress) text update.
    pub async fn emit_partial(&mut self, text: String) {
        let action = self.compute_action(&self.last_partial, &text);
        self.last_partial.clone_from(&text);
        self.send_event(SessionEvent::PartialText(text), &action)
            .await;
    }

    /// Emit finalized text.  Always sent regardless of mode.
    pub async fn emit_final(&mut self, text: String) {
        let action = self.compute_action(&self.last_partial, &text);
        self.last_partial.clear();
        self.send_event(SessionEvent::FinalizedText(text), &action)
            .await;
    }

    /// Clear in-progress state (e.g., on cancel).
    pub fn reset(&mut self) {
        self.last_partial.clear();
    }

    /// Compute the minimal correction action from `old` → `new`.
    ///
    /// For NotesView the `ReplaceAll` variant is always used.
    /// For future keyboard injection, `DeleteThenType` avoids re-typing the
    /// common prefix.
    #[inline]
    pub fn compute_action(&self, old: &str, new: &str) -> OutputAction {
        let (backspaces, suffix) = char_diff(old, new);
        if backspaces == 0 && suffix == new {
            // No common prefix at all — just replace everything.
            OutputAction::ReplaceAll(new.to_owned())
        } else if backspaces == 0 {
            // Pure append.
            OutputAction::DeleteThenType {
                delete_count: 0,
                insert: suffix.to_owned(),
            }
        } else {
            OutputAction::DeleteThenType {
                delete_count: backspaces,
                insert: suffix.to_owned(),
            }
        }
    }

    #[inline]
    async fn send_event(&self, event: SessionEvent, action: &OutputAction) {
        debug!(?action, "output action");
        // Dropping the error is intentional: if the receiver is gone the
        // session is already tearing down.
        let _ = self.event_tx.send(event).await;
    }
}

/// Compute the number of characters to delete and the new suffix when
/// transitioning from `old` to `new`.
///
/// Returns `(delete_count, new_suffix)` where `new_suffix` is a slice of `new`.
#[inline]
pub fn char_diff<'a>(old: &str, new: &'a str) -> (usize, &'a str) {
    let common = old
        .chars()
        .zip(new.chars())
        .take_while(|(a, b)| a == b)
        .count();

    let old_char_count = old.chars().count();
    let delete_count = old_char_count - common;

    // Byte offset of the common prefix end in `new`.
    let suffix_byte_start = new
        .char_indices()
        .nth(common)
        .map(|(i, _)| i)
        .unwrap_or(new.len());

    (delete_count, &new[suffix_byte_start..])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── char_diff ─────────────────────────────────────────────────────────

    #[test]
    fn diff_pure_append() {
        let (del, suffix) = char_diff("hello", "hello world");
        assert_eq!(del, 0);
        assert_eq!(suffix, " world");
    }

    #[test]
    fn diff_replace_tail() {
        // "hello worm" → "hello world": common prefix = "hello wor" (9 chars),
        // delete 'm', then type "ld".
        let (del, suffix) = char_diff("hello worm", "hello world");
        assert_eq!(del, 1); // only 'm' differs
        assert_eq!(suffix, "ld");
    }

    #[test]
    fn diff_full_replacement() {
        let (del, suffix) = char_diff("abc", "xyz");
        assert_eq!(del, 3);
        assert_eq!(suffix, "xyz");
    }

    #[test]
    fn diff_empty_old() {
        let (del, suffix) = char_diff("", "hello");
        assert_eq!(del, 0);
        assert_eq!(suffix, "hello");
    }

    #[test]
    fn diff_empty_new() {
        let (del, suffix) = char_diff("hello", "");
        assert_eq!(del, 5);
        assert_eq!(suffix, "");
    }

    #[test]
    fn diff_identical() {
        let (del, suffix) = char_diff("same", "same");
        assert_eq!(del, 0);
        assert_eq!(suffix, "");
    }

    #[test]
    fn diff_unicode() {
        // "héllo" vs "héllo monde" — é is multi-byte
        let (del, suffix) = char_diff("héllo", "héllo monde");
        assert_eq!(del, 0);
        assert_eq!(suffix, " monde");
    }

    // ── compute_action ────────────────────────────────────────────────────

    #[tokio::test]
    async fn emit_partial_sent() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut ctrl = TextOutputController::new(tx);
        ctrl.emit_partial("hello".into()).await;
        let event = rx.try_recv().unwrap();
        assert!(matches!(event, SessionEvent::PartialText(t) if t == "hello"));
    }

    #[tokio::test]
    async fn emit_final_sent() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut ctrl = TextOutputController::new(tx);
        ctrl.emit_final("done".into()).await;
        let event = rx.try_recv().unwrap();
        assert!(matches!(event, SessionEvent::FinalizedText(t) if t == "done"));
    }

    #[tokio::test]
    async fn reset_clears_partial() {
        let (tx, _rx) = mpsc::channel(4);
        let mut ctrl = TextOutputController::new(tx);
        ctrl.emit_partial("partial text".into()).await;
        ctrl.reset();
        assert!(ctrl.last_partial.is_empty());
    }
}
