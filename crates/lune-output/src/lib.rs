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
    /// Fast path: first emission or after reset — no prefix to preserve.
    /// Otherwise delegates to `char_diff` for incremental editing.
    #[inline]
    pub fn compute_action(&self, old: &str, new: &str) -> OutputAction {
        if old.is_empty() {
            return OutputAction::ReplaceAll(new.to_owned());
        }
        let (delete_count, suffix) = char_diff(old, new);
        OutputAction::DeleteThenType {
            delete_count,
            insert: suffix.to_owned(),
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
///
/// Hot path optimisations:
///   - ASCII text uses byte comparison (~3× faster than char iteration).
///   - Unicode text uses a single combined pass to find prefix byte offsets,
///     then only counts tail chars in `old` rather than the full string.
#[inline]
pub fn char_diff<'a>(old: &str, new: &'a str) -> (usize, &'a str) {
    // ASCII fast path: byte identity implies char identity, no UTF-8 decode.
    if old.is_ascii() && new.is_ascii() {
        let old_b = old.as_bytes();
        let new_b = new.as_bytes();
        let common = old_b.iter().zip(new_b).take_while(|(a, b)| a == b).count();
        return (old_b.len() - common, &new[common..]);
    }

    // Unicode path: single combined pass to locate the common-prefix boundary
    // by byte offset in both strings simultaneously.
    let mut byte_old = 0usize;
    let mut byte_new = 0usize;

    let mut oi = old.char_indices();
    let mut ni = new.char_indices();
    loop {
        match (oi.next(), ni.next()) {
            (Some((ob, oc)), Some((nb, nc))) if oc == nc => {
                byte_old = ob + oc.len_utf8();
                byte_new = nb + nc.len_utf8();
            }
            _ => break,
        }
    }

    // Count only the tail of `old` — avoids a redundant full-string pass.
    let delete_count = old[byte_old..].chars().count();
    (delete_count, &new[byte_new..])
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
