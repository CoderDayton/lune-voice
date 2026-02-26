use tokio::sync::mpsc;

use lune_core::session::SessionEvent;

/// Routes post-processed text to event consumers (D-Bus, delivery, logging).
pub struct TextOutputController {
    event_tx: mpsc::Sender<SessionEvent>,
}

impl TextOutputController {
    pub fn new(event_tx: mpsc::Sender<SessionEvent>) -> Self {
        Self { event_tx }
    }

    /// Emit a partial (in-progress) text update.
    pub async fn emit_partial(&mut self, text: String) {
        let _ = self.event_tx.send(SessionEvent::PartialText(text)).await;
    }

    /// Emit finalized text — triggers delivery to the focused input.
    pub async fn emit_final(&mut self, text: String) {
        let _ = self.event_tx.send(SessionEvent::FinalizedText(text)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
