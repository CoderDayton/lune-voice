use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::asr::AsrBackendKind;
use crate::config::StyleId;

/// A recorded dictation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationSession {
    pub id: Uuid,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub style: StyleId,
    pub backend: AsrBackendKind,
    pub status: SessionStatus,
    /// Final post-processed text.
    pub transcript: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SessionStatus {
    Active,
    Completed,
    Failed(String),
    Cancelled,
}

/// Events emitted during a dictation session.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SessionEvent {
    Started,
    PartialText(String),
    FinalizedText(String),
    /// Async LLM correction for a previously-delivered low-confidence utterance.
    CorrectedText(String),
    CommandDetected(String),
    Error(String),
    Stopped,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictation_session_round_trip() {
        let session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: "general".into(),
            backend: AsrBackendKind::ParakeetTdt,
            status: SessionStatus::Active,
            transcript: None,
        };
        let json = serde_json::to_string(&session).unwrap();
        let parsed: DictationSession = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, session.id);
        assert_eq!(parsed.style, "general");
    }

    #[test]
    fn session_status_failed_round_trip() {
        let status = SessionStatus::Failed("timeout".into());
        let json = serde_json::to_string(&status).unwrap();
        let parsed: SessionStatus = serde_json::from_str(&json).unwrap();
        match parsed {
            SessionStatus::Failed(msg) => assert_eq!(msg, "timeout"),
            _ => panic!("expected Failed variant"),
        }
    }
}
