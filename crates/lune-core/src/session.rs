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

impl DictationSession {
    pub fn is_active(&self) -> bool {
        matches!(self.status, SessionStatus::Active)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            SessionStatus::Completed | SessionStatus::Failed(_) | SessionStatus::Cancelled
        )
    }

    /// Duration in milliseconds, if ended.
    pub fn duration_ms(&self) -> Option<i64> {
        self.ended_at
            .map(|end| (end - self.started_at).num_milliseconds())
    }
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
///
/// Uses `Box<str>` for immutable payloads to avoid excess heap re-allocation
/// when cloning and to reduce per-event allocation size.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SessionEvent {
    Started,
    /// Partial in-flight transcription notification.
    PartialText,
    /// Final delivered transcript for a dictation request. Carries the
    /// session id so the delivery path stays correlated end-to-end.
    FinalizedText {
        session_id: Uuid,
        text: String,
    },
    /// Async LLM correction for a previously-delivered low-confidence utterance.
    CorrectedText(String),
    /// Pipeline is processing accumulated audio (batch ASR + LLM).
    Processing,
    /// RMS audio energy level, normalized to [0.0, 1.0]. Emitted at ~15 fps
    /// while recording so the UI can show a live audio visualization.
    AudioLevel(f32),
    CommandDetected {
        session_id: Uuid,
        payload: String,
    },
    /// Gate decision: true = LLM was used, false = bypassed.
    GateDecision {
        used_llm: bool,
        lang: String,
        unresolved: Vec<(String, usize)>,
    },
    Error(String),
    /// Non-fatal anomaly observable by listeners (e.g. a persist write
    /// failed but the session itself is still progressing). Carries human
    /// text; never terminal.
    Warning(String),
    Stopped,
}

impl SessionEvent {
    /// Returns the text payload if this event carries one, avoiding a match at call sites.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::FinalizedText { text, .. } => Some(text.as_str()),
            Self::CommandDetected { payload, .. } => Some(payload.as_str()),
            Self::CorrectedText(s) | Self::Error(s) | Self::Warning(s) => Some(s.as_str()),
            Self::Started
            | Self::PartialText
            | Self::Processing
            | Self::AudioLevel(_)
            | Self::GateDecision { .. }
            | Self::Stopped => None,
        }
    }

    /// True iff this event marks the logical end of a session.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Stopped | Self::Error(_))
    }
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
            style: "casual".into(),
            backend: AsrBackendKind::Qwen3Asr,
            status: SessionStatus::Active,
            transcript: None,
        };
        let json = serde_json::to_string(&session).unwrap();
        let parsed: DictationSession = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, session.id);
        assert_eq!(parsed.style, "casual");
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

    #[test]
    fn session_active_and_terminal_predicates() {
        let mut s = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: "casual".into(),
            backend: AsrBackendKind::Qwen3Asr,
            status: SessionStatus::Active,
            transcript: None,
        };
        assert!(s.is_active());
        assert!(!s.is_terminal());

        s.status = SessionStatus::Completed;
        assert!(!s.is_active());
        assert!(s.is_terminal());
    }

    #[test]
    fn session_duration_none_when_open() {
        let s = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: "casual".into(),
            backend: AsrBackendKind::Qwen3Asr,
            status: SessionStatus::Active,
            transcript: None,
        };
        assert!(s.duration_ms().is_none());
    }

    #[test]
    fn session_event_text_accessor() {
        assert_eq!(
            (SessionEvent::FinalizedText {
                session_id: Uuid::nil(),
                text: "hello".into(),
            })
            .text(),
            Some("hello")
        );
        assert_eq!(SessionEvent::Started.text(), None);
        assert_eq!(SessionEvent::Stopped.text(), None);
    }

    #[test]
    fn session_event_warning_carries_text_and_is_not_terminal() {
        let evt = SessionEvent::Warning("persist offline".into());
        assert_eq!(evt.text(), Some("persist offline"));
        assert!(
            !evt.is_terminal(),
            "Warning is informational; session must keep running",
        );
    }

    #[test]
    fn session_event_terminal_predicate() {
        assert!(SessionEvent::Stopped.is_terminal());
        assert!(SessionEvent::Error("oops".into()).is_terminal());
        assert!(
            !(SessionEvent::FinalizedText {
                session_id: Uuid::nil(),
                text: "hi".into(),
            })
            .is_terminal()
        );
        assert!(!SessionEvent::Started.is_terminal());
    }
}
