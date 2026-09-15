// SPDX-License-Identifier: GPL-3.0-or-later

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
    /// Styling preset used for this session, or `None` for no styling.
    pub style: Option<StyleId>,
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
    /// Final delivered transcript for a dictation request. Carries the
    /// session id so the delivery path stays correlated end-to-end.
    FinalizedText {
        session_id: Uuid,
        text: String,
    },
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
    /// Session teardown finished.
    ///
    /// `error_dropped` is true when a preceding `Error` could not be delivered
    /// because the listener stalled: the session failed, but the listener never
    /// saw why. A listener that tracks success/failure must treat such a
    /// `Stopped` as a failure rather than a clean stop.
    Stopped {
        error_dropped: bool,
    },
}

impl SessionEvent {
    /// A clean `Stopped`: no terminal error was lost on the way out.
    pub const fn stopped() -> Self {
        Self::Stopped {
            error_dropped: false,
        }
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
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Active,
            transcript: None,
        };
        let json = serde_json::to_string(&session).unwrap();
        let parsed: DictationSession = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, session.id);
        assert_eq!(parsed.style.as_deref(), Some("casual"));
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
    fn stopped_is_a_clean_stop() {
        assert!(matches!(
            SessionEvent::stopped(),
            SessionEvent::Stopped {
                error_dropped: false
            }
        ));
    }
}
