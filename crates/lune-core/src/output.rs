use serde::{Deserialize, Serialize};

/// Where text should be routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OutputSink {
    /// Internal notes view (Tauri event).
    NotesView,
    /// OS keyboard injection into active app.
    KeyboardInjection,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_sink_round_trip() {
        let sink = OutputSink::NotesView;
        let json = serde_json::to_string(&sink).unwrap();
        let parsed: OutputSink = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, OutputSink::NotesView);
    }
}
