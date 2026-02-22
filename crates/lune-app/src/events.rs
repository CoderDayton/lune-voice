/// Commands sent from the OS event loop (main thread) to the tokio runtime.
#[derive(Debug, Clone)]
pub enum AppCommand {
    /// Toggle dictation on/off.
    ToggleDictation,
    /// Cancel the current dictation session.
    CancelDictation,
    /// Shut down the application.
    Quit,
}

/// State updates sent from the tokio runtime to the main thread for UI.
#[derive(Debug, Clone)]
pub enum AppStateUpdate {
    /// No active session.
    Idle,
    /// Microphone is active, listening for speech.
    Listening,
    /// Partial (in-progress) transcription text.
    PartialText(String),
    /// Finalized transcription text.
    FinalText(String),
    /// An error occurred in the pipeline.
    Error(String),
}
