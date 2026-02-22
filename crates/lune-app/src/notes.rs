use std::io::{self, Write};

use crate::events::AppStateUpdate;

/// Renders session state and transcribed text to the terminal.
///
/// Partials overwrite the current line; finals print on a new line.
pub struct NotesView {
    /// Whether we have a partial line that needs clearing on the next update.
    has_partial: bool,
}

impl NotesView {
    pub fn new() -> Self {
        Self { has_partial: false }
    }

    /// Render a state update to stdout.
    pub fn render(&mut self, update: &AppStateUpdate) {
        let stdout = io::stdout();
        let mut out = stdout.lock();

        match update {
            AppStateUpdate::Idle => {
                self.clear_partial(&mut out);
                let _ = writeln!(out, "[idle] Ready — run `lune-voice toggle` to dictate");
            }
            AppStateUpdate::Listening => {
                self.clear_partial(&mut out);
                let _ = writeln!(out, "[listening] Speak now...");
            }
            AppStateUpdate::PartialText(text) => {
                // Overwrite the current line with the partial.
                let _ = write!(out, "\r\x1b[K  ... {text}");
                let _ = out.flush();
                self.has_partial = true;
            }
            AppStateUpdate::FinalText(text) => {
                self.clear_partial(&mut out);
                let _ = writeln!(out, "  > {text}");
            }
            AppStateUpdate::Error(msg) => {
                self.clear_partial(&mut out);
                let _ = writeln!(out, "[error] {msg}");
            }
        }
    }

    fn clear_partial(&mut self, out: &mut io::StdoutLock<'_>) {
        if self.has_partial {
            let _ = write!(out, "\r\x1b[K");
            let _ = out.flush();
            self.has_partial = false;
        }
    }
}

impl Default for NotesView {
    fn default() -> Self {
        Self::new()
    }
}
