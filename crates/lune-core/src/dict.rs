use serde::{Deserialize, Serialize};

/// A user-specific dictionary entry for ASR correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    /// Spoken form (what ASR might produce).
    pub phrase: String,
    /// Correct casing/spelling.
    pub replacement: String,
    /// Higher = applied first.
    pub priority: u8,
}

/// A reusable text template triggered by a spoken phrase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snippet {
    pub id: String,
    pub trigger_phrase: String,
    /// May contain `{date}`, `{name}`, etc.
    pub body: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_entry_round_trip() {
        let entry = DictionaryEntry {
            phrase: "rust lang".into(),
            replacement: "Rust".into(),
            priority: 10,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: DictionaryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.phrase, "rust lang");
        assert_eq!(parsed.replacement, "Rust");
        assert_eq!(parsed.priority, 10);
    }

    #[test]
    fn snippet_round_trip() {
        let snippet = Snippet {
            id: "greeting".into(),
            trigger_phrase: "insert greeting".into(),
            body: "Hello! Today is {date}. How can I help you?".into(),
        };
        let json = serde_json::to_string(&snippet).unwrap();
        let parsed: Snippet = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "greeting");
        assert!(parsed.body.contains("{date}"));
    }
}
