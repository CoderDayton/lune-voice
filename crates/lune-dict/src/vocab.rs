use std::collections::HashSet;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

/// The bundled vocabulary, compiled from `data/vocab/*.txt` by `build.rs`.
///
/// Words are stored both lowercased and in their source casing, so a lookup
/// is an exact match against whichever form the caller holds.
#[derive(Debug, Serialize, Deserialize)]
pub struct SupplementalVocab {
    pub words: HashSet<String>,
}

impl SupplementalVocab {
    pub fn is_known(&self, word: &str) -> bool {
        self.words.contains(word)
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }
}

static BUNDLED: LazyLock<SupplementalVocab> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/supplemental_vocab.bin"));
    if bytes.is_empty() {
        return SupplementalVocab {
            words: HashSet::new(),
        };
    }
    let (vocab, _): (SupplementalVocab, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .expect("decode supplemental vocab");
    vocab
});

pub fn bundled() -> &'static SupplementalVocab {
    &BUNDLED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_vocab_loads() {
        let v = bundled();
        assert!(v.word_count() > 100);
    }

    #[test]
    fn known_and_unknown_words() {
        let v = bundled();
        assert!(v.is_known("WebSocket") || v.is_known("websocket"));
        assert!(v.is_known("Roblox") || v.is_known("roblox"));
        assert!(!v.is_known("xyznonexistent"));
    }
}
