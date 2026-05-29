use std::collections::HashMap;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SupplementalVocab {
    pub categories: Vec<String>,
    pub words: HashMap<String, u8>,
}

impl SupplementalVocab {
    pub fn is_known(&self, word: &str) -> bool {
        self.words.contains_key(word)
    }

    pub fn category_of(&self, word: &str) -> Option<&str> {
        self.words
            .get(word)
            .and_then(|&id| self.categories.get(id as usize))
            .map(|s| s.as_str())
    }

    pub fn category_count(&self) -> usize {
        self.categories.len()
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    pub fn candidates_by_len(&self, lo: usize, hi: usize) -> impl Iterator<Item = &str> {
        self.words
            .keys()
            .filter(move |w| {
                let len = w.len();
                len >= lo && len < hi
            })
            .map(|s| s.as_str())
    }
}

static BUNDLED: LazyLock<SupplementalVocab> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/supplemental_vocab.bin"));
    if bytes.is_empty() {
        return SupplementalVocab {
            categories: Vec::new(),
            words: HashMap::new(),
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
        assert!(v.category_count() >= 4);
    }

    #[test]
    fn lookup_by_category() {
        let v = bundled();
        assert!(v.is_known("WebSocket") || v.is_known("websocket"));
        assert!(v.is_known("Roblox") || v.is_known("roblox"));
        assert!(v.category_of("lune").is_some());
        assert!(!v.is_known("xyznonexistent"));
    }
}
