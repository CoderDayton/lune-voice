//! Vocabulary loader for TDT models (vocab.txt format).
//!
//! Ported from `parakeet-rs/src/vocab.rs`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use lune_core::error::{Error, Result};

/// Token vocabulary mapping IDs → text.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub id_to_token: Vec<String>,
    #[allow(dead_code)]
    pub blank_id: usize,
}

impl Vocabulary {
    /// Load from `vocab.txt` (format: `token_text token_id` per line).
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Asr(format!("open vocab file {}: {e}", path.display())))?;

        let reader = BufReader::new(file);
        let mut id_to_token = Vec::new();
        let mut blank_id = 0;

        for line in reader.lines() {
            let line = line.map_err(|e| Error::Asr(format!("read vocab: {e}")))?;
            let mut parts = line.splitn(2, ' ');
            if let (Some(token_str), Some(id_str)) = (parts.next(), parts.next()) {
                let id: usize = id_str
                    .parse()
                    .map_err(|e| Error::Asr(format!("invalid token ID: {e}")))?;

                if id >= id_to_token.len() {
                    id_to_token.resize(id + 1, String::new());
                }
                id_to_token[id] = token_str.to_string();

                if token_str == "<blk>" || token_str == "<blank>" {
                    blank_id = id;
                }
            }
        }

        // Default: last token is blank if none found.
        if blank_id == 0 && !id_to_token.is_empty() {
            blank_id = id_to_token.len() - 1;
        }

        Ok(Self {
            id_to_token,
            blank_id,
        })
    }

    /// Look up token text by ID.
    #[inline]
    pub fn get(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Number of tokens (including blank).
    #[inline]
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }
}
