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
    #[allow(dead_code)]
    /// Flat byte storage for all tokens; zero-copy slice lookups.
    token_data: Vec<u8>,
    /// (start, end) byte offsets into `token_data` per token ID.
    offsets: Vec<(u32, u32)>,
    pub blank_id: usize,
}

impl Vocabulary {
    /// Load from `vocab.txt` (format: `token_text token_id` per line).
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Asr(format!("open vocab file {}: {e}", path.display())))?;

        // Use a large buffer to minimize syscall overhead on typical vocab files (32–64 KB).
        let reader = BufReader::with_capacity(64 * 1024, file);

        // Pre-allocate assuming ~4 bytes/token average; realloc is cheap.
        let mut token_data: Vec<u8> = Vec::with_capacity(128 * 1024);
        // Typical vocab size ~1024; resize on demand.
        let mut offsets: Vec<(u32, u32)> = Vec::with_capacity(1024);
        let mut blank_id: usize = 0;
        let mut blank_found = false;

        for line in reader.lines() {
            let line = line.map_err(|e| Error::Asr(format!("read vocab: {e}")))?;
            // splitn avoids scanning past the first space for tokens that contain spaces.
            let mut parts = line.splitn(2, ' ');
            if let (Some(token_str), Some(id_str)) = (parts.next(), parts.next()) {
                let id: usize = id_str
                    .trim_end()
                    .parse()
                    .map_err(|e| Error::Asr(format!("invalid token ID: {e}")))?;

                if id >= offsets.len() {
                    // Grow with a sentinel (0, 0) meaning empty string.
                    offsets.resize(id + 1, (0, 0));
                }

                let start = token_data.len() as u32;
                token_data.extend_from_slice(token_str.as_bytes());
                let end = token_data.len() as u32;
                offsets[id] = (start, end);

                if !blank_found && (token_str == "<blk>" || token_str == "<blank>") {
                    blank_id = id;
                    blank_found = true;
                }
            }
        }

        // Default: last token is blank if none found.
        if !blank_found && !offsets.is_empty() {
            blank_id = offsets.len() - 1;
        }

        // Shrink to fit so the allocator can reclaim any over-provisioned memory.
        token_data.shrink_to_fit();
        offsets.shrink_to_fit();

        Ok(Self {
            token_data,
            offsets,
            blank_id,
        })
    }

    /// Look up token text by ID — zero-copy slice into flat byte buffer.
    #[inline]
    pub fn get(&self, id: usize) -> Option<&str> {
        let &(start, end) = self.offsets.get(id)?;
        std::str::from_utf8(&self.token_data[start as usize..end as usize]).ok()
    }

    /// Number of tokens (including blank).
    #[inline]
    pub fn size(&self) -> usize {
        self.offsets.len()
    }

    /// Create an empty vocabulary (for tests).
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            token_data: Vec::new(),
            offsets: Vec::new(),
            blank_id: 0,
        }
    }
}

/// Returns `true` when `token_text` is a special token wrapped in angle
/// brackets (e.g. `<pad>`, `<s>`, `</s>`) that should be skipped during
/// text reconstruction. `<unk>` is **not** considered special — it
/// carries vocabulary-content meaning and must survive into output.
///
/// Centralised here so every backend uses the same definition instead of
/// open-coding the same check in decode loops.
#[inline]
pub fn is_special_token(token_text: &str) -> bool {
    token_text.starts_with('<') && token_text.ends_with('>') && token_text != "<unk>"
}

#[cfg(test)]
mod tests {
    use super::is_special_token;

    #[test]
    fn detects_pad_bos_eos() {
        assert!(is_special_token("<pad>"));
        assert!(is_special_token("<s>"));
        assert!(is_special_token("</s>"));
        assert!(is_special_token("<|endoftext|>"));
    }

    #[test]
    fn unk_is_not_special() {
        // <unk> carries content and must survive into the transcript.
        assert!(!is_special_token("<unk>"));
    }

    #[test]
    fn regular_tokens_not_special() {
        assert!(!is_special_token("hello"));
        assert!(!is_special_token("▁the"));
        assert!(!is_special_token(""));
        assert!(!is_special_token("<"));
        assert!(!is_special_token(">"));
    }
}
