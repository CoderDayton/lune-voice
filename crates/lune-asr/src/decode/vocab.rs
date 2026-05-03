//! Vocabulary loader for TDT models (vocab.txt format).
//!
//! Ported from `parakeet-rs/src/vocab.rs`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use lune_core::error::{Error, Result};

/// Token vocabulary mapping IDs → text, with sorted token list for tokenization.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Flat byte storage for all tokens; zero-copy slice lookups.
    token_data: Vec<u8>,
    /// (start, end) byte offsets into `token_data` per token ID.
    offsets: Vec<(u32, u32)>,
    pub blank_id: usize,
    /// Tokens sorted by descending byte length for maximal-munch tokenization.
    sorted_tokens: Vec<(String, usize)>,
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

        // Build sorted token list for tokenization.
        let mut sorted_tokens = Vec::with_capacity(offsets.len());
        for (id, &(start, end)) in offsets.iter().enumerate() {
            if start == end {
                continue;
            }
            if let Ok(text) = std::str::from_utf8(&token_data[start as usize..end as usize]) {
                if is_special_token(text) {
                    continue;
                }
                let lower = text.to_lowercase();
                sorted_tokens.push((lower, id));
            }
        }
        // Sort by descending length for maximal-munch.
        sorted_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(Self {
            token_data,
            offsets,
            blank_id,
            sorted_tokens,
        })
    }

    /// Look up token text by ID — zero-copy slice into flat byte buffer.
    pub fn get(&self, id: usize) -> Option<&str> {
        let &(start, end) = self.offsets.get(id)?;
        std::str::from_utf8(&self.token_data[start as usize..end as usize]).ok()
    }

    /// Number of tokens (including blank).
    pub fn size(&self) -> usize {
        self.offsets.len()
    }

    /// Tokenize a phrase into token IDs using left-to-right maximal munch.
    /// Returns empty if no in-vocab tokens match.
    pub fn tokenize_greedy(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        // Normalize: lowercase, prepend ▁ for word-initial position.
        let normalized = format!("▁{}", text.to_lowercase().replace(' ', " ▁"));
        let mut pos = 0;
        let mut ids = Vec::new();

        while pos < normalized.len() {
            let remaining = &normalized[pos..];
            let mut matched = false;
            for (token, id) in &self.sorted_tokens {
                if remaining.starts_with(token.as_str()) {
                    ids.push(*id);
                    pos += token.len();
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Skip one UTF-8 character (OOV).
                let c = remaining.chars().next().unwrap();
                pos += c.len_utf8();
            }
        }
        ids
    }

    /// Create an empty vocabulary (for tests).
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            token_data: Vec::new(),
            offsets: Vec::new(),
            blank_id: 0,
            sorted_tokens: Vec::new(),
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
