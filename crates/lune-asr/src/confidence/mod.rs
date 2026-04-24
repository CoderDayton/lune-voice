//! Standardized per-word confidence extraction from subword tokens.
//!
//! Both ASR backends produce confidence scores at the subword token level.
//! This module provides a unified interface for aggregating token-level
//! scores into per-word `WordConfidence` values, regardless of the
//! tokenizer's word-boundary convention.
//!
//! - **SentencePiece** (Parakeet TDT): word boundaries marked by `▁` prefix
//! - **BPE/GPT2** (Qwen ASR): word boundaries marked by `Ġ` prefix
//!
//! Both produce identical `Vec<WordConfidence>` output.

pub mod calibrator;

pub use calibrator::IsotonicCalibrator;

use lune_core::asr::WordConfidence;

/// A decoded token with its confidence score and text.
pub struct ScoredToken {
    pub text: String,
    pub confidence: f32,
}

/// Strategy for detecting word boundaries in a token sequence.
#[derive(Debug, Clone, Copy)]
pub enum WordBoundaryKind {
    /// SentencePiece: `▁` prefix marks word start (Parakeet TDT)
    SentencePiece,
    /// BPE/GPT2: `Ġ` prefix marks word start (Qwen ASR)
    Bpe,
}

/// Extract per-word confidence from a sequence of scored subword tokens.
///
/// Uses min-aggregation: each word's confidence is the minimum confidence
/// of its constituent tokens. This is conservative — a single uncertain
/// subword flags the entire word for potential LLM correction.
pub fn extract_word_confidences(
    tokens: &[ScoredToken],
    kind: WordBoundaryKind,
) -> Vec<WordConfidence> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut words: Vec<WordConfidence> = Vec::new();
    let mut current_word = String::new();
    let mut current_min_conf = f32::INFINITY;

    for (i, tok) in tokens.iter().enumerate() {
        let trimmed = tok.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        let is_boundary = match kind {
            WordBoundaryKind::SentencePiece => trimmed.starts_with('▁') || i == 0,
            WordBoundaryKind::Bpe => tok.text.starts_with('Ġ') || i == 0,
        };

        if is_boundary && !current_word.is_empty() {
            words.push(WordConfidence::new(
                std::mem::take(&mut current_word),
                current_min_conf,
            ));
            current_min_conf = f32::INFINITY;
        }

        // Strip the boundary marker and append to current word
        let clean = match kind {
            WordBoundaryKind::SentencePiece => trimmed.trim_start_matches('▁'),
            WordBoundaryKind::Bpe => tok.text.trim_start_matches('Ġ').trim(),
        };

        if !clean.is_empty() {
            current_word.push_str(clean);
            current_min_conf = current_min_conf.min(tok.confidence);
        }
    }

    // Flush last word
    if !current_word.is_empty() {
        words.push(WordConfidence::new(current_word, current_min_conf));
    }

    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, conf: f32) -> ScoredToken {
        ScoredToken {
            text: text.to_string(),
            confidence: conf,
        }
    }

    #[test]
    fn sentencepiece_basic() {
        let tokens = vec![tok("▁Hello", 0.9), tok("▁world", 0.8)];
        let words = extract_word_confidences(&tokens, WordBoundaryKind::SentencePiece);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello");
        assert_eq!(words[1].word, "world");
        assert!((words[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn sentencepiece_subwords() {
        let tokens = vec![tok("▁un", 0.9), tok("believ", 0.7), tok("able", 0.8)];
        let words = extract_word_confidences(&tokens, WordBoundaryKind::SentencePiece);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "unbelievable");
        assert!((words[0].confidence - 0.7).abs() < 1e-6); // min of 0.9, 0.7, 0.8
    }

    #[test]
    fn bpe_basic() {
        let tokens = vec![tok("Hello", 0.95), tok("Ġworld", 0.85)];
        let words = extract_word_confidences(&tokens, WordBoundaryKind::Bpe);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello");
        assert_eq!(words[1].word, "world");
    }

    #[test]
    fn empty_tokens() {
        let words = extract_word_confidences(&[], WordBoundaryKind::SentencePiece);
        assert!(words.is_empty());
    }

    #[test]
    fn single_token() {
        let tokens = vec![tok("▁test", 0.5)];
        let words = extract_word_confidences(&tokens, WordBoundaryKind::SentencePiece);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "test");
    }
}
