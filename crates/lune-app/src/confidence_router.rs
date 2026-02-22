//! Confidence-based routing for ASR output.
//!
//! Three-tier routing following CognitivePipe patterns:
//! - **High** (>= 0.82): bypass LLM, deliver instantly.
//! - **Mid** (0.50–0.82): accumulate in pending buffer until 12 words, then batch to LLM.
//! - **Low** (< 0.50): accumulate in pending buffer until 8 words, then batch to LLM.

use std::collections::VecDeque;

/// Configuration for the confidence router.
#[derive(Debug, Clone)]
pub struct ConfidenceRouterConfig {
    /// Confidence threshold above which text skips the LLM (default: 0.82).
    pub high_threshold: f32,
    /// Mid-confidence boundary: below this uses 8-word threshold, above uses 12 (default: 0.50).
    pub mid_threshold: f32,
    /// Number of recent confirmed utterances to retain as context (default: 3).
    pub context_window_size: usize,
    /// Reject LLM correction if output/input length ratio exceeds this (default: 2.0).
    pub max_length_ratio: f32,
    /// Reject LLM correction if output/input length ratio is below this (default: 0.5).
    pub min_length_ratio: f32,
    /// Word count threshold for low-confidence text (< mid_threshold).
    pub low_word_threshold: usize,
    /// Word count threshold for mid-confidence text (mid_threshold..high_threshold).
    pub mid_word_threshold: usize,
}

impl Default for ConfidenceRouterConfig {
    fn default() -> Self {
        Self {
            high_threshold: 0.82,
            mid_threshold: 0.50,
            context_window_size: 3,
            max_length_ratio: 2.0,
            min_length_ratio: 0.5,
            low_word_threshold: 8,
            mid_word_threshold: 12,
        }
    }
}

/// Routing decision for a finalized utterance.
#[derive(Debug, Clone)]
pub enum RouteDecision {
    /// ASR output is high-confidence — deliver as-is, skip LLM.
    HighConfidence { text: String, confidence: f32 },
    /// Text accumulated in pending buffer, word threshold not yet met.
    Pending { confidence: f32 },
    /// Pending buffer reached word threshold — dispatch combined text to LLM.
    ReadyForLlm { text: String, confidence: f32 },
}

/// Routes ASR output based on confidence scores.
///
/// Sub-high-confidence text accumulates in `pending_buf` until a word-count
/// threshold is met (8 words for low, 12 for mid), then drains as a single
/// `ReadyForLlm` batch.
pub struct ConfidenceRouter {
    config: ConfidenceRouterConfig,
    context_window: VecDeque<String>,
    pending_buf: String,
}

impl ConfidenceRouter {
    pub fn new(config: ConfidenceRouterConfig) -> Self {
        Self {
            context_window: VecDeque::with_capacity(config.context_window_size + 1),
            pending_buf: String::new(),
            config,
        }
    }

    /// Decide routing for a finalized utterance.
    ///
    /// - High confidence (>= `high_threshold`): instant delivery, no LLM.
    /// - Below high: accumulates in `pending_buf` until word threshold is met.
    ///   Returns `Pending` while accumulating, `ReadyForLlm` when threshold reached.
    ///
    /// If `confidence` is `None` (e.g. OpenAI backend), treats as 0.0 (low).
    #[inline]
    pub fn route(&mut self, text: &str, confidence: Option<f32>) -> RouteDecision {
        let conf = confidence.unwrap_or(0.0);
        if conf >= self.config.high_threshold {
            RouteDecision::HighConfidence {
                text: text.to_owned(),
                confidence: conf,
            }
        } else {
            self.accumulate(text, conf)
        }
    }

    /// Accumulate text in pending buffer. Returns `ReadyForLlm` when word threshold met.
    fn accumulate(&mut self, text: &str, confidence: f32) -> RouteDecision {
        if !self.pending_buf.is_empty() {
            self.pending_buf.push(' ');
        }
        self.pending_buf.push_str(text);

        let word_threshold = if confidence < self.config.mid_threshold {
            self.config.low_word_threshold
        } else {
            self.config.mid_word_threshold
        };

        let word_count = self.pending_buf.split_whitespace().count();
        if word_count >= word_threshold {
            let batch = std::mem::take(&mut self.pending_buf);
            RouteDecision::ReadyForLlm {
                text: batch,
                confidence,
            }
        } else {
            RouteDecision::Pending { confidence }
        }
    }

    /// Drain any remaining pending text (e.g. on session stop).
    /// Returns `None` if the buffer is empty.
    #[inline]
    pub fn flush_pending(&mut self) -> Option<String> {
        if self.pending_buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.pending_buf))
        }
    }

    /// Confirm an utterance (add to context window for future LLM prompts).
    pub fn confirm(&mut self, text: &str) {
        self.context_window.push_back(text.to_owned());
        while self.context_window.len() > self.config.context_window_size {
            self.context_window.pop_front();
        }
    }

    /// Get joined context from recent confirmed utterances.
    pub fn context(&self) -> Option<String> {
        if self.context_window.is_empty() {
            None
        } else {
            let total_len: usize = self.context_window.iter().map(|s| s.len()).sum::<usize>()
                + self.context_window.len().saturating_sub(1);
            let mut out = String::with_capacity(total_len);
            for (i, s) in self.context_window.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(s);
            }
            Some(out)
        }
    }

    /// Length sanity check for LLM corrections.
    #[inline]
    pub fn length_check_ok(&self, original: &str, corrected: &str) -> bool {
        if original.is_empty() {
            return !corrected.is_empty();
        }
        let ratio = corrected.len() as f32 / original.len() as f32;
        ratio >= self.config.min_length_ratio && ratio <= self.config.max_length_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_confidence_routes_direct() {
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig::default());
        match router.route("hello world", Some(0.95)) {
            RouteDecision::HighConfidence { confidence, .. } => {
                assert!(confidence >= 0.82);
            }
            _ => panic!("expected HighConfidence"),
        }
    }

    #[test]
    fn high_confidence_does_not_accumulate() {
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig::default());
        router.route("hello world", Some(0.95));
        assert!(
            router.flush_pending().is_none(),
            "high confidence should not accumulate"
        );
    }

    #[test]
    fn low_confidence_accumulates_until_threshold() {
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig {
            low_word_threshold: 4,
            ..Default::default()
        });
        // 2 words — below threshold of 4
        match router.route("hello world", Some(0.3)) {
            RouteDecision::Pending { .. } => {}
            other => panic!("expected Pending, got {other:?}"),
        }
        // 2 more words — now at 4, meets threshold
        match router.route("foo bar", Some(0.3)) {
            RouteDecision::ReadyForLlm { text, .. } => {
                assert_eq!(text, "hello world foo bar");
            }
            other => panic!("expected ReadyForLlm, got {other:?}"),
        }
        // Buffer should be empty now
        assert!(router.flush_pending().is_none());
    }

    #[test]
    fn mid_confidence_uses_higher_threshold() {
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig {
            mid_word_threshold: 6,
            low_word_threshold: 3,
            ..Default::default()
        });
        // conf=0.6 is mid (>= 0.50, < 0.82), so threshold = 6 words
        match router.route("one two three", Some(0.6)) {
            RouteDecision::Pending { .. } => {} // 3 words < 6
            other => panic!("expected Pending, got {other:?}"),
        }
        match router.route("four five six", Some(0.6)) {
            RouteDecision::ReadyForLlm { text, .. } => {
                assert_eq!(text, "one two three four five six");
            }
            other => panic!("expected ReadyForLlm, got {other:?}"),
        }
    }

    #[test]
    fn none_confidence_treated_as_low() {
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig {
            low_word_threshold: 2,
            ..Default::default()
        });
        match router.route("hello world", None) {
            RouteDecision::ReadyForLlm { .. } => {} // 2 words meets threshold=2
            other => panic!("expected ReadyForLlm for None confidence, got {other:?}"),
        }
    }

    #[test]
    fn flush_pending_drains_buffer() {
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig::default());
        router.route("partial text", Some(0.4)); // accumulates
        let flushed = router.flush_pending();
        assert_eq!(flushed, Some("partial text".to_string()));
        assert!(
            router.flush_pending().is_none(),
            "second flush should be empty"
        );
    }

    #[test]
    fn context_window_bounded() {
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig {
            context_window_size: 2,
            ..Default::default()
        });
        router.confirm("first");
        router.confirm("second");
        router.confirm("third");
        assert_eq!(router.context(), Some("second third".to_string()));
    }

    #[test]
    fn length_check() {
        let router = ConfidenceRouter::new(ConfidenceRouterConfig::default());
        assert!(router.length_check_ok("hello world", "Hello, world."));
        assert!(!router.length_check_ok(
            "hi",
            "This is a very long replacement text that is way too long"
        ));
        assert!(!router.length_check_ok("a long sentence here", "x"));
    }
}
