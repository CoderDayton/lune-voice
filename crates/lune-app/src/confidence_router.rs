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
    /// Flat string accumulator; we track word count separately to avoid
    /// re-scanning the full buffer on every call.
    pending_buf: String,
    /// Incrementally maintained word count for `pending_buf`.
    pending_word_count: usize,
}

impl ConfidenceRouter {
    pub fn new(config: ConfidenceRouterConfig) -> Self {
        // Pre-allocate pending buffer: typical utterance ~64 bytes, threshold up
        // to 12 words ≈ ~80 chars.
        let pending_buf = String::with_capacity(128);
        Self {
            context_window: VecDeque::with_capacity(config.context_window_size + 1),
            pending_buf,
            pending_word_count: 0,
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
    ///
    /// Word count is maintained incrementally — O(new words only), not O(total buffer).
    #[inline(always)]
    fn accumulate(&mut self, text: &str, confidence: f32) -> RouteDecision {
        // Append with separator
        if !self.pending_buf.is_empty() {
            self.pending_buf.push(' ');
        }
        self.pending_buf.push_str(text);

        // Count only the new words, not the whole buffer
        self.pending_word_count += count_words(text);

        let word_threshold = if confidence < self.config.mid_threshold {
            self.config.low_word_threshold
        } else {
            self.config.mid_word_threshold
        };

        if self.pending_word_count >= word_threshold {
            // Reset word count before take so state is consistent
            self.pending_word_count = 0;
            let batch = std::mem::take(&mut self.pending_buf);
            // Restore capacity after take
            self.pending_buf = String::with_capacity(128);
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
            self.pending_word_count = 0;
            let buf = std::mem::take(&mut self.pending_buf);
            self.pending_buf = String::with_capacity(128);
            Some(buf)
        }
    }

    /// Confirm an utterance (add to context window for future LLM prompts).
    pub fn confirm(&mut self, text: &str) {
        self.context_window.push_back(text.to_owned());
        // Pop from front when over capacity — avoids shift cost of Vec
        while self.context_window.len() > self.config.context_window_size {
            self.context_window.pop_front();
        }
    }

    /// Get joined context from recent confirmed utterances.
    ///
    /// Pre-computes exact capacity to avoid reallocation.
    pub fn context(&self) -> Option<String> {
        if self.context_window.is_empty() {
            return None;
        }
        // Exact capacity: sum of string lens + (n-1) spaces
        let n = self.context_window.len();
        let chars: usize = self.context_window.iter().map(|s| s.len()).sum();
        let capacity = chars + n.saturating_sub(1);
        let mut out = String::with_capacity(capacity);
        for (i, s) in self.context_window.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(s);
        }
        Some(out)
    }

    /// Length sanity check for LLM corrections.
    ///
    /// Uses byte lengths (fast). For ASCII-dominant speech transcripts this is
    /// equivalent to char-based ratios; flagging extreme ratios is the goal.
    #[inline]
    pub fn length_check_ok(&self, original: &str, corrected: &str) -> bool {
        let orig_len = original.len();
        if orig_len == 0 {
            return !corrected.is_empty();
        }
        // Integer multiply/divide to stay in integer domain, avoids f32 cast
        // ratio = corrected.len() / orig_len; check min <= ratio <= max
        // Equivalent: corrected.len() * denom_scale within [min*orig, max*orig]
        // But thresholds are 0.5 and 2.0 — simple bit-shift friendly:
        //   min: corrected >= orig / 2  (i.e. corrected*2 >= orig)
        //   max: corrected <= orig * 2
        let corr_len = corrected.len();
        // corrected >= original * min_ratio  →  corrected * 2 >= original
        let above_min = (corr_len << 1) >= orig_len; // works for 0.5 default
        // corrected <= original * max_ratio  →  corrected <= original * 2
        let below_max = corr_len <= (orig_len << 1); // works for 2.0 default

        // Fast path for default thresholds — if non-default, fall back to f32
        if self.config.min_length_ratio == 0.5 && self.config.max_length_ratio == 2.0 {
            above_min && below_max
        } else {
            let ratio = corr_len as f32 / orig_len as f32;
            ratio >= self.config.min_length_ratio && ratio <= self.config.max_length_ratio
        }
    }
}

/// Count whitespace-delimited words in a string slice without allocation.
///
/// Equivalent to `s.split_whitespace().count()` but avoids iterator overhead
/// by using a simple state machine on bytes.
#[inline]
fn count_words(s: &str) -> usize {
    let mut count = 0usize;
    let mut in_word = false;
    // Operate on bytes — ASCII whitespace is single-byte; multi-byte UTF-8 chars
    // are never ASCII whitespace, so this correctly counts word boundaries.
    for b in s.bytes() {
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
        if in_word && ws {
            count += 1;
            in_word = false;
        } else if !ws {
            in_word = true;
        }
    }
    if in_word {
        count += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_words_basic() {
        assert_eq!(count_words(""), 0);
        assert_eq!(count_words("  "), 0);
        assert_eq!(count_words("hello"), 1);
        assert_eq!(count_words("hello world"), 2);
        assert_eq!(count_words("  hello   world  "), 2);
        assert_eq!(count_words("one two three four"), 4);
    }

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
        assert_eq!(router.pending_word_count, 0);
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
        assert_eq!(router.pending_word_count, 0);
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

    #[test]
    fn length_check_empty_original() {
        let router = ConfidenceRouter::new(ConfidenceRouterConfig::default());
        assert!(!router.length_check_ok("", ""));
        assert!(router.length_check_ok("", "something"));
    }

    #[test]
    fn incremental_word_count_matches_split_whitespace() {
        // Invariant: pending_word_count should equal the split_whitespace count
        // of the current pending_buf after each accumulation.
        let mut router = ConfidenceRouter::new(ConfidenceRouterConfig {
            low_word_threshold: 100, // prevent flush
            ..Default::default()
        });
        let phrases = ["hello world", "foo bar baz", "one"];
        for phrase in &phrases {
            router.route(phrase, Some(0.3));
        }
        let expected = router.pending_buf.split_whitespace().count();
        assert_eq!(router.pending_word_count, expected);
    }
}
