use serde::{Deserialize, Serialize};

use crate::Result;

/// Per-word confidence derived from subword BPE token log-probabilities.
/// Confidence = minimum exp(log_prob) across constituent subword tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordConfidence {
    pub word: String,
    pub confidence: f32,
}

/// A segment of recognized speech.
///
/// Fields are ordered hot-first: `id`, `confidence`, `is_final` land on the
/// same cache line so routing decisions don't pay for a `text` load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrSegment {
    pub id: u64,
    /// Mean token-level confidence from the decoder (0.0–1.0).
    /// `None` for backends that don't expose logits (e.g. OpenAI API).
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Minimum single-token confidence in the segment (0.0–1.0).
    /// Used for word-level gating: even if the mean is high, one
    /// low-confidence token flags the segment for LLM review.
    #[serde(default)]
    pub min_token_confidence: Option<f32>,
    /// Per-word confidence from BPE subword aggregation (min of constituent tokens).
    /// Empty for backends that don't expose per-token logits.
    #[serde(default)]
    pub word_confidences: Vec<WordConfidence>,
    /// Immutable once true.
    pub is_final: bool,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub text: String,
}

impl AsrSegment {
    /// Returns `true` when confidence meets or exceeds `threshold`, or when
    /// confidence is unavailable (conservative: assume pass).
    /// Inline so routing hot-loops don't pay a call frame.
    #[inline(always)]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence.is_none_or(|c| c >= threshold)
    }

    /// Two-level confidence gate: sentence-level mean AND min-token check.
    /// Per arXiv:2407.21414, sentence threshold=0.95 + word threshold=0.7
    /// catches segments where one bad token drags overall quality down.
    #[inline(always)]
    pub fn is_confident_gated(&self, mean_threshold: f32, min_token_threshold: f32) -> bool {
        self.is_confident(mean_threshold)
            && self
                .min_token_confidence
                .is_none_or(|m| m >= min_token_threshold)
    }

    /// Duration of the segment in milliseconds, if both timestamps are known.
    #[inline]
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_ms?.checked_sub(self.start_ms?)
    }

    /// Build text with low-confidence words wrapped in `[?word?]` markers.
    /// Falls back to `self.text.clone()` when no per-word data is available.
    pub fn mark_uncertain(&self, threshold: f32) -> String {
        if self.word_confidences.is_empty() {
            return self.text.clone();
        }
        let mut parts: Vec<String> = Vec::with_capacity(self.word_confidences.len());
        let mut prev_word: Option<&str> = None;
        for wc in &self.word_confidences {
            // Never mark a word as uncertain if it repeats the previous word —
            // the LLM incorrectly "corrects" intentional repetitions away.
            let is_repeat = prev_word.map_or(false, |pw| pw.eq_ignore_ascii_case(&wc.word));
            if wc.confidence < threshold && !is_repeat {
                parts.push(format!("[?{}?]", wc.word));
            } else {
                parts.push(wc.word.clone());
            }
            prev_word = Some(&wc.word);
        }
        parts.join(" ")
    }

    /// Contextual confidence marking — adjusts per-word thresholds based
    /// on position and neighbor confidence.
    ///
    /// ASR errors are bursty: noise affects consecutive words. A word with
    /// borderline confidence surrounded by high-confidence neighbors is
    /// likely correct (clean acoustic region). A borderline word surrounded
    /// by low-confidence neighbors is likely part of an error cluster.
    ///
    /// Adjustments:
    /// - **Positional leniency**: first word gets -0.08 (autoregressive
    ///   decoders are less confident at utterance start due to cold context).
    /// - **Neighbor agreement**: avg neighbor confidence shifts threshold by
    ///   `(avg_neighbor - 0.7) * 0.2` — strong neighbors raise the effective
    ///   confidence (protect the word), weak neighbors lower it (flag it).
    /// - **Repetition guard**: repeated adjacent words never marked (LLM
    ///   would incorrectly "correct" intentional repetitions away).
    pub fn mark_uncertain_contextual(&self, base_threshold: f32) -> String {
        if self.word_confidences.is_empty() {
            return self.text.clone();
        }
        let n = self.word_confidences.len();
        let mut parts: Vec<String> = Vec::with_capacity(n);
        let mut prev_word: Option<&str> = None;

        for (idx, wc) in self.word_confidences.iter().enumerate() {
            let is_repeat = prev_word.map_or(false, |pw| pw.eq_ignore_ascii_case(&wc.word));

            let effective = contextual_threshold(&self.word_confidences, idx, base_threshold);

            if wc.confidence < effective && !is_repeat {
                parts.push(format!("[?{}?]", wc.word));
            } else {
                parts.push(wc.word.clone());
            }
            prev_word = Some(&wc.word);
        }
        parts.join(" ")
    }

    /// Returns `true` if any word falls below `threshold`.
    /// Returns `false` when no per-word data is available (conservative: assume pass).
    #[inline]
    pub fn has_uncertain_words(&self, threshold: f32) -> bool {
        !self.word_confidences.is_empty()
            && self
                .word_confidences
                .iter()
                .any(|wc| wc.confidence < threshold)
    }
}

/// Compute context-adjusted confidence threshold for a word at `idx`.
///
/// Returns an effective threshold that is lower (more lenient) when the
/// word's context suggests it's likely correct, and higher (stricter) when
/// context suggests an error cluster.
///
/// Grounded in two observations from ASR confidence calibration literature:
/// 1. Autoregressive decoders underestimate first-token confidence (Siu & Gish 1999)
/// 2. ASR errors cluster in noisy acoustic regions (Jiang 2005)
fn contextual_threshold(words: &[WordConfidence], idx: usize, base: f32) -> f32 {
    let mut adj = 0.0f32;

    // Positional leniency: first word of utterance has less decoder context,
    // so raw confidence is systematically lower. Compensate.
    // Single-word utterances (idx==0, len==1) get double leniency — they're
    // both start-of-sequence and lack any neighbor context.
    if idx == 0 {
        adj -= 0.08;
        if words.len() == 1 {
            adj -= 0.04; // extra leniency for isolated words
        }
    }

    // Last-word leniency: decoder has no right context for attention,
    // causing a weaker but measurable confidence dip. Less severe than
    // first-word cold start, so smaller adjustment.
    if idx + 1 == words.len() && words.len() > 1 {
        adj -= 0.03;
    }

    // Neighbor agreement/disagreement.
    // Default neighbor confidence 0.5 (neutral) when at edge.
    let left = if idx > 0 {
        words[idx - 1].confidence
    } else {
        0.5
    };
    let right = if idx + 1 < words.len() {
        words[idx + 1].confidence
    } else {
        0.5
    };
    let avg_neighbor = (left + right) / 2.0;

    // Strong neighbors (>0.7) → negative adjustment (lower threshold, protect word)
    // Weak neighbors (<0.7) → positive adjustment (raise threshold, flag word)
    // Scale: ±0.2 at the extremes
    adj -= (avg_neighbor - 0.7) * 0.2;

    // Effective range for base=0.65: [0.44, 0.79].
    // Lower bound 0.45: any word below 45% confidence is almost certainly wrong.
    // Upper bound 0.80: prevents flagging very high-confidence words.
    (base + adj).clamp(0.45, 0.80)
}

/// Configuration for an ASR session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrSessionConfig {
    pub sample_rate: u32,
    pub frame_size: usize,
    /// BCP-47 tag or None for auto-detect.
    pub language: Option<String>,
    /// Model name override for remote backends.
    pub model_hint: Option<String>,
}

impl Default for AsrSessionConfig {
    #[inline]
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_size: 480,
            language: Some("en-US".into()),
            model_hint: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AsrBackendKind {
    ParakeetTdt,
    OrtQwenAsr,
    OpenAiCompatible,
}

impl AsrBackendKind {
    /// Recommended `max_segment_ms` for the VadSegmenter.
    ///
    /// All backends handle long audio natively (OrtQwen uses windowed
    /// attention internally). This is a safety ceiling, not an encoder limit.
    #[deprecated(note = "use AsrRegistry::pipeline_config().max_segment_ms instead")]
    pub fn suggested_max_segment_ms(&self) -> u32 {
        match self {
            Self::OrtQwenAsr => 30_000,
            Self::ParakeetTdt => 30_000,
            Self::OpenAiCompatible => 30_000,
        }
    }
}

/// Trait for ASR backend implementations.
#[async_trait::async_trait]
pub trait AsrBackend: Send + Sync {
    type SessionHandle: Send + Sync;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle>;

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()>;

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>>;

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()>;

    /// Final poll that commits all rollback-buffered tokens.
    /// Default: delegates to `poll_results`. Streaming backends override
    /// to pass `finalize=true` through the decode loop.
    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        self.poll_results(session).await
    }

    /// Clear the audio buffer and segment state so the next inference starts
    /// fresh. Called after finalizing an utterance to prevent re-transcribing
    /// already-delivered text.
    #[inline]
    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let _ = session;
        Ok(())
    }

    /// Optional hint: expected maximum number of segments per `poll_results`
    /// call. Callers may use this to pre-allocate buffers.
    #[inline]
    fn segment_capacity_hint(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment() -> AsrSegment {
        AsrSegment {
            id: 1,
            text: "hello world".into(),
            is_final: true,
            start_ms: Some(0),
            end_ms: Some(1500),
            confidence: Some(0.95),
            min_token_confidence: Some(0.80),
            word_confidences: vec![],
        }
    }

    #[test]
    fn asr_segment_serde_round_trip() {
        let seg = make_segment();
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: AsrSegment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 1);
        assert!(parsed.is_final);
        assert_eq!(parsed.text, "hello world");
    }

    #[test]
    fn asr_session_config_defaults() {
        let config = AsrSessionConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.frame_size, 480);
        assert_eq!(config.language, Some("en-US".into()));
    }

    #[test]
    fn asr_backend_kind_round_trip() {
        let kind = AsrBackendKind::ParakeetTdt;
        let json = serde_json::to_string(&kind).unwrap();
        let parsed: AsrBackendKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, AsrBackendKind::ParakeetTdt);
    }

    #[test]
    fn is_confident_no_confidence_passes() {
        let mut seg = make_segment();
        seg.confidence = None;
        assert!(seg.is_confident(0.9));
    }

    #[test]
    fn is_confident_below_threshold_fails() {
        let seg = make_segment(); // confidence = 0.95
        assert!(!seg.is_confident(0.99));
        assert!(seg.is_confident(0.90));
    }

    #[test]
    fn duration_ms_computed_correctly() {
        let seg = make_segment(); // start=0, end=1500
        assert_eq!(seg.duration_ms(), Some(1500));
    }

    #[test]
    fn duration_ms_missing_timestamps() {
        let mut seg = make_segment();
        seg.start_ms = None;
        assert_eq!(seg.duration_ms(), None);
    }

    #[test]
    fn is_confident_gated_passes_both_thresholds() {
        let seg = make_segment(); // confidence=0.95, min_token=0.80
        assert!(seg.is_confident_gated(0.90, 0.7));
    }

    #[test]
    fn is_confident_gated_fails_on_low_min_token() {
        let mut seg = make_segment();
        seg.confidence = Some(0.98); // mean is high
        seg.min_token_confidence = Some(0.5); // but one token is bad
        assert!(!seg.is_confident_gated(0.95, 0.7));
    }

    #[test]
    fn is_confident_gated_none_min_token_passes() {
        let mut seg = make_segment();
        seg.min_token_confidence = None; // backend doesn't expose per-token
        assert!(seg.is_confident_gated(0.90, 0.7));
    }

    #[test]
    fn mark_uncertain_empty_word_confidences_returns_text() {
        let seg = make_segment();
        assert_eq!(seg.mark_uncertain(0.85), "hello world");
    }

    #[test]
    fn mark_uncertain_wraps_low_confidence_words() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "hello".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "world".into(),
                confidence: 0.60,
            },
        ];
        assert_eq!(seg.mark_uncertain(0.85), "hello [?world?]");
    }

    #[test]
    fn mark_uncertain_all_confident() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "hello".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "world".into(),
                confidence: 0.90,
            },
        ];
        assert_eq!(seg.mark_uncertain(0.85), "hello world");
    }

    #[test]
    fn has_uncertain_words_empty_returns_false() {
        let seg = make_segment();
        assert!(!seg.has_uncertain_words(0.85));
    }

    #[test]
    fn has_uncertain_words_detects_low_confidence() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "hello".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "world".into(),
                confidence: 0.60,
            },
        ];
        assert!(seg.has_uncertain_words(0.85));
    }

    #[test]
    fn has_uncertain_words_all_above_threshold() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "hello".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "world".into(),
                confidence: 0.90,
            },
        ];
        assert!(!seg.has_uncertain_words(0.85));
    }

    // ── Contextual confidence tests ─────────────────────────────────────

    #[test]
    fn contextual_first_word_leniency() {
        // First word at 0.60 should NOT be marked when threshold=0.65
        // because positional leniency lowers the effective threshold.
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "the".into(),
                confidence: 0.60,
            },
            WordConfidence {
                word: "meeting".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "starts".into(),
                confidence: 0.92,
            },
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        assert_eq!(result, "the meeting starts");
    }

    #[test]
    fn contextual_strong_neighbors_protect_word() {
        // Middle word at 0.62 should NOT be marked when both neighbors
        // are very confident (>0.9) — acoustic context is clean.
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "the".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "weather".into(),
                confidence: 0.62,
            },
            WordConfidence {
                word: "today".into(),
                confidence: 0.94,
            },
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        assert_eq!(result, "the weather today");
    }

    #[test]
    fn contextual_weak_neighbors_flag_word() {
        // Middle word at 0.63 SHOULD be marked when both neighbors
        // are also low-confidence — error cluster.
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "tha".into(),
                confidence: 0.35,
            },
            WordConfidence {
                word: "wether".into(),
                confidence: 0.63,
            },
            WordConfidence {
                word: "tody".into(),
                confidence: 0.30,
            },
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        // All words should be marked — error cluster
        assert!(result.contains("[?tha?]"));
        assert!(result.contains("[?wether?]"));
        assert!(result.contains("[?tody?]"));
    }

    #[test]
    fn contextual_high_confidence_never_marked() {
        // Words with very high confidence (>0.9) should never be marked
        // regardless of neighbor context.
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "bad".into(),
                confidence: 0.20,
            },
            WordConfidence {
                word: "good".into(),
                confidence: 0.92,
            },
            WordConfidence {
                word: "bad".into(),
                confidence: 0.20,
            },
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        assert_eq!(result, "[?bad?] good [?bad?]");
    }

    #[test]
    fn contextual_falls_back_without_word_confidences() {
        let seg = make_segment();
        assert_eq!(seg.mark_uncertain_contextual(0.65), "hello world");
    }

    #[test]
    fn contextual_single_word() {
        // Single word — no neighbors to use.
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence {
            word: "hello".into(),
            confidence: 0.60,
        }];
        // First-word leniency should protect it at 0.60 vs threshold 0.65
        let result = seg.mark_uncertain_contextual(0.65);
        assert_eq!(result, "hello");
    }

    #[test]
    fn contextual_last_word_leniency() {
        // Last word at 0.63 should NOT be marked when threshold=0.65
        // because last-word leniency lowers the effective threshold
        // (decoder has no right context for attention).
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "the".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "meeting".into(),
                confidence: 0.95,
            },
            WordConfidence {
                word: "today".into(),
                confidence: 0.63,
            },
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        assert_eq!(result, "the meeting today");
    }

    #[test]
    fn contextual_repetition_guard_preserved() {
        // Repeated words should not be marked even when low-confidence
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence {
                word: "very".into(),
                confidence: 0.50,
            },
            WordConfidence {
                word: "very".into(),
                confidence: 0.50,
            },
            WordConfidence {
                word: "good".into(),
                confidence: 0.95,
            },
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        // First "very" may be marked, but second "very" must NOT be (repetition guard)
        assert!(result.contains("very good"));
        assert!(!result.contains("[?very?] [?very?]"));
    }

    #[test]
    fn word_confidence_serde_round_trip() {
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence {
            word: "test".into(),
            confidence: 0.75,
        }];
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: AsrSegment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.word_confidences.len(), 1);
        assert_eq!(parsed.word_confidences[0].word, "test");
    }
}
