use serde::{Deserialize, Serialize};

use crate::Result;

/// Per-word confidence emitted alongside an [`AsrSegment`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordConfidence {
    pub word: String,
    pub confidence: f32,
    #[serde(default)]
    pub word_error_risk: f32,
}

/// A segment of recognized speech.
///
/// Fields are ordered hot-first: `id`, `confidence`, `is_final` land on the
/// same cache line so routing decisions don't pay for a `text` load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrSegment {
    pub id: u64,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub min_token_confidence: Option<f32>,
    #[serde(default)]
    pub word_confidences: Vec<WordConfidence>,

    pub is_final: bool,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub text: String,
    #[serde(default)]
    pub detected_language: Option<String>,
}

impl AsrSegment {
    /// Returns `true` when confidence meets or exceeds `threshold`, or when
    /// confidence is unavailable (conservative: assume pass).
    ///
    /// After the session layer applies the isotonic calibrator, the `confidence`
    /// field carries `P(correct | raw softmax)` rather than the raw softmax
    /// probability. Thresholds passed to this method should be interpreted on
    /// the calibrated scale (i.e. as a minimum acceptance probability), not on
    /// raw softmax magnitudes.
    ///
    /// Inline so routing hot-loops don't pay a call frame.
    #[inline(always)]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence.is_none_or(|c| c >= threshold)
    }

    /// Two-level confidence gate: sentence-level mean AND min-token check.
    /// Per arXiv:2407.21414, sentence threshold=0.95 + word threshold=0.7
    /// catches segments where one bad token drags overall quality down.
    ///
    /// Both `mean_threshold` and `min_token_threshold` are compared against
    /// calibrated `P(correct)` values once `SessionManager` has run the
    /// isotonic calibrator — update tuned thresholds accordingly.
    #[inline(always)]
    pub fn is_confident_gated(&self, mean_threshold: f32, min_token_threshold: f32) -> bool {
        self.is_confident(mean_threshold)
            && self
                .min_token_confidence
                .is_none_or(|m| m >= min_token_threshold)
    }

    /// Returns `true` when the segment's minimum token confidence falls
    /// below `broken_threshold`, signalling that the audio was too degraded
    /// for the LLM corrector to help — invoking it on broken transcripts
    /// triggers the overcorrection failure mode (arXiv:2509.25048 Fig 3).
    ///
    /// Returns `false` when `min_token_confidence` is absent.
    pub fn is_broken_for_llm(&self, broken_threshold: f32) -> bool {
        self.min_token_confidence
            .is_some_and(|m| m < broken_threshold)
    }

    /// Duration of the segment in milliseconds, if both timestamps are known.
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_ms?.checked_sub(self.start_ms?)
    }
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
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_size: 480,
            language: None,
            model_hint: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AsrBackendKind {
    // `OrtQwenAsr` is the pre-rename name still present in older session
    // rows; accept it as an alias so history loads without a fallback warning.
    #[serde(alias = "OrtQwenAsr")]
    Qwen3Asr,
    OpenAiCompatible,
}

impl AsrBackendKind {
    /// Short stable identifier used for per-backend filenames like
    /// `calibrator_{slug}.json`. Must stay lowercase, ASCII, no spaces.
    ///
    /// These strings are part of the on-disk layout — renaming them will
    /// orphan existing calibrator files. Add new variants rather than
    /// changing existing slugs.
    pub fn slug(&self) -> &'static str {
        match self {
            Self::Qwen3Asr => "qwen3asr",
            Self::OpenAiCompatible => "openai",
        }
    }
}

/// Trait for ASR backend implementations.
#[async_trait::async_trait]
pub trait AsrBackend: Send + Sync {
    type SessionHandle: Send + Sync;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle>;

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()>;

    /// Run transcription over all audio buffered since the session started
    /// (or since the last `reset_context`) and return the resulting segments.
    /// Called once per utterance after capture ends. This is a batch pipeline:
    /// there is no incremental/streaming result path.
    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>>;

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()>;

    /// Clear the audio buffer and segment state so the next inference starts
    /// fresh. Called after finalizing an utterance to prevent re-transcribing
    /// already-delivered text.
    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let _ = session;
        Ok(())
    }

    /// Optional hint: expected maximum number of segments per `poll_final`
    /// call. Callers may use this to pre-allocate buffers.
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
            detected_language: None,
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
        assert_eq!(config.language, None);
    }

    #[test]
    fn asr_backend_kind_round_trip() {
        let kind = AsrBackendKind::Qwen3Asr;
        let json = serde_json::to_string(&kind).unwrap();
        let parsed: AsrBackendKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, AsrBackendKind::Qwen3Asr);
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
    fn is_broken_for_llm_uses_min_token_confidence_when_present() {
        let mut seg = make_segment();
        seg.min_token_confidence = Some(0.20);
        assert!(seg.is_broken_for_llm(0.30));
        assert!(!seg.is_broken_for_llm(0.15));
    }
}
