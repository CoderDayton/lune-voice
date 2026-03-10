//! Per-backend pipeline configuration.
//!
//! Each ASR backend provides tuned defaults for VAD timing, confidence
//! thresholds, and segment limits. Users can override any field via
//! [`PipelineOverrides`] in their TOML config.
//!
//! The merge contract: backend provides [`PipelineConfig`] defaults,
//! user provides [`PipelineOverrides`] (all optional), and
//! [`PipelineConfig::with_overrides`] produces the final resolved config.

use serde::{Deserialize, Serialize};

/// Fully resolved pipeline configuration for a single ASR backend.
///
/// Constructed by calling `AsrRegistry::pipeline_config()` and optionally
/// merging user overrides via [`with_overrides`](Self::with_overrides).
///
/// Every field has a concrete value — no `Option`s. The backend is
/// responsible for providing sensible defaults; the user overrides only
/// what they want to change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    // ── VAD segmenter ──────────────────────────────────────────────────
    /// Force-flush VAD segments longer than this (ms).
    ///
    /// Backends with encoder window limits should set this below the
    /// window size. Backends that handle long audio natively use 30_000.
    pub max_segment_ms: u32,

    /// VAD silence duration to close a sentence-level segment (ms).
    ///
    /// Pauses shorter than this are absorbed into the current segment
    /// (natural hesitations, breaths). Longer pauses trigger a segment
    /// boundary.
    pub sentence_close_ms: u32,

    /// VAD silence duration for hard close / topic break (ms).
    ///
    /// Must be ≥ `sentence_close_ms`. When silence exceeds this, the
    /// segment is flushed regardless of other heuristics.
    pub hard_close_ms: u32,

    /// VAD probability threshold for speech detection (0.0–1.0).
    pub speech_threshold: f32,

    /// Minimum speech duration to accept (ms). Rejects noise bursts.
    pub min_speech_ms: u32,

    /// Enable adaptive VAD silence thresholds that learn from the
    /// speaker's pause rhythm (P95-based with EMA smoothing).
    pub adaptive_vad: bool,

    // ── Confidence gating ──────────────────────────────────────────────
    /// Per-word confidence threshold for `[?word?]` marking and LLM gating.
    ///
    /// Words below this trigger LLM correction. When all words are above
    /// threshold, LLM is bypassed entirely. `None` means confidence
    /// gating is disabled — raw ASR text passes through without LLM
    /// (used by backends that don't provide per-token log-probs, e.g.
    /// OpenAI-compatible).
    pub word_confidence_threshold: Option<f32>,

    // ── Session limits ─────────────────────────────────────────────────
    /// Maximum speech-time samples before auto-ending session.
    ///
    /// Only VAD-detected speech counts — silence/thinking pauses are
    /// excluded. At 16 kHz, 960_000 = 60 seconds.
    pub max_speech_samples: usize,
}

impl PipelineConfig {
    /// Merge user overrides on top of backend defaults.
    ///
    /// Each `Some` field in `overrides` replaces the corresponding
    /// default. `None` fields are left unchanged.
    #[must_use]
    pub fn with_overrides(mut self, overrides: &PipelineOverrides) -> Self {
        if let Some(v) = overrides.max_segment_ms {
            self.max_segment_ms = v;
        }
        if let Some(v) = overrides.sentence_close_ms {
            self.sentence_close_ms = v;
        }
        if let Some(v) = overrides.hard_close_ms {
            self.hard_close_ms = v;
        }
        if let Some(v) = overrides.speech_threshold {
            self.speech_threshold = v;
        }
        if let Some(v) = overrides.min_speech_ms {
            self.min_speech_ms = v;
        }
        if let Some(v) = overrides.adaptive_vad {
            self.adaptive_vad = v;
        }
        if let Some(v) = overrides.word_confidence_threshold {
            self.word_confidence_threshold = Some(v);
        }
        if let Some(v) = overrides.max_speech_secs {
            self.max_speech_samples = (v as usize) * 16_000;
        }
        self
    }

    /// Validate invariants. Returns a list of violations (empty = valid).
    pub fn validate(&self) -> Vec<&'static str> {
        let mut issues = Vec::new();
        if self.hard_close_ms < self.sentence_close_ms {
            issues.push("hard_close_ms must be >= sentence_close_ms");
        }
        if self.max_segment_ms < self.hard_close_ms {
            issues.push("max_segment_ms must be >= hard_close_ms");
        }
        if !(0.0..=1.0).contains(&self.speech_threshold) {
            issues.push("speech_threshold must be in 0.0..=1.0");
        }
        if let Some(t) = self.word_confidence_threshold {
            if !(0.0..=1.0).contains(&t) {
                issues.push("word_confidence_threshold must be in 0.0..=1.0");
            }
        }
        if self.max_speech_samples == 0 {
            issues.push("max_speech_samples must be > 0");
        }
        issues
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_segment_ms: 30_000,
            sentence_close_ms: 600,
            hard_close_ms: 1_200,
            speech_threshold: 0.5,
            min_speech_ms: 150,
            adaptive_vad: false,
            word_confidence_threshold: Some(0.65),
            max_speech_samples: 60 * 16_000,
        }
    }
}

/// User-specified overrides for pipeline configuration.
///
/// All fields are optional. `None` = inherit the backend's recommended
/// default. Deserialized from the `[pipeline]` section of the TOML config.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineOverrides {
    pub max_segment_ms: Option<u32>,
    pub sentence_close_ms: Option<u32>,
    pub hard_close_ms: Option<u32>,
    pub speech_threshold: Option<f32>,
    pub min_speech_ms: Option<u32>,
    pub adaptive_vad: Option<bool>,
    pub word_confidence_threshold: Option<f32>,
    /// Convenience alias: converted to `max_speech_samples` at merge time.
    pub max_speech_secs: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_validates() {
        assert!(PipelineConfig::default().validate().is_empty());
    }

    #[test]
    fn override_merges_selectively() {
        let base = PipelineConfig::default();
        let overrides = PipelineOverrides {
            sentence_close_ms: Some(800),
            max_speech_secs: Some(120),
            ..Default::default()
        };
        let merged = base.with_overrides(&overrides);
        assert_eq!(merged.sentence_close_ms, 800);
        assert_eq!(merged.max_speech_samples, 120 * 16_000);
        // Unchanged fields
        assert_eq!(merged.hard_close_ms, 1_200);
        assert_eq!(merged.max_segment_ms, 30_000);
    }

    #[test]
    fn override_none_is_identity() {
        let base = PipelineConfig::default();
        let merged = base.clone().with_overrides(&PipelineOverrides::default());
        assert_eq!(base, merged);
    }

    #[test]
    fn validation_catches_invalid() {
        let mut cfg = PipelineConfig::default();
        cfg.hard_close_ms = 100; // less than sentence_close_ms
        cfg.speech_threshold = 1.5;
        let issues = cfg.validate();
        assert_eq!(issues.len(), 2);
    }

    #[test]
    fn toml_round_trip() {
        let overrides = PipelineOverrides {
            sentence_close_ms: Some(800),
            adaptive_vad: Some(true),
            ..Default::default()
        };
        let toml_str = toml::to_string_pretty(&overrides).unwrap();
        let parsed: PipelineOverrides = toml::from_str(&toml_str).unwrap();
        assert_eq!(overrides, parsed);
    }
}
