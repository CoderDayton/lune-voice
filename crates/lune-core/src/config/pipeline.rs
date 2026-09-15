// SPDX-License-Identifier: GPL-3.0-or-later

//! Per-backend pipeline configuration.
//!
//! Each ASR backend provides tuned defaults for the confidence threshold,
//! the utterance buffer ceiling and the styling timeout. Users can override
//! any field via [`PipelineOverrides`] in their TOML config.
//!
//! The merge contract: backend provides [`PipelineConfig`] defaults,
//! user provides [`PipelineOverrides`] (all optional), and
//! [`PipelineConfig::with_overrides`] produces the final resolved config.

use serde::{Deserialize, Serialize};

use crate::audio::SAMPLE_RATE;

/// Fully resolved pipeline configuration for a single ASR backend.
///
/// Constructed by calling `AsrRegistry::pipeline_config()` and optionally
/// merging user overrides via [`with_overrides`](Self::with_overrides).
///
/// Every field has a concrete value — no `Option`s except where `None` is a
/// meaningful "disabled". The backend is responsible for providing sensible
/// defaults; the user overrides only what they want to change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Per-word confidence threshold for correction spans.
    ///
    /// Words below this are treated as low-confidence, which gates
    /// single-word fuzzy dictionary matches. `None` means confidence
    /// gating is disabled — used by backends that don't provide
    /// per-token log-probs, e.g. OpenAI-compatible.
    pub word_confidence_threshold: Option<f32>,

    /// Maximum speech-time samples before auto-ending session.
    ///
    /// At 16 kHz, 960_000 = 60 seconds.
    pub max_speech_samples: usize,

    /// Timeout for LLM post-processing (seconds).
    ///
    /// If the LLM call exceeds this timeout, the raw ASR text is used
    /// instead. Default is 30 seconds. Bounded to
    /// [`MAX_LLM_TIMEOUT_SECS`] by [`PipelineConfig::clamp`], which is also
    /// what `0` resolves to — there is no way to run the call unbounded.
    #[serde(default = "default_llm_timeout_secs")]
    pub llm_timeout_secs: u32,
}

fn default_llm_timeout_secs() -> u32 {
    30
}

/// Shortest utterance buffer that still records a usable dictation (1 s).
pub const MIN_SPEECH_SAMPLES: usize = SAMPLE_RATE;

/// Longest utterance buffer (10 min). The buffer is `f32` at 16 kHz, so this
/// is about 38 MB — the ceiling exists because the value is a user override
/// and nothing else bounds the allocation.
pub const MAX_SPEECH_SAMPLES: usize = 600 * SAMPLE_RATE;

/// Longest styling-LLM timeout. Also what a `0` ("no timeout") resolves to:
/// the styling call is the only unbounded step between the hotkey and the
/// text appearing, so it always gets an end-to-end bound.
pub const MAX_LLM_TIMEOUT_SECS: u32 = 300;

impl PipelineConfig {
    /// Merge user overrides on top of backend defaults.
    ///
    /// Each `Some` field in `overrides` replaces the corresponding
    /// default. `None` fields are left unchanged.
    #[must_use]
    pub fn with_overrides(mut self, overrides: &PipelineOverrides) -> Self {
        if let Some(v) = overrides.llm_timeout_secs {
            self.llm_timeout_secs = v;
        }

        // Target field is Option<f32>; wrap incoming value in Some.
        if let Some(v) = overrides.word_confidence_threshold {
            self.word_confidence_threshold = Some(v);
        }

        // Different target field + unit conversion.
        if let Some(v) = overrides.max_speech_secs {
            self.max_speech_samples = (v as usize) * SAMPLE_RATE;
        }

        self
    }
}

/// Bring a unit-interval override into range, replacing a non-number with
/// `fallback`.
///
/// `f32::clamp` returns NaN for NaN and every comparison against NaN is false,
/// so clamping alone would leave a NaN threshold — which TOML spells
/// literally — in place.
fn into_unit_interval(value: f32, fallback: f32) -> f32 {
    if value.is_nan() {
        fallback
    } else {
        value.clamp(0.0, 1.0)
    }
}

impl PipelineConfig {
    /// Repair every invariant violation, reporting what was changed.
    ///
    /// [`validate`](Self::validate) only describes, and the caller used the
    /// values anyway — so a single bad override cost the user their
    /// dictations rather than that one setting. This is the half that acts:
    /// after it, `validate` is empty by construction.
    #[must_use]
    pub fn clamp(mut self) -> (Self, Vec<&'static str>) {
        let mut repairs = Vec::new();
        let d = Self::default();

        if self
            .word_confidence_threshold
            .is_some_and(|t| !(0.0..=1.0).contains(&t))
        {
            self.word_confidence_threshold = self
                .word_confidence_threshold
                .map(|t| into_unit_interval(t, d.word_confidence_threshold.unwrap_or(0.5)));
            repairs.push("word_confidence_threshold clamped into 0.0..=1.0");
        }
        if !(MIN_SPEECH_SAMPLES..=MAX_SPEECH_SAMPLES).contains(&self.max_speech_samples) {
            self.max_speech_samples = self
                .max_speech_samples
                .clamp(MIN_SPEECH_SAMPLES, MAX_SPEECH_SAMPLES);
            repairs.push("max_speech_samples clamped to the supported utterance length");
        }
        if self.llm_timeout_secs == 0 || self.llm_timeout_secs > MAX_LLM_TIMEOUT_SECS {
            self.llm_timeout_secs = MAX_LLM_TIMEOUT_SECS;
            repairs.push("llm_timeout_secs bounded to the ceiling (0 is not 'no timeout')");
        }

        (self, repairs)
    }

    /// Validate invariants. Returns a list of violations (empty = valid).
    pub fn validate(&self) -> Vec<&'static str> {
        let mut issues = Vec::new();
        if let Some(t) = self.word_confidence_threshold
            && !(0.0..=1.0).contains(&t)
        {
            issues.push("word_confidence_threshold must be in 0.0..=1.0");
        }
        if !(MIN_SPEECH_SAMPLES..=MAX_SPEECH_SAMPLES).contains(&self.max_speech_samples) {
            issues.push("max_speech_samples is outside the supported utterance length");
        }
        if self.llm_timeout_secs == 0 || self.llm_timeout_secs > MAX_LLM_TIMEOUT_SECS {
            issues.push("llm_timeout_secs must be in 1..=300");
        }
        issues
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            word_confidence_threshold: Some(0.55),
            max_speech_samples: 60 * SAMPLE_RATE,
            llm_timeout_secs: 30,
        }
    }
}

/// User-specified overrides for pipeline configuration.
///
/// All fields are optional. `None` = inherit the backend's recommended
/// default. Deserialized from the `[pipeline]` section of the TOML config.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineOverrides {
    pub word_confidence_threshold: Option<f32>,
    /// Convenience alias: converted to `max_speech_samples` at merge time.
    pub max_speech_secs: Option<u32>,
    /// Override LLM timeout in seconds.
    pub llm_timeout_secs: Option<u32>,
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
            max_speech_secs: Some(120),
            ..Default::default()
        };
        let merged = base.with_overrides(&overrides);
        assert_eq!(merged.max_speech_samples, 120 * 16_000);
        // Unchanged fields
        assert_eq!(merged.word_confidence_threshold, Some(0.55));
        assert_eq!(merged.llm_timeout_secs, 30);
    }

    #[test]
    fn override_none_is_identity() {
        let base = PipelineConfig::default();
        let merged = base.clone().with_overrides(&PipelineOverrides::default());
        assert_eq!(base, merged);
    }

    #[test]
    fn validation_catches_invalid() {
        let cfg = PipelineConfig {
            word_confidence_threshold: Some(1.5),
            llm_timeout_secs: 0,
            ..PipelineConfig::default()
        };
        assert_eq!(cfg.validate().len(), 2);
    }

    /// `validate` only describes. A config that violates an invariant is
    /// still used, so every violation has to be repaired to something usable
    /// before the pipeline is built.
    #[test]
    fn clamp_repairs_every_invariant_validate_reports() {
        let cfg = PipelineConfig {
            word_confidence_threshold: Some(2.0),
            max_speech_samples: 0,
            llm_timeout_secs: 100_000,
        };
        let (fixed, repairs) = cfg.clamp();
        assert_eq!(repairs.len(), 3);
        assert!(
            fixed.validate().is_empty(),
            "clamp must leave a valid config, got {:?}",
            fixed.validate()
        );
    }

    /// Zero means "one syllable per dictation" and 10 hours means a 2.3 GB
    /// buffer. Neither is a choice worth honouring.
    #[test]
    fn clamp_bounds_the_speech_buffer_at_both_ends() {
        let too_small = PipelineConfig {
            max_speech_samples: 0,
            ..PipelineConfig::default()
        }
        .clamp()
        .0;
        assert_eq!(too_small.max_speech_samples, MIN_SPEECH_SAMPLES);

        let too_big = PipelineConfig {
            max_speech_samples: 36_000 * SAMPLE_RATE,
            ..PipelineConfig::default()
        }
        .clamp()
        .0;
        assert_eq!(too_big.max_speech_samples, MAX_SPEECH_SAMPLES);
    }

    /// Zero used to mean "no timeout", which left the styling call with no
    /// end-to-end bound at all.
    #[test]
    fn clamp_turns_a_disabled_llm_timeout_into_the_ceiling() {
        let (fixed, repairs) = PipelineConfig {
            llm_timeout_secs: 0,
            ..PipelineConfig::default()
        }
        .clamp();
        assert_eq!(fixed.llm_timeout_secs, MAX_LLM_TIMEOUT_SECS);
        assert!(repairs.iter().any(|r| r.contains("llm_timeout_secs")));
    }

    /// A NaN threshold — which TOML spells `nan` — passes straight through
    /// `f32::clamp`, so it needs its own repair.
    #[test]
    fn clamp_repairs_a_nan_threshold() {
        let (fixed, _) = PipelineConfig {
            word_confidence_threshold: Some(f32::NAN),
            ..PipelineConfig::default()
        }
        .clamp();
        assert!(fixed.validate().is_empty(), "got {:?}", fixed.validate());
        assert!(fixed.word_confidence_threshold.is_some_and(f32::is_finite));
    }

    #[test]
    fn clamp_leaves_a_valid_config_alone() {
        let cfg = PipelineConfig::default();
        let (fixed, repairs) = cfg.clone().clamp();
        assert_eq!(cfg, fixed);
        assert!(repairs.is_empty());
    }

    #[test]
    fn toml_round_trip() {
        let overrides = PipelineOverrides {
            word_confidence_threshold: Some(0.7),
            max_speech_secs: Some(120),
            llm_timeout_secs: Some(45),
        };
        let toml_str = toml::to_string_pretty(&overrides).unwrap();
        let parsed: PipelineOverrides = toml::from_str(&toml_str).unwrap();
        assert_eq!(overrides, parsed);
    }
}
