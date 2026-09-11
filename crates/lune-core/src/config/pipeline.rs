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

use crate::audio::SAMPLE_RATE;

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

    /// VAD probability threshold for entering speech (0.0–1.0).
    pub speech_threshold: f32,

    /// VAD probability threshold for leaving speech (0.0–1.0).
    ///
    /// Must be ≤ `speech_threshold` to provide hysteresis. Lower values
    /// hold speech open through brief score dips; higher values cut off
    /// faster on pauses.
    #[serde(default = "default_speech_threshold_end")]
    pub speech_threshold_end: f32,

    /// Minimum speech duration to accept (ms). Rejects noise bursts.
    pub min_speech_ms: u32,

    /// Enable adaptive VAD silence thresholds that learn from the
    /// speaker's pause rhythm (P95-based with EMA smoothing).
    pub adaptive_vad: bool,

    // ── Confidence gating ──────────────────────────────────────────────
    /// Per-word confidence threshold for correction spans and LLM gating.
    ///
    /// Words below this trigger LLM correction. When all words are above
    /// threshold, LLM is bypassed entirely. `None` means confidence
    /// gating is disabled — raw ASR text passes through without LLM
    /// (used by backends that don't provide per-token log-probs, e.g.
    /// OpenAI-compatible).
    pub word_confidence_threshold: Option<f32>,

    /// Skip-when-clean LLM bypass threshold, on the per-word confidence scale
    /// (`1 − word_error_risk`).
    ///
    /// The LLM correction pass enters its "attention band" when at least TWO
    /// words in a segment fall below this confidence, protecting clean segments
    /// from over-correction (the dominant failure mode per arXiv:2509.25048).
    /// A single low-confidence word is almost always a correctly-transcribed
    /// rare word (a proper noun or brand), not an error the text-only corrector
    /// could fix — so one dipping word no longer escalates; two do.
    ///
    /// Default `Some(0.40)` — the point that matched oracle routing on the
    /// domain corpus.
    ///
    /// `None` disables the band gate — the LLM then runs only for unresolved
    /// words or ambiguous punctuation.
    #[serde(default)]
    pub llm_skip_threshold: Option<f32>,

    /// Skip-when-broken LLM bypass threshold, on the same per-word confidence
    /// scale.
    ///
    /// Lower bound of `llm_skip_threshold`, evaluated on the SEGMENT-AVERAGE
    /// confidence: when the mean word confidence falls below this value, the
    /// transcript is considered too degraded for the LLM corrector to help
    /// and the whole LLM stage is skipped. Deliberately not the weakest
    /// token — one garbled word in a clean utterance is the prime correction
    /// case, served by the attention band. The paper (arXiv:2509.25048 Fig 3)
    /// shows naive correction on broken transcripts routinely degrades WER —
    /// this gate is the defensive counterpart to skip-when-clean.
    ///
    /// Together with `llm_skip_threshold`, this forms an adaptive,
    /// bimodal invocation gate: the LLM runs only when
    /// `broken ≤ min_token_confidence < clean`. The decision adapts
    /// per-segment to observed confidence metrics, not a static bar.
    ///
    /// `None` disables the lower bound. Default `Some(0.30)` for
    /// confidence-exposing backends.
    #[serde(default)]
    pub llm_skip_noisy_threshold: Option<f32>,

    // ── Session limits ─────────────────────────────────────────────────
    /// Maximum speech-time samples before auto-ending session.
    ///
    /// Only VAD-detected speech counts — silence/thinking pauses are
    /// excluded. At 16 kHz, 960_000 = 60 seconds.
    pub max_speech_samples: usize,

    // ── Post-processing ───────────────────────────────────────────────
    /// Enable Inverse Text Normalization (ITN) on the final transcript.
    ///
    /// Converts spoken-form numbers, dates, and currency to written form
    /// ("two hundred" → "200", "january fifth" → "January 5"). Enabled
    /// by default for production use; disable in benchmarks where the
    /// reference text uses spelled-out numbers.
    #[serde(default = "default_itn_enabled")]
    pub itn_enabled: bool,

    /// Timeout for LLM post-processing (seconds).
    ///
    /// If the LLM call exceeds this timeout, the raw ASR text is used
    /// instead. Default is 30 seconds. Bounded to
    /// [`MAX_LLM_TIMEOUT_SECS`] by [`PipelineConfig::clamp`], which is also
    /// what `0` resolves to — there is no way to run the call unbounded.
    #[serde(default = "default_llm_timeout_secs")]
    pub llm_timeout_secs: u32,
}

fn default_speech_threshold_end() -> f32 {
    0.3
}

fn default_itn_enabled() -> bool {
    false
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
        macro_rules! apply {
            ($field:ident) => {
                if let Some(v) = overrides.$field {
                    self.$field = v;
                }
            };
        }

        apply!(max_segment_ms);
        apply!(sentence_close_ms);
        apply!(hard_close_ms);
        apply!(speech_threshold);
        apply!(speech_threshold_end);
        apply!(min_speech_ms);
        apply!(adaptive_vad);
        apply!(llm_timeout_secs);

        // Target field is Option<f32>; wrap incoming value in Some.
        if let Some(v) = overrides.word_confidence_threshold {
            self.word_confidence_threshold = Some(v);
        }
        if let Some(v) = overrides.llm_skip_threshold {
            self.llm_skip_threshold = Some(v);
        }
        if let Some(v) = overrides.llm_skip_noisy_threshold {
            self.llm_skip_noisy_threshold = Some(v);
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
/// so clamping alone would leave `speech_threshold = nan` — which TOML spells
/// literally — in place, and the segmenter would never enter speech.
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
        macro_rules! fix {
            ($cond:expr, $field:ident = $value:expr, $note:literal) => {
                if $cond {
                    self.$field = $value;
                    repairs.push($note);
                }
            };
        }

        fix!(
            !(0.0..=1.0).contains(&self.speech_threshold),
            speech_threshold = into_unit_interval(self.speech_threshold, d.speech_threshold),
            "speech_threshold clamped into 0.0..=1.0"
        );
        fix!(
            !(0.0..=1.0).contains(&self.speech_threshold_end),
            speech_threshold_end =
                into_unit_interval(self.speech_threshold_end, d.speech_threshold_end),
            "speech_threshold_end clamped into 0.0..=1.0"
        );
        // Hysteresis: leaving speech must not need a higher score than
        // entering it, or the segmenter can never close a segment.
        fix!(
            self.speech_threshold_end > self.speech_threshold,
            speech_threshold_end = self.speech_threshold,
            "speech_threshold_end lowered to speech_threshold"
        );
        fix!(
            self.hard_close_ms < self.sentence_close_ms,
            hard_close_ms = self.sentence_close_ms,
            "hard_close_ms raised to sentence_close_ms"
        );
        fix!(
            self.max_segment_ms < self.hard_close_ms,
            max_segment_ms = self.hard_close_ms,
            "max_segment_ms raised to hard_close_ms"
        );
        fix!(
            self.word_confidence_threshold
                .is_some_and(|t| !(0.0..=1.0).contains(&t)),
            word_confidence_threshold = self
                .word_confidence_threshold
                .map(|t| into_unit_interval(t, d.word_confidence_threshold.unwrap_or(0.5))),
            "word_confidence_threshold clamped into 0.0..=1.0"
        );
        fix!(
            self.llm_skip_threshold
                .is_some_and(|t| !(0.0..=1.0).contains(&t)),
            llm_skip_threshold = self
                .llm_skip_threshold
                .map(|t| into_unit_interval(t, d.llm_skip_threshold.unwrap_or(0.85))),
            "llm_skip_threshold clamped into 0.0..=1.0"
        );
        fix!(
            self.llm_skip_noisy_threshold
                .is_some_and(|t| !(0.0..=1.0).contains(&t)),
            llm_skip_noisy_threshold = self
                .llm_skip_noisy_threshold
                .map(|t| into_unit_interval(t, d.llm_skip_noisy_threshold.unwrap_or(0.6))),
            "llm_skip_noisy_threshold clamped into 0.0..=1.0"
        );
        // The noisy gate has to sit strictly below the clean one; three
        // quarters keeps the two apart without inventing an epsilon. At
        // `clean == 0.0` no fraction of it is strictly below, so the gate is
        // dropped rather than left violating the invariant it was repaired for.
        if let (Some(clean), Some(noisy)) = (self.llm_skip_threshold, self.llm_skip_noisy_threshold)
            && noisy >= clean
        {
            let lowered = clean * 0.75;
            self.llm_skip_noisy_threshold = (lowered < clean).then_some(lowered);
            repairs.push("llm_skip_noisy_threshold lowered below llm_skip_threshold");
        }
        fix!(
            !(MIN_SPEECH_SAMPLES..=MAX_SPEECH_SAMPLES).contains(&self.max_speech_samples),
            max_speech_samples = self
                .max_speech_samples
                .clamp(MIN_SPEECH_SAMPLES, MAX_SPEECH_SAMPLES),
            "max_speech_samples clamped to the supported utterance length"
        );
        fix!(
            self.llm_timeout_secs == 0 || self.llm_timeout_secs > MAX_LLM_TIMEOUT_SECS,
            llm_timeout_secs = MAX_LLM_TIMEOUT_SECS,
            "llm_timeout_secs bounded to the ceiling (0 is not 'no timeout')"
        );

        (self, repairs)
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
        if !(0.0..=1.0).contains(&self.speech_threshold_end) {
            issues.push("speech_threshold_end must be in 0.0..=1.0");
        }
        if self.speech_threshold_end > self.speech_threshold {
            issues.push("speech_threshold_end must be <= speech_threshold");
        }
        if let Some(t) = self.word_confidence_threshold
            && !(0.0..=1.0).contains(&t)
        {
            issues.push("word_confidence_threshold must be in 0.0..=1.0");
        }
        if let Some(t) = self.llm_skip_threshold
            && !(0.0..=1.0).contains(&t)
        {
            issues.push("llm_skip_threshold must be in 0.0..=1.0");
        }
        if let Some(t) = self.llm_skip_noisy_threshold
            && !(0.0..=1.0).contains(&t)
        {
            issues.push("llm_skip_noisy_threshold must be in 0.0..=1.0");
        }
        if let (Some(clean), Some(broken)) =
            (self.llm_skip_threshold, self.llm_skip_noisy_threshold)
            && broken >= clean
        {
            issues.push("llm_skip_noisy_threshold must be < llm_skip_threshold");
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
            max_segment_ms: 30_000,
            sentence_close_ms: 776,
            hard_close_ms: 1_552,
            speech_threshold: 0.5,
            speech_threshold_end: 0.3,
            min_speech_ms: 150,
            adaptive_vad: false,
            // Thresholds are on the calibrated P(correct) scale after the
            // session layer runs the isotonic calibrator. See the per-backend
            // overrides in `lune_asr::AsrRegistry::pipeline_config` for the
            // reasoning behind these values.
            word_confidence_threshold: Some(0.55),
            llm_skip_threshold: Some(0.40),
            llm_skip_noisy_threshold: Some(0.30),
            max_speech_samples: 60 * SAMPLE_RATE,
            itn_enabled: false,
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
    pub max_segment_ms: Option<u32>,
    pub sentence_close_ms: Option<u32>,
    pub hard_close_ms: Option<u32>,
    pub speech_threshold: Option<f32>,
    pub speech_threshold_end: Option<f32>,
    pub min_speech_ms: Option<u32>,
    pub adaptive_vad: Option<bool>,
    pub word_confidence_threshold: Option<f32>,
    pub llm_skip_threshold: Option<f32>,
    pub llm_skip_noisy_threshold: Option<f32>,
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
            sentence_close_ms: Some(800),
            max_speech_secs: Some(120),
            ..Default::default()
        };
        let merged = base.with_overrides(&overrides);
        assert_eq!(merged.sentence_close_ms, 800);
        assert_eq!(merged.max_speech_samples, 120 * 16_000);
        // Unchanged fields
        assert_eq!(merged.hard_close_ms, 1_552);
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
        let cfg = PipelineConfig {
            hard_close_ms: 100, // less than sentence_close_ms
            speech_threshold: 1.5,
            ..PipelineConfig::default()
        };
        let issues = cfg.validate();
        assert_eq!(issues.len(), 2);
    }

    #[test]
    fn llm_skip_threshold_out_of_range_flagged() {
        let cfg = PipelineConfig {
            llm_skip_threshold: Some(1.5),
            ..PipelineConfig::default()
        };
        let issues = cfg.validate();
        assert!(
            issues.iter().any(|i| i.contains("llm_skip_threshold")),
            "expected range violation, got {issues:?}"
        );
    }

    #[test]
    fn llm_skip_threshold_override_applies() {
        let base = PipelineConfig::default();
        let overrides = PipelineOverrides {
            llm_skip_threshold: Some(0.98),
            ..Default::default()
        };
        let merged = base.with_overrides(&overrides);
        assert_eq!(merged.llm_skip_threshold, Some(0.98));
    }

    #[test]
    fn default_has_skip_threshold_at_0_40() {
        // Matches the per-backend registry override for the local ASR backend.
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.llm_skip_threshold, Some(0.40));
        assert_eq!(cfg.word_confidence_threshold, Some(0.55));
        assert!(cfg.validate().is_empty(), "default must validate");
    }

    /// `validate` only describes. A config that violates an invariant is
    /// still used, so every violation has to be repaired to something usable
    /// before the pipeline is built.
    #[test]
    fn clamp_repairs_every_invariant_validate_reports() {
        let cfg = PipelineConfig {
            sentence_close_ms: 2_000,
            hard_close_ms: 100,
            max_segment_ms: 50,
            speech_threshold: 1.5,
            speech_threshold_end: -0.2,
            word_confidence_threshold: Some(2.0),
            llm_skip_threshold: Some(0.2),
            llm_skip_noisy_threshold: Some(0.9),
            max_speech_samples: 0,
            llm_timeout_secs: 100_000,
            ..PipelineConfig::default()
        };
        let (fixed, repairs) = cfg.clamp();
        assert!(!repairs.is_empty());
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

    /// `clamp`'s contract is that `validate` is empty afterwards. Two inputs
    /// used to escape it: a `llm_skip_threshold` of zero, where no fraction of
    /// it is strictly below itself, and a NaN — which TOML spells `nan` and
    /// which `f32::clamp` passes straight through.
    #[test]
    fn clamp_repairs_the_inputs_that_a_plain_clamp_cannot() {
        let (fixed, _) = PipelineConfig {
            llm_skip_threshold: Some(0.0),
            llm_skip_noisy_threshold: Some(0.0),
            ..PipelineConfig::default()
        }
        .clamp();
        assert!(fixed.validate().is_empty(), "got {:?}", fixed.validate());

        let (fixed, _) = PipelineConfig {
            speech_threshold: f32::NAN,
            speech_threshold_end: f32::NAN,
            word_confidence_threshold: Some(f32::NAN),
            ..PipelineConfig::default()
        }
        .clamp();
        assert!(fixed.validate().is_empty(), "got {:?}", fixed.validate());
        assert!(fixed.speech_threshold.is_finite());
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
            sentence_close_ms: Some(800),
            adaptive_vad: Some(true),
            ..Default::default()
        };
        let toml_str = toml::to_string_pretty(&overrides).unwrap();
        let parsed: PipelineOverrides = toml::from_str(&toml_str).unwrap();
        assert_eq!(overrides, parsed);
    }
}
