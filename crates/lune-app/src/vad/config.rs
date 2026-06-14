//! Tunable knobs for the hybrid VAD pipeline.
//!
//! [`VadPipelineConfig`] is the internal, fully-resolved config consumed by
//! [`super::chunker::Chunker`] and the `HybridVad` orchestrator. It is seeded
//! from the user-facing [`lune_core::config::PipelineConfig`] via
//! [`VadPipelineConfig::from_pipeline`]; the Silero/chunker knobs that
//! `PipelineConfig` does not yet surface carry the [`DEFAULT_VAD_CONFIG`]
//! placeholders until the `vad_tune` benchmark locks the real defaults.

use lune_core::config::PipelineConfig;

use super::silero::SILERO_FRAME;

/// Resolved configuration for the two-stage hybrid gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadPipelineConfig {
    /// Audio sample rate (Hz). Drives every ms↔frame/sample conversion.
    pub sample_rate: usize,
    /// Master switch for stage 2. `false` falls back to Earshot-only gating.
    pub silero_enabled: bool,
    /// Use Earshot as a cheap pre-gate deciding which frames reach Silero.
    /// `false` runs Silero on every frame (more CPU, Earshot ignored for
    /// segmentation).
    pub earshot_prefilter: bool,
    /// Permissive Earshot `vad_score` gate — high recall, since Silero filters
    /// the false positives. Only consulted when `earshot_prefilter` is on.
    pub earshot_activity_threshold: f32,
    /// Silero prob ≥ this opens a segment (onset).
    pub silero_onset: f32,
    /// Silero prob ≤ this counts as trailing silence (offset). Keep
    /// `silero_offset < silero_onset` for hysteresis.
    pub silero_offset: f32,
    /// Audio prepended to each segment so the first phoneme is never clipped.
    pub preroll_ms: u32,
    /// Trailing silence tolerated inside a segment before it finalizes.
    pub hangover_ms: u32,
    /// Segments with less voiced audio than this are dropped as noise blips.
    pub min_chunk_ms: u32,
    /// Hard cap — force-finalize a segment once it reaches this length.
    pub max_chunk_ms: u32,
    /// Reconstruct inter-segment pauses by injecting silence into the ASR
    /// buffer. The benchmark decides whether it still helps once Silero
    /// segments cleanly.
    pub pause_injection: bool,
}

/// Placeholder defaults until `cgs bench TYPE=vad` locks the tuned values.
pub const DEFAULT_VAD_CONFIG: VadPipelineConfig = VadPipelineConfig {
    sample_rate: 16_000,
    silero_enabled: true,
    earshot_prefilter: true,
    earshot_activity_threshold: 0.35,
    silero_onset: 0.6,
    silero_offset: 0.35,
    preroll_ms: 300,
    hangover_ms: 500,
    min_chunk_ms: 250,
    max_chunk_ms: 20_000,
    pause_injection: false,
};

impl Default for VadPipelineConfig {
    fn default() -> Self {
        DEFAULT_VAD_CONFIG
    }
}

impl VadPipelineConfig {
    /// Seed from the user-facing pipeline config. Fields the hybrid gate does
    /// not yet surface through `PipelineConfig` keep their tuned defaults.
    pub(crate) fn from_pipeline(p: &PipelineConfig) -> Self {
        Self {
            // The Earshot stage-1 recall gate reuses the existing VAD start
            // threshold; Silero (stage 2) owns the precision decision.
            earshot_activity_threshold: p.speech_threshold,
            ..DEFAULT_VAD_CONFIG
        }
    }

    /// Samples per Silero frame — the chunker's clock tick.
    pub const fn frame_samples(&self) -> usize {
        SILERO_FRAME
    }

    fn ms_to_samples(&self, ms: u32) -> usize {
        (ms as usize * self.sample_rate) / 1_000
    }

    fn ms_to_frames(&self, ms: u32) -> usize {
        self.ms_to_samples(ms) / SILERO_FRAME
    }

    /// Pre-roll length in whole Silero frames.
    pub fn preroll_frames(&self) -> usize {
        self.ms_to_frames(self.preroll_ms)
    }

    /// Pre-roll length in samples (sizes the `HybridVad` ring buffer).
    pub fn preroll_samples(&self) -> usize {
        self.ms_to_samples(self.preroll_ms)
    }

    /// Hangover (trailing-silence tolerance) in whole Silero frames.
    pub fn hangover_frames(&self) -> usize {
        self.ms_to_frames(self.hangover_ms)
    }

    /// Minimum voiced length to keep a segment, in whole Silero frames.
    pub fn min_chunk_frames(&self) -> usize {
        self.ms_to_frames(self.min_chunk_ms)
    }

    /// Force-finalize length, in whole Silero frames.
    pub fn max_chunk_frames(&self) -> usize {
        self.ms_to_frames(self.max_chunk_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ms_conversions_match_16khz_frames() {
        // 16 kHz, 512-sample Silero frames → 1 frame = 32 ms.
        let cfg = VadPipelineConfig {
            hangover_ms: 96,
            min_chunk_ms: 64,
            max_chunk_ms: 320,
            preroll_ms: 96,
            ..VadPipelineConfig::default()
        };
        assert_eq!(cfg.frame_samples(), 512);
        assert_eq!(cfg.hangover_frames(), 3);
        assert_eq!(cfg.min_chunk_frames(), 2);
        assert_eq!(cfg.max_chunk_frames(), 10);
        assert_eq!(cfg.preroll_frames(), 3);
        assert_eq!(cfg.preroll_samples(), 1_536);
    }

    #[test]
    fn from_pipeline_takes_earshot_threshold_from_speech_threshold() {
        let p = PipelineConfig {
            speech_threshold: 0.42,
            ..PipelineConfig::default()
        };
        let v = VadPipelineConfig::from_pipeline(&p);
        assert!((v.earshot_activity_threshold - 0.42).abs() < f32::EPSILON);
        // Untouched Silero knobs keep the tuned defaults.
        assert_eq!(v.silero_onset, DEFAULT_VAD_CONFIG.silero_onset);
        assert!(v.silero_enabled);
    }
}
