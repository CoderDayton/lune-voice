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

/// Gate defaults, tuned by `cgs bench TYPE=vad` against end-to-end ASR WER over
/// the 24-fixture panel (clean + SNR 0/5/10 + multilingual). The tuned hybrid
/// scored 12.4% mean WER vs 13.6% for no-VAD (p90 27.8% vs 31.9%, noisy 19.2%
/// vs 20.7%) — the gate removes ~14% of audio the recognizer was hallucinating
/// on. Re-run the benchmark and update here if the pipeline changes.
pub const DEFAULT_VAD_CONFIG: VadPipelineConfig = VadPipelineConfig {
    sample_rate: 16_000,
    silero_enabled: true,
    earshot_prefilter: true,
    earshot_activity_threshold: 0.35,
    silero_onset: 0.30,
    silero_offset: 0.20,
    preroll_ms: 500,
    hangover_ms: 350,
    min_chunk_ms: 100,
    max_chunk_ms: 20_000,
    pause_injection: false,
};

impl Default for VadPipelineConfig {
    fn default() -> Self {
        DEFAULT_VAD_CONFIG
    }
}

impl VadPipelineConfig {
    /// Resolve the gate config for a session. The hybrid gate is self-tuned
    /// ([`DEFAULT_VAD_CONFIG`], set by `cgs bench TYPE=vad`); it runs on the raw
    /// Earshot `vad_score`, so the legacy `PipelineConfig::speech_threshold`
    /// (which drove the old Earshot-only `is_speech` gate) deliberately no
    /// longer feeds it — mapping it in regressed WER badly (0.50 → 16% vs
    /// 0.35 → 13.8%). Kept as the single bridge point should the knobs ever be
    /// surfaced for user override.
    pub(crate) fn from_pipeline(_pipeline: &PipelineConfig) -> Self {
        DEFAULT_VAD_CONFIG
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
    fn from_pipeline_is_self_tuned_and_ignores_speech_threshold() {
        // The raw-score gate is self-tuned: the legacy `speech_threshold` must
        // not leak into `earshot_activity_threshold` (doing so regressed WER).
        let p = PipelineConfig {
            speech_threshold: 0.42,
            ..PipelineConfig::default()
        };
        let v = VadPipelineConfig::from_pipeline(&p);
        assert_eq!(v, DEFAULT_VAD_CONFIG);
        assert_eq!(
            v.earshot_activity_threshold,
            DEFAULT_VAD_CONFIG.earshot_activity_threshold
        );
    }
}
