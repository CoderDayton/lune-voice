//! Tunable knobs for the hybrid VAD pipeline.
//!
//! [`VadPipelineConfig`] is the internal, fully-resolved config consumed by
//! [`super::chunker::SpanGate`] and the `HybridVad` orchestrator. It is seeded
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
    /// Hard cap — force-finalize a span once it reaches this length.
    pub max_chunk_ms: u32,
    /// Minimum voiced length to KEEP a whole span (contiguous-span gate). Spans
    /// with less verified speech than this are dropped as noise. Replaces the
    /// per-segment `min_chunk_ms`.
    pub min_speech_ms: u32,
    /// A trailing-silence run this long closes the current span (an intentional
    /// stop). Sub-`long_gap` pauses are KEPT inside the span so Qwen3-ASR sees
    /// the real speech rhythm — this is what stops adjacent words fusing.
    pub long_gap_ms: u32,
}

/// Gate defaults for the contiguous-span gate. The acoustic thresholds
/// (`silero_onset/offset`, `earshot_activity_threshold`, `hangover_ms`,
/// `preroll_ms`) are tuned by `cgs bench TYPE=vad` against end-to-end ASR WER;
/// the span gate matches no-VAD mean WER (8.2%) while improving the tail
/// (p90 16.7% vs 19.2%) and cutting segments-per-clip from ~5 to ~1.6 — the
/// fragmentation that stripped inter-word pauses and fused adjacent words
/// (`MakeSure`) is gone. `long_gap_ms` is deliberately NOT WER-tuned: WER is
/// blind to the fusion bug (it normalizes case/spacing away), so a WER sweep
/// would shrink it back toward the old splice behaviour. It is set to the
/// intentional-stop threshold — pauses below it stay inside the span so the
/// recognizer hears the real rhythm; only a deliberate multi-second stop closes
/// a span. Re-run the benchmark and update the acoustic knobs if the pipeline
/// changes.
pub const DEFAULT_VAD_CONFIG: VadPipelineConfig = VadPipelineConfig {
    sample_rate: 16_000,
    silero_enabled: true,
    earshot_prefilter: true,
    earshot_activity_threshold: 0.35,
    silero_onset: 0.30,
    silero_offset: 0.20,
    preroll_ms: 500,
    hangover_ms: 350,
    max_chunk_ms: 20_000,
    min_speech_ms: 100,
    long_gap_ms: 2_000,
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

    /// Hangover (trailing-silence to keep at a span's tail) in samples.
    pub fn hangover_samples(&self) -> usize {
        self.ms_to_samples(self.hangover_ms)
    }

    /// Long-gap cutoff (intentional-stop threshold) in whole Silero frames.
    pub fn long_gap_frames(&self) -> usize {
        self.ms_to_frames(self.long_gap_ms)
    }

    /// Minimum voiced length to keep a span, in whole Silero frames.
    pub fn min_speech_frames(&self) -> usize {
        self.ms_to_frames(self.min_speech_ms)
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
            min_speech_ms: 64,
            max_chunk_ms: 320,
            preroll_ms: 96,
            long_gap_ms: 2_048,
            ..VadPipelineConfig::default()
        };
        assert_eq!(cfg.frame_samples(), 512);
        assert_eq!(cfg.hangover_frames(), 3);
        assert_eq!(cfg.hangover_samples(), 1_536);
        assert_eq!(cfg.min_speech_frames(), 2);
        assert_eq!(cfg.max_chunk_frames(), 10);
        assert_eq!(cfg.preroll_frames(), 3);
        assert_eq!(cfg.preroll_samples(), 1_536);
        assert_eq!(cfg.long_gap_frames(), 64);
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
