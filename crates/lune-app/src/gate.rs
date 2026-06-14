//! The selective-correction gate.
//!
//! Decides whether the expensive LLM pass runs. Extracted verbatim from the
//! inline `process_and_deliver` logic so the decision is testable and tunable in
//! one place. Skip-when-broken: correction degrades a transcript that is shaky
//! across the board, so a low segment-average confidence vetoes it
//! (arXiv:2509.25048).

use lune_core::config::PipelineConfig;

/// Everything the gate decision reads.
pub struct GateInputs<'a> {
    pub pipeline: &'a PipelineConfig,
    /// `CorrectionResult::all_resolved`.
    pub all_resolved: bool,
    /// `PunctuationOutcome::needs_llm`.
    pub needs_punct: bool,
    /// `StylePreset::needs_styling`.
    pub wants_styling: bool,
    /// Per-word ASR risk over the whole utterance (the min-confidence band).
    pub word_risk_scores: &'a [(&'a str, f32)],
    /// `CorrectionResult::word_risk_positions` (the segment-average band).
    pub word_risk_positions: &'a [(usize, f32)],
    /// Whitespace word count of the post-command text (segment-average divisor).
    pub word_count: usize,
}

/// The gate's decision plus the confidence signals it derived (for logging).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateOutcome {
    pub needs_correction: bool,
    pub needs_llm: bool,
    pub conf_attention: bool,
    pub too_broken: bool,
    pub min_conf: f32,
    pub avg_conf: f32,
}

/// Decide whether the LLM correction/styling pass should run. The arithmetic is
/// byte-for-byte the prior inline gate — do not "simplify" the f32 folds.
pub fn decide(g: GateInputs<'_>) -> GateOutcome {
    let max_risk = g
        .word_risk_scores
        .iter()
        .map(|(_, r)| *r)
        .fold(0.0_f32, f32::max);
    let min_conf = 1.0 - max_risk;
    let conf_attention = g.pipeline.llm_skip_threshold.is_some_and(|t| min_conf < t);

    let word_count = g.word_count.max(1);
    let mean_risk = g.word_risk_positions.iter().map(|(_, r)| *r).sum::<f32>() / word_count as f32;
    let avg_conf = 1.0 - mean_risk;
    let too_broken = g
        .pipeline
        .llm_skip_noisy_threshold
        .is_some_and(|t| avg_conf < t);

    let needs_correction = !g.all_resolved || g.needs_punct || conf_attention;
    let needs_llm = (needs_correction || g.wants_styling) && !too_broken;

    GateOutcome {
        needs_correction,
        needs_llm,
        conf_attention,
        too_broken,
        min_conf,
        avg_conf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(skip: Option<f32>, noisy: Option<f32>) -> PipelineConfig {
        PipelineConfig {
            llm_skip_threshold: skip,
            llm_skip_noisy_threshold: noisy,
            ..Default::default()
        }
    }

    fn inputs<'a>(
        pipeline: &'a PipelineConfig,
        all_resolved: bool,
        needs_punct: bool,
        wants_styling: bool,
        scores: &'a [(&'a str, f32)],
        positions: &'a [(usize, f32)],
    ) -> GateInputs<'a> {
        GateInputs {
            pipeline,
            all_resolved,
            needs_punct,
            wants_styling,
            word_risk_scores: scores,
            word_risk_positions: positions,
            word_count: scores.len(),
        }
    }

    #[test]
    fn clean_resolved_transcript_skips_llm() {
        let c = cfg(Some(0.5), Some(0.3));
        let out = decide(inputs(
            &c,
            true,
            false,
            false,
            &[("the", 0.1), ("dog", 0.1)],
            &[(0, 0.1), (1, 0.1)],
        ));
        assert!(!out.needs_correction && !out.needs_llm);
    }

    #[test]
    fn unresolved_word_triggers_correction() {
        let c = cfg(Some(0.5), Some(0.3));
        let out = decide(inputs(&c, false, false, false, &[("x", 0.1)], &[(0, 0.1)]));
        assert!(out.needs_correction && out.needs_llm);
    }

    #[test]
    fn low_confidence_word_triggers_attention() {
        // max_risk 0.6 -> min_conf 0.4 < 0.5 -> conf_attention.
        let c = cfg(Some(0.5), None);
        let out = decide(inputs(&c, true, false, false, &[("a", 0.6)], &[(0, 0.6)]));
        assert!(out.conf_attention && out.needs_llm);
    }

    #[test]
    fn too_broken_vetoes_correction() {
        // avg_conf 0.5 < 0.9 -> too_broken -> needs_llm false despite correction.
        let c = cfg(Some(0.5), Some(0.9));
        let out = decide(inputs(&c, false, false, false, &[("x", 0.5)], &[(0, 0.5)]));
        assert!(out.needs_correction && !out.needs_llm);
    }

    #[test]
    fn styling_forces_llm_on_clean_transcript() {
        let c = cfg(Some(0.5), Some(0.3));
        let out = decide(inputs(&c, true, false, true, &[("ok", 0.1)], &[(0, 0.1)]));
        assert!(!out.needs_correction && out.needs_llm);
    }
}
