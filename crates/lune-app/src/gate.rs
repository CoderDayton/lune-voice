//! The selective-correction gate.
//!
//! Decides whether the expensive LLM pass runs. Extracted verbatim from the
//! inline `process_and_deliver` logic so the decision is testable and tunable in
//! one place. Skip-when-broken: correction degrades a transcript that is shaky
//! across the board, so a low segment-average confidence vetoes it
//! (arXiv:2509.25048).

use lune_core::config::PipelineConfig;

/// Everything the gate decision reads.
pub(crate) struct GateInputs<'a> {
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
pub(crate) struct GateOutcome {
    pub needs_correction: bool,
    pub needs_llm: bool,
    pub conf_attention: bool,
    pub too_broken: bool,
    pub min_conf: f32,
    pub avg_conf: f32,
}

/// Decide whether the LLM correction/styling pass should run. The confidence
/// "attention" band fires only when >=2 words fall below `llm_skip_threshold`
/// (see the field doc); the segment-average "too broken" veto is unchanged.
pub(crate) fn decide(g: GateInputs<'_>) -> GateOutcome {
    let max_risk = g
        .word_risk_scores
        .iter()
        .map(|(_, r)| *r)
        .fold(0.0_f32, f32::max);
    let min_conf = 1.0 - max_risk;
    // Attention requires at least TWO low-confidence words, not one. A single
    // dipping word is almost always a correctly-transcribed rare word (a proper
    // noun or brand); escalating on it routed clean dictation to the LLM, and a
    // text-only corrector cannot recover a proper-noun mishear anyway, so the
    // call was wasted. Two uncertain words is a far better proxy for a genuinely
    // shaky transcript correction can help. Validated by benchmarks/gate_sweep.rs
    // on the domain corpus: gate accuracy 45%->91%, LLM calls 13->4, routed WER
    // 5.23%->4.96% (within 0.07pp of the oracle).
    let conf_attention = g.pipeline.llm_skip_threshold.is_some_and(|t| {
        g.word_risk_scores
            .iter()
            .filter(|(_, r)| 1.0 - *r < t)
            .count()
            >= 2
    });

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
    fn single_low_confidence_word_does_not_trigger_attention() {
        // One dipping word (a rare proper noun) is not enough — the gate now
        // requires two uncertain words before escalating a resolved transcript.
        let c = cfg(Some(0.5), None);
        let out = decide(inputs(
            &c,
            true,
            false,
            false,
            &[("a", 0.6), ("clean", 0.1)],
            &[(0, 0.6), (1, 0.1)],
        ));
        assert!(!out.conf_attention && !out.needs_llm);
    }

    #[test]
    fn two_low_confidence_words_trigger_attention() {
        // Two words below the confidence threshold signal a genuinely shaky
        // transcript, so the correction pass runs.
        let c = cfg(Some(0.5), None);
        let out = decide(inputs(
            &c,
            true,
            false,
            false,
            &[("a", 0.6), ("b", 0.7)],
            &[(0, 0.6), (1, 0.7)],
        ));
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

    #[test]
    fn empty_scores_clamp_word_count() {
        // word_count is the pre-correction whitespace split; the gate clamps it
        // to 1 so an empty risk set never divides by zero.
        let c = cfg(Some(0.5), Some(0.3));
        let out = decide(GateInputs {
            pipeline: &c,
            all_resolved: true,
            needs_punct: false,
            wants_styling: false,
            word_risk_scores: &[],
            word_risk_positions: &[],
            word_count: 0,
        });
        assert_eq!(out.min_conf, 1.0); // max_risk over empty = 0
        assert_eq!(out.avg_conf, 1.0); // mean_risk = 0 / max(1) = 0
        assert!(!out.needs_llm);
    }
}
