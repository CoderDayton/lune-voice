use serde::{Deserialize, Serialize};

use crate::Result;

/// Per-word confidence with optional runner-up alternates.
///
/// `alternates` is a (possibly empty) ordered list of runner-up words
/// produced by backends that expose top-K decoding (e.g. Parakeet TDT via
/// subword swap). Each entry differs from `word` by exactly one subword.
/// Backends without top-K capability (`OpenAi`) leave the vec
/// empty and the legacy `[?word?]` marker is emitted by `mark_uncertain`.
///
/// Confidence is on the calibrated `P(correct | raw)` scale after the
/// session layer runs the isotonic calibrator. Raw per-token confidence
/// comes from Tsallis entropy (α = 0.5) over the full vocab distribution,
/// not softmax-of-argmax — see `tdt_decoder::greedy_decode`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WordConfidence {
    pub word: String,
    pub confidence: f32,
    #[serde(default)]
    pub alternates: Vec<String>,
}

impl WordConfidence {
    /// Convenience constructor for the common "no alternates" case. Prefer
    /// this over struct-literal syntax at call sites that do not populate
    /// the runner-up field.
    #[inline]
    pub fn new(word: impl Into<String>, confidence: f32) -> Self {
        Self {
            word: word.into(),
            confidence,
            alternates: Vec::new(),
        }
    }
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
    /// Ordered N-best list of full-sentence hypotheses for this segment.
    ///
    /// First element (when present) is always the argmax transcript;
    /// subsequent elements are alternate transcriptions derived from the
    /// runner-up tokens via single- or double-word substitution. Empty
    /// when the backend can't produce alternates or `build_hypotheses`
    /// hasn't been called yet.
    ///
    /// Consumed by the n-best LLM rescoring path in `lune-llm` — the
    /// corrector picks one of these hypotheses as the final transcript.
    #[serde(default)]
    pub hypotheses: Vec<String>,
    /// Immutable once true.
    pub is_final: bool,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub text: String,
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

    /// Minimum per-word confidence across this segment, or `None` when
    /// per-word data is not available. After session-layer calibration,
    /// this is `min P(correct | raw)` — the single weakest word.
    #[inline]
    pub fn min_word_confidence(&self) -> Option<f32> {
        self.word_confidences
            .iter()
            .map(|w| w.confidence)
            .reduce(f32::min)
    }

    /// Returns `true` when every word in the segment meets or exceeds
    /// `skip_threshold`, signalling that the transcript is clean enough to
    /// bypass LLM correction entirely. Conservative: if `word_confidences`
    /// is empty, returns `false` so the caller falls back to the regular
    /// path (we never skip LLM on unknown evidence).
    #[inline]
    pub fn is_clean_for_llm(&self, skip_threshold: f32) -> bool {
        if self.word_confidences.is_empty() {
            return false;
        }
        self.word_confidences
            .iter()
            .all(|w| w.confidence >= skip_threshold)
    }

    /// Populate `self.hypotheses` with a ranked N-best list derived from
    /// `word_confidences` and their per-word `alternates`.
    ///
    /// Algorithm:
    /// 1. Hypothesis 0 is always the argmax sentence
    ///    `word_confidences[*].word` joined with spaces.
    /// 2. For every word at position `i` with `confidence < threshold` and
    ///    at least one alternate, generate single-swap candidates:
    ///    `argmax with word[i] replaced by each alternate[j]`. These are
    ///    the positions where the decoder's raw posterior disagrees with
    ///    the argmax most sharply — the most likely sites of an error.
    /// 3. Rank single-swap candidates by `1 - confidence[i]` descending
    ///    (weakest position first), then by alternate index ascending
    ///    (strongest alternate first).
    /// 4. Append one **double-swap** candidate at the two weakest positions
    ///    simultaneously. Covers the Hamming-distance-2 error cases that
    ///    single-swap cannot reach (≈20 % of errored sentences under the
    ///    assumption of per-word error rate ~0.1).
    /// 5. Deduplicate case-insensitively, preserve argmax at index 0, and
    ///    truncate to `k_max` entries.
    ///
    /// This is a post-hoc approximation of beam search that captures most
    /// Hamming-≤1 errors and a fraction of Hamming-2 errors, with zero
    /// extra decoder-inference cost. See `build_hypotheses_tests` for
    /// worked examples.
    pub fn build_hypotheses(&mut self, threshold: f32, k_max: usize) {
        self.hypotheses = compute_hypotheses(&self.word_confidences, threshold, k_max);
    }

    /// Returns `true` when the segment's minimum token confidence falls
    /// below `broken_threshold`, signalling that the audio was too degraded
    /// for the LLM corrector to help — invoking it on broken transcripts
    /// triggers the overcorrection failure mode (arXiv:2509.25048 Fig 3).
    ///
    /// This is the symmetric lower bound of `is_clean_for_llm`: together
    /// they form an adaptive, bimodal LLM gate that skips correction on
    /// both ends of the quality spectrum and runs only in the mid-range
    /// where the LLM has a fighting chance.
    ///
    /// Reads `min_token_confidence` when present, otherwise falls back to
    /// the minimum over `word_confidences`. If neither is available,
    /// returns `false` (never skip on absent evidence).
    #[inline]
    pub fn is_broken_for_llm(&self, broken_threshold: f32) -> bool {
        let min = self
            .min_token_confidence
            .or_else(|| self.min_word_confidence());
        match min {
            Some(m) => m < broken_threshold,
            None => false,
        }
    }

    /// Duration of the segment in milliseconds, if both timestamps are known.
    #[inline]
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_ms?.checked_sub(self.start_ms?)
    }

    /// Build text with low-confidence words annotated for the LLM corrector.
    ///
    /// Emission rules per word:
    /// - confidence ≥ `threshold` → bare word (no annotation)
    /// - confidence < `threshold` + `Some(alternate)` + `alternate != word` →
    ///   `{word|alternate}` (runner-up channel, backends that support it)
    /// - confidence < `threshold` + `None`/identical alternate → `[?word?]`
    ///   (legacy marker, works for every backend)
    ///
    /// Repeated adjacent words never get annotated — the LLM incorrectly
    /// "corrects" intentional repetitions away.
    ///
    /// Falls back to `self.text.clone()` when no per-word data is available.
    pub fn mark_uncertain(&self, threshold: f32) -> String {
        if self.word_confidences.is_empty() {
            return self.text.clone();
        }
        let mut parts: Vec<String> = Vec::with_capacity(self.word_confidences.len());
        let mut prev_word: Option<&str> = None;
        for wc in &self.word_confidences {
            let is_repeat = prev_word.is_some_and(|pw| pw.eq_ignore_ascii_case(&wc.word));
            if wc.confidence < threshold && !is_repeat {
                parts.push(format_uncertain(&wc.word, &wc.alternates));
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
            let is_repeat = prev_word.is_some_and(|pw| pw.eq_ignore_ascii_case(&wc.word));

            let effective = contextual_threshold(&self.word_confidences, idx, base_threshold);

            if wc.confidence < effective && !is_repeat {
                parts.push(format_uncertain(&wc.word, &wc.alternates));
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

/// Pure helper for [`AsrSegment::build_hypotheses`]. Separated so the
/// ranking logic can be exercised by unit tests without constructing a
/// full `AsrSegment`, and so the algorithm is testable end-to-end with
/// synthetic `WordConfidence` inputs.
pub(crate) fn compute_hypotheses(
    words: &[WordConfidence],
    threshold: f32,
    k_max: usize,
) -> Vec<String> {
    if k_max == 0 || words.is_empty() {
        return Vec::new();
    }

    let argmax: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
    let argmax_sentence = argmax.join(" ");
    if k_max == 1 {
        return vec![argmax_sentence];
    }

    // Single-swap candidates: one per (uncertain position, alternate).
    // Score = (1 - confidence[i]) so weakest positions rank highest.
    // For ties (same position, multiple alternates) we emit the strongest
    // alternate first by preserving the slice order from `alternates`.
    let mut candidates: Vec<(f32, usize, String)> = Vec::new();
    let mut uncertain_positions: Vec<(usize, f32)> = Vec::new();
    for (i, wc) in words.iter().enumerate() {
        if wc.confidence >= threshold || wc.alternates.is_empty() {
            continue;
        }
        uncertain_positions.push((i, wc.confidence));
        let position_score = (1.0 - wc.confidence).clamp(0.0, 1.0);
        for (alt_idx, alt) in wc.alternates.iter().enumerate() {
            if alt.is_empty() || alt.eq_ignore_ascii_case(&wc.word) {
                continue;
            }
            // Penalise weaker alternates at the same position so alt[0]
            // outranks alt[1] at the same site.
            let score = position_score - (alt_idx as f32) * 0.05;
            let mut pieces: Vec<String> = argmax.iter().map(|s| s.to_string()).collect();
            pieces[i] = alt.clone();
            candidates.push((score, i, pieces.join(" ")));
        }
    }

    // Sort by score descending (strongest candidate first).
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Assemble: argmax first, then single-swaps in order, deduped.
    let mut result: Vec<String> = vec![argmax_sentence.clone()];
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(argmax_sentence.to_ascii_lowercase());
    for (_, _, sentence) in &candidates {
        if result.len() >= k_max {
            break;
        }
        let key = sentence.to_ascii_lowercase();
        if seen.insert(key) {
            result.push(sentence.clone());
        }
    }

    // Double-swap: combine the strongest alternate at the two weakest
    // positions (if any) to catch Hamming-distance-2 error patterns.
    // Weight is multiplicative in error probability: two independent
    // errors at 10 % each give a joint probability of 1 %, so this path
    // matters less often but is the only way to reach joint corrections.
    if result.len() < k_max && uncertain_positions.len() >= 2 {
        uncertain_positions.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (i0, _) = uncertain_positions[0];
        let (i1, _) = uncertain_positions[1];
        if i0 != i1
            && let (Some(alt0), Some(alt1)) =
                (words[i0].alternates.first(), words[i1].alternates.first())
            && !alt0.is_empty()
            && !alt1.is_empty()
        {
            let mut pieces: Vec<String> = argmax.iter().map(|s| s.to_string()).collect();
            pieces[i0] = alt0.clone();
            pieces[i1] = alt1.clone();
            let double = pieces.join(" ");
            let key = double.to_ascii_lowercase();
            if seen.insert(key) {
                result.push(double);
            }
        }
    }

    result.truncate(k_max);
    result
}

/// Format a below-threshold word as either `{word|alt1|alt2|...}`
/// (runner-up channel) or legacy `[?word?]` marker. Filters out alternates
/// that are empty or case-insensitively identical to the best guess so the
/// small corrector doesn't waste attention on degenerate choices.
///
/// Extracted so the contextual and flat marking paths stay in sync.
#[inline]
fn format_uncertain(word: &str, alternates: &[String]) -> String {
    let useful: Vec<&str> = alternates
        .iter()
        .map(String::as_str)
        .filter(|alt| !alt.is_empty() && !alt.eq_ignore_ascii_case(word))
        .collect();
    if useful.is_empty() {
        format!("[?{word}?]")
    } else {
        let joined = useful.join("|");
        format!("{{{word}|{joined}}}")
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
    OpenAiCompatible,
}

impl AsrBackendKind {
    /// Recommended `max_segment_ms` for the VadSegmenter.
    ///
    /// All backends handle long audio natively. This is a safety ceiling,
    /// not an encoder limit.
    #[deprecated(note = "use AsrRegistry::pipeline_config().max_segment_ms instead")]
    pub fn suggested_max_segment_ms(&self) -> u32 {
        match self {
            Self::ParakeetTdt => 30_000,
            Self::OpenAiCompatible => 30_000,
        }
    }

    /// Short stable identifier used for per-backend filenames like
    /// `calibrator_{slug}.json`. Must stay lowercase, ASCII, no spaces.
    ///
    /// These strings are part of the on-disk layout — renaming them will
    /// orphan existing calibrator files. Add new variants rather than
    /// changing existing slugs.
    #[inline]
    pub fn slug(&self) -> &'static str {
        match self {
            Self::ParakeetTdt => "parakeet",
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

    /// Whether this backend supports streaming (partial results during audio
    /// push). When true, `poll_results` returns incremental text. When false,
    /// all transcription happens in `poll_final` (batch mode).
    ///
    /// The session loop uses this to decide whether to deliver text
    /// progressively or wait for the final batch result.
    #[inline]
    fn supports_streaming(&self) -> bool {
        false
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
            hypotheses: vec![],
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
            WordConfidence::new("hello", 0.95),
            WordConfidence::new("world", 0.60),
        ];
        assert_eq!(seg.mark_uncertain(0.85), "hello [?world?]");
    }

    #[test]
    fn mark_uncertain_all_confident() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("hello", 0.95),
            WordConfidence::new("world", 0.90),
        ];
        assert_eq!(seg.mark_uncertain(0.85), "hello world");
    }

    #[test]
    fn mark_uncertain_emits_alternate_when_below_threshold() {
        // Runner-up alternate present + below threshold → {best|alt}.
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("the", 0.95),
            WordConfidence {
                word: "fox".into(),
                confidence: 0.40,
                alternates: vec!["box".into()],
            },
        ];
        assert_eq!(seg.mark_uncertain(0.85), "the {fox|box}");
    }

    #[test]
    fn mark_uncertain_alternate_collapses_when_identical() {
        // alt == best (case-insensitive) → fall back to legacy [?word?].
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence {
            word: "fox".into(),
            confidence: 0.40,
            alternates: vec!["Fox".into()],
        }];
        assert_eq!(seg.mark_uncertain(0.85), "[?fox?]");
    }

    #[test]
    fn mark_uncertain_emits_multi_alternate_format() {
        // Multiple alternates → `{best|a|b|c}` format, filtered of
        // degenerate (empty/identical) entries.
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence {
            word: "weather".into(),
            confidence: 0.40,
            alternates: vec![
                "whether".into(),
                "Weather".into(), // case-identical, must be filtered
                "wether".into(),
            ],
        }];
        assert_eq!(seg.mark_uncertain(0.85), "{weather|whether|wether}");
    }

    #[test]
    fn mark_uncertain_alternate_collapses_when_empty_string() {
        // Empty-string alternate is filtered; with no useful alternates,
        // format falls back to legacy marker.
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence {
            word: "fox".into(),
            confidence: 0.40,
            alternates: vec![String::new()],
        }];
        assert_eq!(seg.mark_uncertain(0.85), "[?fox?]");
    }

    #[test]
    fn mark_uncertain_alternate_ignored_when_above_threshold() {
        // High-confidence word keeps its alternate hidden — no annotation.
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence {
            word: "fox".into(),
            confidence: 0.99,
            alternates: vec!["box".into()],
        }];
        assert_eq!(seg.mark_uncertain(0.85), "fox");
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
            WordConfidence::new("hello", 0.95),
            WordConfidence::new("world", 0.60),
        ];
        assert!(seg.has_uncertain_words(0.85));
    }

    #[test]
    fn has_uncertain_words_all_above_threshold() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("hello", 0.95),
            WordConfidence::new("world", 0.90),
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
            WordConfidence::new("the", 0.60),
            WordConfidence::new("meeting", 0.95),
            WordConfidence::new("starts", 0.92),
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
            WordConfidence::new("the", 0.95),
            WordConfidence::new("weather", 0.62),
            WordConfidence::new("today", 0.94),
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
            WordConfidence::new("tha", 0.35),
            WordConfidence::new("wether", 0.63),
            WordConfidence::new("tody", 0.30),
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
            WordConfidence::new("bad", 0.20),
            WordConfidence::new("good", 0.92),
            WordConfidence::new("bad", 0.20),
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
        seg.word_confidences = vec![WordConfidence::new("hello", 0.60)];
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
            WordConfidence::new("the", 0.95),
            WordConfidence::new("meeting", 0.95),
            WordConfidence::new("today", 0.63),
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        assert_eq!(result, "the meeting today");
    }

    #[test]
    fn contextual_repetition_guard_preserved() {
        // Repeated words should not be marked even when low-confidence
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("very", 0.50),
            WordConfidence::new("very", 0.50),
            WordConfidence::new("good", 0.95),
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        // First "very" may be marked, but second "very" must NOT be (repetition guard)
        assert!(result.contains("very good"));
        assert!(!result.contains("[?very?] [?very?]"));
    }

    #[test]
    fn contextual_emits_alternate_when_available() {
        // Contextual path must use the same {best|alt} format as the flat path.
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("the", 0.20),
            WordConfidence {
                word: "fox".into(),
                confidence: 0.30,
                alternates: vec!["box".into()],
            },
            WordConfidence::new("jumps", 0.25),
        ];
        let result = seg.mark_uncertain_contextual(0.65);
        // "fox" should be marked with its alternate, not the legacy marker
        assert!(result.contains("{fox|box}"));
    }

    #[test]
    fn word_confidence_serde_round_trip() {
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence {
            word: "test".into(),
            confidence: 0.75,
            alternates: vec!["taste".into(), "text".into()],
        }];
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: AsrSegment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.word_confidences.len(), 1);
        assert_eq!(parsed.word_confidences[0].word, "test");
        assert_eq!(
            parsed.word_confidences[0].alternates,
            vec!["taste".to_string(), "text".to_string()]
        );
    }

    #[test]
    fn min_word_confidence_empty_returns_none() {
        let seg = make_segment();
        assert!(seg.min_word_confidence().is_none());
    }

    #[test]
    fn min_word_confidence_returns_lowest() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("a", 0.9),
            WordConfidence::new("b", 0.3),
            WordConfidence::new("c", 0.7),
        ];
        assert_eq!(seg.min_word_confidence(), Some(0.3));
    }

    #[test]
    fn is_clean_for_llm_empty_returns_false() {
        // Conservative: no per-word data → don't skip, defer to LLM.
        let seg = make_segment();
        assert!(!seg.is_clean_for_llm(0.92));
    }

    #[test]
    fn is_clean_for_llm_all_above_skip_threshold() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("a", 0.95),
            WordConfidence::new("b", 0.93),
            WordConfidence::new("c", 0.99),
        ];
        assert!(seg.is_clean_for_llm(0.92));
    }

    #[test]
    fn is_clean_for_llm_one_below_skip_threshold() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("a", 0.95),
            WordConfidence::new("b", 0.70), // below skip
            WordConfidence::new("c", 0.99),
        ];
        assert!(!seg.is_clean_for_llm(0.92));
    }

    #[test]
    fn is_clean_for_llm_exactly_at_threshold() {
        let mut seg = make_segment();
        seg.word_confidences = vec![WordConfidence::new("a", 0.92)];
        assert!(seg.is_clean_for_llm(0.92));
    }

    #[test]
    fn word_confidence_serde_missing_alternates_defaults_to_empty() {
        // Serde must accept old records without the alternates field.
        let json = r#"{"id":1,"confidence":0.9,"min_token_confidence":null,"word_confidences":[{"word":"hi","confidence":0.9}],"is_final":true,"start_ms":null,"end_ms":null,"text":"hi"}"#;
        let parsed: AsrSegment = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.word_confidences.len(), 1);
        assert!(parsed.word_confidences[0].alternates.is_empty());
    }

    // ── is_broken_for_llm (bimodal gate lower bound) ────────────────────

    #[test]
    fn is_broken_for_llm_uses_min_token_confidence_when_present() {
        let mut seg = make_segment();
        seg.min_token_confidence = Some(0.20);
        assert!(seg.is_broken_for_llm(0.30));
        assert!(!seg.is_broken_for_llm(0.15));
    }

    #[test]
    fn is_broken_for_llm_falls_back_to_word_confidences() {
        let mut seg = make_segment();
        seg.min_token_confidence = None;
        seg.word_confidences = vec![
            WordConfidence::new("a", 0.90),
            WordConfidence::new("b", 0.15), // weakest
            WordConfidence::new("c", 0.80),
        ];
        assert!(seg.is_broken_for_llm(0.30));
    }

    #[test]
    fn is_broken_for_llm_returns_false_with_no_signal() {
        // No min_token_confidence, empty word_confidences → conservative
        // "don't skip" behaviour.
        let mut seg = make_segment();
        seg.min_token_confidence = None;
        seg.word_confidences = vec![];
        assert!(!seg.is_broken_for_llm(0.30));
    }

    // ── compute_hypotheses (sentence-level N-best) ──────────────────────

    fn wc_with_alts(word: &str, confidence: f32, alts: &[&str]) -> WordConfidence {
        WordConfidence {
            word: word.into(),
            confidence,
            alternates: alts.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn compute_hypotheses_empty_input() {
        let result = compute_hypotheses(&[], 0.85, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn compute_hypotheses_returns_argmax_when_all_confident() {
        let words = vec![
            WordConfidence::new("the", 0.99),
            WordConfidence::new("fox", 0.98),
            WordConfidence::new("jumped", 0.97),
        ];
        let result = compute_hypotheses(&words, 0.85, 5);
        assert_eq!(result, vec!["the fox jumped".to_string()]);
    }

    #[test]
    fn compute_hypotheses_single_swap_at_weak_position() {
        let words = vec![
            WordConfidence::new("the", 0.99),
            wc_with_alts("fox", 0.40, &["box", "ox"]),
            WordConfidence::new("jumped", 0.97),
        ];
        let result = compute_hypotheses(&words, 0.85, 5);
        assert_eq!(result[0], "the fox jumped");
        assert_eq!(result[1], "the box jumped");
        assert_eq!(result[2], "the ox jumped");
    }

    #[test]
    fn compute_hypotheses_weakest_position_ranked_first() {
        let words = vec![
            wc_with_alts("big", 0.70, &["pig"]),
            WordConfidence::new("the", 0.99),
            wc_with_alts("fox", 0.30, &["box"]),
            WordConfidence::new("jumped", 0.97),
        ];
        let result = compute_hypotheses(&words, 0.85, 5);
        assert_eq!(result[0], "big the fox jumped");
        assert_eq!(result[1], "big the box jumped");
        assert_eq!(result[2], "pig the fox jumped");
    }

    #[test]
    fn compute_hypotheses_double_swap_at_two_weakest() {
        let words = vec![
            wc_with_alts("big", 0.30, &["pig"]),
            WordConfidence::new("the", 0.99),
            wc_with_alts("fox", 0.35, &["box"]),
        ];
        let result = compute_hypotheses(&words, 0.85, 5);
        assert!(result.contains(&"big the fox".to_string()));
        assert!(result.contains(&"pig the box".to_string()));
    }

    #[test]
    fn compute_hypotheses_dedupe_when_alt_equals_best_word() {
        let words = vec![wc_with_alts("fox", 0.40, &["fox"])];
        let result = compute_hypotheses(&words, 0.85, 5);
        assert_eq!(result, vec!["fox".to_string()]);
    }

    #[test]
    fn compute_hypotheses_respects_k_max() {
        let words = vec![
            wc_with_alts("a", 0.30, &["1", "2", "3"]),
            wc_with_alts("b", 0.35, &["4", "5"]),
            wc_with_alts("c", 0.40, &["6"]),
        ];
        let result = compute_hypotheses(&words, 0.85, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "a b c");
    }

    #[test]
    fn compute_hypotheses_k_max_zero_returns_empty() {
        let words = vec![WordConfidence::new("hi", 0.9)];
        assert!(compute_hypotheses(&words, 0.85, 0).is_empty());
    }

    #[test]
    fn compute_hypotheses_k_max_one_returns_argmax_only() {
        let words = vec![
            wc_with_alts("the", 0.30, &["a"]),
            WordConfidence::new("fox", 0.99),
        ];
        let result = compute_hypotheses(&words, 0.85, 1);
        assert_eq!(result, vec!["the fox".to_string()]);
    }

    #[test]
    fn build_hypotheses_writes_to_segment() {
        let mut seg = make_segment();
        seg.word_confidences = vec![
            WordConfidence::new("hello", 0.99),
            wc_with_alts("world", 0.30, &["word"]),
        ];
        seg.build_hypotheses(0.85, 5);
        assert_eq!(seg.hypotheses.len(), 2);
        assert_eq!(seg.hypotheses[0], "hello world");
        assert_eq!(seg.hypotheses[1], "hello word");
    }
}
