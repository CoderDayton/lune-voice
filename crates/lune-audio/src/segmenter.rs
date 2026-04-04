//! VAD-driven speech segmenter with three-state machine.
//!
//! Replaces binary per-chunk VAD gating with proper segment boundary
//! detection: `Silence → Speech → TrailingSilence → flush → Silence`.
//!
//! Key properties:
//! - `max_segment_ms` (30000) force-flushes at the 60s encoder input limit
//! - Tiered silence: `sentence_close_ms` (600) for sentence boundaries,
//!   `hard_close_ms` (1200) for topic breaks. Pauses < sentence_close_ms
//!   are absorbed to keep segments intact across natural hesitations.
//! - Optional adaptive mode: learns thresholds from the speaker's own
//!   pause rhythm (absorbed vs completed pauses). P95-based with EMA.
//! - `min_speech_ms` (150) ignores sub-150ms noise bursts
//! - `avg_prob` per segment feeds into confidence gating

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use ten_vad_rs::TenVad;
use tracing::warn;

/// Embedded TEN VAD ONNX model (~306KB).
const TEN_VAD_MODEL: &[u8] = include_bytes!("../models/ten-vad.onnx");
const HOP_SIZE: usize = 256; // 16ms at 16kHz
const SAMPLE_RATE: u32 = 16_000;

/// Online percentile estimator using a fixed-size FIFO buffer.
///
/// Stores observations in insertion order (FIFO eviction of oldest).
/// Sorts a temporary copy for percentile queries — O(n log n) but n ≤ 100.
#[derive(Clone, Serialize, Deserialize)]
struct RunningPercentile {
    buf: VecDeque<u32>,
    capacity: usize,
}

impl RunningPercentile {
    fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, value: u32) {
        if self.buf.len() == self.capacity {
            self.buf.pop_front(); // evict oldest observation
        }
        self.buf.push_back(value);
    }

    fn percentile(&self, p: f32) -> Option<u32> {
        if self.buf.is_empty() {
            return None;
        }
        let mut sorted: Vec<u32> = self.buf.iter().copied().collect();
        sorted.sort_unstable();
        let idx = ((p * (sorted.len() - 1) as f32) as usize).min(sorted.len() - 1);
        Some(sorted[idx])
    }

    fn len(&self) -> usize {
        self.buf.len()
    }
}

/// Adaptive silence thresholds learned from the speaker's pause rhythm.
///
/// Tracks two populations of pauses:
/// - **Absorbed**: speech resumed after the pause (within-sentence hesitations)
/// - **Completed**: pause triggered a segment flush (between-sentence breaks)
///
/// `sentence_close = P95(absorbed) × 1.5` — just past the longest hesitation.
/// `hard_close = P95(completed) × 1.2` — just past the longest sentence break.
/// EMA smoothing prevents threshold thrashing on outliers.
///
/// Serializable for cross-session persistence.
#[derive(Clone, Serialize, Deserialize)]
pub struct AdaptiveVadConfig {
    pub min_sentence_close_ms: u32,
    pub max_sentence_close_ms: u32,
    pub min_hard_close_ms: u32,
    pub max_hard_close_ms: u32,

    absorbed_pauses: RunningPercentile,
    completed_pauses: RunningPercentile,

    warmup_samples: usize,

    current_sentence_close: u32,
    current_hard_close: u32,

    ema_alpha: f32,
}

impl AdaptiveVadConfig {
    pub fn new(initial_sentence_close: u32, initial_hard_close: u32) -> Self {
        Self {
            min_sentence_close_ms: 500,
            max_sentence_close_ms: 4_000,
            min_hard_close_ms: 1_200,
            max_hard_close_ms: 8_000,
            absorbed_pauses: RunningPercentile::new(60),
            completed_pauses: RunningPercentile::new(40),
            warmup_samples: 0,
            current_sentence_close: initial_sentence_close,
            current_hard_close: initial_hard_close,
            ema_alpha: 0.6,
        }
    }

    /// Record a pause that was absorbed (speech resumed before threshold fired).
    pub fn record_absorbed_pause(&mut self, duration_ms: u32) {
        self.absorbed_pauses.push(duration_ms);
        self.recompute();
    }

    /// Record a pause that closed a segment (threshold fired).
    pub fn record_completed_pause(&mut self, duration_ms: u32) {
        self.completed_pauses.push(duration_ms);
        self.recompute();
    }

    pub fn sentence_close_ms(&self) -> u32 {
        self.current_sentence_close
    }

    pub fn hard_close_ms(&self) -> u32 {
        self.current_hard_close
    }

    fn recompute(&mut self) {
        if let Some(p95_absorbed) = self.absorbed_pauses.percentile(0.95) {
            let target = ((p95_absorbed as f32 * 1.5) as u32)
                .clamp(self.min_sentence_close_ms, self.max_sentence_close_ms);
            self.current_sentence_close = self.ema(self.current_sentence_close, target);
        }

        // hard_close from completed pauses, or 2.2× sentence_close if insufficient data
        let hard_target = if self.completed_pauses.len() >= 2 {
            if let Some(p95_completed) = self.completed_pauses.percentile(0.95) {
                ((p95_completed as f32 * 1.2) as u32)
                    .clamp(self.min_hard_close_ms, self.max_hard_close_ms)
            } else {
                self.current_sentence_close * 2
            }
        } else {
            (self.current_sentence_close as f32 * 2.2) as u32
        };

        self.current_hard_close = self
            .ema(self.current_hard_close, hard_target)
            .max(self.current_sentence_close + 200); // hard > sentence always
    }

    #[inline]
    fn ema(&self, current: u32, target: u32) -> u32 {
        (self.ema_alpha * target as f32 + (1.0 - self.ema_alpha) * current as f32) as u32
    }

    pub fn reset(&mut self) {
        self.absorbed_pauses = RunningPercentile::new(self.absorbed_pauses.capacity);
        self.completed_pauses = RunningPercentile::new(self.completed_pauses.capacity);
    }

    pub fn debug_state(&self) -> String {
        format!(
            "sentence_close={}ms  hard_close={}ms  absorbed_n={}  completed_n={}  (warmup: {})",
            self.current_sentence_close,
            self.current_hard_close,
            self.absorbed_pauses.len(),
            self.completed_pauses.len(),
            if self.absorbed_pauses.len() < self.warmup_samples {
                "active"
            } else {
                "done"
            },
        )
    }
}

#[derive(Debug, PartialEq)]
enum VadState {
    Silence,
    Speech,
    TrailingSilence,
}

/// Completed speech segment boundary (no PCM — caller handles audio routing).
#[derive(Debug, Clone)]
pub struct SegmentBoundary {
    pub start_ms: u32,
    pub end_ms: u32,
    pub avg_prob: f32,
}

/// Configuration for [`VadSegmenter`].
#[derive(Debug, Clone)]
pub struct VadSegmenterConfig {
    /// VAD probability threshold for speech detection (0.0–1.0).
    pub speech_threshold: f32,
    /// Ignore speech bursts shorter than this (ms).
    pub min_speech_ms: u32,
    /// Silence duration to close a segment at sentence boundary (ms).
    /// Pauses shorter than this are absorbed into the current segment.
    /// Used as initial value during adaptive warmup.
    pub sentence_close_ms: u32,
    /// Force close regardless — topic break / long silence (ms).
    /// Used as initial value during adaptive warmup.
    pub hard_close_ms: u32,
    /// Force-flush segments longer than this (ms). Must be < encoder window.
    pub max_segment_ms: u32,
    /// Enable adaptive silence thresholds that learn from the speaker's
    /// pause rhythm. When true, `sentence_close_ms` and `hard_close_ms`
    /// serve as initial defaults during warmup, then adapt.
    pub adaptive: bool,
}

impl Default for VadSegmenterConfig {
    fn default() -> Self {
        // Conservative defaults for streaming ASR with context reset.
        // Frequent resets (short sentence_close) prevent past_text accumulation
        // which causes truncation on longer utterances.
        Self {
            speech_threshold: 0.5,
            min_speech_ms: 150,
            sentence_close_ms: 600,
            hard_close_ms: 1_200,
            max_segment_ms: 30_000,
            adaptive: false,
        }
    }
}

impl VadSegmenterConfig {
    /// Conversational speech — slightly longer pauses tolerated.
    pub fn conversational() -> Self {
        Self {
            sentence_close_ms: 700,
            hard_close_ms: 2_000,
            ..Default::default()
        }
    }

    /// Interview/podcast — longer sentences, deliberate pauses.
    pub fn interview() -> Self {
        Self {
            sentence_close_ms: 1_000,
            hard_close_ms: 2_500,
            ..Default::default()
        }
    }

    /// Lecture/monologue — long unbroken speech with natural pauses.
    pub fn lecture() -> Self {
        Self {
            sentence_close_ms: 1_200,
            hard_close_ms: 3_000,
            ..Default::default()
        }
    }

    /// Dictation — speaker deliberately pausing between phrases.
    pub fn dictation() -> Self {
        Self {
            sentence_close_ms: 400,
            hard_close_ms: 1_200,
            ..Default::default()
        }
    }

    /// Adaptive mode — learns thresholds from the speaker's pause rhythm.
    /// Starts with given defaults, then adapts after warmup (~8 pauses).
    pub fn adaptive() -> Self {
        Self {
            adaptive: true,
            ..Default::default()
        }
    }
}

/// Three-state VAD segmenter that detects speech segment boundaries.
///
/// Does NOT buffer PCM — only tracks state transitions and emits
/// [`SegmentBoundary`] events. The caller is responsible for routing
/// audio to ASR based on [`is_speech()`].
pub struct VadSegmenter {
    vad: TenVad,
    state: VadState,
    silence_frames: usize,
    frame_index: u64,

    // Segment tracking (no PCM buffer — just metadata)
    segment_start_ms: u32,
    segment_prob_sum: f32,
    segment_frames: usize,

    config: VadSegmenterConfig,
    adaptive: Option<AdaptiveVadConfig>,

    // Precomputed constants (used as fallback when not adaptive)
    ms_per_frame: u32,
    sentence_close_frames: usize,
    hard_close_frames: usize,
    min_speech_frames: usize,
}

impl VadSegmenter {
    pub fn new(config: VadSegmenterConfig) -> Self {
        let adaptive = if config.adaptive {
            Some(AdaptiveVadConfig::new(
                config.sentence_close_ms,
                config.hard_close_ms,
            ))
        } else {
            None
        };
        Self::new_inner(config, adaptive)
    }

    /// Create a segmenter with pre-existing adaptive state (cross-session persistence).
    /// The adaptive config carries learned pause distributions from previous sessions.
    pub fn new_with_adaptive(config: VadSegmenterConfig, adaptive: AdaptiveVadConfig) -> Self {
        Self::new_inner(config, Some(adaptive))
    }

    fn new_inner(config: VadSegmenterConfig, adaptive: Option<AdaptiveVadConfig>) -> Self {
        let vad = TenVad::new_from_bytes(TEN_VAD_MODEL, SAMPLE_RATE)
            .expect("TEN VAD init failed — embedded model is corrupt");

        let ms_per_frame = (HOP_SIZE as f32 / SAMPLE_RATE as f32 * 1000.0) as u32;
        let sentence_close_frames =
            (config.sentence_close_ms / ms_per_frame.max(1)).max(1) as usize;
        let hard_close_frames = (config.hard_close_ms / ms_per_frame.max(1)).max(1) as usize;
        let min_speech_frames = (config.min_speech_ms / ms_per_frame.max(1)).max(1) as usize;

        Self {
            vad,
            state: VadState::Silence,
            silence_frames: 0,
            frame_index: 0,
            segment_start_ms: 0,
            segment_prob_sum: 0.0,
            segment_frames: 0,
            config,
            adaptive,
            ms_per_frame,
            sentence_close_frames,
            hard_close_frames,
            min_speech_frames,
        }
    }

    /// Extract the adaptive config for cross-session persistence.
    /// Returns `None` if adaptive mode is not enabled.
    pub fn take_adaptive(&mut self) -> Option<AdaptiveVadConfig> {
        self.adaptive.take()
    }

    /// Push f32 audio samples (16kHz mono).
    ///
    /// Returns `(boundaries, is_speech)`:
    /// - `boundaries`: completed segment boundaries (0 or more)
    /// - `is_speech`: whether audio should be routed to ASR after this push
    pub fn push(&mut self, samples: &[f32]) -> (SmallVec<[SegmentBoundary; 1]>, bool) {
        let mut boundaries: SmallVec<[SegmentBoundary; 1]> = SmallVec::new();

        for chunk in samples.chunks(HOP_SIZE) {
            // Convert f32 → i16 for TenVad
            let mut frame_i16 = [0i16; HOP_SIZE];
            for (i, &s) in chunk.iter().enumerate() {
                frame_i16[i] = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            }

            let prob = match self.vad.process_frame(&frame_i16) {
                Ok(p) => p,
                Err(e) => {
                    warn!(error = ?e, "TEN VAD frame error");
                    continue;
                }
            };

            let is_speech = prob >= self.config.speech_threshold;
            let frame_ms = self.frame_ms();

            if let Some(boundary) = self.advance_state(is_speech, prob, frame_ms) {
                boundaries.push(boundary);
            }

            self.frame_index += 1;

            // Force-flush if segment exceeds max duration
            if self.state == VadState::Speech {
                let duration = self.frame_ms().saturating_sub(self.segment_start_ms);
                if duration >= self.config.max_segment_ms
                    && let Some(b) = self.flush_segment(self.frame_ms())
                {
                    boundaries.push(b);
                }
            }
        }

        (boundaries, self.is_speech())
    }

    /// Flush any remaining buffered speech at end of stream.
    pub fn finish(&mut self) -> Option<SegmentBoundary> {
        if self.segment_frames >= self.min_speech_frames {
            self.flush_segment(self.frame_ms())
        } else {
            None
        }
    }

    /// Whether currently in an active segment (speech or trailing silence).
    ///
    /// Includes TrailingSilence because audio is processed in chunks (e.g. 1s).
    /// A chunk that transitions speech→silence mid-way must still be pushed to
    /// ASR — otherwise the speech portion at the start of the chunk is lost.
    pub fn is_speech(&self) -> bool {
        matches!(self.state, VadState::Speech | VadState::TrailingSilence)
    }

    fn frame_ms(&self) -> u32 {
        self.frame_index as u32 * self.ms_per_frame
    }

    fn advance_state(
        &mut self,
        is_speech: bool,
        prob: f32,
        frame_ms: u32,
    ) -> Option<SegmentBoundary> {
        match self.state {
            VadState::Silence => {
                if is_speech {
                    self.state = VadState::Speech;
                    self.silence_frames = 0;
                    self.segment_start_ms = frame_ms;
                    self.segment_prob_sum = prob;
                    self.segment_frames = 1;
                }
                None
            }

            VadState::Speech => {
                self.segment_prob_sum += prob;
                self.segment_frames += 1;

                if !is_speech {
                    self.state = VadState::TrailingSilence;
                    self.silence_frames = 1;
                }
                None
            }

            VadState::TrailingSilence => {
                self.segment_frames += 1;
                let silence_ms = self.silence_frames as u32 * self.ms_per_frame;

                if is_speech {
                    // Speech resumed — this pause was absorbed (within-sentence)
                    if let Some(ref mut adaptive) = self.adaptive {
                        adaptive.record_absorbed_pause(silence_ms);
                    }
                    self.state = VadState::Speech;
                    self.silence_frames = 0;
                    self.segment_prob_sum += prob;
                    None
                } else {
                    self.silence_frames += 1;

                    let (sc_frames, hc_frames) = self.current_threshold_frames();

                    if self.silence_frames >= hc_frames {
                        // Long silence — topic break, always flush
                        if let Some(ref mut adaptive) = self.adaptive {
                            adaptive.record_completed_pause(silence_ms);
                        }
                        self.flush_segment(frame_ms)
                    } else if self.silence_frames >= sc_frames
                        && self.segment_frames >= self.min_speech_frames
                    {
                        // Sentence boundary — flush if enough speech content
                        if let Some(ref mut adaptive) = self.adaptive {
                            adaptive.record_completed_pause(silence_ms);
                        }
                        self.flush_segment(frame_ms)
                    } else {
                        // Below sentence_close — absorb pause, keep segment open
                        None
                    }
                }
            }
        }
    }

    /// Current threshold frames — adaptive if enabled, static otherwise.
    fn current_threshold_frames(&self) -> (usize, usize) {
        if let Some(ref adaptive) = self.adaptive {
            let ms_per = self.ms_per_frame.max(1);
            let sc = (adaptive.sentence_close_ms() / ms_per).max(1) as usize;
            let hc = (adaptive.hard_close_ms() / ms_per).max(1) as usize;
            (sc, hc)
        } else {
            (self.sentence_close_frames, self.hard_close_frames)
        }
    }

    /// Current adaptive state for debugging. Returns None if not adaptive.
    pub fn debug_adaptive_state(&self) -> Option<String> {
        self.adaptive.as_ref().map(|a| a.debug_state())
    }

    /// Reset adaptive learning (e.g. on speaker change).
    pub fn reset_adaptive(&mut self) {
        if let Some(ref mut adaptive) = self.adaptive {
            adaptive.reset();
        }
    }

    fn flush_segment(&mut self, end_ms: u32) -> Option<SegmentBoundary> {
        let boundary = if self.segment_frames >= self.min_speech_frames {
            Some(SegmentBoundary {
                start_ms: self.segment_start_ms,
                end_ms,
                avg_prob: self.segment_prob_sum / self.segment_frames.max(1) as f32,
            })
        } else {
            None
        };

        self.segment_prob_sum = 0.0;
        self.segment_frames = 0;
        self.silence_frames = 0;
        self.state = VadState::Silence;

        boundary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_speech(duration_ms: u32) -> Vec<f32> {
        let samples = (duration_ms as usize * SAMPLE_RATE as usize) / 1000;
        (0..samples)
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / SAMPLE_RATE as f32).sin() * 0.8
            })
            .collect()
    }

    fn silence(duration_ms: u32) -> Vec<f32> {
        vec![0.0f32; (duration_ms as usize * SAMPLE_RATE as usize) / 1000]
    }

    #[test]
    fn pure_silence_no_boundaries() {
        let mut seg = VadSegmenter::new(VadSegmenterConfig::default());
        let (boundaries, is_speech) = seg.push(&silence(2000));
        assert!(boundaries.is_empty());
        assert!(!is_speech);
    }

    #[test]
    fn finish_on_empty_returns_none() {
        let mut seg = VadSegmenter::new(VadSegmenterConfig::default());
        assert!(seg.finish().is_none());
    }

    #[test]
    fn max_segment_forces_flush() {
        let mut seg = VadSegmenter::new(VadSegmenterConfig {
            max_segment_ms: 1000,
            sentence_close_ms: 500,
            hard_close_ms: 1000,
            min_speech_ms: 100,
            speech_threshold: 0.01, // very low so sine wave triggers
            ..Default::default()
        });

        // Push 2s of "speech" — should force-flush at 1s
        let (boundaries, _) = seg.push(&sine_speech(2000));
        assert!(
            !boundaries.is_empty(),
            "expected force-flush boundary at max_segment_ms"
        );
    }

    #[test]
    fn running_percentile_basic() {
        let mut rp = RunningPercentile::new(10);
        for v in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] {
            rp.push(v);
        }
        // P50 of [100..1000] = 500 (index 4 of 9)
        assert_eq!(rp.percentile(0.5), Some(500));
        // P95: index = floor(0.95 * 9) = 8 → 900
        assert_eq!(rp.percentile(0.95), Some(900));
    }

    #[test]
    fn running_percentile_fifo_eviction() {
        let mut rp = RunningPercentile::new(5);
        for v in [100, 200, 300, 400, 500] {
            rp.push(v);
        }
        assert_eq!(rp.percentile(0.5), Some(300));
        // Push 5 more — should evict [100,200,300,400,500]
        for v in [600, 700, 800, 900, 1000] {
            rp.push(v);
        }
        // Now contains [600, 700, 800, 900, 1000]
        assert_eq!(rp.percentile(0.5), Some(800));
        assert_eq!(rp.len(), 5);
    }

    #[test]
    fn adaptive_config_instant_adaptation() {
        let mut ac = AdaptiveVadConfig::new(600, 1200);
        // Zero warmup: first pause immediately triggers adaptation
        ac.record_absorbed_pause(200);
        // P95 of single value = 200, × 1.5 = 300 → clamped to floor 500
        // EMA(600, 500, 0.6) = 0.6*500 + 0.4*600 = 540
        let sc = ac.sentence_close_ms();
        assert!(sc < 600, "should adapt immediately: got {sc}");
        assert!(sc >= 500, "should respect floor: got {sc}");
    }

    #[test]
    fn adaptive_config_converges_after_warmup() {
        let mut ac = AdaptiveVadConfig::new(600, 1200);
        // Feed 10 absorbed pauses around 200ms
        for _ in 0..10 {
            ac.record_absorbed_pause(200);
        }
        // P95(200ms×10) = 200ms, × 1.5 = 300ms (clamped to floor 300ms)
        // EMA converges fast with alpha=0.6
        let sc = ac.sentence_close_ms();
        assert!(
            sc < 600,
            "sentence_close should decrease from initial: got {sc}"
        );
        assert!(
            sc >= 500,
            "sentence_close should not go below floor: got {sc}"
        );
    }

    #[test]
    fn adaptive_hard_close_stays_above_sentence() {
        let mut ac = AdaptiveVadConfig::new(600, 1200);
        for _ in 0..10 {
            ac.record_absorbed_pause(200);
        }
        // hard_close should always be > sentence_close + 200
        assert!(
            ac.hard_close_ms() >= ac.sentence_close_ms() + 200,
            "hard_close ({}) must be >= sentence_close ({}) + 200",
            ac.hard_close_ms(),
            ac.sentence_close_ms(),
        );
    }

    #[test]
    fn adaptive_segmenter_has_debug_state() {
        let seg = VadSegmenter::new(VadSegmenterConfig::adaptive());
        let state = seg.debug_adaptive_state();
        assert!(
            state.is_some(),
            "adaptive segmenter should have debug state"
        );
        assert!(state.unwrap().contains("warmup: done"));
    }

    #[test]
    fn adaptive_state_serde_roundtrip_empty() {
        let original = AdaptiveVadConfig::new(600, 1200);
        let json = serde_json::to_string(&original).unwrap();
        let restored: AdaptiveVadConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.sentence_close_ms(), original.sentence_close_ms());
        assert_eq!(restored.hard_close_ms(), original.hard_close_ms());
    }

    #[test]
    fn adaptive_state_serde_roundtrip_with_data() {
        let mut original = AdaptiveVadConfig::new(600, 1200);
        // Fill both buffers past warmup
        for i in 0..60 {
            original.record_absorbed_pause(200 + i * 5);
        }
        for i in 0..40 {
            original.record_completed_pause(800 + i * 10);
        }

        let json = serde_json::to_string(&original).unwrap();
        let restored: AdaptiveVadConfig = serde_json::from_str(&json).unwrap();

        // Thresholds must survive roundtrip exactly
        assert_eq!(restored.sentence_close_ms(), original.sentence_close_ms());
        assert_eq!(restored.hard_close_ms(), original.hard_close_ms());

        // New observations should continue from restored state, not reset
        let pre = restored.sentence_close_ms();
        let mut restored = restored;
        restored.record_absorbed_pause(100);
        // Adding a short pause should nudge threshold down (or stay same)
        assert!(restored.sentence_close_ms() <= pre + 50);
    }

    #[test]
    fn adaptive_state_serde_corrupt_json() {
        let result = serde_json::from_str::<AdaptiveVadConfig>("{ broken json }");
        assert!(result.is_err());
    }
}
