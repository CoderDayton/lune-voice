//! Contiguous-span gate — stage 3 of the hybrid VAD.
//!
//! Pure and model-free: it consumes a stream of Silero speech probabilities
//! (one per [`super::silero::SILERO_FRAME`] frame) and emits span-boundary
//! events. The `HybridVad` orchestrator owns the audio buffers and turns these
//! events into ASR pushes; keeping the machine pure makes the
//! onset/offset/long-gap/min/max logic fully unit-testable without models.
//!
//! Hysteresis: `silero_onset > silero_offset` so brief score dips inside an
//! utterance don't end the span. Unlike a hangover-based chunker, a span closes
//! only on a `long_gap` run of trailing silence (an intentional stop) or the
//! max-length cap — short pauses are kept *inside* the span so the recognizer
//! still hears the real word boundaries.

use super::config::VadPipelineConfig;

/// One boundary decision per frame for the contiguous-span gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpanEvent {
    /// Not in a span and below onset — the frame belongs in the pre-roll ring.
    Idle,
    /// Onset this frame — drain pre-roll and open a span (append this frame).
    Open,
    /// In a span — append this frame (voiced OR sub-`long_gap` silence). This is
    /// the fix: intra-utterance pauses stay inside the span, so the word
    /// boundaries Qwen3-ASR reads from them are never spliced away.
    Append,
    /// A long gap (intentional stop) closed the span. `keep` is `false` for
    /// spans with less voiced audio than `min_speech` (noise blips).
    Close { keep: bool },
    /// The `max_chunk` length cap was hit. A memory FLUSH, not an endpoint: the
    /// orchestrator emits the audio accumulated so far but keeps building the
    /// *same* span — no trailing-silence trim, no pre-roll re-prepend on the
    /// continuation, no segment finalize. A length split therefore rejoins
    /// seamlessly and never injects a spurious silence boundary (which would
    /// bypass the intra-silence cap and read as a sentence break). The span only
    /// truly ends on `Close`.
    Flush,
}

/// Contiguous-span gate — the stage-3 replacement for the old chunked gate.
///
/// Where the old hangover-based chunker ended a segment after a short hangover
/// and let the
/// orchestrator drop the silence between segments, `SpanGate` keeps a span
/// open through short pauses and closes only on a `long_gap_frames` run of
/// trailing silence. The orchestrator appends every in-span frame (speech and
/// silence alike) and trims the closing tail to the hangover window, so the
/// real rhythm of an utterance survives to the recognizer.
pub(crate) struct SpanGate {
    onset: f32,
    offset: f32,
    long_gap_frames: usize,
    min_speech_frames: usize,
    max_frames: usize,
    in_span: bool,
    /// Voiced frames seen in the current span (drives the keep decision).
    voiced_frames: usize,
    /// Total frames in the current span (drives the max-length cap).
    span_frames: usize,
    /// Current trailing-silence run (resets on a voiced frame).
    silence_run: usize,
}

impl SpanGate {
    pub(crate) fn new(cfg: &VadPipelineConfig) -> Self {
        Self {
            onset: cfg.silero_onset,
            offset: cfg.silero_offset,
            // A 0-frame long-gap would close a span the instant any silence
            // appeared, collapsing back to the old splice behaviour.
            long_gap_frames: cfg.long_gap_frames().max(1),
            // A 0-frame min-speech would keep every span (incl. noise blips);
            // require at least one voiced frame, like the guards around it.
            min_speech_frames: cfg.min_speech_frames().max(1),
            max_frames: cfg.max_chunk_frames().max(1),
            in_span: false,
            voiced_frames: 0,
            span_frames: 0,
            silence_run: 0,
        }
    }

    /// True while a span is open (drives the Earshot-prefilter bypass).
    pub(crate) fn in_span(&self) -> bool {
        self.in_span
    }

    /// Advance one frame with its Silero speech probability.
    pub(crate) fn step(&mut self, prob: f32) -> SpanEvent {
        if !self.in_span {
            if prob >= self.onset {
                self.in_span = true;
                self.voiced_frames = 1;
                self.span_frames = 1;
                self.silence_run = 0;
                return SpanEvent::Open;
            }
            return SpanEvent::Idle;
        }

        self.span_frames += 1;
        if prob <= self.offset {
            self.silence_run += 1;
        } else {
            self.silence_run = 0;
            self.voiced_frames += 1;
        }

        // A long trailing-silence run is an intentional stop → close the span.
        if self.silence_run >= self.long_gap_frames {
            let keep = self.voiced_frames >= self.min_speech_frames;
            self.reset();
            return SpanEvent::Close { keep };
        }

        // Length cap reached. This is a memory FLUSH, not an endpoint: emit the
        // audio so far but keep the span open (reset only the per-flush frame
        // count) so the utterance rejoins seamlessly. A real endpoint is still
        // the long-gap run above or `finish`.
        if self.span_frames >= self.max_frames {
            self.span_frames = 0;
            return SpanEvent::Flush;
        }

        SpanEvent::Append
    }

    /// Close any open span at end of stream.
    pub(crate) fn finish(&mut self) -> SpanEvent {
        if !self.in_span {
            return SpanEvent::Idle;
        }
        let keep = self.voiced_frames >= self.min_speech_frames;
        self.reset();
        SpanEvent::Close { keep }
    }

    /// Drop all span state (back to `Idle`).
    pub(crate) fn reset(&mut self) {
        self.in_span = false;
        self.voiced_frames = 0;
        self.span_frames = 0;
        self.silence_run = 0;
    }
}

#[cfg(test)]
mod span_tests {
    use super::*;

    // 16 kHz, 512-sample frames → 1 frame = 32 ms.
    // long_gap 320 ms = 10 frames, min_speech 64 ms = 2 frames, max 3200 ms = 100 frames.
    fn test_gate() -> SpanGate {
        let cfg = VadPipelineConfig {
            sample_rate: 16_000,
            silero_onset: 0.6,
            silero_offset: 0.35,
            min_speech_ms: 64,
            long_gap_ms: 320,
            max_chunk_ms: 3_200,
            ..VadPipelineConfig::default()
        };
        SpanGate::new(&cfg)
    }

    #[test]
    fn stays_idle_below_onset() {
        let mut g = test_gate();
        assert_eq!(g.step(0.10), SpanEvent::Idle);
        assert_eq!(g.step(0.59), SpanEvent::Idle);
        assert!(!g.in_span());
    }

    #[test]
    fn onset_opens_span() {
        let mut g = test_gate();
        assert_eq!(g.step(0.60), SpanEvent::Open);
        assert!(g.in_span());
        assert_eq!(g.step(0.90), SpanEvent::Append);
    }

    /// THE fix: a pause shorter than `long_gap` keeps the span open, so the
    /// silence — and the word boundary it encodes — is preserved, not stripped.
    #[test]
    fn sub_long_gap_pause_keeps_span_open() {
        let mut g = test_gate();
        g.step(0.90); // Open, voiced 1
        g.step(0.90); // Append, voiced 2
        // 9 frames of silence — one short of the 10-frame long gap.
        for i in 0..9 {
            assert_eq!(g.step(0.10), SpanEvent::Append, "silence frame {i}");
        }
        assert!(g.in_span(), "span must survive a sub-long-gap pause");
        // Speech resumes — still the same span.
        assert_eq!(g.step(0.90), SpanEvent::Append);
    }

    #[test]
    fn long_gap_closes_span_and_keeps() {
        let mut g = test_gate();
        g.step(0.90); // Open, voiced 1
        g.step(0.90); // Append, voiced 2 (>= min)
        g.step(0.90); // Append, voiced 3
        // 10 silence frames: frames 1..=9 Append, the 10th hits the long gap.
        for _ in 0..9 {
            assert_eq!(g.step(0.10), SpanEvent::Append);
        }
        assert_eq!(g.step(0.10), SpanEvent::Close { keep: true });
        assert!(!g.in_span());
    }

    #[test]
    fn short_blip_dropped_at_long_gap() {
        let mut g = test_gate();
        assert_eq!(g.step(0.70), SpanEvent::Open); // voiced 1 < min 2
        for _ in 0..9 {
            g.step(0.10);
        }
        assert_eq!(g.step(0.10), SpanEvent::Close { keep: false });
    }

    /// A voiced frame mid-pause resets the long-gap timer — proving only a
    /// *continuous* long silence (an intentional stop) ends the span.
    #[test]
    fn voiced_frame_resets_long_gap_timer() {
        let mut g = test_gate();
        g.step(0.90); // Open
        for _ in 0..9 {
            g.step(0.10); // 9 silence (one short of the long gap)
        }
        assert_eq!(g.step(0.90), SpanEvent::Append); // voiced resets the run
        for _ in 0..9 {
            assert_eq!(g.step(0.10), SpanEvent::Append); // run restarts, still open
        }
        assert!(g.in_span());
    }

    /// The length cap FLUSHES (emits audio for memory) but keeps the span open,
    /// so a long utterance rejoins seamlessly instead of getting an artificial
    /// endpoint that would inject a spurious silence boundary.
    #[test]
    fn max_length_flushes_but_keeps_span_open() {
        let cfg = VadPipelineConfig {
            sample_rate: 16_000,
            silero_onset: 0.6,
            silero_offset: 0.35,
            min_speech_ms: 64,
            long_gap_ms: 10_000, // disabled for this test
            max_chunk_ms: 320,   // 10 frames
            ..VadPipelineConfig::default()
        };
        let mut g = SpanGate::new(&cfg);
        assert_eq!(g.step(0.90), SpanEvent::Open);
        for i in 2..10 {
            assert_eq!(g.step(0.90), SpanEvent::Append, "frame {i}");
        }
        // 10th frame hits the cap: a flush, not a close — the span stays open.
        assert_eq!(g.step(0.90), SpanEvent::Flush);
        assert!(g.in_span(), "a length flush must not end the span");
        // Accumulation continues; another `max_frames` later flushes again.
        for i in 1..10 {
            assert_eq!(g.step(0.90), SpanEvent::Append, "post-flush frame {i}");
        }
        assert_eq!(g.step(0.90), SpanEvent::Flush);
        assert!(g.in_span());
    }

    /// A real endpoint still fires after a flush: the long-gap silence run is
    /// tracked across the flush boundary (only the per-flush frame count resets).
    #[test]
    fn long_gap_after_flush_still_closes() {
        let cfg = VadPipelineConfig {
            sample_rate: 16_000,
            silero_onset: 0.6,
            silero_offset: 0.35,
            min_speech_ms: 64,
            long_gap_ms: 320,  // 10 frames
            max_chunk_ms: 320, // 10 frames
            ..VadPipelineConfig::default()
        };
        let mut g = SpanGate::new(&cfg);
        assert_eq!(g.step(0.90), SpanEvent::Open);
        for _ in 2..10 {
            g.step(0.90);
        }
        assert_eq!(g.step(0.90), SpanEvent::Flush); // 10th voiced frame → flush
        // Now a long-gap run of silence: the 10th silence frame closes for real.
        for _ in 0..9 {
            assert_eq!(g.step(0.10), SpanEvent::Append);
        }
        assert_eq!(g.step(0.10), SpanEvent::Close { keep: true });
        assert!(!g.in_span());
    }

    /// A new onset on the frame right after a close opens a fresh span with
    /// clean counters — no voiced/silence state leaks across the boundary.
    #[test]
    fn onset_right_after_close_opens_fresh_span() {
        let mut g = test_gate();
        g.step(0.90); // Open, voiced 1
        g.step(0.90); // Append, voiced 2 (>= min)
        for _ in 0..9 {
            g.step(0.10); // 9 silence frames -> Append
        }
        assert_eq!(g.step(0.10), SpanEvent::Close { keep: true }); // 10th: long gap
        // Immediately onset again: a brand-new span, not a resumption.
        assert_eq!(g.step(0.90), SpanEvent::Open);
        assert!(g.in_span());
        // It must drop as a short blip (voiced 1 < min 2), proving voiced_frames
        // reset to 1 rather than carrying over the prior span's count.
        assert_eq!(g.finish(), SpanEvent::Close { keep: false });
    }

    #[test]
    fn finish_closes_open_span() {
        let mut g = test_gate();
        g.step(0.90); // Open, voiced 1
        g.step(0.90); // Append, voiced 2
        assert_eq!(g.finish(), SpanEvent::Close { keep: true });
        assert!(!g.in_span());
        assert_eq!(g.finish(), SpanEvent::Idle); // idempotent
    }

    #[test]
    fn finish_when_idle_is_idle() {
        let mut g = test_gate();
        assert_eq!(g.finish(), SpanEvent::Idle);
    }

    #[test]
    fn finish_drops_short_span() {
        let mut g = test_gate();
        g.step(0.90); // Open, voiced 1 < min 2
        assert_eq!(g.finish(), SpanEvent::Close { keep: false });
    }
}
