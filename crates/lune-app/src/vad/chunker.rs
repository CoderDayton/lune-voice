//! Segment state machine — stage 3 of the hybrid gate.
//!
//! Pure and model-free: it consumes a stream of Silero speech probabilities
//! (one per [`super::silero::SILERO_FRAME`] frame) and emits segment-boundary
//! events. The `HybridVad` orchestrator owns the actual audio buffers and
//! turns these events into ASR pushes; keeping the machine pure makes the
//! onset/offset/hangover/min/max logic fully unit-testable without models.
//!
//! Hysteresis: `silero_onset > silero_offset` so brief score dips inside an
//! utterance don't end the segment. A segment finalizes when trailing silence
//! exceeds the hangover window, or when it reaches the max-length cap.

use super::config::VadPipelineConfig;

/// One boundary decision per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkerEvent {
    /// No speech; the frame belongs in the pre-roll ring.
    Idle,
    /// Speech just started this frame — flush pre-roll and open a segment.
    Begin,
    /// Speech continues (or sits inside the hangover window) — append the frame.
    Continue,
    /// Segment closed. `keep` is `false` for sub-`min_chunk` noise blips.
    Finalize { keep: bool },
}

/// Frame-clocked onset/offset/hangover state machine.
pub(crate) struct Chunker {
    onset: f32,
    offset: f32,
    hangover_frames: usize,
    min_speech_frames: usize,
    max_frames: usize,
    in_speech: bool,
    /// Frames since the segment opened (includes trailing silence).
    segment_frames: usize,
    /// Length of the current trailing-silence run (resets on a voiced frame).
    silence_frames: usize,
}

impl Chunker {
    pub(crate) fn new(cfg: &VadPipelineConfig) -> Self {
        Self {
            onset: cfg.silero_onset,
            offset: cfg.silero_offset,
            // Guard against degenerate sweep values: a 0-frame hangover would
            // finalize a segment the instant it opened.
            hangover_frames: cfg.hangover_frames().max(1),
            min_speech_frames: cfg.min_chunk_frames(),
            max_frames: cfg.max_chunk_frames().max(1),
            in_speech: false,
            segment_frames: 0,
            silence_frames: 0,
        }
    }

    /// True while a segment is open (drives the Earshot-prefilter bypass).
    pub(crate) fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Advance one frame with its Silero speech probability.
    pub(crate) fn step(&mut self, prob: f32) -> ChunkerEvent {
        if !self.in_speech {
            if prob >= self.onset {
                self.in_speech = true;
                self.segment_frames = 1;
                self.silence_frames = 0;
                return ChunkerEvent::Begin;
            }
            return ChunkerEvent::Idle;
        }

        self.segment_frames += 1;
        if prob <= self.offset {
            self.silence_frames += 1;
        } else {
            self.silence_frames = 0;
        }

        // Trailing silence ran past the hangover window → close the segment.
        if self.silence_frames >= self.hangover_frames {
            let voiced = self.segment_frames - self.silence_frames;
            let keep = voiced >= self.min_speech_frames;
            self.reset();
            return ChunkerEvent::Finalize { keep };
        }

        // Hard length cap → force-finalize ongoing speech (always kept).
        if self.segment_frames >= self.max_frames {
            self.reset();
            return ChunkerEvent::Finalize { keep: true };
        }

        ChunkerEvent::Continue
    }

    /// Close any in-progress segment at end of stream.
    pub(crate) fn finish(&mut self) -> ChunkerEvent {
        if !self.in_speech {
            return ChunkerEvent::Idle;
        }
        let voiced = self.segment_frames - self.silence_frames;
        let keep = voiced >= self.min_speech_frames;
        self.reset();
        ChunkerEvent::Finalize { keep }
    }

    /// Drop all segment state (back to `Idle`).
    pub(crate) fn reset(&mut self) {
        self.in_speech = false;
        self.segment_frames = 0;
        self.silence_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 16 kHz, 512-sample frames → 1 frame = 32 ms.
    // hangover 96 ms = 3 frames, min 64 ms = 2 frames, max 320 ms = 10 frames.
    fn test_chunker() -> Chunker {
        let cfg = VadPipelineConfig {
            sample_rate: 16_000,
            silero_onset: 0.6,
            silero_offset: 0.35,
            hangover_ms: 96,
            min_chunk_ms: 64,
            max_chunk_ms: 320,
            ..VadPipelineConfig::default()
        };
        Chunker::new(&cfg)
    }

    #[test]
    fn stays_idle_below_onset() {
        let mut c = test_chunker();
        assert_eq!(c.step(0.10), ChunkerEvent::Idle);
        assert_eq!(c.step(0.59), ChunkerEvent::Idle);
        assert!(!c.in_speech());
    }

    #[test]
    fn onset_begins_segment() {
        let mut c = test_chunker();
        assert_eq!(c.step(0.60), ChunkerEvent::Begin);
        assert!(c.in_speech());
        assert_eq!(c.step(0.90), ChunkerEvent::Continue);
    }

    #[test]
    fn hangover_then_finalize_keep() {
        let mut c = test_chunker();
        // 3 voiced frames (≥ min 2), then a full hangover of silence.
        assert_eq!(c.step(0.90), ChunkerEvent::Begin);
        assert_eq!(c.step(0.90), ChunkerEvent::Continue);
        assert_eq!(c.step(0.90), ChunkerEvent::Continue);
        assert_eq!(c.step(0.10), ChunkerEvent::Continue); // silence 1
        assert_eq!(c.step(0.10), ChunkerEvent::Continue); // silence 2
        assert_eq!(c.step(0.10), ChunkerEvent::Finalize { keep: true }); // silence 3 == hangover
        assert!(!c.in_speech());
    }

    #[test]
    fn short_blip_dropped() {
        let mut c = test_chunker();
        // One voiced frame → voiced span 1 < min 2 → dropped as noise.
        assert_eq!(c.step(0.70), ChunkerEvent::Begin); // voiced 1
        assert_eq!(c.step(0.10), ChunkerEvent::Continue); // silence 1
        assert_eq!(c.step(0.10), ChunkerEvent::Continue); // silence 2
        assert_eq!(c.step(0.10), ChunkerEvent::Finalize { keep: false }); // silence 3
    }

    #[test]
    fn voiced_frame_resets_hangover() {
        let mut c = test_chunker();
        c.step(0.90); // Begin
        c.step(0.10); // silence 1
        c.step(0.10); // silence 2
        assert_eq!(c.step(0.90), ChunkerEvent::Continue); // resets the run
        c.step(0.10); // silence 1
        c.step(0.10); // silence 2
        assert_eq!(c.step(0.10), ChunkerEvent::Finalize { keep: true }); // silence 3
    }

    #[test]
    fn max_length_force_finalizes() {
        let mut c = test_chunker();
        // max = 10 frames: frame 1 Begin, frames 2..=9 Continue, frame 10 closes.
        assert_eq!(c.step(0.90), ChunkerEvent::Begin);
        for i in 2..10 {
            assert_eq!(c.step(0.90), ChunkerEvent::Continue, "frame {i}");
        }
        assert_eq!(c.step(0.90), ChunkerEvent::Finalize { keep: true });
        assert!(!c.in_speech());
    }

    #[test]
    fn finish_finalizes_in_progress() {
        let mut c = test_chunker();
        c.step(0.90); // Begin, voiced 1
        c.step(0.90); // Continue, voiced 2 (≥ min)
        assert_eq!(c.finish(), ChunkerEvent::Finalize { keep: true });
        assert!(!c.in_speech());
        // Idempotent — nothing is open now.
        assert_eq!(c.finish(), ChunkerEvent::Idle);
    }

    #[test]
    fn finish_when_idle_is_idle() {
        let mut c = test_chunker();
        assert_eq!(c.finish(), ChunkerEvent::Idle);
    }

    #[test]
    fn finish_drops_short_segment() {
        let mut c = test_chunker();
        c.step(0.90); // Begin, voiced 1 < min 2
        assert_eq!(c.finish(), ChunkerEvent::Finalize { keep: false });
    }
}
