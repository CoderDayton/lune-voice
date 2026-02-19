use std::time::Instant;

use lune_core::asr::AsrSegment;

/// Tracks ASR segments with partial updates and finalization.
pub struct SegmentTracker {
    segments: Vec<AsrSegment>,
    next_id: u64,
    /// Time of last segment update — used for silence-based finalization.
    last_update: Instant,
    /// Silence duration (ms) after which a partial segment is finalized.
    finalize_after_ms: u64,
}

impl SegmentTracker {
    pub fn new(finalize_after_ms: u64) -> Self {
        Self {
            segments: Vec::new(),
            next_id: 1,
            last_update: Instant::now(),
            finalize_after_ms,
        }
    }

    /// Push new decoded text, updating the current partial segment or creating a new one.
    pub fn push_partial(&mut self, text: String, timestamp_ms: Option<u64>) {
        self.last_update = Instant::now();

        // Update existing partial segment or create new one.
        if let Some(seg) = self.segments.last_mut() {
            if !seg.is_final {
                seg.text = text;
                seg.end_ms = timestamp_ms;
                return;
            }
        }

        // Create new partial segment.
        let seg = AsrSegment {
            id: self.next_id,
            text,
            is_final: false,
            start_ms: timestamp_ms,
            end_ms: timestamp_ms,
        };
        self.next_id += 1;
        self.segments.push(seg);
    }

    /// Finalize the current partial segment (mark as immutable).
    pub fn finalize_current(&mut self) {
        if let Some(seg) = self.segments.last_mut() {
            if !seg.is_final {
                seg.is_final = true;
            }
        }
    }

    /// Check if the current partial should be finalized based on silence duration.
    pub fn check_silence_finalization(&mut self) {
        let elapsed = self.last_update.elapsed().as_millis() as u64;
        if elapsed >= self.finalize_after_ms {
            self.finalize_current();
        }
    }

    /// Finalize all segments (end of session).
    pub fn finalize_all(&mut self) {
        for seg in &mut self.segments {
            seg.is_final = true;
        }
    }

    /// Drain all segments that haven't been polled yet.
    ///
    /// Returns segments and clears the internal buffer.
    pub fn drain(&mut self) -> Vec<AsrSegment> {
        std::mem::take(&mut self.segments)
    }

    /// Get all current segments without draining.
    pub fn peek(&self) -> &[AsrSegment] {
        &self.segments
    }

    /// Number of segments tracked.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_partial_creates_segment() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hello".into(), Some(0));
        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.peek()[0].text, "hello");
        assert!(!tracker.peek()[0].is_final);
    }

    #[test]
    fn push_partial_updates_existing() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hel".into(), Some(0));
        tracker.push_partial("hello".into(), Some(100));
        // Should update, not create a new segment.
        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.peek()[0].text, "hello");
    }

    #[test]
    fn finalize_then_new_partial() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hello".into(), Some(0));
        tracker.finalize_current();
        assert!(tracker.peek()[0].is_final);

        tracker.push_partial("world".into(), Some(1000));
        assert_eq!(tracker.len(), 2);
        assert_eq!(tracker.peek()[1].text, "world");
        assert!(!tracker.peek()[1].is_final);
    }

    #[test]
    fn finalize_all() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hello".into(), Some(0));
        tracker.finalize_current();
        tracker.push_partial("world".into(), Some(1000));
        tracker.finalize_all();

        assert!(tracker.peek().iter().all(|s| s.is_final));
    }

    #[test]
    fn drain_clears_buffer() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hello".into(), Some(0));
        let drained = tracker.drain();
        assert_eq!(drained.len(), 1);
        assert!(tracker.is_empty());
    }

    #[test]
    fn segment_ids_increment() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("a".into(), Some(0));
        tracker.finalize_current();
        tracker.push_partial("b".into(), Some(100));

        assert_eq!(tracker.peek()[0].id, 1);
        assert_eq!(tracker.peek()[1].id, 2);
    }

    #[test]
    fn silence_finalization() {
        let mut tracker = SegmentTracker::new(0); // 0ms = immediate finalize
        tracker.push_partial("hello".into(), Some(0));
        std::thread::sleep(std::time::Duration::from_millis(1));
        tracker.check_silence_finalization();
        assert!(tracker.peek()[0].is_final);
    }
}
