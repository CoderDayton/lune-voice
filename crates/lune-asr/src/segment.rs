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
    #[inline]
    pub fn new(finalize_after_ms: u64) -> Self {
        Self {
            // Pre-allocate for typical utterance burst (avoids realloc on first ~8 segments).
            segments: Vec::with_capacity(8),
            next_id: 1,
            last_update: Instant::now(),
            finalize_after_ms,
        }
    }

    /// Push new decoded text, updating the current partial segment or creating a new one.
    #[inline]
    pub fn push_partial(
        &mut self,
        text: String,
        timestamp_ms: Option<u64>,
        confidence: Option<f32>,
    ) {
        self.last_update = Instant::now();

        // Update existing partial segment or create new one.
        // SAFETY: We immediately return after the mutation, so no aliasing issue.
        if let Some(seg) = self.segments.last_mut()
            && !seg.is_final
        {
            seg.text = text;
            seg.end_ms = timestamp_ms;
            seg.confidence = confidence;
            return;
        }

        let id = self.next_id;
        self.next_id += 1;
        // Push constructed in-place to avoid intermediate stack copy.
        self.segments.push(AsrSegment {
            id,
            text,
            is_final: false,
            start_ms: timestamp_ms,
            end_ms: timestamp_ms,
            confidence,
        });
    }

    /// Finalize the current partial segment (mark as immutable).
    #[inline]
    pub fn finalize_current(&mut self) {
        if let Some(seg) = self.segments.last_mut()
            && !seg.is_final
        {
            seg.is_final = true;
        }
    }

    /// Check if the current partial should be finalized based on silence duration.
    ///
    /// Fast-path: returns immediately when there is no partial or silence threshold is zero.
    #[inline]
    pub fn check_silence_finalization(&mut self) {
        // Fast-path: nothing to finalize.
        let Some(seg) = self.segments.last_mut() else {
            return;
        };
        if seg.is_final {
            return;
        }

        // Avoid elapsed() syscall when finalize_after_ms == 0.
        if self.finalize_after_ms == 0
            || self.last_update.elapsed().as_millis() as u64 >= self.finalize_after_ms
        {
            seg.is_final = true;
        }
    }

    /// Finalize all segments (end of session).
    #[inline]
    pub fn finalize_all(&mut self) {
        // Tight loop — avoid branch per element by only writing when needed.
        for seg in &mut self.segments {
            // Branchless write is fine here; `is_final` is always set to true.
            seg.is_final = true;
        }
    }

    /// Drain all segments that haven't been polled yet.
    ///
    /// Returns segments and clears the internal buffer while preserving allocation.
    #[inline]
    pub fn drain(&mut self) -> Vec<AsrSegment> {
        // Take the vec (O(1)) and replace with a fresh pre-allocated one,
        // so the next burst doesn't pay for reallocation.
        std::mem::replace(&mut self.segments, Vec::with_capacity(8))
    }

    /// Get all current segments without draining.
    #[cfg(test)]
    #[inline]
    pub fn peek(&self) -> &[AsrSegment] {
        &self.segments
    }

    /// Number of segments tracked.
    #[inline]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    #[inline]
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
        tracker.push_partial("hello".into(), Some(0), None);
        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.peek()[0].text, "hello");
        assert!(!tracker.peek()[0].is_final);
    }

    #[test]
    fn push_partial_updates_existing() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hel".into(), Some(0), None);
        tracker.push_partial("hello".into(), Some(100), Some(0.9));
        // Should update, not create a new segment.
        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.peek()[0].text, "hello");
        assert_eq!(tracker.peek()[0].confidence, Some(0.9));
    }

    #[test]
    fn finalize_then_new_partial() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hello".into(), Some(0), None);
        tracker.finalize_current();
        assert!(tracker.peek()[0].is_final);

        tracker.push_partial("world".into(), Some(1000), None);
        assert_eq!(tracker.len(), 2);
        assert_eq!(tracker.peek()[1].text, "world");
        assert!(!tracker.peek()[1].is_final);
    }

    #[test]
    fn finalize_all() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hello".into(), Some(0), None);
        tracker.finalize_current();
        tracker.push_partial("world".into(), Some(1000), None);
        tracker.finalize_all();

        assert!(tracker.peek().iter().all(|s| s.is_final));
    }

    #[test]
    fn drain_clears_buffer() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("hello".into(), Some(0), None);
        let drained = tracker.drain();
        assert_eq!(drained.len(), 1);
        assert!(tracker.is_empty());
    }

    #[test]
    fn drain_preserves_capacity() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("a".into(), Some(0), None);
        let _ = tracker.drain();
        // After drain, internal vec should still have capacity reserved.
        assert!(tracker.segments.capacity() >= 8);
    }

    #[test]
    fn segment_ids_increment() {
        let mut tracker = SegmentTracker::new(500);
        tracker.push_partial("a".into(), Some(0), None);
        tracker.finalize_current();
        tracker.push_partial("b".into(), Some(100), None);

        assert_eq!(tracker.peek()[0].id, 1);
        assert_eq!(tracker.peek()[1].id, 2);
    }

    #[test]
    fn silence_finalization() {
        let mut tracker = SegmentTracker::new(0); // 0ms = immediate finalize
        tracker.push_partial("hello".into(), Some(0), None);
        std::thread::sleep(std::time::Duration::from_millis(1));
        tracker.check_silence_finalization();
        assert!(tracker.peek()[0].is_final);
    }

    #[test]
    fn silence_finalization_noop_when_empty() {
        let mut tracker = SegmentTracker::new(0);
        // Should not panic on empty tracker.
        tracker.check_silence_finalization();
        assert!(tracker.is_empty());
    }

    #[test]
    fn silence_finalization_noop_when_already_final() {
        let mut tracker = SegmentTracker::new(0);
        tracker.push_partial("hello".into(), Some(0), None);
        tracker.finalize_current();
        // Already final — should not panic or double-set.
        tracker.check_silence_finalization();
        assert_eq!(tracker.len(), 1);
        assert!(tracker.peek()[0].is_final);
    }
}
