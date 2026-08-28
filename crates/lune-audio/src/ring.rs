//! Lock-free SPSC ring buffer for audio chunks.
//!
//! `push` never blocks and never fails: when the buffer is full it evicts the
//! oldest chunk to make room. That *is* a drop — the newest audio is kept and
//! the oldest is lost — so it is counted and logged rather than passed off as a
//! guarantee. A consumer that keeps up sees none of this; a stalled one
//! silently loses the front of the utterance, which is exactly the failure a
//! counter has to make visible ([`RingBuffer::evicted`]).
//!
//! Eviction only means data loss *while someone is reading*. The pre-warmed
//! capture stream keeps producing between dictations, and with no session
//! attached the ring is a rolling pre-roll window whose oldest chunk aging out
//! is the whole point. So evictions are counted and logged only while a
//! consumer is attached ([`RingBuffer::attach_consumer`]); otherwise an idle
//! app would warn forever and bury the stall this counter exists to catch.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use tokio::sync::Notify;

/// Log the first eviction, then one in every `EVICTION_LOG_INTERVAL`. A stalled
/// consumer would otherwise emit one line per audio chunk.
const EVICTION_LOG_INTERVAL: u64 = 100;

/// Single-producer, single-consumer ring buffer.
pub struct RingBuffer<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
    /// Wakes a waiting `pop().await` when a producer pushes. Replaces an
    /// exponential-backoff sleep loop that added up to 10 ms of latency
    /// on the first miss.
    notify: Notify,
    /// Chunks dropped to make room while a consumer was attached — i.e. audio
    /// a reader actually lost. Reset when the first consumer of a new run
    /// attaches, so it reads per-session rather than per-process.
    evicted: AtomicU64,
    /// Consumers currently draining the ring. Zero means the buffer is acting
    /// as an idle pre-roll window, where overwriting the oldest chunk is the
    /// design rather than a stall.
    consumers: AtomicUsize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("ring buffer capacity must be > 0");
        }
        let (tx, rx) = bounded(capacity);
        Ok(Self {
            tx,
            rx,
            notify: Notify::new(),
            evicted: AtomicU64::new(0),
            consumers: AtomicUsize::new(0),
        })
    }

    /// How many chunks have been evicted to make room for newer ones *while a
    /// consumer was attached*.
    ///
    /// Non-zero means the consumer fell behind and audio was lost from the
    /// front of the buffer — the transcript will be missing its opening.
    /// Evictions from the idle pre-roll window are not counted here.
    pub fn evicted(&self) -> u64 {
        self.evicted.load(Ordering::Relaxed)
    }

    /// Register a consumer. Evictions count and warn only while at least one is
    /// attached; the first of a run also zeroes [`Self::evicted`] so the count
    /// describes this run's loss and not the idle time before it.
    ///
    /// Pair with [`Self::detach_consumer`] — callers should prefer the RAII
    /// guard from `AudioStream::consume` over calling these directly.
    pub fn attach_consumer(&self) {
        if self.consumers.fetch_add(1, Ordering::AcqRel) == 0 {
            self.evicted.store(0, Ordering::Relaxed);
        }
    }

    /// Deregister a consumer attached by [`Self::attach_consumer`].
    pub fn detach_consumer(&self) {
        self.consumers.fetch_sub(1, Ordering::AcqRel);
    }

    /// Count a dropped chunk, logging the first and then every
    /// `EVICTION_LOG_INTERVAL`-th one. Returns whether the eviction counted as
    /// data loss — `false` when no consumer was attached to lose it.
    fn note_eviction(&self) -> bool {
        if self.consumers.load(Ordering::Acquire) == 0 {
            return false;
        }
        let n = self.evicted.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n % EVICTION_LOG_INTERVAL == 0 {
            tracing::warn!(
                evicted = n,
                "ring buffer full — dropping the oldest audio; the consumer is not keeping up"
            );
        }
        true
    }

    /// Push an item, evicting the oldest to make room when full.
    ///
    /// The incoming item always wins; what is lost is the front of the buffer.
    /// Every eviction is counted and periodically logged.
    pub fn push(&self, item: T) {
        let mut item = item;
        // Evict-and-retry until the send lands. Bounded rather than unbounded:
        // this runs on the cpal capture callback, where spinning is worse than
        // losing one chunk, and `handle()` documents that several consumers can
        // coexist — so a lost race is possible even though the producer is
        // single. Each attempt frees a slot, so exhausting the budget means we
        // lost the race MAX_EVICT_RETRIES times in a row.
        const MAX_EVICT_RETRIES: usize = 8;
        for _ in 0..MAX_EVICT_RETRIES {
            match self.tx.try_send(item) {
                Ok(()) => {
                    self.notify.notify_one();
                    return;
                }
                Err(TrySendError::Full(returned)) => {
                    item = returned;
                    if self.rx.try_recv().is_ok() {
                        self.note_eviction();
                    }
                }
                // Channel closed — consumer dropped. Nothing to do.
                Err(TrySendError::Disconnected(_)) => {
                    self.notify.notify_one();
                    return;
                }
            }
        }
        // The incoming chunk itself was lost, not just the oldest one.
        if self.note_eviction() {
            tracing::warn!(
                retries = MAX_EVICT_RETRIES,
                "ring buffer push dropped a chunk after repeated eviction races"
            );
        }
        // Wake one parked consumer (if any). `notify_one` is non-blocking
        // and safe to call from a non-async producer thread.
        self.notify.notify_one();
    }

    /// Pop an item (async, parks until a producer pushes).
    pub async fn pop(&self) -> Option<T> {
        loop {
            match self.rx.try_recv() {
                Ok(item) => return Some(item),
                // `notify_one` calls during the await window are coalesced
                // by `Notify`, so a push that races with the empty check
                // still wakes us instead of being lost.
                Err(TryRecvError::Empty) => self.notify.notified().await,
                Err(TryRecvError::Disconnected) => return None,
            }
        }
    }

    /// Try to pop without blocking.
    pub fn try_pop(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }

    /// Current buffer length.
    pub fn len(&self) -> usize {
        self.rx.len()
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    #[test]
    fn a_capacity_of_zero_is_refused() {
        assert!(RingBuffer::<u8>::new(0).is_err());
    }

    #[test]
    fn a_consumer_that_keeps_up_loses_nothing() {
        let ring = RingBuffer::new(4).expect("capacity > 0");
        for i in 0..4 {
            ring.push(i);
        }
        assert_eq!(ring.len(), 4);
        assert_eq!(ring.evicted(), 0);
        for i in 0..4 {
            assert_eq!(ring.try_pop(), Some(i));
        }
        assert_eq!(ring.try_pop(), None);
    }

    #[test]
    fn overflow_drops_the_oldest_and_counts_it() {
        // The stalled-consumer case. The newest audio survives, the front of
        // the utterance does not — and that has to be visible, because the only
        // other symptom is a transcript quietly missing its opening words.
        let ring = RingBuffer::new(2).expect("capacity > 0");
        ring.attach_consumer();
        ring.push(1);
        ring.push(2);
        ring.push(3);

        assert_eq!(ring.evicted(), 1, "one chunk was dropped to make room");
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.try_pop(), Some(2), "the oldest chunk is the one lost");
        assert_eq!(ring.try_pop(), Some(3));
    }

    #[test]
    fn the_eviction_count_accumulates() {
        let ring = RingBuffer::new(1).expect("capacity > 0");
        ring.attach_consumer();
        for i in 0..5 {
            ring.push(i);
        }
        assert_eq!(ring.evicted(), 4);
        assert_eq!(ring.try_pop(), Some(4), "only the newest survives");
    }

    #[test]
    fn idle_overflow_is_not_counted_as_loss() {
        // A pre-warmed capture stream keeps producing between dictations. With
        // nobody reading, the ring is a rolling pre-roll window: the oldest
        // chunk aging out is the design, so it must neither count nor warn —
        // otherwise an idle app logs forever and buries a real stall.
        let ring = RingBuffer::new(2).expect("capacity > 0");
        for i in 0..50 {
            ring.push(i);
        }
        assert_eq!(ring.evicted(), 0, "no consumer was there to lose anything");
        assert_eq!(ring.try_pop(), Some(48), "the newest chunks still survive");
    }

    #[test]
    fn attaching_rebaselines_the_count_after_idle() {
        // The count feeds a per-session "the transcript lost its opening"
        // diagnostic, so a session must not inherit the previous one's total.
        let ring = RingBuffer::new(1).expect("capacity > 0");
        ring.attach_consumer();
        ring.push(1);
        ring.push(2);
        assert_eq!(ring.evicted(), 1);

        ring.detach_consumer();
        for i in 0..20 {
            ring.push(i);
        }
        assert_eq!(ring.evicted(), 1, "idle pushes leave the count alone");

        ring.attach_consumer();
        assert_eq!(ring.evicted(), 0, "a new run starts from zero");
    }

    #[test]
    fn a_second_consumer_does_not_rebaseline() {
        let ring = RingBuffer::new(1).expect("capacity > 0");
        ring.attach_consumer();
        ring.push(1);
        ring.push(2);
        assert_eq!(ring.evicted(), 1);

        ring.attach_consumer();
        assert_eq!(ring.evicted(), 1, "only the first attach of a run resets");

        ring.detach_consumer();
        ring.push(3);
        assert_eq!(ring.evicted(), 2, "still counted while one consumer holds");
    }

    #[tokio::test]
    async fn pop_parks_until_a_push_arrives() {
        use std::sync::Arc;
        let ring = Arc::new(RingBuffer::new(4).expect("capacity > 0"));
        let producer = Arc::clone(&ring);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            producer.push(7);
        });
        assert_eq!(ring.pop().await, Some(7));
    }
}
