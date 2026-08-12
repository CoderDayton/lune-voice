//! Lock-free SPSC ring buffer for audio chunks.
//!
//! `push` never blocks and never fails: when the buffer is full it evicts the
//! oldest chunk to make room. That *is* a drop — the newest audio is kept and
//! the oldest is lost — so it is counted and logged rather than passed off as a
//! guarantee. A consumer that keeps up sees none of this; a stalled one
//! silently loses the front of the utterance, which is exactly the failure a
//! counter has to make visible ([`RingBuffer::evicted`]).

use std::sync::atomic::{AtomicU64, Ordering};

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
    /// Chunks dropped to make room, for the lifetime of this buffer.
    evicted: AtomicU64,
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
        })
    }

    /// How many chunks have been evicted to make room for newer ones.
    ///
    /// Non-zero means the consumer fell behind and audio was lost from the
    /// front of the buffer — the transcript will be missing its opening.
    pub fn evicted(&self) -> u64 {
        self.evicted.load(Ordering::Relaxed)
    }

    /// Count a dropped chunk, logging the first and then every
    /// `EVICTION_LOG_INTERVAL`-th one.
    fn note_eviction(&self) {
        let n = self.evicted.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n % EVICTION_LOG_INTERVAL == 0 {
            tracing::warn!(
                evicted = n,
                "ring buffer full — dropping the oldest audio; the consumer is not keeping up"
            );
        }
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
        self.note_eviction();
        tracing::warn!(
            retries = MAX_EVICT_RETRIES,
            "ring buffer push dropped a chunk after repeated eviction races"
        );
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
        for i in 0..5 {
            ring.push(i);
        }
        assert_eq!(ring.evicted(), 4);
        assert_eq!(ring.try_pop(), Some(4), "only the newest survives");
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
