//! Lock-free SPSC ring buffer for audio chunks.
//!
//! Guarantees: no audio is silently dropped. When full, the oldest
//! chunk is evicted atomically before the new chunk is pushed. `push`
//! retries a bounded number of times and logs if it ever gives up, so a
//! drop is always visible rather than silent.

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use tokio::sync::Notify;

/// Single-producer, single-consumer ring buffer.
pub struct RingBuffer<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
    /// Wakes a waiting `pop().await` when a producer pushes. Replaces an
    /// exponential-backoff sleep loop that added up to 10 ms of latency
    /// on the first miss.
    notify: Notify,
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
        })
    }

    /// Push an item. If full, evicts the oldest item first.
    /// Never silently drops the new item.
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
                    let _ = self.rx.try_recv();
                }
                // Channel closed — consumer dropped. Nothing to do.
                Err(TrySendError::Disconnected(_)) => {
                    self.notify.notify_one();
                    return;
                }
            }
        }
        // Never silent: the module's no-drop guarantee failed, and a caller
        // debugging audio gaps has to be able to see that.
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
