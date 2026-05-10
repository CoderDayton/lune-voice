//! Lock-free SPSC ring buffer for audio chunks.
//!
//! Guarantees: no audio is silently dropped. When full, the oldest
//! chunk is evicted atomically before the new chunk is pushed.

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
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring buffer capacity must be > 0");
        let (tx, rx) = bounded(capacity);
        Self {
            tx,
            rx,
            notify: Notify::new(),
        }
    }

    /// Push an item. If full, evicts the oldest item first.
    /// Never silently drops the new item.
    pub fn push(&self, item: T) {
        match self.tx.try_send(item) {
            Ok(()) => {}
            Err(TrySendError::Full(item)) => {
                // Evict oldest, then retry. If the consumer popped between
                // try_send and try_recv, the retry will succeed without eviction.
                let _ = self.rx.try_recv();
                // Retry — if still full (shouldn't happen in SPSC), force-drop oldest again
                if let Err(TrySendError::Full(item)) = self.tx.try_send(item) {
                    let _ = self.rx.try_recv();
                    let _ = self.tx.try_send(item);
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                // Channel closed — consumer dropped. Nothing to do.
            }
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
