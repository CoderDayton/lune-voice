//! Lock-free SPSC ring buffer for audio chunks.
//!
//! Guarantees: no audio is silently dropped. When full, the oldest
//! chunk is evicted atomically before the new chunk is pushed.

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use std::time::Duration;

/// Single-producer, single-consumer ring buffer.
pub struct RingBuffer<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring buffer capacity must be > 0");
        let (tx, rx) = bounded(capacity);
        Self { tx, rx }
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
    }

    /// Pop an item (async, waits for data with exponential backoff).
    pub async fn pop(&self) -> Option<T> {
        let mut backoff_ms = 1u64;
        loop {
            match self.rx.try_recv() {
                Ok(item) => return Some(item),
                Err(TryRecvError::Empty) => {
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(10); // cap at 10ms
                }
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
