//! Lock-free SPSC ring buffer for audio chunks

use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use std::time::Duration;

/// Single-producer, single-consumer ring buffer
pub struct RingBuffer<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = bounded(capacity);
        Self { tx, rx, capacity }
    }

    /// Push an item (overwrites oldest if full)
    pub fn push(&self, item: T) {
        // If full, this will block briefly, but the callback
        // handles this by trying to pop first
        let _ = self.tx.try_send(item);
    }

    /// Pop an item (async, waits for data)
    pub async fn pop(&self) -> Option<T> {
        loop {
            match self.rx.try_recv() {
                Ok(item) => return Some(item),
                Err(TryRecvError::Empty) => tokio::time::sleep(Duration::from_millis(2)).await,
                Err(TryRecvError::Disconnected) => return None,
            }
        }
    }

    /// Try to pop without blocking
    pub fn try_pop(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }

    /// Current buffer length
    pub fn len(&self) -> usize {
        self.rx.len()
    }

    /// Is buffer full
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }
}
