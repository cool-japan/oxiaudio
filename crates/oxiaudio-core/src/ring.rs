//! A bounded ring buffer for inter-stage audio buffering.
//!
//! [`AudioRingBuffer`] is a fixed-capacity FIFO of interleaved samples (or any
//! `Copy + Default` element). It is intended for producer/consumer hand-off
//! between pipeline stages (e.g. a decoder feeding a DSP chain feeding an output
//! callback).
//!
//! The implementation is safe Rust (`#![forbid(unsafe_code)]` is inherited from
//! the crate root): internal state is guarded by a [`std::sync::Mutex`], so the
//! structure is `Send + Sync` and correct under any number of producers and
//! consumers. For strict single-producer/single-consumer use it behaves as a
//! standard SPSC queue; the lock is uncontended in that case.
//!
//! Capacity is rounded **up** to the next power of two at construction. Reads and
//! writes are frame-oriented helpers built on top of single-element
//! [`push`](AudioRingBuffer::push) / [`pop`](AudioRingBuffer::pop).

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::OxiAudioError;

/// Overflow policy applied when a write would exceed remaining capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Return [`OxiAudioError::BufferOverflow`] without modifying the buffer.
    Error,
    /// Drop the oldest queued elements to make room for the new ones.
    OverwriteOldest,
    /// Drop the newest (incoming) elements that do not fit.
    DropNewest,
}

/// A bounded, thread-safe ring buffer of `Copy + Default` elements.
#[derive(Debug)]
pub struct AudioRingBuffer<T> {
    inner: Mutex<VecDeque<T>>,
    capacity: usize,
    policy: OverflowPolicy,
}

impl<T: Copy + Default> AudioRingBuffer<T> {
    /// Create a ring buffer whose capacity is rounded up to the next power of two
    /// (minimum 2). The default overflow policy is [`OverflowPolicy::Error`].
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1).next_power_of_two();
        Self {
            inner: Mutex::new(VecDeque::with_capacity(cap)),
            capacity: cap,
            policy: OverflowPolicy::Error,
        }
    }

    /// Set the overflow policy (builder style).
    pub fn with_policy(mut self, policy: OverflowPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the overflow policy (builder style — alias for [`with_policy`]).
    ///
    /// [`with_policy`]: Self::with_policy
    pub fn with_overflow_policy(self, policy: OverflowPolicy) -> Self {
        self.with_policy(policy)
    }

    /// Total capacity (power-of-two rounded).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of elements currently available to read.
    pub fn available_read(&self) -> usize {
        self.inner.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Number of elements that can be written before the buffer is full.
    pub fn available_write(&self) -> usize {
        self.inner
            .lock()
            .map(|q| self.capacity - q.len())
            .unwrap_or(0)
    }

    /// `true` if no elements are queued.
    pub fn is_empty(&self) -> bool {
        self.available_read() == 0
    }

    /// `true` if the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.available_read() >= self.capacity
    }

    /// Push a single element, honoring the configured [`OverflowPolicy`].
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::BufferOverflow`] only under
    /// [`OverflowPolicy::Error`] when the buffer is full.
    pub fn push(&self, value: T) -> Result<(), OxiAudioError> {
        let mut q = self
            .inner
            .lock()
            .map_err(|_| OxiAudioError::BufferOverflow("ring buffer mutex poisoned".into()))?;
        if q.len() >= self.capacity {
            match self.policy {
                OverflowPolicy::Error => {
                    return Err(OxiAudioError::BufferOverflow(format!(
                        "ring buffer full (capacity {})",
                        self.capacity
                    )));
                }
                OverflowPolicy::OverwriteOldest => {
                    q.pop_front();
                }
                OverflowPolicy::DropNewest => {
                    return Ok(());
                }
            }
        }
        q.push_back(value);
        Ok(())
    }

    /// Pop a single element, or `None` if empty.
    pub fn pop(&self) -> Option<T> {
        self.inner.lock().ok().and_then(|mut q| q.pop_front())
    }

    /// Write all of `frames` into the buffer, honoring the [`OverflowPolicy`].
    ///
    /// # Errors
    ///
    /// Under [`OverflowPolicy::Error`], returns [`OxiAudioError::BufferOverflow`]
    /// (without writing anything) if `frames` does not fit in the remaining space.
    pub fn write(&self, frames: &[T]) -> Result<usize, OxiAudioError> {
        let mut q = self
            .inner
            .lock()
            .map_err(|_| OxiAudioError::BufferOverflow("ring buffer mutex poisoned".into()))?;
        let free = self.capacity - q.len();
        match self.policy {
            OverflowPolicy::Error => {
                if frames.len() > free {
                    return Err(OxiAudioError::BufferOverflow(format!(
                        "write of {} elements exceeds {} free slots",
                        frames.len(),
                        free
                    )));
                }
                q.extend(frames.iter().copied());
                Ok(frames.len())
            }
            OverflowPolicy::OverwriteOldest => {
                for &v in frames {
                    if q.len() >= self.capacity {
                        q.pop_front();
                    }
                    q.push_back(v);
                }
                Ok(frames.len())
            }
            OverflowPolicy::DropNewest => {
                let n = frames.len().min(free);
                q.extend(frames[..n].iter().copied());
                Ok(n)
            }
        }
    }

    /// Read exactly `out.len()` elements into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::BufferUnderflow`] (without consuming anything) if
    /// fewer than `out.len()` elements are available.
    pub fn read_exact(&self, out: &mut [T]) -> Result<(), OxiAudioError> {
        let mut q = self
            .inner
            .lock()
            .map_err(|_| OxiAudioError::BufferUnderflow("ring buffer mutex poisoned".into()))?;
        if q.len() < out.len() {
            return Err(OxiAudioError::BufferUnderflow(format!(
                "read of {} elements but only {} available",
                out.len(),
                q.len()
            )));
        }
        for slot in out.iter_mut() {
            *slot = q.pop_front().unwrap_or_default();
        }
        Ok(())
    }

    /// Read up to `max` elements, returning however many were available.
    pub fn read(&self, max: usize) -> Vec<T> {
        let mut q = match self.inner.lock() {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };
        let n = max.min(q.len());
        q.drain(..n).collect()
    }

    /// Discard all queued elements.
    pub fn clear(&self) {
        if let Ok(mut q) = self.inner.lock() {
            q.clear();
        }
    }

    // ------------------------------------------------------------------
    // Frame-count API aliases (mirrors the task specification)
    // ------------------------------------------------------------------

    /// Number of elements currently available to read (alias for [`available_read`]).
    ///
    /// [`available_read`]: Self::available_read
    pub fn available_read_frames(&self) -> usize {
        self.available_read()
    }

    /// Number of elements that can be written (alias for [`available_write`]).
    ///
    /// [`available_write`]: Self::available_write
    pub fn available_write_frames(&self) -> usize {
        self.available_write()
    }

    /// Write exactly `frames` elements from `data`.
    ///
    /// Returns [`OxiAudioError::BufferOverflow`] (without writing anything) if
    /// `frames` exceeds available write space under [`OverflowPolicy::Error`].
    pub fn write_frames(&self, data: &[T], frames: usize) -> Result<(), OxiAudioError> {
        self.write(&data[..frames.min(data.len())]).map(|_| ())
    }

    /// Read exactly `frames` elements.
    ///
    /// Returns [`OxiAudioError::BufferUnderflow`] (without consuming anything) if
    /// fewer than `frames` elements are available.
    pub fn read_frames(&self, frames: usize) -> Result<Vec<T>, OxiAudioError> {
        let mut out = vec![T::default(); frames];
        self.read_exact(&mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_rounds_up_to_pow2() {
        assert_eq!(AudioRingBuffer::<f32>::new(100).capacity(), 128);
        assert_eq!(AudioRingBuffer::<f32>::new(256).capacity(), 256);
        assert_eq!(AudioRingBuffer::<f32>::new(0).capacity(), 1);
    }

    #[test]
    fn push_pop_fifo_order() {
        let rb = AudioRingBuffer::<i32>::new(4);
        rb.push(1).unwrap();
        rb.push(2).unwrap();
        rb.push(3).unwrap();
        assert_eq!(rb.available_read(), 3);
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn overflow_error_policy() {
        let rb = AudioRingBuffer::<f32>::new(2); // capacity 2
        rb.push(1.0).unwrap();
        rb.push(2.0).unwrap();
        assert!(rb.push(3.0).is_err());
        assert!(rb.is_full());
    }

    #[test]
    fn overwrite_oldest_policy() {
        let rb = AudioRingBuffer::<i32>::new(2).with_policy(OverflowPolicy::OverwriteOldest);
        rb.push(1).unwrap();
        rb.push(2).unwrap();
        rb.push(3).unwrap(); // evicts 1
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
    }

    #[test]
    fn drop_newest_policy() {
        let rb = AudioRingBuffer::<i32>::new(2).with_policy(OverflowPolicy::DropNewest);
        let written = rb.write(&[1, 2, 3, 4]).unwrap();
        assert_eq!(written, 2);
        assert_eq!(rb.read(10), vec![1, 2]);
    }

    #[test]
    fn write_read_exact_roundtrip() {
        let rb = AudioRingBuffer::<f32>::new(8);
        rb.write(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        let mut out = [0.0f32; 4];
        rb.read_exact(&mut out).unwrap();
        assert_eq!(out, [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn read_exact_underflow() {
        let rb = AudioRingBuffer::<f32>::new(8);
        rb.write(&[1.0, 2.0]).unwrap();
        let mut out = [0.0f32; 4];
        assert!(matches!(
            rb.read_exact(&mut out),
            Err(OxiAudioError::BufferUnderflow(_))
        ));
        // Nothing consumed on underflow.
        assert_eq!(rb.available_read(), 2);
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AudioRingBuffer<f32>>();
    }

    // -----------------------------------------------------------------------
    // Overflow policy tests using the with_overflow_policy API
    // -----------------------------------------------------------------------

    #[test]
    fn test_overflow_overwrite_oldest() {
        // Capacity rounds to 4 (next power of two ≥ 3).
        let rb =
            AudioRingBuffer::<i32>::new(3).with_overflow_policy(OverflowPolicy::OverwriteOldest);
        let cap = rb.capacity(); // 4
                                 // Fill to capacity.
        for i in 0..cap as i32 {
            rb.push(i).expect("push into non-full buffer must succeed");
        }
        assert!(rb.is_full(), "buffer must be full before overflow push");
        // Push one more — should evict the oldest (0).
        rb.push(99)
            .expect("OverwriteOldest must not return an error");
        // The oldest element was 0; after eviction the new head should be 1.
        let head = rb.pop().expect("buffer must have data after overflow push");
        assert_eq!(
            head, 1,
            "oldest element (0) should have been evicted, next is 1"
        );
        // Last element pushed (99) should be at the tail.
        let contents = rb.read(cap);
        assert!(
            contents.contains(&99),
            "new element 99 must be in the buffer"
        );
    }

    #[test]
    fn test_overflow_drop_newest() {
        let rb = AudioRingBuffer::<i32>::new(3).with_overflow_policy(OverflowPolicy::DropNewest);
        let cap = rb.capacity(); // 4
                                 // Fill to capacity.
        for i in 0..cap as i32 {
            rb.push(i).expect("push into non-full buffer must succeed");
        }
        assert!(rb.is_full());
        // Push one more — should be silently dropped.
        rb.push(99).expect("DropNewest must not return an error");
        // Buffer length must still equal capacity (99 was dropped).
        assert_eq!(rb.available_read(), cap, "buffer length must be unchanged");
        // The dropped element (99) must not appear.
        let contents = rb.read(cap);
        assert!(!contents.contains(&99), "99 must have been dropped");
        // Original elements must still be present in FIFO order.
        for (i, &v) in contents.iter().enumerate() {
            assert_eq!(v, i as i32, "element at position {i} should be {i}");
        }
    }
}
