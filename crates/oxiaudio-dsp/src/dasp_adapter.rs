//! dasp `Signal` adapters for `AudioBuffer<f32>`.
//!
//! This module is gated by the `dasp` feature of `oxiaudio-dsp`.  It exposes two iterator-like
//! adapter types that wrap an `AudioBuffer<f32>` and implement the [`dasp_signal::Signal`] trait:
//!
//! - [`MonoSignal`] — yields `f32` frames (dasp mono, `[f32; 1]`-compatible).
//! - [`StereoSignal`] — yields `[f32; 2]` stereo frames.
//!
//! Both adapters are _exhaustible_: `Signal::is_exhausted()` returns `true` once all frames in the
//! buffer have been consumed, and subsequent `next()` calls return the frame equilibrium value
//! (`[0.0; N]`).
//!
//! # Example — using `MonoSignal` with dasp envelope detection
//!
//! ```rust,ignore
//! use oxiaudio_dsp::dasp_adapter::MonoSignal;
//! use dasp_signal::Signal;
//!
//! let buf = /* AudioBuffer<f32> with Mono channel layout */;
//! let mut sig = MonoSignal::new(&buf);
//!
//! while !sig.is_exhausted() {
//!     let frame: [f32; 1] = sig.next();
//!     // … process frame …
//! }
//! ```
//!
//! # Example — using `StereoSignal` with dasp operators
//!
//! ```rust,ignore
//! use oxiaudio_dsp::dasp_adapter::StereoSignal;
//! use dasp_signal::Signal;
//!
//! let buf = /* AudioBuffer<f32> with Stereo channel layout */;
//! let mut sig = StereoSignal::new(&buf);
//!
//! // Collect all frames into a Vec.
//! let frames: Vec<[f32; 2]> = sig.until_exhausted().collect();
//! ```

use dasp_signal::Signal;
use oxiaudio_core::{AudioBuffer, ChannelLayout};

// ─── MonoSignal ──────────────────────────────────────────────────────────────

/// A dasp [`Signal`] adapter over an `AudioBuffer<f32>` with a mono channel layout.
///
/// Each call to `next()` returns the next `[f32; 1]` frame from the underlying
/// interleaved sample slice.  When all frames are consumed, `is_exhausted()` returns
/// `true` and further calls to `next()` yield `[0.0]`.
pub struct MonoSignal {
    samples: Vec<f32>,
    pos: usize,
}

impl MonoSignal {
    /// Create a new `MonoSignal` from a mono `AudioBuffer<f32>`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `buf.channels` is not `ChannelLayout::Mono`.
    #[must_use]
    pub fn new(buf: &AudioBuffer<f32>) -> Self {
        debug_assert!(
            buf.channels == ChannelLayout::Mono,
            "MonoSignal expects a Mono AudioBuffer, got {:?}",
            buf.channels
        );
        Self {
            samples: buf.samples.clone(),
            pos: 0,
        }
    }

    /// Returns the number of frames remaining in the signal.
    #[must_use]
    pub fn remaining_frames(&self) -> usize {
        self.samples.len().saturating_sub(self.pos)
    }

    /// Returns the total number of frames in the underlying buffer.
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.samples.len()
    }
}

impl Signal for MonoSignal {
    type Frame = [f32; 1];

    fn next(&mut self) -> Self::Frame {
        if self.pos < self.samples.len() {
            let s = self.samples[self.pos];
            self.pos += 1;
            [s]
        } else {
            [0.0_f32]
        }
    }

    fn is_exhausted(&self) -> bool {
        self.pos >= self.samples.len()
    }
}

// ─── StereoSignal ────────────────────────────────────────────────────────────

/// A dasp [`Signal`] adapter over an `AudioBuffer<f32>` with a stereo channel layout.
///
/// The underlying sample slice must be interleaved (L, R, L, R, …).  Each call to
/// `next()` returns the next `[f32; 2]` frame.  When all frames are consumed,
/// `is_exhausted()` returns `true` and further calls yield `[0.0, 0.0]`.
pub struct StereoSignal {
    samples: Vec<f32>,
    /// Byte position measured in *samples* (not frames).
    pos: usize,
}

impl StereoSignal {
    /// Create a new `StereoSignal` from a stereo `AudioBuffer<f32>`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `buf.channels` is not `ChannelLayout::Stereo`.
    #[must_use]
    pub fn new(buf: &AudioBuffer<f32>) -> Self {
        debug_assert!(
            buf.channels == ChannelLayout::Stereo,
            "StereoSignal expects a Stereo AudioBuffer, got {:?}",
            buf.channels
        );
        debug_assert!(
            buf.samples.len() % 2 == 0,
            "Stereo AudioBuffer sample count must be even, got {}",
            buf.samples.len()
        );
        Self {
            samples: buf.samples.clone(),
            pos: 0,
        }
    }

    /// Returns the number of stereo frames remaining in the signal.
    #[must_use]
    pub fn remaining_frames(&self) -> usize {
        self.samples.len().saturating_sub(self.pos) / 2
    }

    /// Returns the total number of stereo frames in the underlying buffer.
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.samples.len() / 2
    }
}

impl Signal for StereoSignal {
    type Frame = [f32; 2];

    fn next(&mut self) -> Self::Frame {
        if self.pos + 1 < self.samples.len() {
            let l = self.samples[self.pos];
            let r = self.samples[self.pos + 1];
            self.pos += 2;
            [l, r]
        } else {
            [0.0_f32, 0.0_f32]
        }
    }

    fn is_exhausted(&self) -> bool {
        self.pos >= self.samples.len()
    }
}

// ─── Conversion helpers ───────────────────────────────────────────────────────

/// Collect all frames from a `MonoSignal` back into an `AudioBuffer<f32>`.
///
/// Drains the signal until `is_exhausted()` is true.
#[must_use]
pub fn mono_signal_to_buffer(sig: MonoSignal, sample_rate: u32) -> AudioBuffer<f32> {
    use oxiaudio_core::SampleFormat;
    let samples: Vec<f32> = sig.until_exhausted().map(|frame| frame[0]).collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Collect all frames from a `StereoSignal` back into an `AudioBuffer<f32>`.
///
/// Drains the signal until `is_exhausted()` is true.
#[must_use]
pub fn stereo_signal_to_buffer(sig: StereoSignal, sample_rate: u32) -> AudioBuffer<f32> {
    use oxiaudio_core::SampleFormat;
    let frames: Vec<[f32; 2]> = sig.until_exhausted().collect();
    let mut samples = Vec::with_capacity(frames.len() * 2);
    for frame in frames {
        samples.push(frame[0]);
        samples.push(frame[1]);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dasp_signal::Signal;
    use oxiaudio_core::{ChannelLayout, SampleFormat};

    fn mono_buf(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn stereo_buf(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_mono_signal_yields_correct_frames() {
        let buf = mono_buf(vec![0.1, 0.2, 0.3]);
        let mut sig = MonoSignal::new(&buf);

        assert!(!sig.is_exhausted());
        assert_eq!(sig.next(), [0.1_f32]);
        assert_eq!(sig.next(), [0.2_f32]);
        assert_eq!(sig.next(), [0.3_f32]);
        assert!(sig.is_exhausted());
        // Post-exhaustion: equilibrium.
        assert_eq!(sig.next(), [0.0_f32]);
    }

    #[test]
    fn test_mono_signal_remaining_frames() {
        let buf = mono_buf(vec![0.1, 0.2, 0.3, 0.4]);
        let mut sig = MonoSignal::new(&buf);
        assert_eq!(sig.total_frames(), 4);
        assert_eq!(sig.remaining_frames(), 4);
        let _ = sig.next();
        assert_eq!(sig.remaining_frames(), 3);
    }

    #[test]
    fn test_mono_signal_to_buffer_roundtrip() {
        let original = vec![0.1_f32, -0.2, 0.3, -0.4];
        let buf = mono_buf(original.clone());
        let sig = MonoSignal::new(&buf);
        let recovered = mono_signal_to_buffer(sig, 44_100);
        assert_eq!(recovered.samples, original);
        assert_eq!(recovered.channels, ChannelLayout::Mono);
        assert_eq!(recovered.sample_rate, 44_100);
    }

    #[test]
    fn test_stereo_signal_yields_correct_frames() {
        // Interleaved: L0=1.0, R0=2.0, L1=3.0, R1=4.0
        let buf = stereo_buf(vec![1.0, 2.0, 3.0, 4.0]);
        let mut sig = StereoSignal::new(&buf);

        assert!(!sig.is_exhausted());
        assert_eq!(sig.next(), [1.0_f32, 2.0]);
        assert_eq!(sig.next(), [3.0_f32, 4.0]);
        assert!(sig.is_exhausted());
        // Post-exhaustion: equilibrium.
        assert_eq!(sig.next(), [0.0_f32, 0.0]);
    }

    #[test]
    fn test_stereo_signal_remaining_frames() {
        let buf = stereo_buf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut sig = StereoSignal::new(&buf);
        assert_eq!(sig.total_frames(), 3);
        assert_eq!(sig.remaining_frames(), 3);
        let _ = sig.next();
        assert_eq!(sig.remaining_frames(), 2);
    }

    #[test]
    fn test_stereo_signal_to_buffer_roundtrip() {
        let original = vec![0.1_f32, -0.1, 0.2, -0.2, 0.3, -0.3];
        let buf = stereo_buf(original.clone());
        let sig = StereoSignal::new(&buf);
        let recovered = stereo_signal_to_buffer(sig, 44_100);
        assert_eq!(recovered.samples, original);
        assert_eq!(recovered.channels, ChannelLayout::Stereo);
        assert_eq!(recovered.sample_rate, 44_100);
    }

    #[test]
    fn test_mono_signal_empty_buffer() {
        let buf = mono_buf(vec![]);
        let mut sig = MonoSignal::new(&buf);
        assert!(sig.is_exhausted());
        assert_eq!(sig.next(), [0.0_f32]);
    }

    #[test]
    fn test_stereo_signal_empty_buffer() {
        let buf = stereo_buf(vec![]);
        let mut sig = StereoSignal::new(&buf);
        assert!(sig.is_exhausted());
        assert_eq!(sig.next(), [0.0_f32, 0.0]);
    }

    #[test]
    fn test_mono_until_exhausted_count() {
        let buf = mono_buf(vec![0.0_f32; 100]);
        let sig = MonoSignal::new(&buf);
        let frames: Vec<_> = sig.until_exhausted().collect();
        assert_eq!(frames.len(), 100);
    }

    #[test]
    fn test_stereo_until_exhausted_count() {
        // 200 samples = 100 stereo frames
        let buf = stereo_buf(vec![0.0_f32; 200]);
        let sig = StereoSignal::new(&buf);
        let frames: Vec<_> = sig.until_exhausted().collect();
        assert_eq!(frames.len(), 100);
    }
}
