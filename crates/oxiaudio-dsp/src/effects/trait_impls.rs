//! # ChannelVocoder - Trait Implementations
//!
//! This module contains trait implementations for `ChannelVocoder`.
//!
//! ## Implemented Traits
//!
//! - `AudioFilter`
//! - `AudioFilter`
//! - `AudioFilter`
//! - `AudioFilter`
//! - `Default`
//! - `AudioFilter`
//! - `AudioFilter`
//! - `AudioFilter`
//! - `AudioFilter`
//! - `AudioFilter`
//! - `AudioFilter`
//! - `AudioFilter`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxiaudio_core::{AudioBuffer, AudioFilter, OxiAudioError};

use super::types::{
    ChannelVocoder, Chorus, ConvolutionReverb, DelayLine, EarlyReflections, Flanger, Freeverb,
    PartitionedConvolutionReverb, Phaser, Tremolo, Vibrato,
};

impl oxiaudio_core::AudioFilter for ChannelVocoder {
    /// Identity pass-through when used as a single-input filter.
    ///
    /// `ChannelVocoder` requires two inputs (modulator + carrier); the two-input
    /// API is exposed via [`ChannelVocoder::process`]. When invoked through the
    /// single-input `AudioFilter` trait, the buffer is returned unchanged.
    fn apply(
        &self,
        buf: &oxiaudio_core::AudioBuffer<f32>,
    ) -> Result<oxiaudio_core::AudioBuffer<f32>, oxiaudio_core::OxiAudioError> {
        Ok(buf.clone())
    }
}

impl AudioFilter for Chorus {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for ConvolutionReverb {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for DelayLine {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl Default for EarlyReflections {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for EarlyReflections {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for Flanger {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for Freeverb {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for PartitionedConvolutionReverb {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for Phaser {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for Tremolo {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl AudioFilter for Vibrato {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}
