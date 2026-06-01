//! BOUNDED_FFI adapter crate: MP3 encoding via LAME.
//!
//! This crate is `#![forbid(unsafe_code)]` in the Rust layer. All C FFI is
//! fully isolated inside `mp3lame-encoder` / `mp3lame-sys`. The `*_to_vec`
//! convenience methods on `mp3lame_encoder::Encoder` handle any `unsafe
//! set_len` internally; callers of this crate never touch unsafe code.
//!
//! The `mp3-encode-lame` feature must be explicitly opted into. It is never
//! compiled as part of the default feature set of the `oxiaudio` facade.
#![forbid(unsafe_code)]

pub use oxiaudio_core::{AudioBuffer, AudioEncoder, ChannelLayout, OxiAudioError};

/// Approximate ReplayGain track gain (dB) using RMS loudness.
///
/// Returns the gain to normalise to –18 LUFS (ReplayGain 2.0 reference).
/// Uses `gain = –18 − (rms_dbfs − 3)` where 3 dB accounts for the typical
/// offset between RMS-dBFS and integrated loudness.
/// For accurate EBU R128, compute via `oxiaudio_dsp::loudness_lufs()` instead.
#[must_use]
pub fn compute_replaygain_gain_approx(buf: &AudioBuffer<f32>) -> f32 {
    const RG2_REFERENCE_LUFS: f32 = -18.0;
    if buf.samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = buf.samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / buf.samples.len() as f32).sqrt();
    let rms_db = if rms > 1e-9_f32 {
        20.0 * rms.log10()
    } else {
        -90.0
    };
    RG2_REFERENCE_LUFS - (rms_db - 3.0)
}

#[cfg(test)]
mod replaygain_tests {
    use super::*;
    use crate::{AudioBuffer, ChannelLayout};
    use oxiaudio_core::SampleFormat;

    fn make_buf(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_compute_replaygain_silence_returns_positive_gain() {
        // Silence → rms clamped → approx_lufs = -93 → gain = 75 dB
        let gain = compute_replaygain_gain_approx(&make_buf(vec![0.0f32; 1024]));
        assert!(
            gain > 50.0,
            "silence should yield large positive gain, got {gain}"
        );
    }

    #[test]
    fn test_compute_replaygain_full_scale_returns_negative_gain() {
        // Full-scale sine RMS ≈ 0.707 → rms_db ≈ -3 → approx_lufs ≈ -6 → gain ≈ -12 dB
        use std::f32::consts::TAU;
        let samples: Vec<f32> = (0..4096)
            .map(|i| (TAU * 440.0 * i as f32 / 44_100.0).sin())
            .collect();
        let gain = compute_replaygain_gain_approx(&make_buf(samples));
        assert!(
            gain < 0.0,
            "full-scale sine should yield negative gain, got {gain}"
        );
    }

    #[test]
    fn test_compute_replaygain_empty_buffer_returns_zero() {
        assert_eq!(compute_replaygain_gain_approx(&make_buf(vec![])), 0.0);
    }
}

#[cfg(feature = "mp3-encode-lame")]
pub mod lame;
