//! Opus hybrid mode encoder (SILK low-band + CELT high-band).
//!
//! RFC 6716 §3.2.3. Hybrid mode uses SILK for the low-frequency range and CELT
//! for the high-frequency range. This module is a structural scaffold; full
//! frequency-domain crossover at 8 kHz is deferred to a future conformant
//! implementation.
//!
//! # TOC byte (RFC 6716 §3.1)
//!
//! Bits 7–3: config (5 bits). Bit 2: stereo. Bits 1–0: frame count code.
//! This module uses config 14 (hybrid mode).

use crate::opus_celt::encode_celt_frame;
use crate::opus_range::RangeEncoder;
use crate::opus_silk::{analyze_silk_frame, encode_silk_frame, SilkBandwidth};

/// TOC byte for hybrid mode (config=14), single frame (code=0).
///
/// Bit layout (RFC 6716 §3.1):
/// - Bits 7–3: config = 14 (hybrid mode)
/// - Bit 2: stereo flag
/// - Bits 1–0: frame count code = 0 (one frame per packet)
pub fn hybrid_toc(stereo: bool) -> u8 {
    let s = u8::from(stereo);
    (14u8 << 3) | (s << 2)
}

/// Select the `SilkBandwidth` appropriate for the given sample rate in hybrid mode.
fn silk_bandwidth(sample_rate: u32) -> SilkBandwidth {
    match sample_rate {
        r if r <= 8_000 => SilkBandwidth::Narrowband,
        r if r <= 12_000 => SilkBandwidth::Mediumband,
        r if r <= 16_000 => SilkBandwidth::Wideband,
        _ => SilkBandwidth::Superwideband,
    }
}

/// Encode one hybrid Opus frame.
///
/// The output is a single Opus packet: TOC byte + SILK layer + CELT layer.
/// The SILK layer encodes the low-frequency content using LP analysis; the
/// CELT layer encodes the full signal using MDCT band energy quantization.
/// Proper 8 kHz crossover filtering is deferred to a future conformant impl.
///
/// # Arguments
/// * `pcm`               — interleaved PCM samples (channels × frame_samples)
/// * `channels`          — 1 or 2
/// * `sample_rate`       — 8000, 12000, 16000, or 24000/48000 Hz
/// * `target_bitrate_kbps` — 16..512 kbps (accepted for API compatibility)
pub fn encode_hybrid_frame(
    pcm: &[f32],
    channels: usize,
    sample_rate: u32,
    target_bitrate_kbps: u32,
) -> Vec<u8> {
    let stereo = channels > 1;
    let mut out = Vec::new();
    out.push(hybrid_toc(stereo));

    // SILK layer: encode the low-frequency band via LP analysis.
    let silk_bytes = encode_silk_layer(pcm, channels, sample_rate);
    out.extend_from_slice(&silk_bytes);

    // CELT layer: encode the high-frequency band via MDCT band quantization.
    let celt_bytes = encode_celt_layer(pcm, channels, target_bitrate_kbps);
    out.extend_from_slice(&celt_bytes);

    out
}

/// Encode the SILK low-band layer.
///
/// Extracts the first (mono) channel, pads or truncates to the expected SILK
/// frame size for the given bandwidth, analyzes LP parameters, and encodes
/// them with the range coder.
fn encode_silk_layer(pcm: &[f32], channels: usize, sample_rate: u32) -> Vec<u8> {
    let bw = silk_bandwidth(sample_rate);
    let frame_size = bw.frame_size();

    // Extract the first channel (or mono signal) and ensure it is `frame_size` samples long.
    let ch_stride = channels.max(1);
    let available = pcm.len() / ch_stride;

    let mono: Vec<f32> = if available >= frame_size {
        pcm.iter()
            .step_by(ch_stride)
            .take(frame_size)
            .copied()
            .collect()
    } else {
        let mut v: Vec<f32> = pcm.iter().step_by(ch_stride).copied().collect();
        v.resize(frame_size, 0.0);
        v
    };

    let frame = analyze_silk_frame(&mono, bw);
    encode_silk_frame(&frame, bw)
}

/// Encode the CELT high-band layer using the existing CELT scaffold.
///
/// Wraps `encode_celt_frame` with a fresh `RangeEncoder` and returns the flushed bytes.
/// The `_target_bitrate_kbps` parameter is accepted for API symmetry but the current
/// structural CELT encoder uses fixed-width quantization regardless of bitrate.
fn encode_celt_layer(pcm: &[f32], channels: usize, _target_bitrate_kbps: u32) -> Vec<u8> {
    let mut enc = RangeEncoder::new();
    encode_celt_frame(pcm, channels, &mut enc);
    enc.finish()
}

/// Heuristic: return `true` when a PCM frame has sufficient energy to benefit
/// from hybrid mode (i.e., the frame is not silent).
///
/// Silence frames are better served by a pure SILK or CELT encoder with
/// explicit silence flags; the threshold of 1e-6 RMS² corresponds to roughly
/// −60 dBFS, well below audible levels.
pub fn should_use_hybrid(pcm: &[f32]) -> bool {
    let len = pcm.len().max(1);
    let energy: f32 = pcm.iter().map(|&x| x * x).sum::<f32>() / len as f32;
    energy > 1e-6
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{encode_hybrid_frame, hybrid_toc, should_use_hybrid};

    fn sine_pcm(freq: f32, n: usize, rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin() * 0.3)
            .collect()
    }

    #[test]
    fn test_hybrid_toc_mono() {
        let toc = hybrid_toc(false);
        // Config must be 14 in the upper 5 bits (bits 7–3).
        assert_eq!(toc >> 3, 14, "config must be 14 for hybrid mode");
        // Stereo bit (bit 2) must be 0 for mono.
        assert_eq!((toc >> 2) & 1, 0, "stereo bit must be 0 for mono");
        // Frame count code (bits 1–0) must be 0 (one frame per packet).
        assert_eq!(toc & 0x03, 0, "frame count code must be 0");
    }

    #[test]
    fn test_hybrid_toc_stereo() {
        let toc = hybrid_toc(true);
        assert_eq!(toc >> 3, 14, "config must be 14 for hybrid mode");
        assert_eq!((toc >> 2) & 1, 1, "stereo bit must be 1 for stereo");
        assert_eq!(toc & 0x03, 0, "frame count code must be 0");
    }

    #[test]
    fn test_hybrid_toc_value_mono() {
        // Explicit byte-value check: config=14 → bits 7-3 = 0b01110 = 112, stereo=0, code=0.
        // TOC = (14 << 3) | 0 = 112 = 0x70
        assert_eq!(hybrid_toc(false), 0x70);
    }

    #[test]
    fn test_hybrid_toc_value_stereo() {
        // TOC = (14 << 3) | (1 << 2) = 112 | 4 = 116 = 0x74
        assert_eq!(hybrid_toc(true), 0x74);
    }

    #[test]
    fn test_encode_hybrid_frame_starts_with_toc_mono() {
        let pcm = sine_pcm(440.0, 960, 48_000);
        let frame = encode_hybrid_frame(&pcm, 1, 48_000, 64);
        assert!(!frame.is_empty(), "hybrid frame must not be empty");
        assert_eq!(
            frame[0],
            hybrid_toc(false),
            "first byte must be the TOC byte"
        );
    }

    #[test]
    fn test_encode_hybrid_frame_starts_with_toc_stereo() {
        let pcm = sine_pcm(440.0, 1920, 48_000);
        let frame = encode_hybrid_frame(&pcm, 2, 48_000, 96);
        assert!(!frame.is_empty(), "stereo hybrid frame must not be empty");
        assert_eq!(frame[0], hybrid_toc(true), "first byte must be stereo TOC");
    }

    #[test]
    fn test_encode_hybrid_frame_non_empty_for_sine() {
        let pcm = sine_pcm(1000.0, 960, 48_000);
        let frame = encode_hybrid_frame(&pcm, 1, 48_000, 128);
        // Must have TOC byte + at least SILK bytes + at least CELT bytes.
        assert!(
            frame.len() > 1,
            "hybrid frame must contain more than just the TOC byte"
        );
    }

    #[test]
    fn test_encode_hybrid_frame_silence_non_empty() {
        let pcm = vec![0.0f32; 960];
        let frame = encode_hybrid_frame(&pcm, 1, 48_000, 64);
        assert!(
            !frame.is_empty(),
            "silence hybrid frame must still produce output"
        );
        assert_eq!(frame[0], hybrid_toc(false), "silence TOC must be correct");
    }

    #[test]
    fn test_encode_hybrid_frame_various_sample_rates() {
        for &(rate, channels) in &[(8_000u32, 1usize), (16_000, 1), (24_000, 2), (48_000, 1)] {
            let n = 960 * channels;
            let pcm = sine_pcm(440.0, n, rate);
            let frame = encode_hybrid_frame(&pcm, channels, rate, 64);
            assert!(
                !frame.is_empty(),
                "hybrid frame must not be empty for rate={rate} ch={channels}"
            );
            let expected_toc = hybrid_toc(channels > 1);
            assert_eq!(
                frame[0], expected_toc,
                "TOC mismatch for rate={rate} ch={channels}"
            );
        }
    }

    #[test]
    fn test_should_use_hybrid_non_silent() {
        let pcm = sine_pcm(440.0, 960, 48_000);
        assert!(
            should_use_hybrid(&pcm),
            "non-silent frame should use hybrid"
        );
    }

    #[test]
    fn test_should_use_hybrid_silence() {
        let pcm = vec![0.0f32; 960];
        assert!(
            !should_use_hybrid(&pcm),
            "silent frame should not use hybrid"
        );
    }

    #[test]
    fn test_should_use_hybrid_very_quiet_signal() {
        // Amplitude 0.001 → energy = 0.001^2 * 0.5 ≈ 5e-7 < 1e-6 → not hybrid.
        let pcm: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.001)
            .collect();
        // RMS² ≈ 5e-7 which is below the 1e-6 threshold.
        assert!(
            !should_use_hybrid(&pcm),
            "very quiet frame should not use hybrid"
        );
    }
}
