//! Extracted from oxiaudio-encode/src/aac.rs to break the encode↔decode dev-dependency cycle.
//!
//! These ADTS round-trip tests encode with `oxiaudio_encode` and decode with
//! `oxiaudio_decode`, so they live in the integration-tests crate that dev-depends on both.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

fn make_buf(channels: ChannelLayout, sample_rate: u32, frames: usize) -> AudioBuffer<f32> {
    let n = frames * channels.channel_count();
    AudioBuffer {
        samples: vec![0.0f32; n],
        sample_rate,
        channels,
        format: SampleFormat::F32,
    }
}

/// ADTS roundtrip: encode silence → decode with `decode_aac` → no error.
///
/// Uses `oxiaudio_decode::decode_aac`, which parses ADTS frames directly
/// without Symphonia's probe step. This validates the silence ICS bitstream is
/// correctly structured (pulse/tns/gain-control flag bits present, section data valid).
#[test]
fn test_aac_adts_symphonia_decode_roundtrip_silence() {
    use oxiaudio_decode::decode_aac;

    let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
    let mut encoded = Vec::new();
    oxiaudio_encode::encode_aac(&buf, &mut encoded).expect("encode silence must succeed");

    // Verify ADTS sync before decoding
    assert_eq!(encoded[0], 0xFF, "ADTS sync byte 0");
    assert_eq!(encoded[1], 0xF1, "ADTS sync byte 1 (no CRC, MPEG-4)");

    let decoded = decode_aac(&encoded).expect("decode_aac must successfully decode silence ADTS");
    assert_eq!(decoded.sample_rate, 44100, "sample rate must round-trip");
    assert_eq!(
        decoded.channels.channel_count(),
        1,
        "mono channel count must round-trip"
    );
}

/// ADTS roundtrip: encode sine wave → decode with `decode_aac` → non-empty samples.
///
/// Tests the real spectral data path (CB11 Huffman coding). If the ICS bitstream
/// has wrong scale factor counts or missing flag bits, `decode_aac` will fail or
/// return empty/zero output.
#[test]
fn test_aac_adts_symphonia_decode_roundtrip_sine() {
    use oxiaudio_decode::decode_aac;

    let sine_samples: Vec<f32> = (0..2048)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
        .collect();
    let buf = AudioBuffer {
        samples: sine_samples,
        sample_rate: 44100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let mut encoded = Vec::new();
    oxiaudio_encode::encode_aac(&buf, &mut encoded).expect("encode sine must succeed");

    assert_eq!(encoded[0], 0xFF, "ADTS sync byte 0");
    assert_eq!(encoded[1], 0xF1, "ADTS sync byte 1");

    let decoded = decode_aac(&encoded)
        .expect("decode_aac must successfully decode sine ADTS with CB11 coding");
    assert_eq!(
        decoded.sample_rate, 44100,
        "sample rate must round-trip for sine"
    );
    assert_eq!(
        decoded.channels.channel_count(),
        1,
        "mono must round-trip for sine"
    );
    // Decoded output must be non-trivial (some samples must be produced)
    assert!(
        !decoded.samples.is_empty(),
        "decoded samples must be non-empty"
    );
}
