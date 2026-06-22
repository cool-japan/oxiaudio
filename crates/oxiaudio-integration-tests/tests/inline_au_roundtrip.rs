//! Extracted from oxiaudio-encode/src/au.rs to break the encode↔decode dev-dependency cycle.
//!
//! This AU round-trip test encodes with `oxiaudio_encode` and decodes with `oxiaudio_decode`,
//! so it lives in the integration-tests crate that dev-depends on both.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

fn sine_mono(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let samples = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Encode a mono 44100 Hz sine and decode it back, verifying sample_rate and fidelity.
#[test]
fn test_au_roundtrip_i16() {
    let buf = sine_mono(44_100, 0.25);
    let tmp = std::env::temp_dir().join("oxiaudio_au_roundtrip_i16.au");

    oxiaudio_encode::encode_au_file(&buf, &tmp, oxiaudio_encode::AuEncoding::I16)
        .expect("encode_au_file failed");

    let decoded = oxiaudio_decode::decode_au_file(&tmp).expect("decode_au_file failed");
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(
        decoded.sample_rate, buf.sample_rate,
        "sample_rate mismatch after roundtrip"
    );
    assert_eq!(
        decoded.samples.len(),
        buf.samples.len(),
        "sample count mismatch after roundtrip"
    );

    // Find the first non-silent sample and verify it is within 2e-4 (I16 quantisation noise).
    let first_nonsilent = buf
        .samples
        .iter()
        .zip(decoded.samples.iter())
        .find(|(&orig, _)| orig.abs() > 1e-3);
    if let Some((&orig, &dec)) = first_nonsilent {
        assert!(
            (orig - dec).abs() < 2e-4,
            "roundtrip error too large: orig={orig} dec={dec} diff={}",
            (orig - dec).abs()
        );
    }
}
