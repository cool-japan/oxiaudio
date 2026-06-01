//! M2 facade integration tests: FLAC encode, format probe, decode-with-metadata.

use oxiaudio::{AudioBuffer, ChannelLayout, SampleFormat};
use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// Generate a short sine-wave buffer (0.1 s at the given rate/layout).
fn sine_buffer(sample_rate: u32, channels: ChannelLayout) -> AudioBuffer<f32> {
    let n_ch = channels.channel_count();
    let n_frames = (sample_rate as f32 * 0.1) as usize;
    let mut samples = Vec::with_capacity(n_frames * n_ch);
    for i in 0..n_frames {
        let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5;
        for _ in 0..n_ch {
            samples.push(s);
        }
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels,
        format: SampleFormat::F32,
    }
}

#[test]
#[cfg(feature = "pure")]
fn test_m2_encode_flac_roundtrip() {
    let original = sine_buffer(44_100, ChannelLayout::Mono);
    let path = temp_path("oxiaudio_m2_flac_roundtrip.flac");

    oxiaudio::encode_flac(&original, &path).expect("encode_flac should succeed");
    let decoded = oxiaudio::decode_file(&path).expect("decode_file of FLAC should succeed");

    let _ = std::fs::remove_file(&path);

    assert_eq!(
        decoded.sample_rate, original.sample_rate,
        "sample_rate mismatch"
    );

    let min_len = original.samples.len().min(decoded.samples.len());
    assert!(min_len > 0, "decoded buffer is empty");

    for (idx, (o, d)) in original.samples[..min_len]
        .iter()
        .zip(decoded.samples[..min_len].iter())
        .enumerate()
    {
        assert!(
            (o - d).abs() < 1e-4,
            "sample[{idx}] mismatch: original={o} decoded={d}"
        );
    }
}

#[test]
#[cfg(feature = "pure")]
fn test_m2_detect_format_wav() {
    let buf = sine_buffer(48_000, ChannelLayout::Stereo);
    let path = temp_path("oxiaudio_m2_detect_format.wav");

    oxiaudio::encode_wav(&buf, &path).expect("encode_wav should succeed");
    let fmt = oxiaudio::detect_format(&path).expect("detect_format should succeed");

    let _ = std::fs::remove_file(&path);

    assert_eq!(fmt.sample_rate, 48_000, "sample_rate mismatch: {fmt:?}");
    assert_eq!(
        fmt.channels,
        ChannelLayout::Stereo,
        "channels mismatch: {fmt:?}"
    );
}

#[test]
#[cfg(feature = "pure")]
fn test_m2_decode_file_with_metadata_wav() {
    let buf = sine_buffer(44_100, ChannelLayout::Stereo);
    let path = temp_path("oxiaudio_m2_meta_wav.wav");

    oxiaudio::encode_wav(&buf, &path).expect("encode_wav should succeed");
    let (decoded, meta) =
        oxiaudio::decode_file_with_metadata(&path).expect("decode_file_with_metadata failed");

    let _ = std::fs::remove_file(&path);

    assert_eq!(decoded.sample_rate, buf.sample_rate, "sample_rate mismatch");
    assert_eq!(decoded.channels, buf.channels, "channels mismatch");
    // WAV files rarely embed metadata tags — just assert the struct is returned without error
    let _ = meta; // metadata fields may all be None for a plain WAV
}
