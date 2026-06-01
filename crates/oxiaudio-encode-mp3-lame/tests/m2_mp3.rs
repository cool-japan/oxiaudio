#![cfg(feature = "mp3-encode-lame")]

use oxiaudio_core::{AudioBuffer, AudioDecoder, AudioEncoder, ChannelLayout, SampleFormat};
use oxiaudio_decode::SymphoniaDecoder;
use oxiaudio_encode_mp3_lame::lame::{LameMode, LameMp3Encoder};
use std::io::Cursor;

fn sine_stereo_buffer(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
    let n_frames = (sample_rate as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(n_frames * 2);
    for i in 0..n_frames {
        let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5;
        samples.push(s);
        samples.push(s);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

#[test]
fn test_lame_default_values() {
    let enc = LameMp3Encoder::default();
    assert_eq!(enc.bitrate, 128);
    assert_eq!(enc.mode, LameMode::JointStereo);
}

#[test]
fn test_lame_roundtrip() {
    let original = sine_stereo_buffer(44_100, 0.5);
    let mut enc = LameMp3Encoder::default();
    let mut mp3_bytes = Cursor::new(Vec::new());
    enc.encode(&original, &mut mp3_bytes)
        .expect("encode should succeed");

    let mp3_data = mp3_bytes.into_inner();
    assert!(!mp3_data.is_empty(), "MP3 output should not be empty");

    // Decode the MP3 and verify basic properties.
    let cursor = Cursor::new(mp3_data);
    let mut dec = SymphoniaDecoder;
    let decoded = dec.decode(cursor).expect("decode MP3 should succeed");
    assert_eq!(decoded.sample_rate, 44_100, "sample rate must be preserved");

    // RMS > 0 (not silent).
    let rms: f32 =
        (decoded.samples.iter().map(|s| s * s).sum::<f32>() / decoded.samples.len() as f32).sqrt();
    assert!(rms > 1e-3, "decoded audio should not be silent, rms={rms}");
}

#[test]
fn test_lame_mono_input() {
    let mono_buf = AudioBuffer {
        samples: vec![0.1f32; 4410],
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let mut enc = LameMp3Encoder {
        bitrate: 128,
        mode: LameMode::Mono,
        id3_tags: None,
    };
    let mut out = Cursor::new(Vec::new());
    enc.encode(&mono_buf, &mut out)
        .expect("mono encode should succeed");
    assert!(!out.into_inner().is_empty());
}

#[test]
fn test_lame_unsupported_bitrate_errors() {
    let buf = sine_stereo_buffer(44_100, 0.1);
    let mut enc = LameMp3Encoder {
        bitrate: 999,
        mode: LameMode::JointStereo,
        id3_tags: None,
    };
    let mut out = Cursor::new(Vec::new());
    let result = enc.encode(&buf, &mut out);
    assert!(
        result.is_err(),
        "unsupported bitrate should return an error"
    );
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("999"),
        "error message should mention the bad bitrate, got: {err_str}"
    );
}
