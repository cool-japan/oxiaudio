use std::io::{Cursor, Write as IoWrite};

use oxiaudio_core::{AudioBuffer, AudioDecoder, AudioEncoder, ChannelLayout, SampleFormat};
use oxiaudio_decode::SymphoniaDecoder;
use oxiaudio_encode::FlacEncoder;

fn sine_buffer(freq: f32, sample_rate: u32, channels: u16, duration_secs: f32) -> AudioBuffer<f32> {
    let n_frames = (sample_rate as f32 * duration_secs) as usize;
    let n_channels = channels as usize;
    let mut samples = Vec::with_capacity(n_frames * n_channels);
    for i in 0..n_frames {
        let s = (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5;
        for _ in 0..n_channels {
            samples.push(s);
        }
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: if channels == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        },
        format: SampleFormat::F32,
    }
}

/// Encode an AudioBuffer to FLAC bytes using FlacEncoder, then decode via SymphoniaDecoder
/// using a temp file (avoids the byte_len=None probe limitation with in-memory Cursors).
fn encode_and_decode(buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
    // Encode to in-memory Vec
    let mut enc = FlacEncoder::default();
    let mut flac_bytes = Cursor::new(Vec::new());
    enc.encode(buf, &mut flac_bytes)
        .expect("encode should succeed");
    let data = flac_bytes.into_inner();

    // Write to a temp file so symphonia gets a proper seekable source with byte_len
    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_test_{}.flac",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));
    {
        let mut f = std::fs::File::create(&tmp_path).expect("create temp file");
        f.write_all(&data).expect("write temp file");
    }

    let f = std::fs::File::open(&tmp_path).expect("open temp file");
    let mut dec = SymphoniaDecoder;
    let decoded = dec
        .decode(std::io::BufReader::new(f))
        .expect("decode should succeed");

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp_path);

    decoded
}

#[test]
fn test_flac_default_compression_level() {
    assert_eq!(FlacEncoder::default().compression_level, 5);
}

#[test]
fn test_flac_roundtrip() {
    let original = sine_buffer(440.0, 44_100, 1, 0.1);

    let decoded = encode_and_decode(&original);

    assert_eq!(decoded.sample_rate, original.sample_rate);
    assert!(!decoded.samples.is_empty());
    // f32 → 24-bit i32 → f32: tolerance 1e-4
    let min_len = original.samples.len().min(decoded.samples.len());
    for (orig, dec_s) in original.samples[..min_len]
        .iter()
        .zip(decoded.samples[..min_len].iter())
    {
        assert!(
            (orig - dec_s).abs() < 1e-4,
            "sample mismatch: orig={orig}, dec={dec_s}"
        );
    }
}

#[test]
fn test_flac_silent_buffer() {
    let silent = AudioBuffer {
        samples: vec![0.0_f32; 4410],
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };

    let decoded = encode_and_decode(&silent);

    for &s in &decoded.samples {
        assert!(s.abs() < 1e-4, "expected zero, got {s}");
    }
}
