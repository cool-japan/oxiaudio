//! M13 facade integration tests:
//! `encode_aiff` roundtrip, `encode_stream_flac` FLAC validity,
//! `TranscodeStream` WAV→FLAC, `TranscodeStream` unsupported extension.

use oxiaudio::{AudioBuffer, ChannelLayout, SampleFormat};
use std::f32::consts::PI;

/// Build a mono sine-wave buffer at 44100 Hz, 0.1 s duration.
fn sine_mono_44100(dur_secs: f32) -> AudioBuffer<f32> {
    let sample_rate = 44_100u32;
    let n = (sample_rate as f32 * dur_secs) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Test 1: `encode_aiff` roundtrip — encode a sine buffer to AIFF then decode it back.
#[test]
#[cfg(feature = "pure")]
fn test_encode_aiff_roundtrip() {
    let original = sine_mono_44100(0.1);
    let path = std::env::temp_dir().join("oxiaudio_m13_aiff.aiff");

    // Encode to AIFF.
    oxiaudio::encode_aiff(&original, &path).expect("encode_aiff failed");

    // Decode back using the dedicated AIFF decoder (Symphonia does not have
    // aiff feature enabled in this workspace).
    let decoded = oxiaudio::decode_aiff_file(&path).expect("decode_aiff_file failed");

    // Clean up before assertions.
    let _ = std::fs::remove_file(&path);

    // Sample rate must match exactly.
    assert_eq!(
        decoded.sample_rate, 44_100,
        "sample rate mismatch: got {}",
        decoded.sample_rate
    );

    // Decoded sample count must be within 10% of the original.
    let orig_len = original.samples.len();
    let dec_len = decoded.samples.len();
    let tolerance = orig_len / 10;
    assert!(
        dec_len.abs_diff(orig_len) <= tolerance,
        "sample count too far off: orig={orig_len} decoded={dec_len}"
    );

    // Find the first non-silent sample pair and compare within 1e-3.
    // The sine starts at 0.0 for i=0, so we skip exact-zero samples.
    let eps_silence = 1e-6_f32;
    let first_idx = original.samples.iter().position(|&s| s.abs() > eps_silence);

    if let Some(idx) = first_idx {
        let orig_s = original.samples[idx];
        let dec_s = decoded.samples[idx];
        assert!(
            (orig_s - dec_s).abs() < 1e-3,
            "first non-silent sample mismatch at [{idx}]: orig={orig_s} decoded={dec_s}"
        );
    }
}

/// Test 2: `encode_stream_flac` writes a valid FLAC file (starts with "fLaC" magic).
#[test]
#[cfg(feature = "pure")]
fn test_encode_stream_flac_valid_flac() {
    let sample_rate = 44_100u32;
    let n_chunk = 2205usize; // 0.05 s at 44100 Hz

    let chunk1 = {
        let samples: Vec<f32> = (0..n_chunk)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    };

    let chunk2 = {
        let samples: Vec<f32> = (n_chunk..n_chunk * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    };

    let path = std::env::temp_dir().join("oxiaudio_m13_flac_stream.flac");

    {
        let file = std::fs::File::create(&path).expect("create flac file");
        let writer = std::io::BufWriter::new(file);
        oxiaudio::encode_stream_flac(&[&chunk1, &chunk2], writer, 5)
            .expect("encode_stream_flac failed");
        // writer is dropped here, flushing the BufWriter.
    }

    let bytes = std::fs::read(&path).expect("read flac bytes");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        &bytes[..4],
        b"fLaC",
        "FLAC magic bytes missing; got {:?}",
        &bytes[..4.min(bytes.len())]
    );
    assert!(
        bytes.len() > 4,
        "FLAC output too short: {} bytes",
        bytes.len()
    );
}

/// Test 3: `TranscodeStream` WAV→FLAC produces a valid FLAC file.
#[test]
#[cfg(feature = "pure")]
fn test_transcode_stream_wav_to_flac() {
    let buf = sine_mono_44100(0.1);
    let dir = std::env::temp_dir();
    let wav_path = dir.join("oxiaudio_m13_transcode_src.wav");
    let flac_path = dir.join("oxiaudio_m13_transcode_dst.flac");

    // Encode source WAV.
    oxiaudio::encode_wav(&buf, &wav_path).expect("encode_wav for TranscodeStream test");

    // Transcode WAV → FLAC via TranscodeStream.
    oxiaudio::TranscodeStream::new(&wav_path, &flac_path)
        .expect("TranscodeStream::new")
        .run()
        .expect("TranscodeStream::run");

    let flac_bytes = std::fs::read(&flac_path).expect("read flac output");

    // Clean up both temp files.
    let _ = std::fs::remove_file(&wav_path);
    let _ = std::fs::remove_file(&flac_path);

    assert_eq!(
        &flac_bytes[..4],
        b"fLaC",
        "TranscodeStream output is not a valid FLAC file"
    );
}

/// Test 4: `TranscodeStream` with an unsupported output extension returns `Err` on `.run()`.
#[test]
#[cfg(feature = "pure")]
fn test_transcode_stream_unsupported_extension_errors() {
    // TranscodeStream::new always succeeds (lazy validation); the error surfaces on run().
    let out = std::env::temp_dir().join("output.xyz");
    let ts = oxiaudio::TranscodeStream::new("input.wav", &out)
        .expect("TranscodeStream::new should not fail on construction");

    let result = ts.run();
    assert!(
        result.is_err(),
        "TranscodeStream::run should return Err for unsupported '.xyz' extension"
    );
}
